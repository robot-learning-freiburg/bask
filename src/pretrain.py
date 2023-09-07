# import warnings
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm

import config as default_config
import utils.logging  # noqa
import wandb
from dataset.dc import DenseCorrespondenceDataset
from encoder import encoder_names, encoder_switch
from encoder.representation_learner import RepresentationLearner
from utils.data_loading import (InfiniteDataIterator,
                                build_infinte_data_iterators)
from utils.misc import (apply_machine_config, import_config_file,
                        load_replay_memory, pretrain_checkpoint_name)
from utils.observation import MaskTypes
from utils.random import configure_seeds
from utils.select_gpu import device

# warnings.filterwarnings("error")  # For getting stack trace on warnings.


def run_training(encoder: RepresentationLearner,
                 replay_memory: DenseCorrespondenceDataset,
                 config: dict, snapshot_prefix: Path) -> None:

    encoder.train()
    current_loss = np.inf

    collate_func = replay_memory.get_collate_func()

    train_iterator, val_iterator = build_infinte_data_iterators(
        replay_memory, collate_func, config['dataset_config'],
        train_workers=0, val_workers=0)

    obs_total = len(replay_memory)
    train_frac = config['dataset_config']['train_split']
    no_train_obs = int(train_frac * obs_total)
    no_eval_obs = int((1 - train_frac) * obs_total)

    save_freq = config["training_config"]["save_freq"]
    eval_freq = config["training_config"]["eval_freq"]
    early_stopping = config["training_config"]["auto_early_stopping"]
    n_steps = config["training_config"]["steps"]

    try:
        for i in tqdm(range(n_steps)):
            training_metrics = run_training_step(
                encoder, config, train_iterator, no_train_obs)

            wandb.log(training_metrics, step=i)

            if i % eval_freq == 0 and val_iterator is not None:
                last_loss = current_loss

                eval_metrics = run_eval_step(
                    encoder, config, val_iterator, no_eval_obs)

                wandb.log(eval_metrics, step=i)

                current_loss = eval_metrics['eval-loss'].cpu().numpy()

                if current_loss > last_loss and early_stopping:
                    logger.info("Started to overfit. Interrupting training.")
                    break

            if save_freq and (i % save_freq == 0):
                file_name = (snapshot_prefix.parent / (
                    snapshot_prefix.name + "_step_" + str(i))
                ).with_suffix(".pt")

                logger.info("Saving intermediate encoder at {}", file_name)
                encoder.to_disk(file_name)

    except KeyboardInterrupt:
        logger.info("Interrupted training.")


def run_training_step(encoder: RepresentationLearner, config: dict,
                      train_iterator: InfiniteDataIterator,
                      no_train_obs: int) -> dict:
    while True:  # try until non-empty correspondence batch
        batch = next(train_iterator)

        training_metrics = encoder.update_params(
                    batch, dataset_size=no_train_obs,
                    batch_size=config["dataset_config"]["batch_size"])

        if training_metrics is not None:
           break

    return training_metrics


def run_eval_step(encoder: RepresentationLearner, config: dict,
                  val_iterator: InfiniteDataIterator,
                  no_eval_obs: int) -> dict:
    encoder.eval()

    with torch.no_grad():
        while True:
            batch = next(val_iterator)

            eval_metrics = encoder.evaluate(
                batch, dataset_size=no_eval_obs,
                batch_size=config["dataset_config"]["eval_batchsize"])

            if eval_metrics is not None:
                break

    encoder.train()

    return eval_metrics


def main(config: dict, save_model: bool = True, path: str | None = None
         ) -> None:

    replay_memory = load_replay_memory(config["dataset_config"], path=path)

    replay_memory.update_camera_crop(config["dataset_config"]["image_crop"])

    Encoder = encoder_switch[config["policy_config"]["encoder"]]
    encoder = Encoder(config["policy_config"]["encoder_config"]).to(device)

    wandb.watch(encoder, log_freq=100)

    replay_memory = DenseCorrespondenceDataset(replay_memory,
                                               config["dataset_config"],
                                               sample_type=encoder.sample_type)

    encoder.initialize_image_normalization(
        replay_memory, config["dataset_config"]["cameras"])

    if config["dataset_config"]["contrast_set"]:
        raise NotImplementedError("Need to update contrast set loading. "
                                  "See release-branche for old code.")

    file_name = pretrain_checkpoint_name(config)
    snapshot_prefix = file_name.with_suffix('')
    logger.info("Using snapshot prefix " + str(snapshot_prefix))

    run_training(encoder, replay_memory, config, snapshot_prefix)

    if save_model:
        logger.info("Saving current model checkpoint.")
        encoder.to_disk(file_name)



if __name__ == "__main__":
    parser = ArgumentParser()
    # TODO: pretrain on multiple tasks? Q-scheduling?
    parser.add_argument(
        "-t", "--task",
        default="PhoneOnBase",
        help="Name of the task to train on.",
    )
    parser.add_argument(
        "-f", "--feedback_type",
        default="pretrain_manual",
        help="The training data type. Cloning, dcm, ..."
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Path to a dataset. May be provided instead of f-t.",
    )
    parser.add_argument(
        "--contrast_set",
        default=None,
        help="Optional: path to an additional contrast dataset.",
    )
    parser.add_argument(
        "-e", "--encoder",
        help=f"Options: {str(encoder_names)[1:-1]}"
    )
    parser.add_argument(
        "--encoder_suffix",
        default=None,
        help="Pass a suffix to append to the name of the encoder checkpoint."
    )
    parser.add_argument(
        "--steps",
        default=None,
        help="Number of gradient update steps."
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        help="Config file to use. Uses default if None provided.",
    )
    parser.add_argument(
        "--cam",
        required=True,
        nargs='+',
        help="The camera(s) to use. Options: wrist, overhead, ..."
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=None,
        help="Specify random seed for fair evalaution."
    )
    # TODO: have some more elegant solution to determining the obs size
    # Can extract from data set once loaded.
    parser.add_argument(
        "--panda",
        dest="panda",
        action="store_true",
        default=False,
        help="Data comes from a real world panda -> 480px obs.",
    )
    parser.add_argument(
        "--offline_wandb",
        action="store_true",
        default=False,
        help="Run wandb in offline mode.",
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    seed = configure_seeds(args)

    if args.config is not None:
        conf_file = import_config_file(args.config)
    else:
        conf_file = default_config

    image_dim = default_config.realsense_cam_resolution_cropped if args.panda \
        else default_config.sim_cam_resolution
    image_crop = default_config.realsense_cam_crop if args.panda else None

    selected_encoder_config = conf_file.encoder_configs[args.encoder]
    selected_encoder_config["obs_config"] = {"image_dim": image_dim}

    steps = None if args.steps is None else int(args.steps)

    mask_type = MaskTypes.GT  # MaskTypes.TSDF

    config = {
        "training_config": {
            "steps": steps or int(1e5),
            "eval_freq": 5,
            "save_freq": int(steps // 5) if steps else 20000,
            "auto_early_stopping": False,
            "seed": seed,
        },

        "policy_config": {  # wrapping encoder conf for unified structure
            "encoder": args.encoder,
            "encoder_config": selected_encoder_config,
            "encoder_suffix": args.encoder_suffix,
        },
        "dataset_config": {
            "feedback_type": args.feedback_type,
            "task": args.task,

            "only_use_first_object_label": (args.encoder == "keypoints_gt" and
                                            args.task == "TakeLidOffSaucepan"),
            "conflate_so_object_labels": mask_type is MaskTypes.GT,

            "mask_type": mask_type,

            "use_object_pose": False,  # NOTE: only works in simulation

            "cameras": tuple(args.cam),
            "contr_cam": ("overhead",),

            "contrast_set": args.contrast_set,

            "data_root": "data",

            "image_crop": image_crop,

            "train_split": 0.7,
            "batch_size": 16,
            "eval_batchsize": 160,
        },

    }

    config["dataset_config"].update(selected_encoder_config.get("pretrain", {}).get(
        "dataset_config", {}))
    custom_training_config = selected_encoder_config.get("pretrain", {}).get(
        "training_config", {})
    if args.steps is not None:
        if "steps" in custom_training_config:
            custom_training_config.pop("steps")  # dont override cli arg
    config["training_config"].update(custom_training_config)

    config = apply_machine_config(config)

    wandb_mode = "disabled" if args.disable_wandb else \
        'offline' if args.offline_wandb else "online"
    wandb.init(config=config, project="bask_pretrain", mode=wandb_mode)
    wandbconfig = wandb.config  # in case the sweep gives different values

    # NOTE: pass raw config so wandb does not fuck up the values
    main(config, save_model=True, path=args.path)
