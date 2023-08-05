import random
# import warnings
from argparse import ArgumentParser

import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm

import config as default_config
import utils.logging  # noqa
import wandb
from dataset.dc import DenseCorrespondenceDataset
from encoder import encoder_switch
from utils.constants import MaskTypes
from utils.data_loading import build_data_loaders
from utils.misc import (apply_machine_config, import_config_file,
                        load_replay_memory, load_replay_memory_from_path,
                        pretrain_checkpoint_name, set_seeds)
from utils.select_gpu import device

# from utils.tasks import tasks

# warnings.filterwarnings("error")  # For getting stack trace on warnings.


def run_training(encoder, replay_memory, config):
    encoder.train()
    current_loss = np.inf

    collate_func = replay_memory.get_collate_func()

    train_loader, val_loader = build_data_loaders(replay_memory,
                                                  collate_func,
                                                  config['dataset_config'],
                                                  train_workers=0,
                                                  val_workers=0)

    obs_total = len(replay_memory)
    train_frac = config['dataset_config']['train_split']
    no_train_obs = int(train_frac * obs_total)
    no_eval_obs = int((1 - train_frac) * obs_total)

    train_generator = iter(train_loader)
    val_generator = iter(val_loader) if val_loader else None

    try:
        for i in tqdm(range(config["training_config"]["steps"])):
            try:
                batch = next(train_generator)
            except StopIteration:
                train_generator = iter(train_loader)
                batch = next(train_generator)

            training_metrics = encoder.update_params(
                batch, dataset_size=no_train_obs,
                batch_size=config["dataset_config"]["batch_size"])
            wandb.log(training_metrics, step=i)

            if i % config["training_config"]["eval_freq"] == 0 \
                    and val_generator is not None:
                last_loss = current_loss
                encoder.eval()
                with torch.no_grad():
                    try:
                        batch = next(val_generator)
                    except StopIteration:
                        val_generator = iter(val_loader)
                        batch = next(val_generator)
                    eval_metrics = encoder.evaluate(
                        batch, dataset_size=no_eval_obs,
                        batch_size=config["dataset_config"]["eval_batchsize"])
                    wandb.log(eval_metrics, step=i)

                encoder.train()

                current_loss = eval_metrics['eval-loss'].cpu().numpy()
                if current_loss > last_loss \
                        and config["training_config"]["auto_early_stopping"]:
                    logger.info("Started to overfit. Interrupting training.")
                    break
    except KeyboardInterrupt:
        logger.info("Interrupted training.")
    return


def main(config, raw_config, save_model, path=None):
    # TODO: use unified load_replay_memory call (see behavior_cloning.py)
    if path:
        replay_memory = load_replay_memory_from_path(path)
    else:
        replay_memory = load_replay_memory(config["dataset_config"])
    replay_memory.update_camera_crop(raw_config["dataset_config"]["crop_left"])

    Encoder = encoder_switch[config["policy_config"]["encoder"]]
    encoder = Encoder(raw_config["policy_config"]["encoder_config"]).to(device)
    # print(encoder)
    # breakpoint()
    # torch.autograd.set_detect_anomaly(True)
    wandb.watch(encoder, log_freq=100)

    replay_memory = DenseCorrespondenceDataset(replay_memory,
                                               raw_config["dataset_config"],
                                               sample_type=encoder.sample_type)

    if config["policy_config"]["encoder"] == "keypoints":
        encoder.initialize_image_normalization(replay_memory, args.cam[0])

        if path := raw_config["dataset_config"]["arm_scan"]:
            logger.info("Loading contrast set from {}", path)
            contrast_memory = torch.load(path)
            # contrast_memory.patch_missing_members()
            robot_labels = [31, 34, 35, 39, 40, 41, 42, 43, 46]
            contrast_memory = DenseCorrespondenceDataset(
                contrast_memory, raw_config["policy_config"]["encoder_config"],
                object_labels=robot_labels)
            replay_memory.contrast_set = contrast_memory

    run_training(encoder, replay_memory, config)
    file_name = pretrain_checkpoint_name(config)
    if save_model:
        logger.info("Saving current model checkpoint.")
        encoder.to_disk(file_name)
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    # TODO: pretrain on multiple tasks? Q-scheduling?
    parser.add_argument(
        "-f",
        "--feedback_type",
        dest="feedback_type",
        default="cloning_10",
        help="options: cloning_10, cloning_100",
    )
    parser.add_argument(
        "-t",
        "--task",
        dest="task",
        default="CloseMicrowave",
        # help="options: {}, 'Mixed'".format(str(tasks)[1:-1]),
    )
    parser.add_argument(
        "-e",
        "--encoder",
        dest="encoder",
        default="transporter",
        help="options: transporter, bvae, monet, keypoints"
    )
    parser.add_argument(
        "-w",
        "--disable_wandb",
        dest="disable_wandb",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-d",
        "--disable_model_save",
        dest="disable_model_save",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-m",
        "--mask",
        dest="mask",
        action="store_true",
        default=False,
        help="Use ground truth object masks.",
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default=None,
        help="Config file to use. Uses default if None provided.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        dest="seed",
        default=None,
        help="Specify random seed for fair evalaution."
    )
    parser.add_argument(
        "-a",
        "--arm_scan",
        dest="arm_scan",
        default=None,
        help="Optional: path to a dataset with robot arm scans to contrast "
        "with.",
    )
    parser.add_argument(
        "-o",
        "--object_pose",
        dest="object_pose",
        action="store_true",
        default=False,
        help="Use data with ground truth object positions.",
    )
    parser.add_argument(
        "--path",
        dest="path",
        default=None,
        help="Path to a dataset. May be provided instead of f-t-m.",
    )
    parser.add_argument(
        "--encoder_suffix",
        dest="encoder_suffix",
        default=None,
        help="Pass a suffix to append to the name of the encoder checkpoint."
    )
    parser.add_argument(
        "--cam",
        dest="cam",
        required=True,
        nargs='+',
        help="The camera(s) to use. Options: wrist, overhead."
    )
    parser.add_argument(
        "--steps",
        dest="steps",
        default=None,
        help="Number of gradient update steps."
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
        "--offline",
        dest="offline",
        action="store_true",
        default=False,
        help="Run wandb in offline mode.",
    )
    args = parser.parse_args()

    seed = int(args.seed) if args.seed else random.randint(0, 2000)
    logger.info("Seed: {}", seed)
    set_seeds(seed)

    if args.config is not None:
        conf_file = import_config_file(args.config)
    else:
        conf_file = default_config

    image_dim = (480, 480) if args.panda else (256, 256)
    crop_left = 160 if args.panda else 0

    selected_encoder_config = conf_file.encoder_configs[args.encoder]
    selected_encoder_config["obs_config"] = {
        "image_dim": image_dim
    }

    steps = None if args.steps is None else int(args.steps)

    mask_type = MaskTypes.GT if args.mask or args.object_pose else \
        MaskTypes.TSDF
    # mask_type = MaskTypes.GT

    config = {
        "training_config": {
            "steps": steps or int(1e5),
            "eval_freq": 5,
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

            "ground_truth_mask": args.mask or args.object_pose,
            "ground_truth_object_pose": args.object_pose,

            "only_use_first_object_label": (args.encoder == "keypoints_gt" and
                                            args.task == "TakeLidOffSaucepan"),
            "conflate_so_object_labels": mask_type is MaskTypes.GT,

            "mask_type": mask_type,
            # "mask_type": MaskTypes.TSDF,

            "use_object_pose": False,  # NOTE: only works in simulation

            "cams": args.cam,
            "contr_cam": ("overhead",),

            "arm_scan": args.arm_scan,

            "data_root": "data",

            "crop_left": crop_left,

            "train_split": 1.0,  # 0.7
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

    wandb_mode = "disabled" if args.disable_wandb else 'offline' if args.offline else "online"
    wandb.init(config=config, project="ceiling_pretrain", mode=wandb_mode)
    wandbconfig = wandb.config  # in case the sweep gives different values
    # HACK: also pass raw config so wandb does not fuck up the values
    main(wandbconfig, config, save_model=not args.disable_model_save,
         path=args.path)
