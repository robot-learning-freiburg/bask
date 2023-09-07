import random
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm

import config as default_config
import utils.logging  # noqa
import wandb
from dataset.bc import BCDataset
from encoder import encoder_names
from encoder.keypoints import PriorTypes
from policy import get_policy_class, policy_switch
from policy.policy import Policy
from utils.data_loading import (InfiniteDataIterator,
                                build_infinte_data_iterators)
from utils.logging import indent_logs
from utils.misc import (apply_machine_config, import_config_file,
                        load_replay_memory, policy_checkpoint_name,
                        pretrain_checkpoint_name)
from utils.observation import MaskTypes, collate
from utils.random import configure_seeds
from utils.select_gpu import device
from viz.particle_filter import ParticleFilterViz

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # for better CUDA debugging


def run_training(policy: Policy, replay_memory: BCDataset, config: dict,
                 snapshot_prefix: str) -> None:

    if config['dataset_config']['fragment_length'] is None:
        collate_func = partial(collate, pad=True)
        raise NotImplementedError(
            "Didn't yet implement padded sequence training. Need to pack and "
            "unpack sequences. Also everything we feed into encoder? Or only "
            "image sequence as LSTM will take care of stopping gradients? See "
            "https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html"  # noqa 501
            "https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e")  # noqa 501
    else:
        collate_func = collate

    train_iterator, val_iterator = build_infinte_data_iterators(
        replay_memory, collate_func, config['dataset_config'],
        full_set_eval=True)

    policy.train()

    current_loss = np.inf

    tconfig = config["training_config"]
    eval_freq = tconfig["eval_freq"]
    save_freq = tconfig.get("save_freq", None)
    early_stopping = tconfig["auto_early_stopping"]
    n_steps = tconfig["steps"]

    logger.info("Beginning training.")

    try:
        for i in tqdm(range(1, n_steps + 1)):
            training_metrics = run_training_step(policy, train_iterator)
            wandb.log(training_metrics, step=i)

            if i % eval_freq == 0 and val_iterator is not None:
                last_loss = current_loss

                eval_metrics = run_eval_step(policy, val_iterator)

                wandb.log(eval_metrics, step=i)

                current_loss = eval_metrics['eval-loss'].cpu().numpy()

                if current_loss > last_loss and early_stopping:
                    logger.info("Started to overfit. Interrupting training.")
                    break

            if tconfig.get("save_freq") and (i % save_freq == 0):
                file_name = snapshot_prefix + "_step_" + str(i) + ".pt"
                logger.info("Saving intermediate policy:", file_name)
                with indent_logs():
                    policy.to_disk(file_name)

    except KeyboardInterrupt:
        logger.info("Interrupted training. Saving current model checkpoint.")


def run_eval_step(policy: Policy, val_iterator: InfiniteDataIterator) -> dict:
    policy.eval()

    with torch.no_grad():
        eval_metrics = []

        for batch in val_iterator:
            eval_metrics.append(policy.evaluate(batch.to(device)))

        eval_metrics = {
                        k: torch.cat([d[k].unsqueeze(0)
                                      for d in eval_metrics]).mean()
                        for k in eval_metrics[0]}

    policy.train()

    return eval_metrics

def run_training_step(policy: Policy, train_iterator: InfiniteDataIterator
                      ) -> dict:
    batch = next(train_iterator)

    training_metrics = policy.update_params(batch)

    return training_metrics


def main(config: dict, path: str | None = None) -> None:
    replay_memory = load_replay_memory(config["dataset_config"], path=path)
    replay_memory.update_camera_crop(config["dataset_config"]["image_crop"])

    encoder_checkpoint = pretrain_checkpoint_name(config)
    encoder_name = encoder_checkpoint.with_suffix('').parts[-1]
    config['dataset_config']['encoder_name'] = encoder_name
    replay_memory = BCDataset(replay_memory, config['dataset_config'])

    Policy = get_policy_class(
        config["policy_config"]["policy"],
        disk_read=config["policy_config"]["disk_read"])

    # In kp_encoder_trajectories, we append the kp selection name to the
    # encoder checkpoint. Need to do here AFTER instantiating the replay buffer
    if kp_selection_name := config["dataset_config"]["kp_pre_encoding"]:
        enc_suffix = config["policy_config"]["encoder_suffix"]
        enc_suffix = "" if enc_suffix is None else enc_suffix + "-"
        config["policy_config"]["encoder_suffix"] = \
            enc_suffix + kp_selection_name
    encoder_checkpoint = pretrain_checkpoint_name(config)

    policy = Policy(config["policy_config"],
                        encoder_checkpoint=encoder_checkpoint).to(device)

    # TODO: init_params per train step to enable multi-task learning?
    with indent_logs():
        policy.initialize_parameters_via_dataset(
            replay_memory, config["dataset_config"]["cameras"])

    wandb.watch(policy, log_freq=100)

    file_name, suffix = policy_checkpoint_name(
        config, create_suffix=config["policy_config"]["suffix"] is None)

    wandb.log({"suffix": suffix}, step=0)

    snapshot_prefix = file_name[:-3]

    # TODO: this is quite hacky. Make more elegant.
    if config["policy_config"]["encoder"] == "keypoints" and \
        config["policy_config"]["encoder_config"]["encoder"][
            "prior_type"] is PriorTypes.PARTICLE_FILTER and \
            config["policy_config"]["encoder_config"]["training"].get(
                "debug"):
        policy.encoder.particle_filter_viz = ParticleFilterViz()
        policy.encoder.particle_filter_viz.run()

    run_training(policy, replay_memory, config, snapshot_prefix)

    policy.to_disk(file_name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-t", "--task",
        default="PhoneOnBase",
        help="Name of the task to train on.",
    )
    parser.add_argument(
        "-f", "--feedback_type",
        default="cloning",
        help="The training data type. Usually cloning."
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Path to a dataset. May be provided instead of f-t.",
    )
    parser.add_argument(
        "--pretrained_on",
        default=None,
        help="task on which the encoder was pretrained."
    )
    parser.add_argument(
        "--pretrain_feedback",
        default=None,
        help="The data on which the model was pretrained, eg. pretrain_manual."
    )
    parser.add_argument(
        "-e", "--encoder",
        required=True,
        # default=None,
        help=f"Options: {str(encoder_names)[1:-1]}"
    )
    parser.add_argument(
        "--encoder_suffix",
        default=None,
        help="Pass a suffix to append to the name of the encoder checkpoint."
    )
    parser.add_argument(
        "-p", "--policy",
        default='encoder',
        help="Options: {}".format(str(list(policy_switch.keys()))[1:-1]),
        required=True
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="Pass a suffix to append to the name of the policy checkpoint."
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
        "-s", "--seed",
        default=None,
        help="Specify random seed for fair evalaution."
    )
    parser.add_argument(
        "--pre_embedding",
        action="store_true",
        default=False,
        help="Load the pre-computed embedding instead of running the encoder. "
             "Still specify the encoder as usual to know where to load from."
             "ONLY for NON-kp encoders. For KP encoder use kp_pre_complete."
    )
    parser.add_argument(
        "--kp_pre_complete",
        default=None,
        help="Load these pre-computed keypoint locations instead of running "
             "the encoder. Specify kp-selection via name. ONLY for kp encoder."
             " Still specify the encoder as usual to know where to load from."
    )
    parser.add_argument(
        "--split",
        default="1.0",
    )
    # TODO: have some more elegant solution to determining the obs size
    # Can extract from data set once loaded.
    parser.add_argument(
        "--panda",
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

    logger.info("Processing config:")
    with indent_logs():
        if args.config is not None:
            logger.info("Using config {}", args.config)
            conf_file = import_config_file(args.config)
            encoder_configs = conf_file.encoder_configs
            logger.info("Overwriting encoder config with external values.")
            try:
                policy_config = conf_file.policy_config
                logger.info("Overwriting policy config with external values.")
            except AttributeError:
                logger.info("Found no external policy config, using default.")
                policy_config = default_config.policy_config
        else:
            encoder_configs = default_config.encoder_configs
            policy_config = default_config.policy_config

    encoder_for_conf = args.encoder  # or "dummy"

    steps = None if args.steps is None else int(args.steps)

    image_dim = default_config.realsense_cam_resolution_cropped if args.panda \
        else default_config.sim_cam_resolution

    image_crop = default_config.realsense_cam_crop if args.panda else None

    selected_encoder_config = encoder_configs[encoder_for_conf]
    selected_encoder_config["obs_config"] = {
        "image_dim": image_dim,
        "n_cams": len(args.cam),
        "disk_read_embedding": args.pre_embedding,
        "disk_read_keypoints": args.kp_pre_complete}

    config = {
        "training_config": {
            "steps": steps or 1000,
            "eval_freq": 25,
            "save_freq": 1000,
            "auto_early_stopping": False,
            "seed": seed,
        },

        "policy_config": {
            "policy": args.policy,
            "disk_read": \
                args.pre_embedding or (args.kp_pre_complete is not None),
            "pre_embedding": args.pre_embedding,
            "kp_pre_encoded": args.kp_pre_complete is not None,

            "lstm_layers": 2,
            "n_cams": len(args.cam),
            "use_ee_pose": True,  # else uses joint angles
            "add_gripper_state": False,
            "action_dim": 7,
            "learning_rate": 3e-4,
            "weight_decay": 3e-6,

            "suffix": args.suffix,

            "encoder": args.encoder,
            "encoder_config": selected_encoder_config,
            "encoder_suffix": args.encoder_suffix,
            "end-to-end": False,  # whether or not to train the encoder too
            },

        "dataset_config": {
            "feedback_type": args.feedback_type,
            "task": args.task,

            # strictly, the next two keys are encoder-attributes, but we only
            # need them for checkpoint loading, so keep them here instead.
            "pretrained_on": args.pretrained_on or args.task,
            "pretrain_feedback": args.pretrain_feedback,

            "cameras": tuple(args.cam),
            "mask_type": MaskTypes.NONE,

            "sample_freq": None,  # downsample the trajectories to this freq

            "data_root": "data",

            "force_skip_rgb": args.encoder == "keypoints_gt",

            "extra_attr": {},

            "only_use_labels": [13],

            "image_crop": image_crop,

            "train_split": float(args.split),
            "batch_size": 16,  # NOTE batches won't be larger than n_trajs
            "eval_batchsize": 15,  # NOTE batches won't be larger than n_trajs
            "fragment_length": 30,  # None,

            "pre_embedding": args.pre_embedding,
            "kp_pre_encoding": args.kp_pre_complete,
        },
    }

    # TODO: it's a bit inconsistent that I update the policy_config, but
    # completely overwrite the encoder config
    config["policy_config"].update(policy_config)

    if args.encoder is None and args.policy == "encoder":
        raise ValueError("Please specify the desired encoder type.")

    config = apply_machine_config(config)

    if config["policy_config"]["use_ee_pose"]:
        config["policy_config"]["proprio_dim"] = 7
    else:
        config["policy_config"]["proprio_dim"] = 8

    if config["policy_config"]["add_gripper_state"]:
        config["policy_config"]["proprio_dim"] += 1
        config["dataset_config"]["extra_attr"].append("gripper_state")

    if config["policy_config"]["encoder_config"].get("encoder", {}).get(
            "prior_type") is PriorTypes.PARTICLE_FILTER and \
            not args.kp_pre_complete:
        config["dataset_config"]["force_load_raw"] = True

    wandb_mode = "disabled" if args.disable_wandb else \
        'offline' if args.offline_wandb else "online"
    wandb.init(config=config, project="bask_bc", mode=wandb_mode)
    wandb_config = wandb.config  # in case the sweep gives different values

    main(config, args.path)
