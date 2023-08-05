import random
from argparse import ArgumentParser
from functools import partial

import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm

import config as default_config
import utils.logging  # noqa
import wandb
from dataset.bc import BCDataset
from dataset.dataclasses import collate_bc
from encoder.keypoints import PriorTypes
from policy import get_policy_class, policy_switch
from utils.constants import MaskTypes
from utils.data_loading import build_data_loaders
from utils.misc import (apply_machine_config, import_config_file,
                        load_replay_memory, policy_checkpoint_name,
                        pretrain_checkpoint_name, set_seeds)
from utils.select_gpu import device
# from utils.tasks import tasks
from viz.particle_filter import ParticleFilterViz

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # for better CUDA debugging


def run_training(policy, replay_memory, config, snapshot_prefix):

    if config['dataset_config']['fragment_length'] is None:
        collate_func = partial(collate_bc, pad=True)
        raise NotImplementedError(
            "Didn't yet implement padded sequence training. Need to pack and "
            "unpack sequences. Also everything we feed into encoder? Or only "
            "image sequence as LSTM will take care of stopping gradients? See "
            "https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html"  # noqa 501
            "https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e")  # noqa 501
    else:
        collate_func = collate_bc

    train_loader, val_loader = build_data_loaders(replay_memory,
                                                  collate_func,
                                                  config['dataset_config'])
    train_generator = iter(train_loader)
    val_generator = iter(val_loader) if val_loader else None

    if config["policy_config"]["encoder"] == "keypoints_gt":
        skip = ["cam_rgb", "cam_rgb2"]
    else:
        skip = None
    logger.info("Tensors being skipped in moving batch to device: {}", skip)

    policy.train()

    current_loss = np.inf

    tconfig = config["training_config"]

    logger.info("Beginning training.")
    try:
        for i in tqdm(range(1, tconfig["steps"] + 1)):
            try:
                batch = next(train_generator)
            except StopIteration:
                train_generator = iter(train_loader)
                batch = next(train_generator)
            # from utils.debug import summarize_class, summarize_dataclass
            # summarize_dataclass(batch)
            # exit()
            training_metrics = policy.update_params(
                batch.to(device, skip=skip))
            wandb.log(training_metrics, step=i)

            if i % tconfig["eval_freq"] == 0 and val_generator is not None:
                last_loss = current_loss
                policy.eval()
                with torch.no_grad():
                    eval_metrics = []
                    for batch in val_generator:
                        eval_metrics.append(policy.evaluate(
                            batch.to(device, skip=skip)))
                    eval_metrics = {
                        k: torch.cat([d[k].unsqueeze(0)
                                      for d in eval_metrics]).mean()
                        for k in eval_metrics[0]}
                    wandb.log(eval_metrics, step=i)
                val_generator = iter(val_loader)
                policy.train()

                current_loss = eval_metrics['eval-loss'].cpu().numpy()
                if current_loss > last_loss and tconfig["auto_early_stopping"]:
                    logger.info("Started to overfit. Interrupting training.")
                    break

            if tconfig.get("save_freq") and (i % tconfig["save_freq"] == 0):
                file_name = snapshot_prefix + "_step_" + str(i) + ".pt"
                logger.info("Saving intermediate policy at {}", file_name)
                policy.to_disk(file_name)

    except KeyboardInterrupt:
        logger.info("Interrupted training. Saving current model checkpoint.")
    return


def main(config, raw_config, path=None):
    replay_memory = load_replay_memory(raw_config["dataset_config"], path=path)
    replay_memory.update_camera_crop(raw_config["dataset_config"]["crop_left"])

    encoder_checkpoint = pretrain_checkpoint_name(raw_config)
    encoder_name = encoder_checkpoint.with_suffix('').parts[-1]
    raw_config['dataset_config']['encoder_name'] = encoder_name
    replay_memory = BCDataset(replay_memory, raw_config['dataset_config'])

    Policy = get_policy_class(
        config["policy_config"]["policy"],
        pre_embd=config["policy_config"]["pre_embedded"],
        pre_enc=config["policy_config"]["kp_pre_encoded"])

    # In kp_encoder_trajectories, we append the kp selection name to the
    # encoder checkpoint. Need to do here AFTER instantiating the replay buffer
    if kp_selection_name := raw_config["dataset_config"]["kp_pre_encoding"]:
        enc_suffix = raw_config["policy_config"]["encoder_suffix"]
        enc_suffix = "" if enc_suffix is None else enc_suffix + "-"
        raw_config["policy_config"]["encoder_suffix"] = \
            enc_suffix + kp_selection_name
    encoder_checkpoint = pretrain_checkpoint_name(raw_config)

    policy = Policy(raw_config["policy_config"],
                    encoder_checkpoint=encoder_checkpoint).to(device)

    # TODO: init_params per train step to enable multi-task learning?
    policy.initialize_parameters_via_dataset(replay_memory)

    wandb.watch(policy, log_freq=100)
    file_name, suffix = policy_checkpoint_name(
        raw_config,
        create_suffix=raw_config["policy_config"]["suffix"] is None)
    wandb.log({"suffix": suffix}, step=0)

    snapshot_prefix = file_name[:-3]

    if raw_config["policy_config"]["encoder"] == "keypoints" and \
        raw_config["policy_config"]["encoder_config"]["encoder"][
            "prior_type"] is PriorTypes.PARTICLE_FILTER and \
            raw_config["policy_config"]["encoder_config"]["training"].get(
                "debug"):
        policy.encoder.particle_filter_viz = ParticleFilterViz()
        policy.encoder.particle_filter_viz.run()

    run_training(policy, replay_memory, config, snapshot_prefix)

    logger.info("Saving policy at {}", file_name)
    policy.to_disk(file_name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--feedback_type",
        dest="feedback_type",
        default="cloning_10",
        help="options: cloning_10, cloning_200",
    )
    parser.add_argument(
        "-t",
        "--task",
        dest="task",
        default="CloseMicrowave",
        # help="options: {}, 'Mixed'".format(str(tasks)[1:-1]),
    )
    parser.add_argument(
        "--pretrained_on",
        dest="pretrained_on",
        default=None,
        help="task on which the encoder was pretrained, "
        #      "options: {}, 'Mixed'".format(str(tasks)[1:-1]),
    )
    parser.add_argument(
        "--pretrain_feedback",
        dest="pretrain_feedback",
        default=None,
        help="The data on which the model was pretrained, eg. dcs_20",
    )
    parser.add_argument(
        "-e",
        "--encoder",
        dest="encoder",
        default=None,
        help="options: transporter, bvae, monet, keypoints"
    )
    parser.add_argument(
        "-p",
        "--policy",
        dest="policy",
        default=None,
        help="options: {}".format(str(list(policy_switch.keys()))[1:-1]),
        required=True
    )
    parser.add_argument(
        "-d",
        "--disable_wandb",
        dest="disable_wandb",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-m",
        "--mask",
        dest="mask",
        action="store_true",
        default=False,
        help="Use data with ground truth object masks.",
    )
    parser.add_argument(
        "--path",
        dest="path",
        default=None,
        help="Path to a dataset. May be provided instead of f-t-m.",
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
        "--suffix",
        dest="suffix",
        default=None,
        help="Pass a suffix to append to the name of the policy checkpoint."
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
        "-o",
        "--object_pose",
        dest="object_pose",
        action="store_true",
        default=False,
        help="Use data with ground truth object positions.",
    )
    parser.add_argument(
        "--steps",
        dest="steps",
        default=None,
        help="Number of gradient update steps."
    )
    parser.add_argument(
        "--pre_embedding",
        dest="pre_embedding",
        action="store_true",
        default=False,
        help="Load the pre-computed embedding instead of running the encoder. "
             "Still specify the encoder as usual to know where to load from."
             "ONLY for NON-kp encoders. For KP encoder use kp_pre_complete."
    )
    parser.add_argument(
        "--kp_pre_complete",
        dest="kp_pre_complete",
        default=None,
        help="Load these pre-computed keypoint locations instead of running "
             "the encoder. Specify kp-selection via name. ONLY for kp encoder."
             " Still specify the encoder as usual to know where to load from."
    )
    parser.add_argument(
        "--split",
        dest="split",
        default="1.0",
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

    encoder_for_conf = args.encoder or "dummy"
    if encoder_for_conf == "keypoints_var":
        encoder_for_conf = "keypoints"

    encoder_config = encoder_configs[encoder_for_conf]
    # encoder_config["pre_embedding"] = args.pre_embedding

    steps = None if args.steps is None else int(args.steps)

    # mask_type = MaskTypes.GT if args.mask or args.object_pose else \
    #     MaskTypes.TSDF
    mask_type = MaskTypes.NONE if args.panda else MaskTypes.GT  # only needed for GT_KP model anyway.

    image_dim = (480, 480) if args.panda else (256, 256)
    crop_left = 160 if args.panda else 0

    selected_encoder_config = encoder_configs[encoder_for_conf]
    selected_encoder_config["obs_config"] = {
        "image_dim": image_dim
    }

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
            "pre_embedded": args.pre_embedding,
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
            "encoder_config": encoder_config,
            "encoder_suffix": args.encoder_suffix,
            "end-to-end": False,  # whether or not to train the encoder too
            },

        "dataset_config": {
            "feedback_type": args.feedback_type,
            "task": args.task,

            "ground_truth_mask": args.mask or args.object_pose,
            "ground_truth_object_pose": args.object_pose,
            # strictly, the next two keys are encoder-attributes, but we only
            # need them for checkpoint loading, so keep them here instead.
            "pretrained_on": args.pretrained_on or args.task,
            "pretrain_feedback": args.pretrain_feedback,

            "mask_type": mask_type,

            "cams": args.cam,

            "sample_freq": None,  # downsample the trajectories to this freq

            "data_root": "data",

            "extra_attr": [],

            "crop_left": crop_left,

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

    wandb_mode = "disabled" if args.disable_wandb else 'offline' if args.offline else "online"
    wandb.init(config=config, project="ceiling_bc", mode=wandb_mode)

    wandb_config = wandb.config  # in case the sweep gives different values
    main(wandb_config, config, args.path)
