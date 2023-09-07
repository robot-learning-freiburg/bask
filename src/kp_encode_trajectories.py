from argparse import ArgumentParser

import matplotlib.pyplot as plt
from loguru import logger
from tqdm.auto import tqdm

import config as default_config
import utils.logging  # noqa
from dataset.bc import BCDataset
from encoder.keypoints import PriorTypes
from policy.encoder import EncoderPseudoPolicy
from utils.data_loading import build_data_loaders
from utils.misc import (apply_machine_config, import_config_file,
                        load_replay_memory, pretrain_checkpoint_name)
from utils.observation import MaskTypes, collate
from utils.random import configure_seeds
from utils.select_gpu import device
from viz.image_single import figure_emb_with_points_overlay

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # for better CUDA debugging


def encode_trajectories(policy: EncoderPseudoPolicy,
                        replay_memory: BCDataset, config: dict,
                        encoder_name: str, kp_selection_name: str) -> None:

    collate_func = collate  # works as we don't need padding for bs=1.

    train_loader, _ = build_data_loaders(replay_memory,
                                         collate_func,
                                         config['dataset_config'],
                                         shuffle=False)
    train_iterator = iter(train_loader)

    cam_names = config["dataset_config"]["cameras"]
    n_cams = len(cam_names)


    using_pf = config["policy_config"]["encoder_config"]["encoder"].get(
        "prior_type", None) is PriorTypes.PARTICLE_FILTER
    dbg = config["policy_config"]["encoder_config"]["training"].get("debug")

    logger.info("Beginning encoding.")

    for traj_no, batch in tqdm(enumerate(train_iterator)):
        batch = batch.to(device)

        policy.reset_episode()

        time_steps = batch.shape[1]

        for step in tqdm(range(time_steps), leave=False):

            obs = batch[:, step, ...]

            encoding, info = policy.encoder.encode(obs)

            replay_memory.add_encoding(traj_no, step, None, "kp", encoding,
                                       encoder_name, kp_selection_name)

            if using_pf and dbg:
                save_particle_debug(replay_memory, encoder_name,
                                    kp_selection_name, cam_names, traj_no,
                                    step, info)

            elif dbg:
                save_discrete_filter_debug(
                    replay_memory, encoder_name, kp_selection_name, cam_names,
                    traj_no, step, info)


def save_particle_debug(replay_memory: BCDataset, encoder_name: str,
                        kp_selection_name:str, cam_names: tuple[str],
                        traj_no: int, obs_no: int, info: dict) -> None:

    for i, cn in enumerate(cam_names):
        replay_memory.add_encoding(
            traj_no, obs_no, cn, "heatmaps",
            info["particles_2d"][i].squeeze(0), encoder_name,
            kp_selection_name)
        replay_memory.add_encoding(
            traj_no, obs_no, cn, "2d_locations",
            info["keypoints_2d"][i].squeeze(0), encoder_name,
            kp_selection_name)

        # for j, (heatmap, kp_pos) in enumerate(zip(
        #     info["particles_2d"][i].squeeze(0),
        #     info["keypoints_2d"][i].squeeze(0))):
        #     fig, extent = figure_emb_with_points_overlay(
        #         heatmap, kp_pos, None, None, None, is_img=False,
        #         rescale=False, colors='y')

        #     replay_memory.add_encoding_fig(
        #         traj_no, obs_no, cn, "heatmap_" + str(j), fig,
        #         encoder_name, kp_selection_name, bbox=extent)
        #     plt.close(fig)

    replay_memory.add_encoding(
        traj_no, obs_no, None, "particle_var",
        info["particle_var"].squeeze(0), encoder_name,
        kp_selection_name)


def save_discrete_filter_debug(replay_memory: BCDataset, encoder_name: str,
                               kp_selection_name: str, cam_names: tuple[str],
                               traj_no: int, step: int, info: dict) -> None:

    if (prior := info["prior"])[0] is not None:
        prior = (p.squeeze(0) for p in prior)
    else:
        prior = tuple(None for _ in range(len(cam_names)))

    sm = (s.squeeze(0) for s in info["sm"])
    post = (p.squeeze(0) for p in info["post"])

    for cn, pr, so, po, in zip(cam_names, prior, sm, post):
        replay_memory.add_encoding(traj_no, step, cn, "prior", pr,
                                   encoder_name, kp_selection_name)
        replay_memory.add_encoding(traj_no, step, cn, "sm", so,
                                   encoder_name, kp_selection_name)
        replay_memory.add_encoding(traj_no, step, cn, "post", po,
                                   encoder_name, kp_selection_name)

def main(config: dict, path: str | None, kp_selection_name: str,
         copy_selection_from: str | None = None) -> None:

    replay_memory = load_replay_memory(config["dataset_config"], path=path)

    encoder_checkpoint = pretrain_checkpoint_name(config)
    encoder_name = encoder_checkpoint.with_suffix('').parts[-1]
    config['dataset_config']['encoder_name'] = encoder_name

    replay_memory = BCDataset(replay_memory, config['dataset_config'])

    policy = EncoderPseudoPolicy(
        config["policy_config"], encoder_checkpoint=encoder_checkpoint,
        copy_selection_from=copy_selection_from)

    policy.initialize_parameters_via_dataset(
        replay_memory, config["dataset_config"]["cameras"])

    # if config["policy_config"]["encoder"] == "keypoints" and \
    #      config["policy_config"]["encoder_config"]["encoder"][
    #          "prior_type"] is PriorTypes.PARTICLE_FILTER and \
    #         config["policy_config"]["encoder_config"]["training"].get("debug"):
    #     policy.encoder.particle_filter_viz = ParticleFilterViz()
    #     policy.encoder.particle_filter_viz.run()

    encode_trajectories(policy, replay_memory, config, encoder_name,
                        kp_selection_name)

    enc_suffix = config["policy_config"]["encoder_suffix"]
    enc_suffix = "" if enc_suffix is None else enc_suffix + "-"
    config["policy_config"]["encoder_suffix"] = enc_suffix + kp_selection_name

    file_name = pretrain_checkpoint_name(config)

    policy.encoder_to_disk(file_name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-t", "--task",
        help="Name of the task from which to load the data.",
    )
    parser.add_argument(
        "-f", "--feedback_type",
        default="demos",
        help="Name of data type to load.",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Path to a dataset. May be provided instead of f-t.",
    )
    parser.add_argument(
        "--pretrained_on",
        default=None,
        help="Task on which the encoder was pretrained. Defaults to task. "
    )
    parser.add_argument(
        "--pretrain_feedback",
        default='pretrain_manual',
        help="The data on which the model was pretrained.",
    )
    parser.add_argument(
        "--encoder_suffix",
        default=None,
        help="Pass a suffix to append to the name of the encoder checkpoint."
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        help="Config file to use. Uses default if None provided.",
    )
    parser.add_argument(
        "-s", "--seed",
        default=None,
        help="Specify random seed for fair evalaution."
    )
    parser.add_argument(
        "--cam",
        required=True,
        nargs='+',
        help="The camera(s) to use. Options: wrist, overhead, ..."
    )
    parser.add_argument(
        "--selection_name",
        required=True,
        help="Name by which to identify the kp selection."
    )
    parser.add_argument(
        "--copy_selection_from",
        required=False,
        help="Path to an encoder from which to copy the reference uvs, vecs."
    )
    # TODO: have some more elegant solution to determining the obs size
    # Can extract from data set once loaded.
    parser.add_argument(
        "--panda",
        action="store_true",
        default=False,
        help="Data comes from a real world panda -> 480px obs.",
    )
    args = parser.parse_args()

    seed = configure_seeds(args)

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

    mask_type = None if args.panda else MaskTypes.GT

    image_dim = default_config.realsense_cam_resolution_cropped if args.panda \
        else default_config.sim_cam_resolution
    image_crop = default_config.realsense_cam_crop if args.panda else None

    n_cams = len(args.cam)

    selected_encoder_config = encoder_configs["keypoints"]
    selected_encoder_config["obs_config"] = {
        "image_dim": image_dim,
        "n_cams": n_cams,
        "disk_read_embedding": True,
        }

    config = {
        "training_config": {
            "seed": seed,
        },

        "policy_config": {
            "pre_embedded": True,
            "kp_pre_encoded": False,

            "n_cams": n_cams,

            "encoder": "keypoints",
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

            "mask_type": mask_type,

            "cameras": tuple(args.cam),
            "image_crop": image_crop,

            "sample_freq": None,  # downsample the trajectories to this freq

            "only_use_labels": [13],

            "data_root": "data",

            "train_split": 1.0,
            "batch_size": 1,  # did not implement sequence padding
            "eval_batchsize": None,  # no eval done for pure embedding
            "fragment_length": -1,

            "pre_embedding": True,
            "kp_pre_encoding": False,
            "debug_encoding": False,
        },
    }

    config = apply_machine_config(config)

    # if config["policy_config"]["encoder_config"].get("encoder", {}).get(
    #         "prior_type") is PriorTypes.PARTICLE_FILTER:
    #     config["dataset_config"]["force_load_raw"] = True
    # NOTE: also need int, ext for 3D projection. For now just forcing always.
    # TODO: figure out exactly what is needed when.
    config["dataset_config"]["force_load_raw"] = True

    main(config, args.path, args.selection_name, args.copy_selection_from)
