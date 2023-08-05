import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
from loguru import logger
from tqdm.auto import tqdm

import config as default_config
import utils.logging  # noqa
from dataset.bc import BCDataset
from dataset.dataclasses import collate_bc
from encoder.keypoints import PriorTypes
from policy.encoder import PseudoPreEmbeddedEncoderPolicy
from utils.constants import MaskTypes
from utils.data_loading import build_data_loaders
from utils.misc import (apply_machine_config, import_config_file,
                        load_replay_memory, pretrain_checkpoint_name,
                        set_seeds)
from utils.select_gpu import device
# from utils.tasks import tasks
from viz.image_single import figure_emb_with_points_overlay
from viz.particle_filter import ParticleFilterViz

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # for better CUDA debugging

def encode_trajectories(policy, replay_memory, config, encoder_name,
                        kp_selection_name):

    collate_func = collate_bc  # works as we don't need padding for bs=1.

    train_loader, _ = build_data_loaders(replay_memory,
                                         collate_func,
                                         config['dataset_config'],
                                         shuffle=False)
    train_generator = iter(train_loader)

    cam_names = config["dataset_config"]["cams"]
    n_cams = len(cam_names)

    logger.info("Beginning training.")
    for traj_no, batch in tqdm(enumerate(train_generator)):
        # if traj_no > 2:
        #     break
        batch = batch.to(device)

        try:
            policy.encoder.reset_traj()
        except AttributeError:
            pass

        for step, obs in tqdm(enumerate(batch)):

            encoding, info = policy.encoder.encode(obs.cam_rgb, full_obs=obs)

            replay_memory.add_encoding(traj_no, step, None, "kp", encoding,
                                       encoder_name, kp_selection_name)

            if config["policy_config"]["encoder_config"]["encoder"].get(
                    "prior_type", None) is PriorTypes.PARTICLE_FILTER:
                if config["policy_config"]["encoder_config"][
                    "training"].get("debug"):

                    for i, cn in enumerate(cam_names):
                        replay_memory.add_encoding(
                            traj_no, step, cn, "heatmaps",
                            info["particles_2d"][i].squeeze(0), encoder_name,
                            kp_selection_name)
                        replay_memory.add_encoding(
                            traj_no, step, cn, "2d_locations",
                            info["keypoints_2d"][i].squeeze(0), encoder_name,
                            kp_selection_name)

                        # for j, (heatmap, kp_pos) in enumerate(zip(
                        #     info["particles_2d"][i].squeeze(0),
                        #     info["keypoints_2d"][i].squeeze(0))):
                        #     fig, extent = figure_emb_with_points_overlay(
                        #         heatmap, kp_pos, None, None, None, is_img=False,
                        #         rescale=False, colors='y')

                        #     replay_memory.add_encoding_fig(
                        #         traj_no, step, cn, "heatmap_" + str(j), fig,
                        #         encoder_name, kp_selection_name, bbox=extent)
                        #     plt.close(fig)

                    replay_memory.add_encoding(
                        traj_no, step, None, "particle_var",
                        info["particle_var"].squeeze(0), encoder_name,
                        kp_selection_name)

                continue

            if (prior := info["prior"])[0] is not None:
                prior = (p.squeeze(0) for p in prior)
            else:
                prior = tuple(None for _ in range(n_cams))
            sm = (s.squeeze(0) for s in info["sm"])
            post = (p.squeeze(0) for p in info["post"])

            if config["dataset_config"]["debug_encoding"]:
                dist_per_cam = torch.stack(
                    [d.squeeze(0) for d in info["distance"]])
                best_cam = torch.argmin(dist_per_cam, dim=0)

            for i, cn, pr, so, po, in zip(
                    range(n_cams), cam_names, prior, sm, post):

                replay_memory.add_encoding(traj_no, step, cn, "prior", pr,
                                           encoder_name, kp_selection_name)
                replay_memory.add_encoding(traj_no, step, cn, "sm", so,
                                           encoder_name, kp_selection_name)
                replay_memory.add_encoding(traj_no, step, cn, "post", po,
                                           encoder_name, kp_selection_name)

                if config["dataset_config"]["debug_encoding"]:
                    kp2d = info["kp_raw_2d"][i].squeeze(0)
                    distance = info["distance"][i].squeeze(0)
                    rgb = [obs.cam_rgb, obs.cam_rgb2][i].squeeze(0)

                    threshold = 1  # TODO: threshold kp distance for plt
                    # TODO: scale sm for heatmaps (similar to keypoint select?)

                    best = best_cam == i

                    fig, extent = figure_emb_with_points_overlay(
                        rgb, kp2d, distance, best, threshold=threshold,
                        is_img=True)

                    replay_memory.add_encoding_fig(
                        traj_no, step, cn, "rgb", fig, encoder_name,
                        kp_selection_name, bbox=extent)
                    plt.close(fig)

                    for en, em in zip(tuple(("prior", "sm", "post")),
                                      tuple((pr, so, po))):
                        if em is not None:
                            for c in range(em.shape[0]):
                                kp = kp2d.reshape(2, -1)[:, c]
                                fig, extent = figure_emb_with_points_overlay(
                                    em[c], kp, distance[c].unsqueeze(0),
                                    best[c].unsqueeze(0),
                                    threshold=threshold, is_img=False)

                                replay_memory.add_encoding_fig(
                                    traj_no, step, cn, en, fig, encoder_name,
                                    kp_selection_name, bbox=extent, channel=c)
                                plt.close(fig)


def main(config, path, kp_selection_name, copy_selection_from=None):
    replay_memory = load_replay_memory(config["dataset_config"],
                                       path=path)

    encoder_checkpoint = pretrain_checkpoint_name(config)
    encoder_name = encoder_checkpoint.with_suffix('').parts[-1]
    config['dataset_config']['encoder_name'] = encoder_name

    replay_memory = BCDataset(replay_memory, config['dataset_config'])

    policy = PseudoPreEmbeddedEncoderPolicy(
        config["policy_config"], encoder_checkpoint=encoder_checkpoint,
        copy_selection_from=copy_selection_from)

    # TODO: does this override anything or diverge from what we do in emb_traj?
    policy.initialize_parameters_via_dataset(replay_memory)

    # if config["policy_config"]["encoder"] == "keypoints" and \
    #     config["policy_config"]["encoder_config"]["encoder"][
    #         "prior_type"] is PriorTypes.PARTICLE_FILTER and \
    #         config["policy_config"]["encoder_config"]["training"].get("debug"):
    #     policy.encoder.particle_filter_viz = ParticleFilterViz()
    #     policy.encoder.particle_filter_viz.run()

    encode_trajectories(policy, replay_memory, config, encoder_name,
                        kp_selection_name)

    enc_suffix = config["policy_config"]["encoder_suffix"]
    enc_suffix = "" if enc_suffix is None else enc_suffix + "-"
    config["policy_config"]["encoder_suffix"] = enc_suffix + kp_selection_name

    file_name = pretrain_checkpoint_name(config)
    logger.info("Saving encoder at {}", file_name)
    policy.encodder_to_disk(file_name)


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
        #     "options: {}, 'Mixed'".format(str(tasks)[1:-1]),
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
        "--selection_name",
        dest="selection_name",
        required=True,
        help="Name by which to identify the kp selection."
    )
    parser.add_argument(
        "--copy_selection_from",
        dest="copy_selection_from",
        required=False,
        help="Path to an encoder from which to copy the reference uvs, vecs."
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

    mask_type = None if args.panda else MaskTypes.GT

    if args.encoder != "keypoints":
        logger.error("Use embed_trajectories.py for other encoders.")
        exit(1)

    image_dim = (480, 480) if args.panda else (256, 256)

    selected_encoder_config = encoder_configs[encoder_for_conf]
    selected_encoder_config["obs_config"] = {
        "image_dim": image_dim
    }

    config = {
        "training_config": {
            "seed": seed,
        },

        "policy_config": {
            "pre_embedded": True,
            "kp_pre_encoded": False,

            "n_cams": len(args.cam),

            "encoder": args.encoder,
            "encoder_config": selected_encoder_config,
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

            "train_split": 1.0,
            "batch_size": 1,  # did not implement sequence padding
            "eval_batchsize": None,  # no eval done for pure embedding
            "fragment_length": -1,

            "pre_embedding": True,
            "kp_pre_encoding": False,
            "debug_encoding": False,

            # Keys should not be needed here. NOTE: pass dummy values if crash
            # "only_use_first_object_label": False,  # only use true on Lid, gtkp
            # "conflate_so_object_labels": mask_type is MaskTypes.GT,
        },
    }

    if args.encoder is None and args.policy == "encoder":
        raise ValueError("Please specify the desired encoder type.")

    config = apply_machine_config(config)

    # if config["policy_config"]["encoder_config"].get("encoder", {}).get(
    #         "prior_type") is PriorTypes.PARTICLE_FILTER:
    #     config["dataset_config"]["force_load_raw"] = True
    # NOTE: also need int, ext for 3D projection. For now just forcing always.
    # TODO: figure out exactly what is needed when.
    config["dataset_config"]["force_load_raw"] = True

    main(config, args.path, args.selection_name, args.copy_selection_from)
