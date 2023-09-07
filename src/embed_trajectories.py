import random
from argparse import ArgumentParser

import torch
from loguru import logger
from tqdm.auto import tqdm

import config as default_config
import utils.logging  # noqa
from dataset.bc import BCDataset
from encoder import encoder_names, encoder_switch
from encoder.representation_learner import RepresentationLearner
from utils.data_loading import build_data_loaders
from utils.misc import (apply_machine_config, import_config_file,
                        load_replay_memory, pretrain_checkpoint_name)
from utils.observation import MaskTypes, collate
from utils.random import configure_seeds
from utils.select_gpu import device


@torch.no_grad()  # otherwise there's a memory leak somwhere. shoulf fix!
def embed_trajectories(encoder: RepresentationLearner, replay_memory: BCDataset,
                       config: dict, encoder_name: str) -> None:

    collate_func = collate  # works as we don't need padding for bs=1.

    train_loader, _ = build_data_loaders(replay_memory, collate_func,
                                         config['dataset_config'],
                                         shuffle=False)
    train_iterator = iter(train_loader)

    cam_names = config["dataset_config"]["cameras"]
    n_cams = len(cam_names)

    logger.info("Beginning embedding.")

    for traj_no, batch in tqdm(enumerate(train_iterator)):
        batch = batch.to(device)

        time_steps = batch.shape[1]

        for step in tqdm(range(time_steps), leave=False):

            obs = batch[:, step, ...]
            embedding, info = encoder.encode(obs)

            if config["policy_config"]["encoder"] == "keypoints":

                save_descriptor(replay_memory, encoder_name, cam_names,
                                traj_no, step, info)

            else:
                save_encoding(replay_memory, config, encoder_name, cam_names,
                              n_cams, traj_no, step, embedding, info)

def save_encoding(replay_memory: BCDataset, config: dict, encoder_name:str,
                  cam_names: tuple[str], n_cams: int, traj_no: int,
                  obs_no: int, embedding: torch.Tensor, info: dict) -> None:

    cam_embeddings = embedding.squeeze(0).detach().chunk(n_cams, -1)

    if config["policy_config"]["encoder"] == "transporter":
        heatmaps = [h.squeeze(0).detach().cpu() for h in info["heatmap"]]

    for i, cn, e in zip(range(n_cams), cam_names, cam_embeddings):
        replay_memory.add_embedding(traj_no, obs_no, cn, "descriptor", e,
                                    encoder_name)

        if config["policy_config"]["encoder"] == "transporter":
            replay_memory.add_embedding(traj_no, obs_no, cn, "heatmap",
                                        heatmaps[i], encoder_name)


def save_descriptor(replay_memory: BCDataset, encoder_name: str,
                    cam_names: tuple[str], traj_no: int, obs_no: int,
                    info: dict) -> None:

    descriptor = (e.squeeze(0).detach() for e in info["descriptor"])

    for cn, d in zip(cam_names, descriptor):
        replay_memory.add_embedding(traj_no, obs_no, cn, "descriptor", d,
                                    encoder_name)


def main(config: dict, path: str | None = None,
         copy_selection_from: str | None = None) -> None:

    replay_memory = load_replay_memory(config["dataset_config"], path=path)
    replay_memory.update_camera_crop(config["dataset_config"]["image_crop"])

    replay_memory = BCDataset(replay_memory, config['dataset_config'])

    Encoder = encoder_switch[config["policy_config"]["encoder"]]
    encoder = Encoder(config["policy_config"]["encoder_config"]).to(device)

    file_name = pretrain_checkpoint_name(config)
    encoder.from_disk(file_name)

    encoder_name = file_name.with_suffix('').parts[-1]

    if config["policy_config"]["encoder"] == "keypoints":
        # Set ref descriptors to zero to avoid out-of-bounds erros.
        # That's easier than skipping the kp-computation for embedding.
        encoder._reference_descriptor_vec = torch.zeros_like(
            encoder._reference_descriptor_vec)
    else:
        # kp_gt needs dataset init
        if (ckpt := copy_selection_from) is not None:
            logger.info("Copying reference positions and descriptors from {}",
                        ckpt)
            state_dict = torch.load(ckpt, map_location='cpu')
            for attr in ['ref_pixels_uv', 'ref_pixel_world',
                         'ref_object_pose', 'ref_depth', 'ref_int', 'ref_ext']:
                setattr(encoder, attr, state_dict[attr].to(device))
        else:
            encoder.initialize_parameters_via_dataset(
                replay_memory, config["dataset_config"]["cameras"][0])

    encoder.eval()

    embed_trajectories(encoder, replay_memory, config, encoder_name)

    # save if kp_gt model as we need the reference selection
    if config["policy_config"]["encoder"] == 'keypoints_gt':
        encoder.to_disk(file_name)


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
        "-e", "--encoder",
        default=None,
        help=f"Options: {str(encoder_names)[1:-1]}"
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
        "--copy_selection_from",
        required=False,
        help="Path to an encoder from which to copy the reference uvs, vecs."
             "For GT-KP model only (to compare different projections)."
    )
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

    encoder_for_conf = args.encoder or "dummy"

    # only needed for GT_KP model anyway.
    mask_type = None if args.panda else MaskTypes.GT

    if args.encoder == "keypoints_gt" and len(args.cam) > 1:
        logger.error("Should only use one camera for GT KP encoder.")
        exit(1)

    image_dim = default_config.realsense_cam_resolution_cropped if args.panda \
        else default_config.sim_cam_resolution
    image_crop = default_config.realsense_cam_crop if args.panda else None

    selected_encoder_config = encoder_configs[encoder_for_conf]
    selected_encoder_config["obs_config"] = {"image_dim": image_dim}

    config = {
        "training_config": {
            "seed": seed,
        },

        "policy_config": {
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

            "only_use_first_object_label": (args.encoder == "keypoints_gt" and
                                            args.task == "TakeLidOffSaucepan"),
            "conflate_so_object_labels":  mask_type is MaskTypes.GT,

            "mask_type": mask_type,

            "cameras": tuple(args.cam),
            "image_crop": image_crop,

            "force_load_raw": True,

            "sample_freq": None,  # downsample the trajectories to this freq

            "only_use_labels": [13],

            "data_root": "data",

            "train_split": 1.0,
            "batch_size": 1,  # did not implement sequence padding
            "eval_batchsize": None,  # no eval done for pure embedding
            "fragment_length": -1,
        },
    }

    if args.encoder is None and args.policy == "encoder":
        raise ValueError("Please specify the desired encoder type.")

    config = apply_machine_config(config)

    main(config, args.path, args.copy_selection_from)
