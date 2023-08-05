import random
from argparse import ArgumentParser

import torch
from loguru import logger
from tqdm.auto import tqdm

import config as default_config
import utils.logging  # noqa
from dataset.bc import BCDataset
from dataset.dataclasses import collate_bc
from encoder import encoder_switch
from utils.constants import MaskTypes
from utils.data_loading import build_data_loaders
from utils.misc import (apply_machine_config, import_config_file,
                        load_replay_memory, pretrain_checkpoint_name,
                        set_seeds)
from utils.select_gpu import device

# from utils.tasks import tasks


@torch.no_grad()  # otherwise there's a memory leak somwhere. shoulf fix!
def embed_trajectories(encoder, replay_memory, config, encoder_name):

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
        batch = batch.to(device)

        for step, obs in tqdm(enumerate(batch), leave=False):
            if config["policy_config"]["encoder"] == "keypoints":
                encoding, info = encoder.encode(obs.cam_rgb, full_obs=obs)

                embedding = (e.squeeze(0).detach() for e in info["descriptor"])

                for i, cn, e in zip(range(n_cams), cam_names, embedding):

                    replay_memory.add_embedding(traj_no, step, cn,
                                                "descriptor", e,
                                                encoder_name)

            else:
                embedding, info = encoder.encode(obs.cam_rgb, full_obs=obs)
                embedding = embedding.squeeze(0).detach().chunk(n_cams, -1)

                if config["policy_config"]["encoder"] == "transporter":
                    heatmaps = [h.squeeze(0).detach().cpu()
                                for h in info["heatmap"]]

                for i, cn, e in zip(range(n_cams), cam_names, embedding):

                    replay_memory.add_embedding(traj_no, step, cn,
                                                "descriptor", e,
                                                encoder_name)
                    if config["policy_config"]["encoder"] == "transporter":

                        replay_memory.add_embedding(traj_no, step, cn,
                                                    "heatmap", heatmaps[i],
                                                    encoder_name)


def main(config, path=None, copy_selection_from=None):
    replay_memory = load_replay_memory(config["dataset_config"],
                                       path=path)
    replay_memory = BCDataset(replay_memory, config['dataset_config'])

    Encoder = encoder_switch[config["policy_config"]["encoder"]]
    encoder = Encoder(config["policy_config"]["encoder_config"]).to(device)
                    #   image_size=(replay_memory.scene_data.image_height,
                    #               replay_memory.scene_data.image_width)).to(

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
            init_encoder = getattr(encoder,
                                   "initialize_parameters_via_dataset", None)
            if callable(init_encoder):
                init_encoder(replay_memory)
            else:
                logger.info("This encoder does not use dataset initialization."
                            )

    encoder.eval()

    embed_trajectories(encoder, replay_memory, config, encoder_name)

    # save if kp_gt model as we need the reference selection
    if config["policy_config"]["encoder"] == 'keypoints_gt':
        encoder.to_disk(file_name)


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
            #  "options: {}, 'Mixed'".format(str(tasks)[1:-1]),
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
        "--copy_selection_from",
        dest="copy_selection_from",
        required=False,
        help="Path to an encoder from which to copy the reference uvs, vecs."
             "For GT-KP model only (to compare different projections)."
    )
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

    # mask_type = MaskTypes.GT if args.mask or args.object_pose else \
    #     MaskTypes.TSDF
    mask_type = None if args.panda else MaskTypes.GT  # only needed for GT_KP model anyway.

    if args.encoder == "keypoints_gt" and len(args.cam) > 1:
        logger.error("Should only use one camera for GT KP encoder.")
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

            "only_use_first_object_label": (args.encoder == "keypoints_gt" and
                                            args.task == "TakeLidOffSaucepan"),
            "conflate_so_object_labels":  mask_type is MaskTypes.GT,

            "mask_type": mask_type,

            "cams": args.cam,

            "sample_freq": None,  # downsample the trajectories to this freq

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

    # if config["policy_config"]["encoder_config"].get("encoder", {}).get(
    #         "prior_type") is PriorTypes.PARTICLE_FILTER:
    #     config["dataset_config"]["force_load_raw"] = True

    main(config, args.path, args.copy_selection_from)
