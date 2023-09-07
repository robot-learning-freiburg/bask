from argparse import ArgumentParser

import torch
from loguru import logger

import config as default_config
import utils.logging  # noqa
import viz.image_series as viz_image_series
from dataset.dc import DenseCorrespondenceDataset
from encoder import KeypointsPredictor, VitFeatureEncoder
from utils.misc import (apply_machine_config, import_config_file,
                        load_replay_memory, pretrain_checkpoint_name)
from utils.random import set_seeds
from utils.select_gpu import device
from viz.live_heatmap_visualization import HeatmapVisualization


def main(config: dict, path: str | None = None, vit_extr: bool = False) -> None:

    replay_memory = load_replay_memory(config["dataset_config"], path=path)
    replay_memory.update_camera_crop(config["dataset_config"]["image_crop"])

    Encoder = VitFeatureEncoder if vit_extr else KeypointsPredictor

    replay_memory = DenseCorrespondenceDataset(
        replay_memory, config["dataset_config"],
        sample_type=Encoder.sample_type)

    if contr_path := config["contrast_set"]:
        raise NotImplementedError("Need to update contrast set loading. "
                                  "See release-branche for old code.")

    file_name = config["encoder_path"] or pretrain_checkpoint_name(config)
    encoder = Encoder(config["policy_config"]["encoder_config"]).to(device)
    logger.info("Loading encoder from {}", file_name)
    encoder.from_disk(file_name)

    live_heatmap = HeatmapVisualization(
        replay_memory, encoder, config["contrast_set"],
        norm_by_descr_dim=config["policy_config"]["encoder_config"][
            "pretrain"]["training_config"]["loss_function"][
                "norm_by_descriptor_dim"],
        cams=config["dataset_config"]["cameras"])

    live_heatmap.run()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-f", "--feedback_type",
        default="pretrain_manual",
        help="The data type to load.",
    )
    parser.add_argument(
        "-t", "--task",
        default="PhoneOnBase",
        help="Name of the task of the dataset.",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Path to a dataset. May be provided instead of f-t.",
    )
    # parser.add_argument(
    #     "--contrast_set",
    #     default=None,
    #     help="Optional: path to a contrast data set.",
    # )
    parser.add_argument(
        "--pretrained_on",
        default=None,
        help="Task on which the encoder was pretrained. Defaults to task.",
    )
    parser.add_argument(
        "--encoder_suffix",
        dest="encoder_suffix",
        default=None,
        help="Pass a suffix to append to the name of the encoder checkpoint."
    )
    parser.add_argument(
        "--encoder_path",
        default=None,
        help="Pass the path to a specific encoder. Useful if there's multiple "
             "snapshot with non-standard names."
    )
    parser.add_argument(
        "-c",
        "--config",
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
        "--panda",
        action="store_true",
        default=False,
        help="Data comes from a real world panda -> 480px obs.",
    )
    parser.add_argument(
        "-v", "--vit_extr",
        action="store_true",
        default=False,
        help="Visualize the features of the ViT feature extractor."
    )
    parser.add_argument(
        "-s", "--show_trajectory",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    set_seeds()

    if args.config is not None:
        conf_file = import_config_file(args.config)
    else:
        conf_file = default_config

    enc_type = "vit_extractor" if args.vit_extr else "keypoints"

    encoder_configs = conf_file.encoder_configs[enc_type]

    image_dim = default_config.realsense_cam_resolution_cropped if args.panda \
        else default_config.sim_cam_resolution
    image_crop = default_config.realsense_cam_crop if args.panda else None

    encoder_configs["obs_config"] = {"image_dim": image_dim}

    config = {
        "encoder_path": args.encoder_path,
        "contrast_set": None,  # args.contrast_set,

        "policy_config": {
            "encoder": enc_type,
            "encoder_config": encoder_configs,
            "encoder_suffix": args.encoder_suffix,
        },
        "dataset_config": {
            "feedback_type": args.feedback_type,
            "task": args.task,

            "pretrained_on": args.pretrained_on or args.task,
            # "pretrain_feedback": args.pretrain_feedback,

            "cameras": args.cam,
            "image_crop": image_crop,

            "data_root": "data",
        }
    }

    config["dataset_config"].update(encoder_configs.get("pretrain", {}).get(
        "dataset_config", {}))

    config = apply_machine_config(config)

    main(config, path=args.path, vit_extr=args.vit_extr)
