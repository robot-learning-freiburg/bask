import pickle
from argparse import ArgumentParser

import numpy as np
import torch
from loguru import logger

import config as default_config
import utils.logging  # noqa
import viz.image_series as viz_image_series
from dataset.dc import DenseCorrespondenceDataset
from encoder import KeypointsPredictor
from utils.misc import (apply_machine_config, import_config_file,
                        load_replay_memory, policy_checkpoint_name,
                        pretrain_checkpoint_name, set_seeds)
from utils.select_gpu import device
from viz.live_heatmap_visualization import HeatmapVisualization


def load_mask(file_name):
    return pickle.load(open(file_name, "rb"))


def main(config, show_trajectory, path=None):

    replay_memory = load_replay_memory(config["dataset_config"], path=path)
    replay_memory.update_camera_crop(config["dataset_config"]["crop_left"])

    Encoder = KeypointsPredictor

    replay_memory = DenseCorrespondenceDataset(
        replay_memory, config["dataset_config"],
        sample_type=Encoder.sample_type)

    if contr_path := config["arm_scan"]:
        logger.info("Loading contrast set from {}", contr_path)
        contrast_memory = torch.load(contr_path)
        robot_labels = [31, 34, 35, 39, 40, 41, 42, 43, 46]
        contrast_memory = DenseCorrespondenceDataset(
            contrast_memory, config["policy_config"]["encoder_config"],
            object_labels=robot_labels
        )
        replay_memory.contrast_set = contrast_memory

    if show_trajectory:
        t = replay_memory.scene_data.sample_traj_idx()
        mask = replay_memory.scene_data.masks_w[t]

        viz_image_series.vis_series_w_mask(
            replay_memory.scene_data.camera_obs_w[t], mask)

    file_name = config["encoder_path"] or pretrain_checkpoint_name(config)
    encoder = Encoder(config["policy_config"]["encoder_config"]).to(device)
    logger.info("Loading encoder from {}", file_name)
    print(file_name)
    encoder.from_disk(file_name)

    live_heatmap = HeatmapVisualization(
        replay_memory, encoder, config["arm_scan"],
        norm_by_descr_dim=config["policy_config"]["encoder_config"][
            "pretrain"]["training_config"]["loss_function"][
                "norm_by_descriptor_dim"],
        cams=config["dataset_config"]["cams"])
    live_heatmap.run()


if __name__ == "__main__":
    set_seeds(1996)
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--feedback_type",
        dest="feedback_type",
        default="dcs_20",
        help="For interactivate heatmap (loads encoder only), pass the DC data"
             " e.g. dcs_20. For static heatmap with keypoints (loads policy )"
             "pass the BC data, eg. cloning_16.",
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
        help="task on which it was pretrained."
        # "options: {}, 'Mixed'".format(str(tasks)[1:-1]),
    )
    parser.add_argument(
        "--pretrain_feedback",
        dest="pretrain_feedback",
        default=None,
        help="The data on which the model was pretrained, eg. dcs_20. "
             "Only needed for the static heatmap.",
    )
    parser.add_argument(
        "-s",
        "--show_trajectory",
        dest="show_trajectory",
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
        "-o",
        "--object_pose",
        dest="object_pose",
        action="store_true",
        default=False,
        help="Use data with ground truth object positions.",
    )
    parser.add_argument(
        "--suffix",
        dest="suffix",
        default=None,
        help="Pass a suffix to load a specific policy checkpoint."
    )
    parser.add_argument(
        "--encoder_path",
        dest="encoder_path",
        default=None,
        help="Pass the path to a specific encoder. Useful if there's multiple "
             "snapshot with non-standard names."
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
        "-c",
        "--config",
        dest="config",
        default=None,
        help="Config file to use. Uses default if None provided.",
    )
    parser.add_argument(
        "--cam",
        dest="cam",
        required=True,
        nargs='+',
        help="The camera(s) to use. Options: wrist, overhead."
    )
    parser.add_argument(
        "--encoder_suffix",
        dest="encoder_suffix",
        default=None,
        help="Pass a suffix to append to the name of the encoder checkpoint."
    )
    parser.add_argument(
        "--path",
        dest="path",
        default=None,
        help="Path to a dataset. May be provided instead of f-t-m.",
    )
    parser.add_argument(
        "--panda",
        dest="panda",
        action="store_true",
        default=False,
        help="Data comes from a real world panda -> 480px obs.",
    )
    args = parser.parse_args()

    if args.config is not None:
        conf_file = import_config_file(args.config)
    else:
        conf_file = default_config
    enc_type = "keypoints"
    encoder_configs = conf_file.encoder_configs[enc_type]

    image_dim = (480, 480) if args.panda else (256, 256)
    crop_left = 160 if args.panda else 0

    encoder_configs["obs_config"] = {
        "image_dim": image_dim
    }


    config = {
        "encoder_path": args.encoder_path,
        "arm_scan": args.arm_scan,

        "policy_config": {
            "policy": "encoder",

            "lstm_layers": 2,
            "n_cams": len(args.cam),
            "use_ee_pose": True,  # else uses joint angles
            "action_dim": 7,
            "learning_rate": 3e-4,
            "weight_decay": 3e-6,

            "suffix": args.suffix,

            "encoder": enc_type,
            "encoder_config": encoder_configs,
            "encoder_suffix": args.encoder_suffix,
        },
        "dataset_config": {
            "feedback_type": args.feedback_type,
            "task": args.task,

            "ground_truth_mask": args.mask or args.object_pose,
            "ground_truth_object_pose": args.object_pose,

            "pretrained_on": args.pretrained_on or args.task,
            "pretrain_feedback": args.pretrain_feedback,

            "cams": args.cam,

            "crop_left": crop_left,


            "data_root": "data",

        }
    }

    config["dataset_config"].update(encoder_configs.get("pretrain", {}).get(
        "dataset_config", {}))

    config = apply_machine_config(config)

    main(config, args.show_trajectory, path=args.path)
