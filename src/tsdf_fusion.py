from argparse import ArgumentParser

import numpy as np
import torch
from loguru import logger

from env import Environment
import utils.logging  # noqa
from dataset.scene import SubSampleTypes

from tsdf.cluster import Cluster  # type: ignore
import tsdf.fusion as fusion
from tsdf.filter import (filter_plane_from_mesh_and_pointcloud,
                         gripper_dists, coordinate_boxes)
from utils.cuda import try_empty_cuda_cache, try_make_context, try_destroy_context, try_debug_memory
from utils.select_gpu import device
from utils.misc import apply_machine_config, load_replay_memory
from utils.random import configure_seeds

# import viz.image_series as viz_image_series


@torch.no_grad()
@logger.contextualize(filter=False)
def main(config: dict, path: str | None = None):
    replay_memory = load_replay_memory(config["dataset_config"], path=path)
    out_dir = replay_memory.initialize_scene_reconstruction()

    if out_dir is None:
        return

    scene_cams = config["dataset_config"]["cameras"]
    fusion_cams = config["fusion_config"]["cams_for_fusion"]

    coordinate_box = coordinate_boxes[config["dataset_config"]["env"]]
    gripper_dist = gripper_dists[config["dataset_config"]["env"]]

    for t in range(len(replay_memory)):
        # Load scene only with fusion cams and subsampled for reconstruction.
        fusion_views = replay_memory.get_scene(
            traj_idx=t,
            cams=fusion_cams,
            subsample_types=config["fusion_config"]["subsample_type"])
        fusion_scene = torch.cat([v for _, v in fusion_views.items()])

        # fusion_scene is SingleCamSceneObservation (stacked from all cams)
        rgb = fusion_scene.rgb.numpy() # type: ignore
        depth = fusion_scene.depth.numpy()  # type: ignore
        extrinsics = fusion_scene.extr.numpy()  # type: ignore
        intrinsics = fusion_scene.intr.numpy()  # type: ignore
        n_imgs = len(rgb)

        H, W = rgb.shape[-2:]

        try_empty_cuda_cache()

        logger.info("Images to fuse: {} from cams {}", n_imgs, fusion_cams)

        context = try_make_context(device)

        tsdf_vol = fusion.fuse(rgb, depth, intrinsics, extrinsics,
                               coordinate_box, gripper_dist)
        tsdf_point_cloud = tsdf_vol.get_point_cloud()[:, :3]

        # We do not need the meshes and point clouds in the dataset, so
        # save them externally for verification purposes. Can be skipped.
        traj_name = replay_memory._paths[t].parts[-1]
        mesh_path = out_dir / ("mesh_" + traj_name + ".ply")
        _, faces = fusion.write_mesh(tsdf_vol, mesh_path)
        fusion.write_pc(tsdf_vol, out_dir / ("pc_" + traj_name + ".ply"))

        try_destroy_context(context)

        vertices, faces = \
            filter_plane_from_mesh_and_pointcloud(tsdf_point_cloud, faces)

        logger.info("Clustering point cloud of size {}, type {}, ...",
                    vertices.shape[0], vertices.dtype)
        cluster = Cluster(eps=0.03, min_samples=5000)
        fitted_cluster = cluster.fit(vertices)
        # cluster labels start at zero, noisy is -1, so + 1 for object labels
        pc_labels = fitted_cluster.labels_ + 1


        # Load entire scene for mask generation.
        scene_views = replay_memory.get_scene(
            traj_idx=t,
            cams=scene_cams,
            subsample_types=None)
        scene = torch.cat([v for _, v in scene_views.items()])

        extrinsics = scene.extr.to(device)  # type: ignore
        intrinsics = scene.intr  # type: ignore
        n_imgs = extrinsics.shape[0]

        del scene

        try_empty_cuda_cache()

        try_debug_memory(device)

        logger.info("Generating masks for {} images from cams {}",
                    n_imgs, scene_cams)
        mask, labels = fusion.build_masks_via_mesh(
            vertices, faces, pc_labels, intrinsics, extrinsics, H, W)

        labels = np.delete(labels, np.argwhere(labels == 0)).tolist()
        logger.info("Generated labels {}", labels)

        logger.info("Writing masks to disk ...")
        traj_lens = [len(t.rgb) for _, t in scene_views.items()]
        traj_ends = np.cumsum(traj_lens)
        masks_per_cam = np.split(mask, traj_ends)

        sanity_check = masks_per_cam.pop(-1)
        assert len(sanity_check) == 0

        for c, m in zip(scene_cams, masks_per_cam):
            replay_memory.add_tsdf_masks(t, c, m, labels)  # type: ignore

        # NOTE: these are useful for debugging/inspection
        # viz_image_series.vis_series_w_mask(replay_memory.camera_obs_w[t],
        #                                    mask)
        # write_mask(mask, out_dir + "/masks_" + str(t) + ".pkl")

        try_empty_cuda_cache()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e", "--environment",
        help="RLBench, Maniskill or Panda."
    )
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
        "--cam",
        dest="cam",
        required=True,
        nargs='+',
        help="The camera(s) to use. Options: wrist, overhead."
    )

    args = parser.parse_args()

    env = Environment[args.environment.upper()]

    config = {
        "fusion_config": {
            "subsample_type": {  # if applicable, ie. cam was passed as arg
                "wrist": SubSampleTypes.POSE,
                "overhead": SubSampleTypes.CONTENT,
            },
            "cams_for_fusion": ["wrist"],  # generating masks for all cams
        },
        "dataset_config": {
            "env": env,
            "feedback_type": args.feedback_type,
            "task": args.task,

            "data_root": "data",

            "cameras": args.cam,
        }
    }

    args.seed = None

    seed = configure_seeds(args)

    config = apply_machine_config(config)

    main(config, args.path)
