from tsdf.cluster import Cluster
import tsdf.fusion as fusion

import pickle
import time
from argparse import ArgumentParser

import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm

import pyrender
import trimesh


import utils.logging  # noqa
from dataset.scene import SubSampleTypes

from tsdf.filter import cut_volume_with_box, filter_background, filter_gripper
from utils.select_gpu import device
from utils.misc import (apply_machine_config, load_replay_memory,
                        set_seeds)
# from utils.tasks import tasks
from utils.torch import (batched_project_onto_cam, batched_rigid_transform,
                         batched_pinhole_projection_image_to_world_coordinates,
                         stack_trajs, invert_homogenous_transform)
# import viz.image_series as viz_image_series
from viz.operations import np_channel_front2back


def estimate_volume_bounds(rgb, depth, intrinsics, extrinsics):
    vol_bnds = np.zeros((3, 2))
    n_imgs = rgb.shape[0]
    for i in range(n_imgs):
        cam_intr = intrinsics[i]
        cam_pose = extrinsics[i]
        depth_im = depth[i]

        # Compute camera view frustum and extend convex hull
        view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(
          vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(
          vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

    return vol_bnds


def fuse(rgb, depth, intrinsics, extrinsics, remove_background=True,
         remove_gripper=True):

    if remove_background:
        depth = filter_background(depth, extrinsics, intrinsics)
    if remove_gripper:
        depth = filter_gripper(depth, extrinsics, intrinsics)

    logger.info("Estimating voxel volume bounds...")
    vol_bnds = estimate_volume_bounds(rgb, depth, intrinsics, extrinsics)
    logger.info("Refining voxel volume bounds with box filter...")
    vol_bnds = cut_volume_with_box(vol_bnds)

    logger.info("Doing rough TSDF pass to refine voxel volume bounds...")
    rough_vol = integrate_volume(rgb, depth, intrinsics, extrinsics,
                                 vol_bnds, voxel_size=0.02)
    pc = rough_vol.get_point_cloud()[:, :3]
    rough_voxel_bnds = np.stack((pc.min(axis=0) - .02, pc.max(axis=0) + .02)).T
    vol_bnds = cut_volume_with_box(vol_bnds, box=rough_voxel_bnds)

    logger.info("Doing fine TSDF pass...")
    fine_vol = integrate_volume(rgb, depth, intrinsics, extrinsics,
                                vol_bnds, voxel_size=0.001)

    return fine_vol


def integrate_volume(rgb, depth, intrinsics, extrinsics, vol_bnds, voxel_size):
    logger.info("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size)

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    n_imgs = rgb.shape[0]
    for i in tqdm(range(n_imgs)):

        color_image = np_channel_front2back(rgb[i])
        cam_intr = intrinsics[i]
        depth_im = depth[i]
        cam_pose = extrinsics[i]

        tsdf_vol.integrate(color_image, depth_im, cam_intr,
                           cam_pose, obs_weight=1.)

    fps = n_imgs / (time.time() - t0_elapse)
    logger.info("Average FPS: {:.2f}".format(fps))

    return tsdf_vol


def write_mesh(tsdf_vol, mesh_filename):
    logger.info("Saving mesh to mesh.ply...")
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    fusion.meshwrite(mesh_filename, verts, faces, norms, colors)

    return verts, faces


def write_pc(tsdf_vol, pc_filename):
    logger.info("Saving point cloud to pc.ply...")
    point_cloud = tsdf_vol.get_point_cloud()
    fusion.pcwrite(pc_filename, point_cloud)


def build_masks(point_cloud, pc_labels, depth, intrinsics, extrinsics, batch_no=4):
    logger.info("Generating masks ...")

    pc_labels = pc_labels.to(device)

    CLIP_VALUE = -2

    depth_batches = torch.chunk(depth, batch_no)
    extr_batches = torch.chunk(extrinsics, batch_no)
    intr_batches = torch.chunk(intrinsics, batch_no)

    masks_all = []

    for j, (d, e, i )in enumerate(zip(depth_batches, extr_batches, intr_batches)):
        n_imgs = d.shape[0]
        masks = torch.zeros(d.shape, dtype=torch.int8, device=device)

        pixel = batched_project_onto_cam(
            point_cloud, d, i, e, clip_value=CLIP_VALUE)

        pixel = pixel.long()

        # either coordinate is outside the view frustrum
        px_mask = (pixel == CLIP_VALUE).sum(dim=-1) > 0

        for i in range(n_imgs):
            masked_pc_labels = torch.where(
                px_mask[i], torch.tensor(0, dtype=torch.int8, device=device), pc_labels)
            masks[i][pixel[i, :, 1], pixel[i, :, 0]] = masked_pc_labels

        masks_all.append(masks)

    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    return torch.cat(masks_all)


def build_masks_via_mesh(vertices, faces, pc_labels, intrinsics, extrinsics):

    intrinsics = intrinsics.cpu().numpy()

    labels = np.unique(pc_labels)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    B = extrinsics.shape[0]

    # Long trajectory can still fill up GPU memory. So batchify projection.
    vertices = torch.from_numpy(mesh.vertices).float().to(device).unsqueeze(0)

    batch_no = 4
    extr_batches = torch.chunk(extrinsics, batch_no)

    projected_all = []

    for j, e in enumerate(extr_batches):
        b = e.shape[0]
        proj = batched_rigid_transform(vertices.expand(b, -1, -1),
                                       invert_homogenous_transform(e))

        projected_all.append(proj)

    vertices = torch.cat(projected_all)

    masks = []

    for j in tqdm(range(B)):
        mesh.vertices = vertices[j].cpu().numpy()

        # split mesh according to clustering
        meshes = [mesh.copy() for _ in range(len(labels))]
        for i in range(len(labels)):
            meshes[i].update_vertices(pc_labels == labels[i])

        # convert to pyrender scene instead of trimesh to use their segmentation mask renderer
        pyrender_scene = pyrender.Scene()
        node_labels = {}
        for i in range(len(labels)):
            sub_mesh = pyrender.Mesh.from_trimesh(meshes[i])
            sm_node = pyrender.Node(mesh=sub_mesh)
            pyrender_scene.add_node(sm_node)
            node_labels[sm_node] = labels[i]  # add to dict for consistent labels
        K = intrinsics[j]
        # as we projected the pc into camera frame, use neutral cam pose
        # need to fix axis direction though
        RT = np.eye(4)
        RT[1][1] = -1
        RT[2][2] = -1
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        zfar = 1000
        znear = 0.01
        cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear, zfar)
        cam_node = pyrender_scene.add(cam, pose=RT)
        renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
        mask = renderer.render(pyrender_scene, pyrender.RenderFlags.SEG, node_labels)[0]
        masks.append(mask[..., 0])

    return torch.from_numpy(np.stack(masks)), labels


def build_masks_via_depth(vol_dim, vol_origin, vox_size, point_cloud,
                          pc_labels, depth, intrinsics,
                          extrinsics, eps=50, batch_no=4):
    logger.info("Generating masks ...")
    fusion.debug_memory(device)

    depth_batches = torch.chunk(depth, batch_no)
    extr_batches = torch.chunk(extrinsics, batch_no)
    intr_batches = torch.chunk(intrinsics, batch_no)

    masks_all = []

    for i, (d, e, i )in enumerate(zip(depth_batches, extr_batches, intr_batches)):
        # create pixel coordinates and project into scene, then voxel space
        B, H, W = d.shape
        px_u = torch.arange(0, W, device=device)
        px_v = torch.arange(0, H, device=device)
        px_vu = torch.cartesian_prod(px_u, px_v).unsqueeze(0).expand(B, -1, -1)

        px_world = batched_pinhole_projection_image_to_world_coordinates(
            px_vu[..., 1], px_vu[..., 0], d.reshape(B, H*W),
            i.unsqueeze(0), e).reshape((B, H, W, 3))
        vox_origin = torch.from_numpy(vol_origin).to(device)
        px_voxel = torch.round((px_world - vox_origin) / vox_size)

        # get point cloud in voxel space and label voxel according to pc labels
        # there is some error in the depth due to round etc, so label neighborhood
        vol = torch.zeros(*vol_dim, dtype=torch.int8).to(device)
        for coord, label in zip(point_cloud, pc_labels):
            if label > 0:
                x, y, z = coord
                vol[x-eps:x+eps+1, y-eps:y+eps+1, z-eps:z+eps+1] = label

        with torch.cuda.device(device):
            torch.cuda.empty_cache()

        # get coordinates of each pixel in voxel space
        B_idx = torch.arange(B).unsqueeze(1).unsqueeze(2)
        h_idx = torch.arange(H).unsqueeze(0).unsqueeze(2)
        w_idx = torch.arange(W).unsqueeze(0).unsqueeze(0)
        vol_idx = px_voxel[B_idx, h_idx, w_idx]
        vol_idx = vol_idx.reshape(B*H*W, 3).long()

        # mask coordinates outside the voxel volume
        valid_idx = torch.prod(torch.logical_and(
            vol_idx >= 0, vol_idx < torch.tensor(vol_dim).to(device)), -1).bool()

        masks = torch.zeros((B*H*W), dtype=torch.int8, device=device)
        masked_masks = torch.masked_select(masks, valid_idx)
        masked_vol_idx = torch.masked_select(
            vol_idx, valid_idx.unsqueeze(-1)).reshape(-1, 3)

        # assign labels from volume to mask
        masked_masks = vol[masked_vol_idx[:, 0], masked_vol_idx[:, 1],
                        masked_vol_idx[:, 2]]
        masks[valid_idx] = masked_masks
        masks = masks.reshape(B, H, W)

        masks_all.append(masks)

    return torch.cat(masks_all)


def write_mask(mask, file_name):
    logger.info("Saving mask to mask.pkl...")
    pickle.dump(mask, open(file_name, "wb"))


@torch.no_grad()
@logger.contextualize(filter=False)
def main(config, path=None):
    replay_memory = load_replay_memory(config["dataset_config"], path=path)
    out_dir = replay_memory.initialize_scene_reconstruction()

    if out_dir is None:
        return

    scene_cams = config["dataset_config"]["cams"]
    fusion_cams = config["fusion_config"]["cams_for_fusion"]

    for t in range(len(replay_memory)):
        # Load scene only with fusion cams and subsampled for reconstruction.
        fusion_views = replay_memory.get_scene(
            traj_idx=t,
            cams=fusion_cams,
            subsample_types=config["fusion_config"]["subsample_type"])
        fusion_scene = stack_trajs([v for _, v in fusion_views.items()])

        rgb = fusion_scene.cam_rgb.numpy()
        depth = fusion_scene.cam_d.numpy()
        extrinsics = fusion_scene.cam_ext.numpy()
        intrinsics = fusion_scene.cam_int.numpy()
        n_imgs = len(rgb)

        with torch.cuda.device(device):
            torch.cuda.empty_cache()

        logger.info("Images to fuse: {} from cams {}", n_imgs, fusion_cams)

        context = fusion.make_context(device)
        fusion.debug_memory(device)

        tsdf_vol = fuse(rgb, depth, intrinsics, extrinsics)
        tsdf_point_cloud = tsdf_vol.get_point_cloud()[:, :3]

        # We do not need the meshes and point clouds in the dataset, so
        # save them externally for verification purposes. Can be skipped.
        traj_name = replay_memory._paths[t].parts[-1]
        mesh_path = out_dir / ("mesh_" + traj_name + ".ply")
        vertices, faces = write_mesh(tsdf_vol, mesh_path)
        write_pc(tsdf_vol, out_dir / ("pc_" + traj_name + ".ply"))

        fusion.destroy_context(context)

        # continue

        logger.info("Clustering point cloud of size {}, type {}, ...",
                    tsdf_point_cloud.shape[0], tsdf_point_cloud.dtype)
        cluster = Cluster(eps=0.03, min_samples=5000)
        fitted_cluster = cluster.fit(tsdf_point_cloud)
        # cluster labels start at zero, noisy is -1, so + 1 for object labels
        # pc_labels = torch.from_numpy(fitted_cluster.labels_ + 1).to(torch.int8)
        pc_labels = fitted_cluster.labels_ + 1


        # import matplotlib.pyplot as plt

        # fig = plt.figure(figsize=(12,7))
        # ax = fig.add_subplot(projection='3d')
        # img = ax.scatter(tsdf_point_cloud[:, 0], tsdf_point_cloud[:, 1], tsdf_point_cloud[:, 2],
        #                  c=pc_labels, cmap='Set1')
        # fig.colorbar(img)

        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')

        # plt.show()

        # Load entire scene for mask generation.
        scene_views = replay_memory.get_scene(
            traj_idx=t,
            cams=scene_cams,
            subsample_types=None)
        scene = stack_trajs([v for _, v in scene_views.items()])

        # rgb = scene.cam_rgb.to(device)
        # depth = scene.cam_d.to(device)
        extrinsics = scene.cam_ext.to(device)
        intrinsics = scene.cam_int  # to(device)
        n_imgs = extrinsics.shape[0]

        del scene

        with torch.cuda.device(device):
            torch.cuda.empty_cache()

        fusion.debug_memory(device)

        logger.info("Generating masks for {} images from cams {}",
                    n_imgs, scene_cams)
        mask, labels = build_masks_via_mesh(
            vertices, faces, pc_labels, intrinsics, extrinsics)

        labels = np.delete(labels, np.argwhere(labels == 0)).tolist()
        logger.info("Generated labels {}", labels)

        logger.info("Writing masks to disk ...")
        traj_lens = [len(t.cam_rgb) for _, t in scene_views.items()]
        traj_ends = np.cumsum(traj_lens)
        masks_per_cam = np.split(mask, traj_ends)

        sanity_check = masks_per_cam.pop(-1)
        assert len(sanity_check) == 0

        for c, m in zip(scene_cams, masks_per_cam):
            replay_memory.add_tsdf_masks(t, c, m, labels)

        # NOTE: these are useful for debugging/inspection
        # viz_image_series.vis_series_w_mask(replay_memory.camera_obs_w[t],
        #                                    mask)
        # write_mask(mask, out_dir + "/masks_" + str(t) + ".pkl")

        with torch.cuda.device(device):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    set_seeds(1996)
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--feedback_type",
        dest="feedback_type",
        default="dcr_10",
        help="options: dcr_10, dcr_100, dcm_10, dcm_100, dcs_10, ed_10",
    )
    parser.add_argument(
        "-t",
        "--task",
        dest="task",
        default="CloseMicrowave",
        # help="options: {}, 'Mixed'".format(str(tasks)[1:-1]),
    )
    parser.add_argument(
        "--cam",
        dest="cam",
        required=True,
        nargs='+',
        help="The camera(s) to use. Options: wrist, overhead."
    )
    parser.add_argument(
        "--path",
        dest="path",
        default=None,
        help="Path to a dataset. May be provided instead of f-t-m.",
    )
    parser.add_argument(
        "-m",
        "--mask",
        dest="mask",
        action="store_true",
        default=False,
        help="Load dataset with ground truth object masks.",
    )
    parser.add_argument(
        "-o",
        "--object_pose",
        dest="object_pose",
        action="store_true",
        default=False,
        help="Use data with ground truth object positions.",
    )
    # NOTE: shoulder cams can capture the robot arm as well. Not filtered yet.

    args = parser.parse_args()
    config = {
        "fusion_config": {
            "subsample_type": {  # if applicable, ie. cam was passed as arg
                "wrist": SubSampleTypes.POSE,
                "overhead": SubSampleTypes.CONTENT,
            },
            "cams_for_fusion": ["wrist"],  # generating masks for all cams
        },
        "dataset_config": {
            "feedback_type": args.feedback_type,
            "task": args.task,

            "ground_truth_mask": args.mask or args.object_pose,
            "ground_truth_object_pose": args.object_pose,

            "data_root": "data",

            "cams": args.cam,
        }
    }

    config = apply_machine_config(config)

    main(config, args.path)
