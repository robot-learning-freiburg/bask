import numpy as np
from pyrep.objects.vision_sensor import VisionSensor

project = VisionSensor.pointcloud_from_depth_and_camera_params

# RLBench
# coordinate_box = np.array([[-.15, 1.35],  # x
#                            [-.85, 0.85],  # y
#                            [0.76, 1.75]])  # z

# Real panda
coordinate_box = np.array([[-0.6, 1.35],  # x
                           [-.85, .85],   # y
                           [0.59, 1.9]])  # z

# gripper_dist = 0.1  # RLBench
gripper_dist = 0.05  # Real panda


def cut_volume_with_box(vol_bnd, box=None):
    if box is None:
        box = coordinate_box

    refined = np.zeros_like(vol_bnd)
    refined[:, 0] = np.maximum(vol_bnd[:, 0], box[:, 0])
    refined[:, 1] = np.minimum(vol_bnd[:, 1], box[:, 1])

    return refined


def filter_background(depth, extrinsics, intrinsics):
    """
    Project depth images into world coordinates and zero-out the depth image
    where the point is outside the given coordinate_box.

    Parameters
    ----------
    depth : np.array(N, H, W, 1)
    extrinsics : np.array(N, 4, 4)
    intrinsics : np.array(N, 3, 3)

    Returns
    -------
    type
        The filtered depth map. Filtered values are zeroed out.
    """
    filtered_depth = np.empty_like(depth)

    for i in range(depth.shape[0]):
        pointcloud = project(depth[i], extrinsics[i], intrinsics[i])

        shape = pointcloud.shape
        pointcloud = pointcloud.reshape(-1, shape[-1])
        lower = coordinate_box[:, 0]
        upper = coordinate_box[:, 1]

        point_is_outside = np.any(
            np.logical_or(lower >= pointcloud, pointcloud >= upper), axis=1)

        point_is_outside = point_is_outside.reshape(shape[0], shape[1])

        filtered_depth[i] = np.where(point_is_outside, 0, depth[i])

    return filtered_depth


def filter_gripper(depth, extrinsics, intrinsics):
    """
    Remove gripper artifacts via filtering all points below the defined depth
    threshold.

    Parameters
    ----------
    depth : np.array(N, H, W, 1)
    extrinsics : np.array(N, 4, 4)
    intrinsics : np.array(N, 3, 3)

    Returns
    -------
    type
        The filtered depth map. Filtered values are zeroed out.
    """
    filtered_depth = np.empty_like(depth)

    for i in range(depth.shape[0]):
        point_is_gripper = depth[i] <= gripper_dist

        filtered_depth[i] = np.where(point_is_gripper, 0, depth[i])

    return filtered_depth
