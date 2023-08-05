import json

import numpy as np
import torch
from torchvision.utils import save_image

from utils.constants import (ANGLE_DISTANCE_THRESHOLD,
                             LINEAR_DISTANCE_THRESHOLD, METADATA_FILENAME)
from utils.geometry import (compute_angle_between_poses,
                            compute_angle_between_quaternions,
                            compute_distance_between_poses)


class Trajectory():
    def __init__(self, camera_names, subsample_by_difference,
                 subsample_to_length):
        self.camera_names = camera_names

        self.subsample_by_difference = subsample_by_difference
        self.subsample_to_length = subsample_to_length
        self.reset()

    def reset(self):
        for c in self.camera_names:
            setattr(self, f"current_camera_obs_{c}", [])
            setattr(self, f"current_camera_depth_{c}", [])
            setattr(self, f"current_mask_{c}", [])
            setattr(self, f"current_extrinsics_{c}", [])
            setattr(self, f"intrinsics_{c}", None)

        self.current_ee_pose = []
        self.current_object_poses = []
        self.current_proprio_obs = []
        self.current_action = []
        self.current_feedback = []

        return

    def add(self, obs, action, feedback):
        for c in self.camera_names:
            getattr(self, f"current_camera_obs_{c}").append(obs.cam_rgb[c])
            getattr(self, f"current_camera_depth_{c}").append(obs.cam_depth[c])
            getattr(self, f"current_mask_{c}").append(obs.cam_mask[c])
            getattr(self, f"current_extrinsics_{c}").append(obs.cam_ext[c])

            if getattr(self, c_int := f"intrinsics_{c}") is None:
                setattr(self, c_int, torch.tensor(obs.cam_int[c]))
            else:
                assert torch.equal(getattr(self, c_int),
                                   torch.tensor(obs.cam_int[c]))

        # TODO: might need checks for existance for obj_poses, masks
        self.current_ee_pose.append(obs.ee_pose)
        self.current_object_poses.append(obs.object_poses)
        self.current_proprio_obs.append(obs.proprio_obs)
        self.current_action.append(action)
        self.current_feedback.append(feedback)

    def save(self, dir):
        if self.subsample_by_difference:
            assert self.wrist_on, "taking wrist for thresholding"
            indeces = get_idx_by_pose_difference_threshold(
                self.current_wrist_pose)
        elif self.subsample_to_length:
            indeces = get_idx_by_target_len(
                len(self.current_proprio_obs), self.subsample_to_length)
        else:
            indeces = list(range(len(self.current_ee_pose)))

        object_label_gt = []

        for c in self.camera_names:
            cam_rgb = downsample_traj_by_idx(
                getattr(self, f"current_camera_obs_{c}"), indeces)
            cam_d = downsample_traj_by_idx(
                getattr(self, f"current_camera_depth_{c}"), indeces)
            mask = downsample_traj_by_idx(
                getattr(self, f"current_mask_{c}"), indeces)
            extrinsics = downsample_traj_by_idx(
                getattr(self, f"current_extrinsics_{c}"), indeces)

            intrinsics = getattr(self, f"intrinsics_{c}").float()

            (dir / f"cam_{c}_rgb").mkdir()
            (dir / f"cam_{c}_d").mkdir()
            (dir / f"cam_{c}_mask_gt").mkdir()
            (dir / f"cam_{c}_ext").mkdir()

            for t in range(len(self.current_ee_pose)):
                save_image(torch.tensor(cam_rgb[t], dtype=torch.float32),
                           dir / f"cam_{c}_rgb" / f"{t}.png")
                torch.save(torch.tensor(cam_d[t], dtype=torch.float32),
                           dir / f"cam_{c}_d" / f"{t}.dat")
                torch.save(torch.tensor(mask[t].astype(np.int32), dtype=torch.uint8)
                           if mask[t] is not None else None,
                           dir / f"cam_{c}_mask_gt" / f"{t}.dat")
                torch.save(torch.tensor(extrinsics[t], dtype=torch.float32),
                           dir / f"cam_{c}_ext" / f"{t}.dat")
            torch.save(intrinsics, dir / f"cam_{c}_int.dat")
            object_label_gt.append(self.get_object_labels(mask))

        ee_pose = downsample_traj_by_idx(self.current_ee_pose, indeces)
        object_poses = downsample_traj_by_idx(self.current_object_poses,
                                              indeces)
        proprio_obs = downsample_traj_by_idx(self.current_proprio_obs,
                                             indeces)
        action = downsample_traj_by_idx(self.current_action, indeces)
        feedback = downsample_traj_by_idx(self.current_feedback, indeces)

        (dir / "ee_pose").mkdir()
        (dir / "object_poses").mkdir()
        (dir / "proprio_obs").mkdir()
        (dir / "action").mkdir()
        (dir / "feedback").mkdir()

        for t in range(len(self.current_ee_pose)):
            torch.save(torch.tensor(ee_pose[t], dtype=torch.float32),
                       dir / "ee_pose" / f"{t}.dat")
            op = {
                k: torch.tensor(v, dtype=torch.float32)
                for k, v in object_poses[t].items()
            }
            torch.save(op, dir / "object_poses" / f"{t}.dat")
            torch.save(torch.tensor(proprio_obs[t], dtype=torch.float32),
                       dir / "proprio_obs" / f"{t}.dat")
            torch.save(torch.tensor(action[t], dtype=torch.float32),
                       dir / "action" / f"{t}.dat")
            torch.save(torch.tensor(feedback[t], dtype=torch.float32),
                       dir / "feedback" / f"{t}.dat")

        object_label_gt = sorted(list(set().union(*object_label_gt)))
        object_label_gt = [] if object_label_gt == [None] else object_label_gt

        metadata = {
            "len": len(indeces),
            "object_label_gt": object_label_gt,
            }

        with open(dir / METADATA_FILENAME, 'w') as f:
            json.dump(metadata, f)

        self.reset()

    def get_object_labels(self, tensors):
        return sorted(
            list(set().union(*[np.unique(t).tolist() for t in tensors])))


def downsample_traj_by_idx(traj, indeces):
    return np.array([traj[i] for i in indeces])


def get_idx_by_target_len(traj_len, target_len):
    if traj_len == target_len:
        return list(range(traj_len))
    elif traj_len < target_len:
        return list(range(traj_len)) + [traj_len - 1] * (
            target_len - traj_len)
    else:
        indeces = np.linspace(start=0, stop=traj_len - 1, num=target_len)
        indeces = np.round(indeces).astype(int)
        return indeces


def get_idx_by_pose_difference_threshold(poses):  # expects 7-dim poses
    idx = [0]
    current_pose = poses[0]
    for i in range(1, len(poses)):
        next_pose = poses[i]
        dist = np.linalg.norm(current_pose[:3] - next_pose[:3])
        angle = compute_angle_between_quaternions(
            current_pose[3:], next_pose[3:])
        if (dist > LINEAR_DISTANCE_THRESHOLD
                or angle > ANGLE_DISTANCE_THRESHOLD):
            idx.append(i)
            current_pose = next_pose

    return idx


def get_idx_by_pose_difference_threshold_matrix(poses):  # expects 4x4 pose
    idx = [0]
    current_pose = poses[0]
    for i in range(1, len(poses)):
        next_pose = poses[i]
        dist = compute_distance_between_poses(current_pose, next_pose)
        angle = compute_angle_between_poses(current_pose, next_pose)
        if (dist > LINEAR_DISTANCE_THRESHOLD
                or angle > ANGLE_DISTANCE_THRESHOLD):
            idx.append(i)
            current_pose = next_pose

    return idx


def get_idx_by_img_difference_threshold(rgb, depth):
    assert rgb.shape[0] == depth.shape[0]

    idx = [0]
    current_rgb = rgb[0]
    current_depth = depth[0]
    for i in range(1, rgb.shape[0]):
        next_rgb = rgb[i]
        next_depth = depth[i]
        if not (torch.isclose(current_rgb, next_rgb).all()
                and torch.isclose(current_depth, next_depth).all()):
            idx.append(i)
            current_rgb, current_depth = next_rgb, next_depth

    return idx


def downsample_traj_by_change(rgb, depth, extrinsics):
    indeces = get_idx_by_img_difference_threshold(rgb, depth)

    return rgb[indeces], depth[indeces], extrinsics[indeces]
