import numpy as np
from pyrep.const import RenderMode
from rlbench.observation_config import CameraConfig, ObservationConfig

from env.observation import GTObservation
from env.trajectory import BaseEnvironment, CustomEnv


class GTEnv(CustomEnv):
    def __init__(self, config, eval=None):
        BaseEnvironment.__init__(self, config)

        self.shoulders_on = config["shoulders_on"]
        self.wrist_on = config["wrist_on"]
        self.overhead_on = config["overhead_on"]

        if "image_size" in config and config["image_size"] is not None:
            image_size = config["image_size"]
        else:
            image_size = (256, 256)

        obs_config = ObservationConfig(
            left_shoulder_camera=CameraConfig(
                rgb=config["shoulders_on"], depth=config["shoulders_on"],
                mask=config["shoulders_on"], render_mode=RenderMode.OPENGL,
                depth_in_meters=True, image_size=image_size,
                point_cloud=False),
            right_shoulder_camera=CameraConfig(
                rgb=config["shoulders_on"], depth=config["shoulders_on"],
                mask=config["shoulders_on"], render_mode=RenderMode.OPENGL,
                depth_in_meters=True, image_size=image_size,
                point_cloud=False),
            front_camera=CameraConfig(
                rgb=False, depth=False, mask=False, image_size=image_size,
                point_cloud=False),
            wrist_camera=CameraConfig(
                rgb=config["wrist_on"], depth=config["wrist_on"],
                mask=config["wrist_on"], render_mode=RenderMode.OPENGL,
                depth_in_meters=True, image_size=image_size,
                point_cloud=False),
            overhead_camera=CameraConfig(
                rgb=config["overhead_on"], depth=config["overhead_on"],
                mask=config["overhead_on"], render_mode=RenderMode.OPENGL,
                depth_in_meters=True, image_size=image_size,
                point_cloud=False),
            joint_positions=True,
            joint_velocities=True,
            joint_forces=False,
            gripper_pose=True,
            gripper_matrix=True,
            task_low_dim_state=True,
        )

        self.launch_simulation_env(config, obs_config)

        self.setup_camera_controls(config)

    def obs_split(self, obs, live=True):
        cam_l_rgb = obs.left_shoulder_rgb.transpose(
            (2, 0, 1)) / 255 if self.shoulders_on else None
        cam_r_rgb = obs.right_shoulder_rgb.transpose(
            (2, 0, 1)) / 255 if self.shoulders_on else None
        cam_w_rgb = obs.wrist_rgb.transpose(
            (2, 0, 1)) / 255 if self.wrist_on else None

        cam_o_rgb = obs.overhead_rgb.transpose(
            (2, 0, 1)) / 255 if self.overhead_on else None
        cam_l_d = obs.left_shoulder_depth if self.shoulders_on else None
        cam_r_d = obs.right_shoulder_depth if self.shoulders_on else None
        cam_w_d = obs.wrist_depth if self.wrist_on else None
        cam_o_d = obs.overhead_depth if self.overhead_on else None

        cam_l_mask = obs.left_shoulder_mask.astype(
            int) if self.shoulders_on else None
        cam_r_mask = obs.right_shoulder_mask.astype(
            int) if self.shoulders_on else None
        cam_w_mask = obs.wrist_mask.astype(int) if self.wrist_on else None
        cam_o_mask = obs.overhead_mask.astype(int) \
            if self.overhead_on else None

        # if not interacting live with the env (eg. when using expert demos),
        # the camera pose cannot be extracted. In that case we do not
        # downsample the trajectory by pose change, so it should be fine to
        # just set the wrist pose to zero.
        if live and self.wrist_on:
            wrist_pose = self.get_camera_pose()["wrist"]
        else:
            wrist_pose = [0] * 7

        cam_r_ext = \
            obs.misc['right_shoulder_camera_extrinsics'] if self.shoulders_on else None  # noqa 501
        cam_r_int = \
            obs.misc['right_shoulder_camera_intrinsics'] if self.shoulders_on else None  # noqa 501
        cam_l_ext = \
            obs.misc['left_shoulder_camera_extrinsics'] if self.shoulders_on else None  # noqa 501
        cam_l_int = \
            obs.misc['left_shoulder_camera_intrinsics'] if self.shoulders_on else None  # noqa 501
        cam_w_ext = \
            obs.misc['wrist_camera_extrinsics'] if self.wrist_on else None
        cam_w_int = \
            obs.misc['wrist_camera_intrinsics'] if self.wrist_on else None
        cam_o_ext = obs.misc['overhead_camera_extrinsics'] \
            if self.overhead_on else None
        cam_o_int = obs.misc['overhead_camera_intrinsics'] \
            if self.overhead_on else None

        proprio_obs = np.append(obs.joint_positions, obs.gripper_open)
        gripper_pose = obs.gripper_pose

        # print("oh", self.env._scene._cam_overhead.get_pose())
        # print("wrist", self.env._scene._cam_wrist.get_pose())

        object_pose = obs.task_low_dim_state

        return GTObservation(gripper_pose, proprio_obs, wrist_pose,
                             cam_l_rgb, cam_r_rgb, cam_w_rgb, cam_o_rgb,
                             cam_l_d, cam_r_d, cam_w_d, cam_o_d,
                             cam_l_mask, cam_r_mask, cam_w_mask, cam_o_mask,
                             cam_r_ext, cam_r_int,
                             cam_l_ext, cam_l_int,
                             cam_w_ext, cam_w_int,
                             cam_o_ext, cam_o_int,
                             object_pose)
