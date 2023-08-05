import os

import numpy as np
from loguru import logger
from pyrep.const import RenderMode
from pyrep.errors import IKError
from rlbench.action_modes import ActionMode, ArmActionMode
from rlbench.environment import Environment as RLBenchEnvironment
from rlbench.observation_config import CameraConfig, ObservationConfig
from rlbench.task_environment import InvalidActionError

from env.environment import BaseEnvironment
from env.observation import CeilingObservation
from utils.tasks import task_switch

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.environ["COPPELIASIM_ROOT"]


class CustomEnv(BaseEnvironment):
    def __init__(self, config):
        super().__init__(config)

        if "image_size" in config and config["image_size"] is not None:
            image_size = config["image_size"]
        else:
            image_size = (256, 256)

        obs_config = ObservationConfig(
            left_shoulder_camera=CameraConfig(
                rgb=False, depth=False, mask=False, image_size=image_size),
            right_shoulder_camera=CameraConfig(
                rgb=False, depth=False, mask=False, image_size=image_size),
            front_camera=CameraConfig(
                rgb=False, depth=False, mask=False, image_size=image_size),
            wrist_camera=CameraConfig(
                rgb=True, depth=False, mask=False,
                render_mode=RenderMode.OPENGL, image_size=image_size),
            overhead_camera=CameraConfig(
                rgb=config["overhead_on"], depth=config["overhead_on"],
                mask=False, render_mode=RenderMode.OPENGL,
                depth_in_meters=True, image_size=image_size,
                point_cloud=False),
            joint_positions=True,
            joint_velocities=True,
            joint_forces=False,
            gripper_pose=True,
            task_low_dim_state=False,
        )

        self.launch_simulation_env(config, obs_config)

        self.setup_camera_controls(config)

    def launch_simulation_env(self, config, obs_config):
        # sphere policy uses custom action mode, ABS_EE_POSE_PLAN_WORLD_FRAME
        # for others: everything like in parent class
        if "action_mode" in config.keys():
            action_mode = ActionMode(config["action_mode"])
            self.do_postprocess_actions = False
        else:
            action_mode = ActionMode(ArmActionMode.EE_POSE_EE_FRAME)
            self.do_postprocess_actions = True

        self.env = RLBenchEnvironment(
            action_mode,
            obs_config=obs_config,
            static_positions=config["static_env"],
            headless=config["headless_env"],
        )

        self.env.launch()

        self.task = self.env.get_task(task_switch[config["task"]])

    def shutdown(self):
        self.env.shutdown()

    def setup_camera_controls(self, config):
        self.camera_pose = config["camera_pose"]

        self.camera_assoc = {}
        if config["shoulders_on"]:
            self.camera_assoc["shoulder_left"] = \
                self.env._scene._cam_over_shoulder_left
            self.camera_assoc["shoulder_right"] = \
                self.env._scene._cam_over_shoulder_right
        if config["wrist_on"]:
            self.camera_assoc["wrist"] = self.env._scene._cam_wrist
        if config["overhead_on"]:
            self.camera_assoc["overhead"] = self.env._scene._cam_overhead

    def reset(self):
        super().reset()

        descriptions, obs = self.task.reset()

        if self.camera_pose:
            self.set_camera_pose(self.camera_pose)

        obs = self.obs_split(obs)

        return obs

    def reset_to_demo(self, demo):
        super().reset()

        descriptions, obs = self.task.reset_to_demo(demo)

        if self.camera_pose:
            self.set_camera_pose(self.camera_pose)

        obs = self.obs_split(obs)

        return obs

    def step(self, action, manual_demo=False):
        action_delayed = self.postprocess_action(action,
                                                 manual_demo=manual_demo)

        zero_action = [0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, action_delayed[-1]]

        if np.isnan(action_delayed).any():
            logger.warning("NaN action, skipping")
            action_delayed = zero_action

        try:
            next_obs, reward, done = self.task.step(action_delayed)
        except (IKError and InvalidActionError):
            logger.info("Skipping invalid action {}.".format(action_delayed))

            next_obs, reward, done = self.task.step(zero_action)
        except RuntimeError as e:
            print(action_delayed)
            print(action)
            raise e

        obs = self.obs_split(next_obs)

        info = {}

        return obs, reward, done, info

    def close(self):
        self.env.shutdown()

    def obs_split(self, obs):
        # Transpose img into torch order (CHW) and scale to [0,1]
        camera_obs = obs.wrist_rgb.transpose((2, 0, 1)) / 255
        proprio_obs = np.append(obs.joint_positions, obs.gripper_open)

        return CeilingObservation(camera_obs, proprio_obs)

    def get_camera_pose(self):

        return {
            name: cam.get_pose() for name, cam in self.camera_assoc.items()
        }

    def set_camera_pose(self, pos_dict):
        for camera_name, pos in pos_dict.items():
            if camera_name in self.camera_assoc:
                camera = self.camera_assoc[camera_name]
                camera.set_pose(pos)
