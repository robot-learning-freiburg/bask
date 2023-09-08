import os

import numpy as np
import torch
from loguru import logger
from pyrep.const import RenderMode
from pyrep.errors import IKError
import rlbench
from rlbench.action_modes import ActionMode, ArmActionMode
from rlbench.backend.observation import Observation as RLBenchObservation
from rlbench.observation_config import CameraConfig, ObservationConfig
from rlbench.task_environment import InvalidActionError
from rlbench.tasks import (ArmScan, CloseMicrowave, PhoneBase, PhoneOnBase,
                           PhoneReceiver, PutRubbishInBin, TakeLidOffSaucepan)

from env.environment import BaseEnvironment
from utils.observation import (CameraOrder, SceneObservation,
                               SingleCamObservation, dict_to_tensordict,
                               empty_batchsize)

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.environ["COPPELIASIM_ROOT"]


task_switch = {
    "CloseMicrowave": CloseMicrowave,
    "TakeLidOffSaucepan": TakeLidOffSaucepan,

    "PhoneOnBase": PhoneOnBase,
    "PutRubbishInBin": PutRubbishInBin,
    "ArmScan": ArmScan,

    "PhoneBaseOnly": PhoneBase,
    "PhoneReceiverOnly": PhoneReceiver,
}


class RLBenchEnvironment(BaseEnvironment):
    def __init__(self, config):
        super().__init__(config)

        self.cameras = config["cameras"]

        assert set(self.cameras).issubset(
            {"left_shoulder", "right_shoulder", "wrist", "overhead", "front"})

        left_shoulder_on = "left_shoulder" in self.cameras
        right_shoulder_on = "right_shoulder" in self.cameras
        wrist_on = "wrist" in self.cameras
        overhead_on = "overhead" in self.cameras
        front_on = "front" in self.cameras

        render_mode = RenderMode.OPENGL
        image_size = tuple((self.image_height, self.image_width))

        obs_config = ObservationConfig(
            left_shoulder_camera=CameraConfig(
                rgb=left_shoulder_on, depth=left_shoulder_on,
                mask=left_shoulder_on, render_mode=render_mode,
                depth_in_meters=True, image_size=image_size,
                point_cloud=False),
            right_shoulder_camera=CameraConfig(
                rgb=right_shoulder_on, depth=right_shoulder_on,
                mask=right_shoulder_on, render_mode=render_mode,
                depth_in_meters=True, image_size=image_size,
                point_cloud=False),
            front_camera=CameraConfig(
                rgb=front_on, depth=front_on,
                mask=front_on, render_mode=render_mode,
                depth_in_meters=True, image_size=image_size,
                point_cloud=False),
            wrist_camera=CameraConfig(
                rgb=wrist_on, depth=wrist_on,
                mask=wrist_on, render_mode=render_mode,
                depth_in_meters=True, image_size=image_size,
                point_cloud=False),
            overhead_camera=CameraConfig(
                rgb=overhead_on, depth=overhead_on,
                mask=overhead_on, render_mode=render_mode,
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

    def launch_simulation_env(self, config: dict,
                              obs_config: ObservationConfig) -> None:
        # sphere policy uses custom action mode, ABS_EE_POSE_PLAN_WORLD_FRAME
        # for others: everything like in parent class
        if "action_mode" in config.keys():
            action_mode = ActionMode(config["action_mode"])
            self.do_postprocess_actions = False
        else:
            action_mode = ActionMode(ArmActionMode.EE_POSE_EE_FRAME)
            self.do_postprocess_actions = True

        self.env = rlbench.environment.Environment(
            action_mode,
            obs_config=obs_config,
            static_positions=config["static_env"],
            headless=config["headless_env"],
        )

        self.env.launch()

        self.task = self.env.get_task(task_switch[config["task"]])

    def close(self):
        self.env.shutdown()

    def setup_camera_controls(self, config):
        self.camera_pose = config["camera_pose"]

        camera_map = {
            "left_shoulder": self.env._scene._cam_over_shoulder_left,
            "right_shoulder": self.env._scene._cam_over_shoulder_right,
            "wrist": self.env._scene._cam_wrist,
            "overhead": self.env._scene._cam_overhead,
            "front": self.env._scene._cam_front,
        }

        self.camera_map = {k: v for k, v in camera_map.items()
                             if k in self.cameras}

    def reset(self):
        super().reset()

        descriptions, obs = self.task.reset()

        if self.camera_pose:
            self.set_camera_pose(self.camera_pose)

        obs = self.process_observation(obs)

        return obs

    def reset_to_demo(self, demo):
        super().reset()

        descriptions, obs = self.task.reset_to_demo(demo)

        if self.camera_pose:
            self.set_camera_pose(self.camera_pose)

        obs = self.process_observation(obs)

        return obs

    def _step(self, action: np.ndarray, postprocess: bool = True,
              delay_gripper: bool = True, scale_action: bool = True,
              ) -> tuple[SceneObservation, float, bool, dict]:
        """
        Postprocess the action and execute it in the environment.
        Catches invalid actions and executes a zero action instead.

        Parameters
        ----------
        action : np.ndarray
            The raw action predicted by a policy.
        postprocess : bool, optional
            Whether to postprocess the action at all, by default True
        delay_gripper : bool, optional
            Whether to delay the gripper action. Usually needed for ML
            policies, by default True
        scale_action : bool, optional
            Whether to scale the action. Usually needed for ML policies,
            by default True

        Returns
        -------
        SceneObservation, float, bool, dict
            The observation, reward, done flag and info dict.

        Raises
        ------
        RuntimeError
            If raised by the environment.
        """
        if postprocess:
            action_delayed = self.postprocess_action(
                action, scale_action=scale_action, delay_gripper=delay_gripper,
                return_euler=False)
        else:
            action_delayed = action

        # NOTE: Quaternion seems to be real-last order.
        zero_action = [0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 1.0, action_delayed[-1]]

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

        obs = self.process_observation(next_obs)

        info = {}

        return obs, reward, done, info

    def get_camera_pose(self) -> dict[str, np.ndarray]:

        return {
            name: cam.get_pose() for name, cam in self.camera_map.items()
        }

    def set_camera_pose(self, pos_dict: dict[str, np.ndarray]) -> None:
        for camera_name, pos in pos_dict.items():
            if camera_name in self.camera_map:
                camera = self.camera_map[camera_name]
                camera.set_pose(pos)

    def process_observation(self, obs: RLBenchObservation) -> SceneObservation:
        """
        Convert the observation from the environment to a SceneObservation.

        Parameters
        ----------
        obs : RLBenchObservation
            Observation as RLBench's Observation class.

        Returns
        -------
        SceneObservation
            The observation in common format as SceneObservation.
        """
        camera_obs = {}

        for cam in self.cameras:
            rgb = getattr(obs, cam + "_rgb").transpose((2, 0, 1)) / 255
            depth = getattr(obs, cam + "_depth")
            mask = getattr(obs, cam + "_mask").astype(int)
            extr = obs.misc[cam + "_camera_extrinsics"]
            intr = obs.misc[cam + "_camera_intrinsics"].astype(float)

            camera_obs[cam] = SingleCamObservation(**{
                "rgb": torch.Tensor(rgb),
                "depth": torch.Tensor(depth),
                "mask": torch.Tensor(mask).to(torch.uint8),
                "extr": torch.Tensor(extr),
                "intr": torch.Tensor(intr),
            }, batch_size=empty_batchsize)

        multicam_obs = dict_to_tensordict(
            {'_order ': CameraOrder._create(self.cameras)} | camera_obs)

        joint_pos = torch.Tensor(obs.joint_positions)
        joint_vel = torch.Tensor(obs.joint_velocities)

        ee_pose = torch.Tensor(obs.gripper_pose)
        gripper_open = torch.Tensor([obs.gripper_open])

        object_poses = dict_to_tensordict(
            {'stacked': torch.Tensor(obs.task_low_dim_state)})

        obs = SceneObservation(cameras=multicam_obs, ee_pose=ee_pose,
                               object_poses=object_poses,
                               joint_pos=joint_pos, joint_vel=joint_vel,
                               gripper_state=gripper_open,
                               batch_size=empty_batchsize)

        return obs
