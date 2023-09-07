from types import MappingProxyType

import cv2
import gym
import mani_skill2.envs  # noqa: F401
import numpy as np
import torch
from loguru import logger

from env.environment import BaseEnvironment
from utils.geometry import np_invert_homogenous_transform
from utils.misc import invert_dict
from utils.observation import (CameraOrder, SceneObservation,
                               SingleCamObservation, dict_to_tensordict,
                               empty_batchsize)

ACTION_MODE = "pd_ee_delta_pose"
OBS_MODE = "state_dict+image"


default_cameras = tuple(("hand_camera", "base_camera"))

cam_name_tranlation = MappingProxyType({
    "hand_camera": "wrist",
    "base_camera": "base",
    "overhead": "overhead",
})

inv_cam_name_tranlation = invert_dict(cam_name_tranlation)


class ManiSkillEnv(BaseEnvironment):
    def __init__(self, config):
        super().__init__(config)

        # NOTE: removed ms_config dict. Just put additional kwargs into the
        # config dict and treat them here and in launch_simulation_env.

        # ManiSkill controllers have action space normalized to [-1,1].
        # Max speed is a bit fast for teleop, so scale down.
        self._delta_pos_scale = 0.25
        self._delta_angle_scale = 0.5

        config['real_depth'] = False

        self.cameras = config["cameras"]
        self.cameras_ms = [inv_cam_name_tranlation[c] for c in self.cameras]

        image_size = tuple((self.image_height, self.image_width))

        self.camera_cfgs = {
            "width": image_size[1],
            "height": image_size[0],
            "use_stereo_depth": config.get("real_depth", False),
            "add_segmentation": True,
            # NOTE: these are examples of how to pass camera params.
            # Should specify these in the config file.
            # "overhead": {  # can pass specific params per cam as well
            #     'p': [0.2, 0, 0.2],
            #     # Quaternions are [w, x, y, z]
            #     'q': [7.7486e-07, -0.194001, 7.7486e-07, 0.981001]
            # },
            # "base_camera": {
            #     'p': [0.2, 0, 0.2],
            #     'q': [0, 0.194, 0, -0.981]  # Quaternions are [w, x, y, z]
            # }
        }

        self.extra_cams = []

        for c, pq in config['camera_pose'].items():
            ms_name = inv_cam_name_tranlation[c]
            if self.camera_cfgs.get(ms_name) is None:
                self.camera_cfgs[ms_name] = {}
            if ms_name not in default_cameras:
                self.extra_cams.append(ms_name)
            self.camera_cfgs[ms_name]['p'] = pq[:3]
            self.camera_cfgs[ms_name]['q'] = pq[3:]

        self.task_name = config["task"]
        self.headless = config["headless_env"]

        self.gym_env = None

        self.render_sapien = config.get("render_sapien", False)
        self.bg_name = config.get("background", None)
        self.model_ids = config.get("model_ids", [])

        self.seed = config.get("seed", None)

        if config.get("static_env"):
            raise NotImplementedError

        # NOTE: would like to make the horizon configurable, but didn't figure
        # it out how to make this work with the Maniskill env registry. TODO
        # self.horizon = -1

        if self.model_ids is None:
            self.model_ids = []

        if not self.render_sapien and not self.headless:
            self.cam_win_title = "Observation"
            self.camera_rgb_window = cv2.namedWindow(self.cam_win_title,
                                                     cv2.WINDOW_AUTOSIZE)

        self._patch_register_cameras()
        self.launch_simulation_env(config)

    @property
    def camera_names(self):
        return tuple([cam_name_tranlation[c]
                      for c in self.gym_env.env._camera_cfgs.keys()])

    @property
    def agent(self):
        return self.gym_env.agent

    @property
    def robot(self):
        return self.agent.robot

    def get_solution_sequence(self):
        return self.gym_env.env._get_solution_sequence()

    def _patch_register_cameras(self):
        from mani_skill2.sensors.camera import CameraConfig
        from mani_skill2.utils.sapien_utils import look_at

        # from sapien.core import Pose as SapienPose

        envs = [mani_skill2.envs.pick_and_place.pick_clutter.PickClutterEnv,
                mani_skill2.envs.pick_and_place.pick_cube.PickCubeEnv,
                mani_skill2.envs.pick_and_place.pick_cube.LiftCubeEnv,
                mani_skill2.envs.pick_and_place.pick_clutter.PickClutterYCBEnv,
                mani_skill2.envs.pick_and_place.stack_cube.StackCubeEnv,
                mani_skill2.envs.pick_and_place.pick_single.PickSingleEGADEnv,
                mani_skill2.envs.pick_and_place.pick_single.PickSingleYCBEnv,
                # mani_skill2.envs.assembly.assembling_kits.AssemblingKitsEnv,
                # TODO: for some reason, these two break upon patching
                # mani_skill2.envs.assembly.peg_insertion_side.PegInsertionSideEnv,
                # mani_skill2.envs.assembly.plug_charger.PlugChargerEnv
                ]

        if self.task_name in ['PegInsertionSide-v0', 'PlugCharger-v0']:
            logger.opt(ansi=True).warning(
                f"Skipping camera patching for {self.task_name}. "
                "<red>This disables camera customization, including the "
                "overhead camera.</red> See code for details.")
            self.camera_cfgs.pop("overhead")

        for env in envs:
            _orig_register_cameras = env._register_cameras

            def _register_cameras(self):
                cfgs = _orig_register_cameras(self)
                if type(cfgs) is CameraConfig:
                    cfgs = [cfgs]
                pose = look_at([0, 0, 0], [0, 0, 0])
                for c in self._extra_camera_names:
                    if c == "base_camera":
                        continue
                    else:
                        logger.info(f"Registering camera {c}")
                        cfgs.append(CameraConfig(
                            c, pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10)
                        )
                return cfgs

            env._extra_camera_names = self.extra_cams
            env._register_cameras = _register_cameras

    def launch_simulation_env(self, config):

        if config.get("action_mode", None) is not None:
            raise NotImplementedError

        env_name = self.task_name

        kwargs = {
            "obs_mode": OBS_MODE,
            "control_mode": ACTION_MODE,
            "camera_cfgs": self.camera_cfgs,
            "shader_dir": "rt" if config.get("real_depth", False) else "ibl",
            # "render_camera_cfgs": dict(width=640, height=480)
            "bg_name": self.bg_name,
            "model_ids": self.model_ids,
            # "max_episode_steps": self.horizon,
        }

        if not self.task_name.startswith("Pick"):
            kwargs.pop('model_ids')  # model_ids only needed for pick tasks

        # NOTE: full list of arguments
        # obs_mode = None,
        # reward_mode = None,  Don't need a reward when imitating.
        # control_mode = None,
        # sim_freq: int = 500,
        # control_freq: int = 20, That's what I use already.
        # renderer: str = "sapien",
        # renderer_kwargs: dict = None,
        # shader_dir: str = "ibl",
        # render_config: dict = None,
        # enable_shadow: bool = False,
        # camera_cfgs: dict = None,
        # render_camera_cfgs: dict = None,
        # bg_name: str = None,

        self.gym_env = gym.make(env_name, **kwargs)

        if self.seed is not None:
            self.gym_env.seed(self.seed)

    def render(self):
        if not self.headless:
            if self.render_sapien:
                self.gym_env.render('human')
            else:
                obs = self.gym_env.render("cameras")
                cv2.imshow(self.cam_win_title, obs)
                cv2.waitKey(1)

    def reset(self, **kwargs):
        super().reset()

        obs = self.gym_env.reset(**kwargs)

        obs = self.obs_split(obs)

        return obs

    def reset_to_demo(self, demo):
        self.gym_env.reset(**demo["reset_kwargs"])

    def set_state(self, state):
        self.gym_env.set_state(state)

    def _step(self, action: np.ndarray, postprocess: bool = True,
              delay_gripper: bool = True, scale_action: bool = True,
              invert_xy: bool = True
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
        invert_xy : bool, optional
            Whether to invert x and y translation. Makes it easier to teleop
            in ManiSkill because of the base camera setup, by default True

        Returns
        -------
        SceneObservation, float, bool, dict
            The observation, reward, done flag and info dict.

        Raises
        ------
        Exception
            Do not yet know how ManiSkill handles invalid actions, so raise
            an exception if it occurs in stepping the action.
        """
        # TODO: don't postprocess for GMMs.
        if postprocess:
            action = self.postprocess_action(
                action, scale_action=scale_action, delay_gripper=delay_gripper,
                return_euler=True)
        else:
            action = action

        if invert_xy:
            # Invert x, y movement and rotation, but not gripper and z.
            action[:2] = -action[:2]
            action[3:-2] = -action[3:-2]

        zero_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, action[-1]]

        if np.isnan(action).any():
            logger.warning("NaN action, skipping")
            action = zero_action

        try:
            next_obs, reward, done, info = self.gym_env.step(action)
        except Exception as e:
            logger.info("Skipping invalid action {}.".format(action))

            logger.warning("Don't yet know how ManiSkill handles invalid actions")
            raise e

            next_obs, reward, done, info = self.gym_env.step(zero_action)
        except RuntimeError as e:
            print(action)
            print(action)
            raise e

        obs = self.obs_split(next_obs)

        self.render()

        return obs, reward, done, info

    def close(self):
        self.gym_env.close()

    def obs_split(self, obs: dict) -> SceneObservation:
        """
        Convert the observation dict from ManiSkill to a SceneObservation.

        Parameters
        ----------
        obs : dict
            The observation dict from ManiSkill.

        Returns
        -------
        SceneObservation
            The observation in common format as a TensorClass.
        """
        cam_obs = obs['image']
        cam_names = cam_obs.keys()

        translated_names = [cam_name_tranlation[c] for c in cam_names]
        assert set(self.cameras).issubset(set(translated_names))

        cam_rgb = {
            cam_name_tranlation[c]: cam_obs[c]['Color'][:, :, :3].transpose(
                (2, 0, 1))
            for c in cam_names
        }

        # Negative depth is channel 2 in the position tensor.
        # See https://insiders.vscode.dev/github/vonHartz/ManiSkill2/blob/main/mani_skill2/sensors/depth_camera.py#L100-L101
        cam_depth = {
            cam_name_tranlation[c]: -cam_obs[c]['Position'][:, :, 2]
            for c in cam_names
        }

        # NOTE channel 0 is mesh-wise, channel 1 is actor-wise, see
        # https://sapien.ucsd.edu/docs/latest/tutorial/rendering/camera.html#visualize-segmentation
        cam_mask = {
            cam_name_tranlation[c]: cam_obs[c]['Segmentation'][:, :, 0]
            for c in cam_names
        }

        # Invert extrinsics for consistency with RLBench, Franka. cam2world vs world2cam.
        cam_ext = {
            cam_name_tranlation[c]: np_invert_homogenous_transform(
                obs['camera_param'][c]['extrinsic_cv'])
            for c in cam_names
        }

        cam_int = {
            cam_name_tranlation[c]: obs['camera_param'][c]['intrinsic_cv']
            for c in cam_names
        }

        ee_pose = torch.Tensor(obs['extra']['tcp_pose'])
        object_poses = dict_to_tensordict({
            k: torch.Tensor(v) for k, v in obs['extra'].items()
                if k.endswith('pose') and k != 'tcp_pose'
        })

        joint_pos = torch.Tensor(obs['agent']['qpos'])
        joint_vel = torch.Tensor(obs['agent']['qvel'])

        # NOTE: the last two dims seem to be the individual fingers
        joint_pos, finger_pose = joint_pos.split([7, 2])
        joint_vel, finger_vel = joint_vel.split([7, 2])

        multicam_obs = dict_to_tensordict(
            {'_order': CameraOrder._create(self.cameras)} | {
                c: SingleCamObservation(**{
                    'rgb': torch.Tensor(cam_rgb[c]),
                    'depth': torch.Tensor(cam_depth[c]),
                    'mask': torch.Tensor(cam_mask[c].astype(np.uint8)).to(
                        torch.uint8),
                    'extr': torch.Tensor(cam_ext[c]),
                    'intr': torch.Tensor(cam_int[c])
            }, batch_size=empty_batchsize
            ) for c in self.cameras
        }
        )

        obs = SceneObservation(cameras=multicam_obs, ee_pose=ee_pose,
                               object_poses=object_poses,
                               joint_pos=joint_pos, joint_vel=joint_vel,
                               gripper_state=finger_pose,
                               batch_size=empty_batchsize)

        return obs


    def get_replayed_obs(self):
        # To be used from extract_demo.py
        obs = self.gym_env._episode_data[0]['o']
        print(obs)
        done = self.gym_env._episode_data[0]['d']
        reward = self.gym_env._episode_data[0]['r']
        info = self.gym_env._episode_data[0]['info']
