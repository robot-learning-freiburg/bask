import cv2
import gym
import mani_skill2.envs  # noqa: F401
import numpy as np
from loguru import logger

from env.environment import BaseEnvironment
from env.observation import MSObservation
from utils.geometry import np_invert_homogenous_transform

ACTION_MODE = "pd_ee_delta_pose"
OBS_MODE = "state_dict+image"


cam_name_tranlation = {
    "hand_camera": "w",
    "base_camera": "b",
    "overhead": "o",
}

class ManiSkillEnv(BaseEnvironment):
    def __init__(self, config, ms_config=None):
        super().__init__(config)

        # ManiSkill controllers have action space normalized to [-1,1].
        # Max speed is a bit fast for teleop, so scale down.
        self._delta_pos_scale = 0.25
        self._delta_angle_scale = 0.5

        if "image_size" in config and config["image_size"] is not None:
            image_size = config["image_size"]
        else:
            image_size = (256, 256)

        config['real_depth'] = False

        self.camera_cfgs = {
            "width": image_size[1],
            "height": image_size[0],
            "use_stereo_depth": config.get("real_depth", False),
            "add_segmentation": True,
            "overhead": {  # can pass specific params per cam as well
                # 'p': [0.2, 0, 0.2],
                # Quaternions are [w, x, y, z]
                # 'q': [7.7486e-07, -0.194001, 7.7486e-07, 0.981001]
            },
            "base_camera": {
                'p': [0.2, 0, 0.2],
                'q': [0, 0.194, 0, -0.981]  # Quaternions are [w, x, y, z]
            }
        }

        self.extra_cams = []

        for c, pq in config['camera_pose'].items():
            if self.camera_cfgs.get(c) is None:
                self.camera_cfgs[c] = {}
            self.extra_cams.append(c)
            self.camera_cfgs[c]['p'] = pq[:3]
            self.camera_cfgs[c]['q'] = pq[3:]

        self.task_name = config["task"]
        self.headless = config["headless_env"]
        self.gym_env = None
        self.camera_pose = None

        self.render_sapien = config.get("render_sapien", False)
        self.bg_name = config.get("background", None)
        self.model_ids = config.get("model_ids", [])

        self.seed = config.get("seed", None)

        # self.horizon = -1

        if self.model_ids is None:
            self.model_ids = []

        self.ms_config = ms_config or {}

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

        # TODO: need recursive update? Also switch order to be sure?
        kwargs.update(self.ms_config)

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

        # if self.camera_pose:
        #     self.set_camera_pose(self.camera_pose)

        obs = self.obs_split(obs)

        return obs

    def reset_to_demo(self, demo):
        self.gym_env.reset(**demo["reset_kwargs"])

    def set_state(self, state):
        self.gym_env.set_state(state)

    def step(self, action, manual_demo=False, postprocess=True, invert=True):
        # TODO: don't postprocess for GMMs.
        if postprocess:
            action_delayed = self.postprocess_action(action,
                                                     manual_demo=manual_demo,
                                                     return_euler=True)
        else:
            action_delayed = action


        if invert:
            # Invert x, y movement and rotation, but not gripper and z.
            action_delayed[:2] = -action_delayed[:2]
            action_delayed[3:-2] = -action_delayed[3:-2]

        zero_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, action_delayed[-1]]

        if np.isnan(action_delayed).any():
            logger.warning("NaN action, skipping")
            action_delayed = zero_action

        try:
            next_obs, reward, done, info = self.gym_env.step(action_delayed)
        except Exception as e:
            logger.info("Skipping invalid action {}.".format(action_delayed))

            logger.warning("Don't yet know how ManiSkill handles invalid actions")
            raise e

            next_obs, reward, done, info = self.gym_env.step(zero_action)
        except RuntimeError as e:
            print(action_delayed)
            print(action)
            raise e

        obs = self.obs_split(next_obs)

        self.render()

        return obs, reward, done, info

    def shutdown(self):
        self.gym_env.close()

    def close(self):
        self.gym_env.close()

    def obs_split(self, obs):
        cam_obs = obs['image']
        cam_names = cam_obs.keys()

        cam_rgb = {
            cam_name_tranlation[c]: cam_obs[c]['Color'][:, :, :3].transpose(
                (2, 0, 1))
            for c in cam_names
        }

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

        cam_ext = {
            cam_name_tranlation[c]: np_invert_homogenous_transform(
                obs['camera_param'][c]['extrinsic_cv'])
            for c in cam_names
        }

        cam_int = {
            cam_name_tranlation[c]: obs['camera_param'][c]['intrinsic_cv']
            for c in cam_names
        }

        ee_pose = obs['extra']['tcp_pose']
        object_poses = {
            k: v for k, v in obs['extra'].items() \
                if k.endswith('pose') and k != 'tcp_pose'
        }

        # NOTE: the last two dims seem to be the individual fingers
        # TODO: use avg of both?
        joint_pose = obs['agent']['qpos']
        joint_vel = obs['agent']['qvel']

        proprio_obs = np.append(joint_pose, joint_vel)

        # print("ext", obs['camera_param']['base_camera']['extrinsic_cv'])
        # print("ext", obs['camera_param']['overhead']['extrinsic_cv'])
        # print('c2w', obs['camera_param']['base_camera']['cam2world_gl'])
        # # print(obs['extra'].keys())
        # print("robo", obs['agent']['base_pose'])
        # print("obj", obs['extra']['obj_pose'])
        # print("ee", ee_pose)

        return MSObservation(ee_pose, proprio_obs, object_poses,
                             cam_rgb, cam_depth, cam_mask, cam_ext, cam_int)


    def get_replayed_obs(self):
        # To be used from extract_demo.py
        obs = self.gym_env._episode_data[0]['o']
        print(obs)
        done = self.gym_env._episode_data[0]['d']
        reward = self.gym_env._episode_data[0]['r']
        info = self.gym_env._episode_data[0]['info']
