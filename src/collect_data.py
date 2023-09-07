import pathlib
# import pprint
import random
import time
from argparse import ArgumentParser

import torch
from loguru import logger
from tqdm.auto import tqdm

import utils.logging  # noqa
from config import camera_pose, realsense_cam_resolution, sim_cam_resolution
from dataset.scene import SceneDataset
from env import Environment, import_env
from policy import PolicyEnum, get_policy_class
from utils.keyboard_observer import KeyboardObserver
from utils.misc import (apply_machine_config, get_dataset_name,
                        get_full_task_name, loop_sleep)
from utils.random import configure_seeds


def main(config):
    Env = import_env(config)

    Policy = get_policy_class(config["policy_config"]["policy"].value)

    task_name = get_full_task_name(config)

    save_path = pathlib.Path(
        config["dataset_config"]["data_root"]) / task_name
    if not save_path.is_dir():
        logger.warning("Creating save path. This should only be needed for "
                       "new tasks.")
        save_path.mkdir(parents=True)

    env = Env(config["env_config"])

    keyboard_obs = KeyboardObserver()

    replay_memory = SceneDataset(
        allow_creation=True, subsample_by_difference=False,
        camera_names=config["env_config"]["cameras"],
        subsample_to_length=config["dataset_config"]["sequence_len"],
        data_root=save_path / config["dataset_config"]["dataset_name"],
        image_size=config["env_config"]["image_size"])


    env.reset()  # extra reset to correct set up of camera poses in first obs

    # TODO: policy config?
    policy = Policy(config, env=env, keyboard_obs=keyboard_obs)

    obs = env.reset()

    time.sleep(5)

    logger.info("Go!")

    episodes_count = 0
    lstm_state = None

    try:
        with tqdm(total=config["n_episodes"]) as pbar:
            while episodes_count < config["n_episodes"]:
                start_time = time.time()

                action, lstm_state, _ = policy.predict(obs, lstm_state)
                next_obs, _ , done, _ = env.step(action)
                obs.action = torch.Tensor(action)
                obs.feedback = torch.Tensor([1])
                replay_memory.add_observation(obs)
                obs = next_obs

                if done or keyboard_obs.success:
                    logger.info("Saving trajectory.")
                    replay_memory.save_current_traj()

                    obs = env.reset()
                    keyboard_obs.reset()
                    policy.reset_episode(env)

                    episodes_count += 1
                    pbar.update(1)

                    done = False
                    lstm_state = None

                elif keyboard_obs.reset_button:
                    logger.info("Resetting without saving traj.")
                    replay_memory.reset_current_traj()

                    obs = env.reset()
                    keyboard_obs.reset()
                    policy.reset_episode(env)

                    lstm_state = None
                else:
                    loop_sleep(start_time)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Attempting graceful shutdown of env...")
        env.close()

    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--pretraining",
        action="store_true",
        default=False,
        help="Wether the data is for pretraining. Used to name the data dir."
    )
    parser.add_argument(
        "-e", "--environment",
        default="rlbench",
        help="RLBench, Maniskill or Panda."
    )
    parser.add_argument(
        "-t", "--task",
        default="PhoneOnBase",
        help="The task to perform."
    )
    parser.add_argument(
        "-n", "--number",
        default="10",
        help="Number of episodes to collect."
    )
    parser.add_argument(
        "-b", "--background",
        default=None,
        help="Maniskill only. Environment background to use."
    )
    parser.add_argument(
        "--model_ids",
        nargs="+",
        default=[],
        help="Maniskill only. Model ids to use for YCB."
    )
    parser.add_argument(
        "-p", "--policy",
        default="manual",
    )
    parser.add_argument(
        "--cam",
        nargs="+",
        default=['wrist', 'base', 'overhead'],
        help="The cameras to collect."
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=None,
        help="Specify random seed. NOTE: not yet tested if leads to "
             "reproducible envs."
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        help="Config file to use. Uses default if not given."
    )

    args = parser.parse_args()

    env = Environment[args.environment.upper()]
    env_is_panda = env is Environment.PANDA

    policy = PolicyEnum[args.policy.upper()]

    image_dim = realsense_cam_resolution if env_is_panda else \
        sim_cam_resolution

    n_episodes = int(args.number)

    config = {
        "n_episodes": n_episodes,

        "env_config": {
            "env": env,
            "task": args.task,

            "cameras": tuple(args.cam),
            "camera_pose": camera_pose,
            "image_size": image_dim,

            "static_env": False,
            "headless_env": False,

            "scale_action": True,
            "delay_gripper": True,

            "gripper_plot": True,

            # Panda keys
            "teleop": True,
            "eval": False,

            # ManiSkill keys
            "render_sapien": False,
            "background": args.background,
            "model_ids": tuple(args.model_ids),
        },
        "dataset_config": {
            "data_root": "data",
            "dataset_name": None,
            "pretraining": args.pretraining,

            "sequence_len": None,
        },
        "policy_config": {
            "policy": policy,

        },
    }

    if args.policy == "sphere":
        assert env is Environment.RLBENCH
        from rlbench.action_modes import ArmActionMode
        config["env_config"]["action_mode"] \
            = ArmActionMode.ABS_JOINT_POSITION

    config["dataset_config"]["dataset_name"] = get_dataset_name(config)

    seed = configure_seeds(args)
    config["seed"] = seed

    config = apply_machine_config(config)

    # logger.info("Config:")
    # logger.info(pprint.pformat(config))

    main(config)
