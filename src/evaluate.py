import pathlib
import random
import time
from argparse import ArgumentParser

import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm

import utils.logging  # noqa
import wandb
from config import (camera_pose, encoder_configs, realsense_cam_crop,
                    realsense_cam_resolution_cropped, sim_cam_resolution)
from encoder import encoder_names
from env import Environment, get_env, import_env
from env.environment import BaseEnvironment
from policy import policy_names, policy_switch
from policy.policy import Policy
from utils.keyboard_observer import (KeyboardObserver,
                                     wait_for_environment_reset)
from utils.misc import import_config_file  # loop_sleep
from utils.misc import apply_machine_config, policy_checkpoint_name
from utils.observation import SceneObservation, random_obs_dropout
from utils.random import configure_seeds
from utils.select_gpu import device
from utils.tasks import get_task_horizon
from viz.live_keypoint import LiveKeypoints

init_griper_state = 0.9 * torch.ones(1, device=device)


def calculate_repeat_action(config, def_freq=20):
    if config["dataset_config"]["sample_freq"]:
        repeat_action = int(config["dataset_config"]["sample_correction"]
                            * def_freq/config["dataset_config"]["sample_freq"])
        logger.info(
            "Sample freq {}, correction {}, thus repeating actions {}x.",
            config["dataset_config"]["sample_freq"],
            config["dataset_config"]["sample_correction"], repeat_action)
    else:
        repeat_action = 1

    return repeat_action


def run_simulation(env: BaseEnvironment, policy: Policy, episodes: int,
                   keypoint_viz: LiveKeypoints | None = None,
                   horizon: int | None = 300, repeat_action: int = 1,
                   fragment_len: int = -1, obs_dropout: float | None = None,
                   keyboard_obs: KeyboardObserver | None = None):

    successes = 0

    time.sleep(10)

    env.reset()  # extra reset to ensure proper camera placement in RLBench

    for episode in tqdm(range(episodes)):
        wait_for_environment_reset(env, keyboard_obs)

        episode_reward, episode_length = run_episode(
            env, keyboard_obs, policy, horizon, keypoint_viz, obs_dropout,
            repeat_action, fragment_len)

        if episode_reward > 0:
            successes += 1

        wandb.log({
            "reward": episode_reward,
            "episode": episode,
            "eps_lenght": episode_length
            })

        if keypoint_viz is not None:
            keypoint_viz.reset()

    success_rate = successes / episodes
    wandb.run.summary["success_rate"] = success_rate

    env.close()

    return

def run_episode(env: BaseEnvironment, keyboard_obs: KeyboardObserver | None,
                policy: Policy, horizon: int | None,
                keypoint_viz: LiveKeypoints | None,
                obs_dropout: float | None,  repeat_action: int,
                fragment_len: int) -> tuple[float, int]:

    episode_reward = 0
    done = False
    obs = env.reset()
    lstm_state = None

    if keyboard_obs is not None:
        keyboard_obs.reset()

    policy.reset_episode()

    action = None
    step_no = 0

    pbar = tqdm(total=horizon or 1000)
    while True:
        action, step_reward, obs, done, lstm_state = process_step(
            obs, obs_dropout, policy, action, lstm_state, repeat_action,
            keypoint_viz, fragment_len, env, keyboard_obs, horizon, step_no)

        episode_reward += step_reward

        if done:
            break

        step_no += 1
        pbar.update(1)

    if keyboard_obs is not None:
        keyboard_obs.reset()

    return episode_reward, step_no + 1


def process_step(obs: SceneObservation, obs_dropout: float | None,
                 policy: Policy, action: np.ndarray | None,
                 lstm_state: tuple[torch.Tensor, torch.Tensor] | None,
                 repeat_action: int, keypoint_viz: LiveKeypoints | None,
                 fragment_len: int, env: BaseEnvironment,
                 keyboard_obs: KeyboardObserver | None,
                 horizon: int | None, step_no: int):
    # start_time = time.time()

    if fragment_len != -1 and step_no % fragment_len == 0:
        logger.info("Resetting LSTM state.")
        lstm_state = None

    obs = random_obs_dropout(obs, obs_dropout)

    # TODO: still need the case where action is None. But can rewrite else?
    obs.gripper_state = init_griper_state if action is None else \
        action[-1, None]

    action, lstm_state, info = policy.predict(obs, lstm_state)

    assert repeat_action > 0

    step_reward = 0
    done = False

    for _ in range(repeat_action):

        if keypoint_viz is not None:
            keypoint_viz.update_from_info(info, obs)

        next_obs, reward, done, env_info = env.step(action)

        step_reward += reward

        if step_no == horizon:
            done = True

        if keyboard_obs is not None:  # For real robot only
            if keyboard_obs.success:
                reward = 1
                done = True
            elif keyboard_obs.reset_button:
                reward = 0
                done = True
            else:
                reward = 0

        env.update_visualization(info)

        obs = next_obs

        # loop_sleep(start_time)

        if done:
            break

    return action, step_reward, obs, done, lstm_state


def main(config):
    Env = import_env(config)

    if config["env_config"]["env"] is Environment.PANDA:
        # from utils.keyboard_observer import KeyboardObserver
        keyboard_obs = KeyboardObserver()
    else:
        keyboard_obs = None

    Policy = policy_switch[config["policy_config"]["policy"]]
    policy = Policy(config["policy_config"]).to(device)

    file_name, _ = policy_checkpoint_name(config)
    logger.info("Loading policy checkpoint from {}", file_name)
    policy.from_disk(file_name)
    policy.eval()

    logger.info("Creating env.")
    env = Env(config["env_config"])

    keypoint_viz = LiveKeypoints.setup_from_conf(config, policy)

    task_horizon = get_task_horizon(config)

    repeat_action = calculate_repeat_action(config)

    logger.info("Running simulation.")

    run_simulation(env, policy, config["eval_config"]["episodes"],
                   keypoint_viz=keypoint_viz,
                   horizon=task_horizon,
                   repeat_action=repeat_action,
                   fragment_len=config["dataset_config"]["fragment_length"],
                   obs_dropout=config["eval_config"]["obs_dropout"],
                   keyboard_obs=keyboard_obs)
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--environment",
        default="maniskill",
        help="RLBench, Maniskill or Panda."
    )
    parser.add_argument(
        "-t", "--task",
        default="CloseMicrowave",
        help="Name of the task trained on."
    )
    # TODO: rename the feedback_type to eg data_type everywhere
    # And change the confusing name translation business
    parser.add_argument(
        "-f", "--feedback_type",
        default="cloning",
        help="The training data type. Cloning, dcm, ..."
    )
    parser.add_argument(
        "--pretrained_on",
        default=None,
        help="The task on which the encoder was pretrained. Defaults to the "
             "task on which the policy was trained."
    )
    parser.add_argument(
        "--pretrain_feedback",
        default="pretrain_manual",
        help="The data type on which the model was pretrained."
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
        default="encoder",
        help=f"Should usually be encoder. Options: {str(policy_names)[1:-1]}"
    )
    parser.add_argument(
        "-e", "--encoder",
        default=None,
        help=f"Options: {str(encoder_names)[1:-1]}"
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="Pass a suffix to load a specific policy checkpoint."
    )
    parser.add_argument(
        "--bc_step",
        default=None,
        help="Pass the number of an intermediate snapeshot to append to the "
             "policy suffix name."
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        help="Config file to use. Uses default if None provided.",
    )
    parser.add_argument(
        "--cam",
        required=True,
        nargs='+',
        help="The camera(s) to use. Options: wrist, overhead, ..."
    )
    parser.add_argument(
        "--obs_dropout",
        default=None,
        help="Probability of dropping out an observation. Default: zero.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render the simulated scene.",
    )
    parser.add_argument(
        "--no_viz",
        dest="viz",
        action="store_false",
        default=True,
        help="Disable visualization.",
    )
    parser.add_argument(
        "-d", "--disable_wandb",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-s", "--seed",
        default=None,
        help="Specify random seed for fair, reproducible evaluation."
    )

    args = parser.parse_args()
    env = get_env(args.environment)
    env_is_panda = env is Environment.PANDA

    seed = configure_seeds(args)

    if args.config is not None:
        logger.info("Using config {}", args.config)
        conf_file = import_config_file(args.config)
        encoder_configs = conf_file.encoder_configs  # noqa 811
        camera_pose = conf_file.camera_pose  # noqa 811
        try:
            policy_config = conf_file.policy_config
            logger.info("Overwriting policy config with external values.")
        except AttributeError:
            logger.info("Found no external policy config.")
            policy_config = {}
    else:
        policy_config = {}

    encoder_for_conf = args.encoder or "dummy"

    suffix = args.suffix
    if args.bc_step is not None:
        suffix += "_step_" + args.bc_step

    image_dim = realsense_cam_resolution_cropped if env_is_panda \
        else sim_cam_resolution
    image_crop = realsense_cam_crop if env_is_panda else None

    selected_encoder_config = encoder_configs[encoder_for_conf]
    selected_encoder_config["obs_config"] = {"image_dim": image_dim}

    obs_dropout = float(args.obs_dropout) if args.obs_dropout else None

    config = {
        "eval_config": {
            "episodes": 25 if env_is_panda else 200,
            "seed": seed,

            "viz": args.viz and args.render,

            "kp_per_channel_viz": False,
            "show_channels": [0, 4, 8, 12],  # None,

            "obs_dropout": obs_dropout,
        },
        "env_config": {
            "env": env,
            "task": args.task,

            "cameras": tuple(args.cam),
            "camera_pose": camera_pose,
            "image_size": image_dim,
            "image_crop": image_crop,

            "static_env": False,
            "headless_env": not args.render,

            "scale_action": True,
            "delay_gripper": True,

            "gripper_plot": True,

            # Panda keys
            "teleop": True,
            "eval": True,

            # ManiSkill keys
            "render_sapien": False,
            "background": args.background,
            "model_ids": tuple(args.model_ids),
        },

        "policy_config": {
            "policy": args.policy,

            "lstm_layers": 2,
            "n_cams": len(args.cam),
            "use_ee_pose": True,  # else uses joint angles
            "add_gripper_state": False,
            "action_dim": 7,
            "learning_rate": 3e-4,
            "weight_decay": 3e-6,

            "visual_embedding_dim": 256,

            "encoder": args.encoder,
            "encoder_config": selected_encoder_config,
            "suffix": suffix,
            },
        "dataset_config": {
            "feedback_type": args.feedback_type,
            "task": args.task,

            "pretrained_on": args.pretrained_on or args.task,
            "pretrain_feedback": args.pretrain_feedback,

            "cameras": args.cam,

            "sample_freq": None,  # 5  # downsample the trajectories to this freq
            "sample_correction": 1,  # correction factor for subsampled data
            # NOTE: not resetting the LSTM state seems to work better
            "fragment_length": -1,  # 30,

            "data_root": "data",
        },
    }

    config["policy_config"].update(policy_config)

    config = apply_machine_config(config)

    if config["policy_config"]["use_ee_pose"]:
        config["policy_config"]["proprio_dim"] = 7
    else:
        config["policy_config"]["proprio_dim"] = 8

    if config["policy_config"]["add_gripper_state"]:
        config["policy_config"]["proprio_dim"] += 1

    wandb_mode = "offline" if env_is_panda else \
        "disabled" if args.disable_wandb else "online"
    wandb.init(config=config, project="bask_eval", mode=wandb_mode)
    wandb.run.summary["bc_step"] = args.bc_step

    wandb_config = wandb.config  # in case the sweep gives different values
    main(config)
