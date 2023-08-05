import itertools
import pathlib
import random
import time
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm

import utils.logging  # noqa
import wandb
from config import camera_pose, encoder_configs
from policy import policy_switch
from utils.kp_dist_tracker import KP_Dist_Tracker
from utils.misc import import_config_file  # loop_sleep
from utils.misc import (apply_machine_config, get_is_manual_demo,
                        policy_checkpoint_name, set_seeds)
from utils.select_gpu import device
# from utils.tasks import tasks
from viz.live_keypoint import LiveKeypoints

task_horizons = {
     "CloseMicrowave": 300,  # i.e. 15 seconds
     "TakeLidOffSaucepan": 300,
     "PhoneOnBase": 600,
     "PutRubbishInBin": 600,
}

task_horizons = defaultdict(lambda: 300, task_horizons)


init_griper_state = 0.9 * torch.ones(1, device=device)


def get_task_horizon(config):
    if config["eval_config"]["panda"]:
        return None
    else:
        return task_horizons[config["eval_config"]["task"]]


def dropout_single_camera(obs, cam_attributes):
    for attr in cam_attributes:
        val = getattr(obs, attr)
        if val is not None:
            setattr(obs, attr, np.zeros_like(val))

    return obs


# TODO: need to drop out sequence of obs, eg a second, hence 20 obs?
def random_obs_dropout(obs, p=None, drop_all=False):
    if p is None:
        return obs

    attr = (("cam_w_rgb", "cam_w_d"),
            ("cam_o_rgb", "cam_o_d"),
            ("cam_r_rgb", "cam_r_d"),
            ("cam_l_rgb", "cam_l_d"))
    # attr = (("cam_o_rgb", "cam_o_d"),)

    if drop_all:
        sample = np.random.binomial(1, p)
        if sample:
            for camera_attrs in attr:
                obs = dropout_single_camera(obs, camera_attrs)
    else:
        for camera_attrs in attr:
            sample = np.random.binomial(1, p)
            if sample:
                obs = dropout_single_camera(obs, camera_attrs)

    return obs


@logger.contextualize(filter=False)
def wait_for_environment_reset(env, keyboard_obs):
    if keyboard_obs is not None:
        env.reset()
        logger.info("Waiting for env reset. Confirm via input ...")
        while True:
            env.get_obs()
            time.sleep(0.5)
            if keyboard_obs.reset_button or keyboard_obs.success:
                keyboard_obs.reset()
                break


def run_simulation(env, policy, episodes, manual_demo, cam, keypoint_viz=None,
                   horizon=300, repeat_action=1, fragment_len=-1,
                   track_kp_dist=False, obs_dropout=None, keyboard_obs=None):

    successes = 0

    time.sleep(10)

    kp_dist_tracker = KP_Dist_Tracker() if track_kp_dist else None

    env.reset()  # extra reset to ensure proper camera placement

    for episode in tqdm(range(episodes)):
        wait_for_environment_reset(env, keyboard_obs)

        episode_reward, episode_length = run_episode(
            env, keyboard_obs, policy, horizon, keypoint_viz, kp_dist_tracker,
            obs_dropout, cam, repeat_action, fragment_len, manual_demo)

        if episode_reward > 0:
            successes += 1

        wandb.log({"reward": episode_reward, "episode": episode,
                   "eps_lenght": episode_length})

        if keypoint_viz is not None:
            keypoint_viz.reset()

    if track_kp_dist:
        save_path = pathlib.Path("tmp")
        kp_dist_tracker.aggregate_and_save_episodes(save_path)

    success_rate = successes / episodes
    wandb.run.summary["success_rate"] = success_rate

    return

def run_episode(env, keyboard_obs, policy, horizon, keypoint_viz,
                kp_dist_tracker, obs_dropout, cam, repeat_action,
                fragment_len, manual_demo):

    episode_reward = 0
    done = False
    obs = env.reset()
    lstm_state = None

    if keyboard_obs is not None:
        keyboard_obs.reset()

    try:
        policy.encoder.reset_traj()
    except AttributeError:
        pass

    if kp_dist_tracker is not None:
        kp_dist_tracker.add_episode()

    action = None
    step_no = 0

    pbar = tqdm(total=horizon or 1000)
    while True:
        action, episode_reward, obs, done, lstm_state = process_step(
            obs, obs_dropout, policy, action, lstm_state, cam, kp_dist_tracker,
            repeat_action, keypoint_viz, fragment_len, env, keyboard_obs,
            manual_demo, horizon, step_no, episode_reward)

        if done:
            break

        step_no += 1
        pbar.update(1)

    if kp_dist_tracker is not None:
        kp_dist_tracker.process_episode()

    if keyboard_obs is not None:
        keyboard_obs.reset()

    return episode_reward, step_no + 1


def process_step(obs, obs_dropout, policy, action, lstm_state, cam,
                 kp_dist_tracker, repeat_action, keypoint_viz, fragment_len,
                 env, keyboard_obs, manual_demo, horizon, step_no,
                 episode_reward):
    # start_time = time.time()

    # NOTE: could reset lstm state, but found better not to do that
    # if fragment_len != -1 and step_no % fragment_len == 0:
    #     lstm_state = None

    obs = random_obs_dropout(obs, obs_dropout)

    obs.gripper_state = init_griper_state if action is None else \
        action[-1, None]

    action, lstm_state, info = policy.predict(obs, lstm_state, cam=cam)

    if kp_dist_tracker is not None:
        kp_dist_tracker.add_step(info)

    for _ in range(repeat_action):

        if keypoint_viz is not None:
            keypoint_viz.update_from_info(info, obs)

        next_obs, reward, done = env.step(action, manual_demo=manual_demo)

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

            env.update_viz(obs.cam_w_rgb, info['vis_encoding'][0])  # HACK

        obs = next_obs
        episode_reward += reward

        # loop_sleep(start_time)

        if done:
            break

    return action, episode_reward, obs, done, lstm_state


def main(config, raw_config, refine_path=None):
    if config["eval_config"]["panda"]:
        from env.franka import FrankaEnv as Env
        from utils.keyboard_observer import KeyboardObserver
        keyboard_obs = KeyboardObserver()
    else:
        from env.ground_truth import GTEnv as Env
        keyboard_obs = None

    Policy = policy_switch[config["policy_config"]["policy"]]
    policy = Policy(raw_config["policy_config"]).to(device)
    file_name, _ = policy_checkpoint_name(raw_config)
    logger.info("Loading policy checkpoint from {}", file_name)
    policy.from_disk(file_name)
    # HACK: hotfix. TODO: do this properly
    if refine_path is not None:
        try:
            policy.encoder.particle_filter.ref_pixel_world = \
                torch.load(refine_path)['ref_pixel_world']
        except Exception as e:
            raise e
    policy.eval()

    logger.info("Creating env.")
    env = Env(raw_config["eval_config"], eval=True)

    manual_demo = get_is_manual_demo(config)
    logger.info("Using postprocessing for manual demo: {}", manual_demo)

    keypoint_viz = LiveKeypoints.setup_from_conf(raw_config, policy)

    if config["policy_config"]["encoder"] == "keypoints_var":
        track_kp_dist = True
        logger.info("Tracking keypoint distances.")
    else:
        track_kp_dist = False

    task_horizon = get_task_horizon(config)

    if config["dataset_config"]["sample_freq"]:
        repeat_action = int(config["dataset_config"]["sample_correction"]
                            * 20/config["dataset_config"]["sample_freq"])
        logger.info(
            "Sample freq {}, correction {}, thus repeating actions {}x.",
            config["dataset_config"]["sample_freq"],
            config["dataset_config"]["sample_correction"], repeat_action)
    else:
        repeat_action = 1

    logger.info("Running simulation.")

    run_simulation(env, policy, config["eval_config"]["episodes"], manual_demo,
                   config["dataset_config"]["cams"],
                   keypoint_viz=keypoint_viz, horizon=task_horizon,
                   repeat_action=repeat_action,
                   fragment_len=config["dataset_config"]["fragment_length"],
                   track_kp_dist=track_kp_dist,
                   obs_dropout=config["eval_config"]["obs_dropout"],
                   keyboard_obs=keyboard_obs)
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--feedback_type",
        dest="feedback_type",
        default="cloning",
        help="options: cloning",
    )
    parser.add_argument(
        "-t",
        "--task",
        dest="task",
        default="CloseMicrowave",
        # help="options: {}, 'Mixed'".format(str(tasks)[1:-1]),
    )
    parser.add_argument(
        "--pretrained_on",
        dest="pretrained_on",
        default=None,
        help="task on which the encoder was pretrained, "
        #      "options: {}, 'Mixed'".format(str(tasks)[1:-1]),
    )
    parser.add_argument(
        "--pretrain_feedback",
        dest="pretrain_feedback",
        default=None,
        help="The data on which the model was pretrained, eg. dcs_20",
    )
    parser.add_argument(
        "-e",
        "--encoder",
        dest="encoder",
        default=None,
        help="options: transporter, bvae, monet, keypoints"
    )
    parser.add_argument(
        "-p",
        "--policy",
        dest="policy",
        default="ceiling",
        help="options: ceiling, encoder"
    )
    parser.add_argument(
        "-m",
        "--mask",
        dest="mask",
        action="store_true",
        default=False,
        help="Use data with ground truth object masks.",
    )
    parser.add_argument(
        "-d",
        "--disable_wandb",
        dest="disable_wandb",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-s",
        "--seed",
        dest="seed",
        default=None,
        help="Specify random seed for fair evalaution."
    )
    parser.add_argument(
        "--suffix",
        dest="suffix",
        default=None,
        help="Pass a suffix to load a specific policy checkpoint."
    )
    parser.add_argument(
        "--bc_step",
        dest="bc_step",
        default=None,
        help="Pass a number of an intermediate snapeshot to append to the "
             "suffix name."
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default=None,
        help="Config file to use. Uses default if None provided.",
    )
    parser.add_argument(
        "--cam",
        dest="cam",
        required=True,
        nargs='+',
        help="The camera(s) to use. Options: wrist, overhead."
    )
    parser.add_argument(
        "-o",
        "--object_pose",
        dest="object_pose",
        action="store_true",
        default=False,
        help="Use data with ground truth object positions.",
    )
    parser.add_argument(
        "-r",
        "--refine_init",
        dest="refine_init",
        default=None,
        help="Load ref world coordinates from a file for pf init refinement.",
    )
    parser.add_argument(
        "--obs_dropout",
        dest="obs_dropout",
        help="Probability of dropping out an observation. Default: zero.",
    )
    parser.add_argument(
        "--render",
        dest="render",
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
        "--panda",
        dest="panda",
        action="store_true",
        default=False,
        help="Use the real panda robot.",
    )
    args = parser.parse_args()

    seed = int(args.seed) if args.seed else random.randint(0, 2000)
    logger.info("Seed: {}", seed)
    set_seeds(seed)

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
    if encoder_for_conf == "keypoints_var":
        encoder_for_conf = "keypoints"

    suffix = args.suffix
    if args.bc_step is not None:
        suffix += "_step_" + args.bc_step

    image_dim = (480, 480) if args.panda else (256, 256)

    selected_encoder_config = encoder_configs[encoder_for_conf]
    selected_encoder_config["obs_config"] = {
        "image_dim": image_dim
    }

    config_defaults = {
        "eval_config": {
            "episodes": 20 if args.panda else 200,
            "seed": seed,

            "panda": args.panda,
            "teleop": True,

            "crop_left": 160 if args.panda else None,

            "task": args.task,

            "shoulders_on": False,
            "wrist_on": True,
            "overhead_on": True,

            # TODO: this should be set automatically or configured.
            # does this currently lead to issues with the real robot?
            "image_size": image_dim,
            "camera_pose": camera_pose,

            "static_env": False,
            "headless_env": not args.render,
            "viz": args.viz and args.render,

            "kp_per_channel_viz": False,
            "show_channels": [0, 4, 8, 12],  # None,

            "obs_dropout": float(
                args.obs_dropout) if args.obs_dropout else None,
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

            "ground_truth_mask": args.mask or args.object_pose,
            "ground_truth_object_pose": args.object_pose,

            "cams": args.cam,

            "sample_freq": None,  # 5  # downsample the trajectories to this freq
            "sample_correction": 1,  # correction factor for subsampled data
            "fragment_length": 30,

            "data_root": "data",
        },
    }

    config_defaults["policy_config"].update(policy_config)

    config_defaults = apply_machine_config(config_defaults)

    if config_defaults["policy_config"]["use_ee_pose"]:
        config_defaults["policy_config"]["proprio_dim"] = 7
    else:
        config_defaults["policy_config"]["proprio_dim"] = 8

    if config_defaults["policy_config"]["add_gripper_state"]:
        config_defaults["policy_config"]["proprio_dim"] += 1

    wandb_mode = "offline" if args.panda else \
        "disabled" if args.disable_wandb else "online"
    wandb.init(config=config_defaults, project="ceiling_eval", mode=wandb_mode)
    wandb.run.summary["bc_step"] = args.bc_step

    config = wandb.config  # in case the sweep gives different values
    main(config, config_defaults, args.refine_init)
