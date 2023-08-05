import pathlib
import pprint
import random
import time
from argparse import ArgumentParser
from enum import Enum

import numpy as np
from loguru import logger
from tqdm.auto import tqdm

import utils.logging  # noqa
# from collect_expert_demos import sequence_length
from config import camera_pose
from dataset.scene import SceneDataset
from policy import policy_switch
from utils.keyboard_observer import KeyboardObserver
from utils.misc import apply_machine_config, loop_sleep, set_seeds


class Environment(Enum):
    PANDA = "panda"
    MANISKILL = "maniskill"
    RLBENCH = "rlbench"


class Policy(Enum):
    RANDOM = "random"
    MANUAL = "manual"
    SPHERE = "sphere"


def main(config):
    if config["env"] is Environment.PANDA:
        from env.franka import FrankaEnv as Env
    elif config["env"] is Environment.MANISKILL:
        from env.mani_skill import ManiSkillEnv as Env
    else:
        if config["collect_object_pose"]:
            from env.ground_truth import GTEnv as Env
        elif config["collect_mask"]:
            from env.mask import MaskEnv as Env
        else:
            from env.correspondence import DCEnv as Env

        # from utils.tasks import tasks

    Policy = policy_switch[config["policy"].value]

    task_name = config["task"]
    task_name += "-" + config["background"] if config["background"] else ""
    task_name += "-" + "_".join([m for m in config["model_ids"]])

    save_path = pathlib.Path(
        config["dataset_config"]["data_root"]) / task_name
    if not save_path.is_dir():
        logger.warning("Creating save path. This should only be needed for "
                       "new tasks.")
        save_path.mkdir(parents=True)

    env = Env(config)
    camera_names = env.camera_names
    keyboard_obs = KeyboardObserver()
    dir_name = config['dataset_name'] + \
        ("_gt" if config["collect_object_pose"] else
         "_mm" if config["collect_mask"] else "")
    replay_memory = SceneDataset(camera_names=camera_names,
                                 subsample_by_difference=False,
                                 subsample_to_length=config["sequence_len"],
                                 data_root=save_path / dir_name,
                                 image_size=config["image_size"])
    if config["task"] == "ArmScan":
        save_all_trajs = True
        logger.info("Task is ArmScan. Saving trajectorys also when pressing "
                    "reset button.")
    else:
        save_all_trajs = False

    env.reset()  # extra reset to correct set up of camera poses in first obs
    # TODO: policy config?
    policy = Policy(config, env, keyboard_obs=keyboard_obs)

    obs = env.reset()

    time.sleep(5)
    logger.info("Go!")
    episodes_count = 0
    lstm_state = None
    try:
        with tqdm(total=config["episodes"]) as pbar:
            while episodes_count < config["episodes"]:
                start_time = time.time()
                action, lstm_state, _ = policy.predict(obs, lstm_state)
                next_obs, _ , done, _ = env.step(action, manual_demo=True)
                replay_memory.add(obs, action, [1])
                obs = next_obs
                if done or keyboard_obs.success or (save_all_trajs and keyboard_obs.reset_button):
                    if keyboard_obs.success or save_all_trajs:
                        logger.info("Saving traj.")
                        replay_memory.save_current_traj()
                    else:
                        logger.info("Resetting without saving traj.")
                    obs = env.reset()
                    episodes_count += 1
                    keyboard_obs.reset()
                    policy.reset_episode(env)
                    done = False
                    lstm_state = None
                    pbar.update(1)
                elif keyboard_obs.reset_button:
                    replay_memory.reset_current_traj()
                    obs = env.reset()
                    keyboard_obs.reset()
                    policy.reset_episode(env)
                    lstm_state = None
                else:
                    loop_sleep(start_time)
                    pass
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Attempting graceful shutdown of env...")
        env.shutdown()

    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pretraining", action="store_true", default=False,
                        help="Wether the data is for pretraining. Used to "
                             "name the dataset folder.")
    parser.add_argument("-e", "--environment", default="rlbench",
                        help="RLBench, Maniskill or Panda.")
    parser.add_argument("-t", "--task", default="PhoneOnBase",
                        help="The task to perform.")
    parser.add_argument( "-n", "--number", default="10",
                        help="Number of demos to collect.")
    parser.add_argument("-m", "--mask", action="store_true",
                        default=True,
                        help="Collect ground truth object masks. Only for "
                             "simulated environments. For quick experiments, "
                             "removes need for TSDF fusion.")
    # parser.add_argument("-o", "--object_pose", action="store_true",
    #                     help="Collect ground truth object positions.")
    parser.add_argument("-b", "--background", default=None,
                        help="Maniskill only. Environment background to use.")
    parser.add_argument("--model_ids", nargs="+", default=[],
                        help="Maniskill only. Model ids to use for YCB.")
    parser.add_argument("-p", "--policy", default="manual")

    args = parser.parse_args()
    env = Environment[args.environment.upper()]
    env_is_panda = env is Environment.PANDA
    policy = Policy[args.policy.upper()]

    config = {
        "env": env,
        "task": args.task,
        "policy": policy,
        "static_env": False,
        "headless_env": env_is_panda,  # set to True for gripper plt
        "viz": False,
        "image_size": (480, 640) if env_is_panda else (256, 256),
        "save_demos": True,
        "episodes": int(args.number),
        "sequence_len": None,
        "shoulders_on": False,
        "camera_pose": camera_pose,
        "collect_mask": (args.mask or args.object_pose) and not env_is_panda,
        "collect_object_pose": not env_is_panda,
            # args.object_pose and not env_is_panda,
        "maniskill": True,
        "teleop": True,
        "dataset_config": {
            "data_root": "data",
        },
        "render_sapien": False,
        "background": args.background,
        "model_ids": args.model_ids,
    }
    if args.policy == "sphere":
        assert not env_is_panda
        from rlbench.action_modes import ArmActionMode
        config["action_mode"] \
            = ArmActionMode.ABS_JOINT_POSITION

    if args.pretraining:
        dataset_name = "densecorr_" + args.policy + "_" + str(
            config["episodes"])
    else:
        dataset_name = "demos"

    config['dataset_name'] = dataset_name

    seed = random.randint(0, 2000)
    logger.info("Seed: {}", seed)
    set_seeds(seed)
    config["seed"] = seed

    config = apply_machine_config(config)

    # logger.info("Config:")
    # logger.info(pprint.pformat(config))

    main(config)
