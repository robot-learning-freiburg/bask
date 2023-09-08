import collections.abc
import importlib
import os
import pathlib
import random
import string
import sys
import time
from functools import reduce

import numpy as np
import torch
from loguru import logger

from utils.logging import indent_logs


def invert_dict(dictionary):
    return {v: k for k, v in dictionary.items()}

def random_string(str_len=8):
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(random.choices(alphabet, k=str_len))


def get_full_task_name(config):
    task_name = config["env_config"]["task"]

    if config["env_config"]["background"]:
        task_name += "-" + config["env_config"]["background"]

    if config["env_config"]["model_ids"]:
        task_name += "-" + "_".join(
            [m for m in config["env_config"]["model_ids"]])

    return task_name


def get_dataset_name(config):
    if config["dataset_config"]["pretraining"]:
        name = "pretrain_" + config["policy_config"]["policy"].value.lower()
    else:
        name = "demos"

    return name


def get_data_path(config, task, root=None):
    if root is None:
        root = "data"

    file_name = config["feedback_type"]

    return root + "/" + task + "/" + file_name


def load_replay_memory(config, path=None):
    import dataset.scene

    logger.info("Loading dataset(s): ")

    with indent_logs():
        if path:
            memory = dataset.scene.SceneDataset(data_root=pathlib.Path(path))
        else:
            task = config["task"]
            if task == "Mixed":
                # task = task_switch.keys()
                raise NotImplementedError(
                    "Disabled to avoid QT mixup between CoppeliaSim and ROS.")
            else:
                task = [task]

            data = []

            for t in task:
                data_path = pathlib.Path(get_data_path(
                    config, t, root=config["data_root"]))
                dset = dataset.scene.SceneDataset(data_root=data_path)
                data.append(dset)

            memory = dataset.scene.SceneDataset.join_datasets(*data)

        logger.info("Done! Data contains {} trajectories.", len(memory))

    return memory


def policy_checkpoint_name(config: dict, create_suffix: bool =False
                           ) -> tuple[str, str]:
    if create_suffix and config["policy_config"].get("suffix"):
        raise ValueError("Should not pass suffix AND ask to create one.")
    if (pt := config["dataset_config"].get("pretrained_on", None)) is not None:
        pretrained_on = pt
    else:
        pretrained_on = "None"
    if v := config["policy_config"].get("suffix"):
        suffix = "-" + v
    elif create_suffix:
        suffix = "-" + random_string()
    else:
        suffix = ""
    return config["dataset_config"]["data_root"] + "/" + \
        config["dataset_config"]["task"] + "/" + \
        config["dataset_config"]["feedback_type"] + "_" + \
        str(config["policy_config"]["encoder"]) + \
        "_pon-" + pretrained_on + "_policy" + suffix + ".pt", suffix


def pretrain_checkpoint_name(config):
    if (pt := config["dataset_config"].get("pretrained_on", None)) is not None:
        task = pt
    else:
        task = config["dataset_config"]["task"]
    if (pf := config["dataset_config"].get("pretrain_feedback", None)) \
            is not None:
        feedback = pf
    else:
        feedback = config["dataset_config"]["feedback_type"]
    suffix = config["policy_config"].get("encoder_suffix")
    suffix = "-" + suffix if suffix else ""
    encoder = config["policy_config"]["encoder"]
    encoder_name = feedback + "_" + encoder + "_encoder" + suffix
    return (pathlib.Path(config["dataset_config"]["data_root"]) / task /
            encoder_name).with_suffix(".pt")


def even(int_number):
    return int_number//2*2


def correct_action(keyboard_obs, action):
    if keyboard_obs.has_joints_cor():
        ee_step = keyboard_obs.get_ee_action()
        action[:-1] = action[:-1] * 0.5 + ee_step
        action = np.clip(action, -0.9, 0.9)
    if keyboard_obs.has_gripper_update():
        action[-1] = keyboard_obs.get_gripper()
    return action


def loop_sleep(start_time, dt=0.05):
    sleep_time = dt - (time.time() - start_time)
    if sleep_time > 0.0:
        time.sleep(sleep_time)
    return


def import_config_file(config_file):
    """
    Import a config file as a python module.

    Parameters
    ----------
    config_file : str or path
        Path to the config file.

    Returns
    -------
    module
        Python module containing the config file.

    """
    config_file = str(config_file)
    if config_file[-3:] == '.py':
        config_file = config_file[:-3]

    config_file_path = os.path.abspath('/'.join(config_file.split('/')[:-1]))
    sys.path.insert(1, config_file_path)
    config = importlib.import_module(config_file.split('/')[-1], package=None)

    return config


def get_variables_from_module(module):
    """
    Get (non-magic, non-private) variables defined in a module.

    Parameters
    ----------
    module : module
        The python module from which to extract the variables.

    Returns
    -------
    dict
        Dict of the vars and their values.
    """
    confs = {k: v for k, v in module.__dict__.items()
             if not (k.startswith('__') or k.startswith('_'))}
    return confs


def recursive_dict_update(base, update):
    for k, v in update.items():
        if isinstance(v, collections.abc.Mapping):
            base[k] = recursive_dict_update(base.get(k, {}), v)
        else:
            base[k] = v
    return base


def apply_machine_config(config, machine_config_path='src/_machine_config.py'):
    path = pathlib.Path(machine_config_path)
    if path.is_file():
        mc = import_config_file(machine_config_path).config
        logger.info("Applying machine config to config dict.")
        config = recursive_dict_update(config, mc)
    else:
        logger.info(
            "No machine config found at {}.".format(machine_config_path))
    return config


def configure_class_instance(instance, class_keys, config):
    for k in class_keys:
        v = config.get(k, None)
        if v is None:
            logger.warning(
                    "Key {} not in encoder config. Assuming False.", k)
            v = False
        setattr(instance, k, v)


def get_and_log_failure(dictionary, key, default=None):
    if key in dictionary:
        return dictionary[key]
    else:
        logger.info("Key {} not in config. Assuming {}.", key, default)
        return default

def multiply_iterable(l):
    return reduce(lambda x, y: x*y, l)
