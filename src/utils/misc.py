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

# from utils.tasks import task_switch


def random_string(str_len=8):
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(random.choices(alphabet, k=str_len))


file_name_roots = {
    "cloning": "demos",
    "dc": "densecorr",
    "dcr": "densecorr_random",
    "dcm": "densecorr_manual",
    "dcs": "densecorr_sphere",
    "ed": "expert_demos",
}

is_manual_demo = {
    "cloning": True,
    "ed": False,
}


def get_is_manual_demo(config):
    feedback_type = config["dataset_config"]["feedback_type"]
    root = feedback_type.split("_")[0]  # root, number or just root
    if root in is_manual_demo.keys():
        return is_manual_demo[root]
    else:
        raise ValueError(
            "Unexpected name of training data {}.".format(feedback_type))


def get_file_name(feedback_type, with_mask=False, with_obj_pose=False):
    components = feedback_type.split("_")  # root, number or just root
    num = "_" + components[1] if len(components) == 2 else ""
    return file_name_roots[components[0]] + num + ("_gt" if with_obj_pose
                                                   else"_mm" if with_mask
                                                   else "")


# TODO: use root-conf also in other funcs
def get_data_path(config, task, root=None):
    if root is None:
        root = "data"
    with_ground_truth_mask = config.get("ground_truth_mask")
    with_ground_truth_pose = config.get("ground_truth_object_pose")
    file_name = get_file_name(config["feedback_type"], with_ground_truth_mask,
                              with_ground_truth_pose)
    return root + "/" + task + "/" + file_name


def get_fusion_dir(config):
    return "data/" + config["task"] + "/" + get_file_name(
        config["feedback_type"])


def load_replay_memory(config, path=None):
    import dataset.scene

    logger.info("Loading dataset(s): ")

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
            dset = dataset.scene.SceneDataset(
                data_root=data_path,
                ground_truth_object_pose=config['ground_truth_object_pose'])
            data.append(dset)

        memory = dataset.scene.SceneDataset.join_datasets(*data)

    logger.info("  Done! Data contains {} trajectories.", len(memory))

    return memory


# TODO: unify with normal load_replay_memory
def load_replay_memory_from_path(data_path):
    import dataset.scene

    data_path = pathlib.Path(data_path)
    logger.info("Loading dataset(s): ")

    if data_path.is_dir():
        logger.info(
            "  Data path is a dir. Creating mulifile set with:")
        load_paths = sorted(list(data_path.rglob('*.dat')))
        for p in load_paths:
            logger.info("    {}", p)
        memory = dataset.multifile.MultiFileSet(load_paths)
    else:
        logger.info("  {}", data_path)
        memory = torch.load(data_path)
        if type(memory) == dataset.scene.SceneDataset:
            # TODO: these were hacks for old data. Can removed now!
            memory.patch_missing_members()

    logger.info("  Done! Data contains {} trajectories.", len(memory))

    return memory


def save_replay_memory(replay_memory, config):
    file_name = get_data_path(config, config["task"])
    logger.info("Saving dataset: ")
    logger.info("  {}", file_name)
    torch.save(replay_memory, file_name)
    logger.info("Done!")


def policy_checkpoint_name(config, create_suffix=False):
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
    if encoder == "keypoints_var":
        encoder = "keypoints"
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


def set_seeds(seed=0):
    """Sets all seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


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


def multiply_iterable(l):
    return reduce(lambda x, y: x*y, l)
