import math
from abc import ABC
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from utils.geometry import euler_to_quaternion

# NOTE: these values are used to fit the labels (actions) of expert policies as
# closely into [-1,1] as possible to improve learning performance. To decide
# the values, inspect the dataset and look at min and max of the action distr.
trans_scale = 50
rot_scale = 12.5


def squash(array, ord=20):
    # map to [-1, 1], but more linear than tanh
    return np.sign(array)*np.power(
        np.tanh(np.power(np.power(array, 2), ord/2)), 1/ord)


class GripperPlot:
    def __init__(self, headless):
        self.headless = headless
        if headless:
            return
        self.displayed_gripper = 0.9
        self.fig = plt.figure()
        ax = self.fig.add_subplot(111)
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        horizontal_patch = plt.Rectangle((-1, 0), 2, 0.6)
        self.left_patch = plt.Rectangle((-0.9, -1), 0.4, 1, color="black")
        self.right_patch = plt.Rectangle((0.5, -1), 0.4, 1, color="black")
        ax.add_patch(horizontal_patch)
        ax.add_patch(self.left_patch)
        ax.add_patch(self.right_patch)
        self.fig.canvas.draw()
        plt.show(block=False)
        plt.pause(0.1)
        for _ in range(2):
            self.set_data(0)
            plt.pause(0.1)
            self.set_data(1)
            plt.pause(0.1)
        return

    def set_data(self, last_gripper_open):
        if self.headless:
            return
        if self.displayed_gripper == last_gripper_open:
            return
        if last_gripper_open == 0.9:
            self.displayed_gripper = 0.9
            self.left_patch.set_xy((-0.9, -1))
            self.right_patch.set_xy((0.5, -1))
        elif last_gripper_open == -0.9:
            self.displayed_gripper = -0.9
            self.left_patch.set_xy((-0.4, -1))
            self.right_patch.set_xy((0, -1))
        self.fig.canvas.draw()
        plt.pause(0.01)
        return

    def reset(self):
        self.set_data(1)


class BaseEnvironment(ABC):
    def __init__(self, config):
        if "image_size" in config and config["image_size"] is not None:
            image_size = config["image_size"]
        else:
            image_size = (256, 256)

        self.image_height, self.image_width = image_size

        self.gripper_plot = GripperPlot(not config["viz"])
        self.gripper_open = 0.9
        self.gripper_deque = deque([0.9] * 4, maxlen=4)

        # Scale actions from [-1,1] to the actual action space, ie transtions
        # in meters etc.
        self._delta_pos_scale = 0.01
        self._delta_angle_scale = 0.04

    def reset(self):
        self.gripper_plot.reset()
        self.gripper_open = 0.9
        self.gripper_deque = deque([0.9] * 4, maxlen=4)

    def step(self, action, manual_demo=False):
        raise NotImplementedError

    def render(self):
        return

    def close(self):
        raise NotImplementedError

    def postprocess_action(self, action, manual_demo=False, return_euler=False):
        if manual_demo:
            delta_position = action[:3] * self._delta_pos_scale
            euler = action[3:6] * self._delta_angle_scale
            if return_euler:
                rot = euler
            else:
                delta_angle_quat = euler_to_quaternion(euler)
                rot = normalize(np.array(delta_angle_quat))
            gripper_delayed = self.delay_gripper(action[-1])
        else:
            action[:3] /= trans_scale
            action[3:6] /= rot_scale
            # action[:6] = squash(action[:6])

            delta_position = action[:3]
            euler = action[3:6]
            if return_euler:
                rot = euler
            else:
                rot = euler_to_quaternion(euler)
            gripper_delayed = action[-1]

        action_post = np.concatenate(
            (delta_position, rot, [gripper_delayed]))

        return action_post

    def delay_gripper(self, gripper_action):
        if gripper_action >= 0.0:
            gripper_action = 0.9
        elif gripper_action < 0.0:
            gripper_action = -0.9
        self.gripper_plot.set_data(gripper_action)
        self.gripper_deque.append(gripper_action)
        if all([x == 0.9 for x in self.gripper_deque]):
            self.gripper_open = 1
        elif all([x == -0.9 for x in self.gripper_deque]):
            self.gripper_open = 0
        return self.gripper_open

    def obs_split(self, obs):
        raise NotImplementedError


def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.
    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    """Return spherical linear interpolation between two quaternions.
    """
    _EPS = np.finfo(float).eps * 4.0
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        np.negative(q1, q1)
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0

def normalize(quat):
    return quat / np.sqrt(np.sum(quat**2))
