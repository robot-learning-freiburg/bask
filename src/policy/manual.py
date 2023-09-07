import numpy as np

from env.environment import BaseEnvironment
from utils.human_feedback import correct_action
from utils.keyboard_observer import KeyboardObserver


class ManualPolicy():
    def __init__(self, config, env, keyboard_obs, ** kwargs):
        self.keyboard_obs = keyboard_obs

        self.gripper_open = 0.9


    def eval(self):
        pass

    def from_disk(self, file_name):
        pass

    def predict(self, obs, lstm_state):
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.gripper_open])
        if self.keyboard_obs.has_joints_cor() or \
                self.keyboard_obs.has_gripper_update():
            action = correct_action(self.keyboard_obs, action)
            self.gripper_open = action[-1]
        return action, None, None

    def reset_episode(self, env: BaseEnvironment | None = None):
        # TODO: add this to all other policies as well and use it to store
        # the LSTM state as well?
        self.gripper_open = 0.9
        return
