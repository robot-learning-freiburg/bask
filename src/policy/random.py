import random

import numpy as np


class RandomPolicy():
    def __init__(self, config, **kwargs):
        self.action_dim = config['action_dim']

    def eval(self):
        pass

    def from_disk(self, file_name):
        pass

    def predict(self, obs, lstm_state):
        action = np.asarray([random.random() for _ in range(self.action_dim)])
        return action, None, None
