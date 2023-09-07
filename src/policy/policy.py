import abc

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.distributions.normal import Normal

from dataset.bc import BCDataset
from env.environment import BaseEnvironment
from utils.logging import log_constructor
from utils.observation import SceneObservation
from utils.select_gpu import device  # , normalize_quaternion


class Policy(nn.Module):
    @abc.abstractmethod  # Policy object misses encoder.
    @log_constructor
    def __init__(self, config: dict, skip_module_init: bool = False, **kwargs):

        if not skip_module_init:
            super().__init__()

        lstm_dim = config["visual_embedding_dim"] + config["proprio_dim"]

        self.lstm = nn.LSTM(lstm_dim, lstm_dim,
                            num_layers=config["lstm_layers"])

        self.linear_out = nn.Linear(lstm_dim, config["action_dim"])

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        self.std = 0.1 * torch.ones(config["action_dim"], dtype=torch.float32)
        self.std = self.std.to(device)

        if "use_ee_pose" in config:
            self.use_ee_pose = config["use_ee_pose"]
        else:
            self.use_ee_pose = False
            logger.info("use_ee_pose not specified in config. Assuming False.")

        if "add_gripper_state" in config:
            self.add_gripper_state = config["add_gripper_state"]
        else:
            self.add_gripper_state = False
            logger.info("add_gripper_state not specified in config. "
                        "Assuming False.")

        return

    def forward_step(self, obs: SceneObservation,  # type: ignore
                     lstm_state: tuple[torch.Tensor, torch.Tensor] | None,
                     ) -> tuple[torch.Tensor,
                                tuple[torch.Tensor, torch.Tensor], dict]:
        vis_encoding, info = self.encoder(obs) # type: ignore

        vis_encoding = torch.flatten(vis_encoding, start_dim=1)

        robo_state = obs.ee_pose if self.use_ee_pose else obs.proprio_obs

        if self.add_gripper_state:
            robo_state = torch.cat((robo_state, obs.gripper_state), dim=1)

        low_dim_input = torch.cat(
            (vis_encoding, robo_state), dim=-1).unsqueeze(0)

        lstm_out, (h, c) = self.lstm(low_dim_input, lstm_state)
        lstm_state = (h, c)

        out = torch.tanh(self.linear_out(lstm_out))

        info['vis_encoding'] = vis_encoding

        return out, lstm_state, info

    def forward(self, batch: SceneObservation) -> torch.Tensor:  # type: ignore

        losses = []
        lstm_state = None

        self.encoder.reset_episode()  # type: ignore

        time_steps = batch.shape[1]

        for t in range(time_steps):
            obs = batch[:, t, ...]
            mu, lstm_state, _ = self.forward_step(obs, lstm_state)
            distribution = Normal(mu, self.std)
            log_prob = distribution.log_prob(obs.action)
            loss = -log_prob * obs.feedback
            losses.append(loss)

        total_loss = torch.cat(losses).mean()

        return total_loss

    def update_params(self, batch: SceneObservation) -> dict:  # type: ignore
        batch = batch.to(device)
        self.optimizer.zero_grad()
        loss = self.forward(batch)
        loss.backward()
        self.optimizer.step()
        training_metrics = {"train-loss": loss}
        return training_metrics

    def evaluate(self, batch: SceneObservation) -> dict:  # type: ignore
        batch = batch.to(device)
        loss = self.forward(batch)
        training_metrics = {"eval-loss": loss}
        return training_metrics

    def predict(self, obs: SceneObservation,  # type: ignore
                lstm_state: tuple[torch.Tensor, torch.Tensor] | None,
                ) -> tuple[np.ndarray, tuple[torch.Tensor, torch.Tensor],
                           dict]:
        with torch.no_grad():
            obs = obs.unsqueeze(0).to(device)
            action_th, lstm_state, info = \
                self.forward_step(obs, lstm_state)
            action = action_th.detach().cpu().squeeze(0).squeeze(0).numpy()
            action[-1] = binary_gripper(action[-1])
        return action, lstm_state, info

    def from_disk(self, chekpoint_path: str) -> None:
        self.load_state_dict(
            torch.load(chekpoint_path, map_location=device))

    def to_disk(self, checkpoint_path: str) -> None:
        logger.info("Saving policy at {}", checkpoint_path)
        torch.save(self.state_dict(), checkpoint_path)

    def initialize_parameters_via_dataset(self, replay_memory: BCDataset,
                                           cameras: tuple[str]) -> None:
        logger.info("This policy does not use dataset initialization.")

    def reset_episode(self, env: BaseEnvironment | None = None) -> None:
        pass


def binary_gripper(gripper_action: float) -> float:
    if gripper_action >= 0.0:
        gripper_action = 0.9
    elif gripper_action < 0.0:
        gripper_action = -0.9
    return gripper_action
