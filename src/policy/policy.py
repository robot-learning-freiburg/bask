import abc

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.distributions.normal import Normal

from env.observation import DCEvalObs  # , CeilingAdapter
from utils.select_gpu import device  # , normalize_quaternion


class Policy(nn.Module):
    @abc.abstractmethod  # Policy object misses encoder.
    def __init__(self, config, **kwargs):
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

    def forward_step(self, obs, lstm_state):
        # from utils.debug import summarize_tensor
        # print(summarize_tensor(obs.cam_rgb, "obs"))

        vis_obs = obs.cam_rgb

        # if hasattr(obs, "cam_rgb2") and obs.cam_rgb2 is not None:
        #     from viz.image_single import image_with_points_overlay, \
        #         channel_front2back
        #     image_with_points_overlay(
        #         channel_front2back(obs.cam_rgb2.squeeze(0)), [])

        # The following lines are only for debugging.
        # vis_obs = nn.functional.interpolate(
        #     vis_obs, size=(128, 128), mode='bilinear', align_corners=True)
        # from viz.image_single import image_with_points_overlay
        # from viz.operations import channel_front2back
        # image_with_points_overlay(channel_front2back(vis_obs.squeeze(0)), [])

        robo_state = obs.wrist_pose if self.use_ee_pose else obs.proprio_obs

        if self.add_gripper_state:
            robo_state = torch.cat((robo_state, obs.gripper_state), dim=1)

        vis_encoding, info = self.encoder(vis_obs, full_obs=obs)
        vis_encoding = torch.flatten(vis_encoding, start_dim=1)
        low_dim_input = torch.cat(
            (vis_encoding, robo_state), dim=-1).unsqueeze(0)
        lstm_out, (h, c) = self.lstm(low_dim_input, lstm_state)
        lstm_state = (h, c)
        out = torch.tanh(self.linear_out(lstm_out))

        info['vis_encoding'] = vis_encoding

        return out, lstm_state, info

    def forward(self, batch):
        # print(self.encoder._reference_descriptor_vec.requires_grad)
        # print(self.encoder.ref_pixels_uv.requires_grad)
        # print(self.encoder._reference_descriptor_vec)
        # print(self.encoder.ref_pixels_uv)
        losses = []
        lstm_state = None
        try:
            self.encoder.reset_traj()
        except AttributeError:
            pass

        # action_store = torch.cat(
        #     [o.action for o in batch]).cpu().detach().numpy()
        # action_store = np.swapaxes(action_store, 0, 1)
        # print(action_store.shape)
        # import viz.action_distribution as action_distribution
        # action_distribution.make_all(action_store)
        # exit()

        for step, obs in enumerate(batch):
            # print("{}: {}".format(step, obs.action))
            mu, lstm_state, _ = self.forward_step(obs, lstm_state)
            distribution = Normal(mu, self.std)
            log_prob = distribution.log_prob(obs.action)
            loss = -log_prob * obs.feedback
            losses.append(loss)

        total_loss = torch.cat(losses).mean()

        return total_loss

    def update_params(self, batch):
        batch = batch.to(device)
        self.optimizer.zero_grad()
        loss = self.forward(batch)
        loss.backward()
        self.optimizer.step()
        training_metrics = {"train-loss": loss}
        return training_metrics

    def evaluate(self, batch):
        batch = batch.to(device)
        loss = self.forward(batch)
        training_metrics = {"eval-loss": loss}
        return training_metrics

    def predict(self, obs, lstm_state, cam):
        # obs = CeilingAdapter(obs, device)  # that was build for traj env
        obs = DCEvalObs(obs, device, cam=cam)
        with torch.no_grad():
            action_th, lstm_state, info = \
                self.forward_step(obs, lstm_state)
            action = action_th.detach().cpu().squeeze(0).squeeze(0).numpy()
            action[-1] = binary_gripper(action[-1])
        return action, lstm_state, info

    def from_disk(self, chekpoint_path):
        self.load_state_dict(
            torch.load(chekpoint_path, map_location=device))

    def to_disk(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)

    def initialize_parameters_via_dataset(self, replay_memory):
        logger.info("This policy does not use dataset initialization.")


def binary_gripper(gripper_action):
    if gripper_action >= 0.0:
        gripper_action = 0.9
    elif gripper_action < 0.0:
        gripper_action = -0.9
    return gripper_action


def clamp_displacement(displacement, max_dist=0.1):
    norm = np.linalg.norm(displacement)
    if norm > max_dist:
        displacement *= max_dist/norm

    return displacement
