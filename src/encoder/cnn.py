import torch
import torch.nn as nn
from loguru import logger

import encoder.representation_learner


class CNN(encoder.representation_learner.RepresentationLearner):
    def __init__(self, config=None):
        self.config = config

        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, padding=1,
                      stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1,
                      stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1,
                      stride=2),
            nn.ELU(),
        )

    def encode(self, camera_obs, full_obs=None):
        camera_obs = nn.functional.interpolate(
            camera_obs, size=(128, 128), mode='bilinear', align_corners=True)
        cam_emb = self.model(camera_obs)

        if (cam_obs2 := full_obs.cam_rgb2) is not None:
            cam_obs2 = nn.functional.interpolate(
                cam_obs2, size=(128, 128), mode='bilinear', align_corners=True)
            cam_emb2 = self.model(cam_obs2)

            cam_emb = torch.cat((cam_emb, cam_emb2), dim=-1)

        info = {}

        return cam_emb, info

    @classmethod
    def get_latent_dim(self, config, n_cams=1, image_dim=None):
        dim_mapping = {(256, 256): 256,
                       # (480, 480): ???
                       }
        return dim_mapping[image_dim] * n_cams

    def from_disk(self, chekpoint_path):
        logger.info("  CNN encoder does not need snapshot loading. Skipping.")


class CNNDepth(CNN):
    def __init__(self, config=None):
        encoder.representation_learner.RepresentationLearner.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, padding=1,
                      stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1,
                      stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1,
                      stride=2),
            nn.ELU(),
        )

    def forward(self, batch, full_obs=None):
        depth = full_obs.cam_d.unsqueeze(1)
        batch = torch.cat((batch, depth), dim=-3)

        if (cam_obs2 := full_obs.cam_rgb2) is not None:
            depth2 = full_obs.cam_d2.unsqueeze(1)
            full_obs.cam_rgb2 = torch.cat((cam_obs2, depth2), dim=-3)

        return self.encode(batch, full_obs=full_obs), {}
