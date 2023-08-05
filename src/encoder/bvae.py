import torch
from torch import nn
from torch.nn import functional as F

import encoder.representation_learner
import models.bvae.beta_vae as bvae
from utils.constants import SampleTypes
from utils.select_gpu import device


class BVAE(encoder.representation_learner.RepresentationLearner):

    sample_type = SampleTypes.CAM_SINGLE

    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self, config=None):
        super().__init__(config=config)

        self.config = config

        self.model = bvae.BetaVAE(config["encoder"])

        pretrain_conf = config["pretrain"]["training_config"]
        self.beta = pretrain_conf["beta"]
        self.kld_correction = pretrain_conf["kld_correction"]
        self.gamma = pretrain_conf["gamma"]
        self.loss_type = pretrain_conf["loss_type"]
        self.C_max = torch.Tensor([pretrain_conf["max_capacity"]])
        self.C_stop_iter = pretrain_conf["Capacity_max_iter"]

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          pretrain_conf["lr"])

    def loss(self, recons, input, mu, log_var, kld_weight=None) -> dict:
        self.num_iter += 1

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(),
                             dim=1), dim=0)

        if self.loss_type == 'H':  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B':  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter
                            * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss': recons_loss,
                'KLD': kld_loss,
                'kLD_weighted': self.beta * kld_weight * kld_loss}

    def calc_kld_weight(self, batch_size, dataset_size):
        return batch_size / dataset_size if self.kld_correction else 1

    def process_batch(self, batch, dataset_size, batch_size):
        batch = nn.functional.interpolate(
            batch, size=(128, 128), mode='bilinear', align_corners=True)
        reconstruction, input, mu, log_var = self.model.forward(batch)
        kld_weight = self.calc_kld_weight(batch_size, dataset_size)
        metrics = self.loss(
            reconstruction, input, mu, log_var, kld_weight=kld_weight)

        return metrics

    # TODO: should be same in every encoder -> remove
    def update_params(self, batch, dataset_size=None, batch_size=None,
                      **kwargs):
        batch = batch.to(device)

        self.optimizer.zero_grad()
        training_metrics = self.process_batch(batch, dataset_size, batch_size)

        training_metrics['loss'].backward()
        self.optimizer.step()

        training_metrics = {
            "train-{}".format(k): v for k, v in training_metrics.items()}

        return training_metrics

    def encode(self, camera_obs, full_obs=None):
        camera_obs = nn.functional.interpolate(
            camera_obs, size=(128, 128), mode='bilinear', align_corners=True)

        mu, log_var = self.model.encode(camera_obs)

        if full_obs is not None:
            if (cam_obs2 := full_obs.cam_rgb2) is not None:
                cam_obs2 = nn.functional.interpolate(
                    cam_obs2, size=(128, 128), mode='bilinear', align_corners=True)

                mu2, log_var2 = self.model.encode(cam_obs2)

                mu = torch.cat((mu, mu2), dim=-1)

        info = {}

        return mu, info  # see paper: treating mu as embedding

    # TODO: should be same in every encoder -> remove
    def evaluate(self, batch, dataset_size=None, batch_size=None,
                 **kwargs):
        batch = batch.to(device)

        eval_metrics = self.process_batch(batch, dataset_size, batch_size)

        eval_metrics = {
            "eval-{}".format(k): v for k, v in eval_metrics.items()}

        return eval_metrics

    def reconstruct(self, batch):
        batch = batch.to(device)
        batch = nn.functional.interpolate(
            batch, size=(128, 128), mode='bilinear', align_corners=True)
        return self.model(batch)[0]

    @classmethod
    def get_latent_dim(self, config, n_cams=1, image_dim=None):
        return config["latent_dim"] * n_cams
