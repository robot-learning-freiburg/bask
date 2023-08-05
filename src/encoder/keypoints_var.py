import pathlib

import torch
from loguru import logger

import encoder.representation_learner
import models.keypoints.keypoints as keypoints
from encoder.keypoints import KeypointsPredictor
from encoder.keypoints_gt import GTKeypointsPredictor

KeypointsTypes = keypoints.KeypointsTypes


class VarKeypointsPredictor(
        encoder.representation_learner.RepresentationLearner):
    def __init__(self, config=None):
        super().__init__(config=config)

        self.kp_model = KeypointsPredictor(config)
        self.gt_model = GTKeypointsPredictor(config)

        self.config = config["encoder"]

        self.distance_hist = []

        n_keypoints = self.kp_model.get_no_keypoints()

        self.register_buffer('ref_pixels_uv',
                             torch.Tensor(2, n_keypoints))
        self.register_buffer('_reference_descriptor_vec',
                             torch.Tensor(n_keypoints,
                                          self.kp_model.descriptor_dimension))

    def __del__(self):
        self.save_dist_hist()

    def save_dist_hist(self):
        if len(self.distance_hist) > 0:
            self.distance_hist = torch.stack(self.distance_hist)

        save_path = pathlib.Path("tmp")
        save_path.mkdir(exist_ok=True)
        kp_dist_file_name = save_path / "kp_var_hist.dat"

        logger.info("Saving kp var to file {}", kp_dist_file_name)

        self.distance_hist = self.distance_hist.to('cpu')

        return torch.save(self.distance_hist, kp_dist_file_name)

    def reset_traj(self):
        self.kp_model.reset_traj()

        if len(self.distance_hist) > 0:
            self.distance_hist[-1] = torch.stack(self.distance_hist[-1])

        self.distance_hist.append([])

    @classmethod
    def get_latent_dim(cls, config, n_cams=1, image_dim=None):
        return KeypointsPredictor.get_latent_dim(config, n_cams)

    def initialize_parameters_via_dataset(self, replay_memory):
        self.select_reference_descriptors(replay_memory)

    def select_reference_descriptors(self, replay_memory):
        self.gt_model.select_reference_descriptors(replay_memory)

        self.kp_model.ref_pixels_uv = self.gt_model.ref_pixels_uv
        self.gt_model._reference_descriptor_vec = \
            self.kp_model._reference_descriptor_vec

        self.ref_pixels_uv = self.gt_model.ref_pixels_uv
        self._reference_descriptor_vec = \
            self.kp_model._reference_descriptor_vec

    def from_disk(self, chekpoint_path, ignore=None):
        self.kp_model.from_disk(chekpoint_path, ignore=ignore)

    def encode(self, camera_obs, full_obs=None):

        # NOTE: encoded keypoints are projected. For 3D accuracy use hard glob
        # projection.
        # NOTE: currently does not work with uvd projection as there the
        # coordinate order differs. Not a problem though as there the distance
        # is not really interpretable anyway.
        kp_pred, kp_info = self.kp_model.encode(camera_obs, full_obs)
        gt_pred, gt_info = self.gt_model.encode(camera_obs, full_obs)

        # GT is still per cam. Just use first cam.
        if full_obs.cam_rgb2 is not None:
            gt_pred = torch.chunk(gt_pred, 2, dim=1)[0]

        # 3D projection stacks x,y,z features, not flattten them.
        kp_pred = torch.stack(kp_pred.chunk(self.kp_model.keypoint_dimension,
                                            dim=1), dim=2)
        gt_pred = torch.stack(gt_pred.chunk(self.gt_model.keypoint_dimension,
                                            dim=1), dim=2)

        distance = self.get_keypoint_distance(kp_pred, gt_pred)

        self.distance_hist[-1].append(distance)

        info = {"distance": distance}

        return kp_pred, info

    @staticmethod
    def get_keypoint_distance(set1, set2):
        # pairwise distance needs both inputs to have shape (N, D), so flatten
        B, N_kp, d_kp = set1.shape
        set1 = set1.reshape((B * N_kp, d_kp))
        set2 = set2.reshape((B * N_kp, d_kp))

        distance = torch.nn.functional.pairwise_distance(set1, set2)
        distance = distance.reshape((B, N_kp))

        return distance
