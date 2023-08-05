# import torch
# from functools import partial
from loguru import logger
from torch.utils.data import Dataset

from dataset.dataclasses import BCTraj, collate_bc
from utils.constants import MaskTypes


class BCDataset(Dataset):
    def __init__(self, scene_dataset=None, config=None):
        self.scene_data = scene_dataset
        self.config = config

        self.fragment_length = config["fragment_length"]

        if (T := self.fragment_length) is None:
            logger.info("Training on full trajectories. Padding.")
        else:
            logger.info("Training on fragments of length {}.", format(T))

        self.object_labels = self.get_object_labels()

        self.load_pre_embedding = \
            ['descriptor'] if config.get("pre_embedding") else []
        self.encoder_name = config.get("encoder_name")

        self.pre_encoding_attr = ["kp"] if config.get("kp_pre_encoding") \
            else []
        self.pre_encoding_name = enc if (enc := config.get("kp_pre_encoding"))\
            else None

        self.extra_attr = config.get("extra_attr", [])

        if self.pre_encoding_name:
            logger.info(
                "Loading attr {} from encoding {} from encoder named {}.",
                self.pre_encoding_attr, self.pre_encoding_name,
                self.encoder_name)
        elif self.load_pre_embedding:
            logger.info("Loading embedding {} from encoder named {}.",
                        self.load_pre_embedding, self.encoder_name)
        else:
            logger.info("Loading raw data for encoder.")

    def __len__(self):
        return len(self.scene_data)

    def __getitem__(self, index) -> BCTraj:
        return self._get_bc_traj(
            index, cams=self.config["cams"],
            skip_rgb=not (self.config.get("force_load_raw")
                          or self.config.get("debug_encoding")),
            extra_attr=self.config.get("extra_attr", []))

    def _get_bc_traj(self, index, cams=("wrist",), fragment_length=None,
                     skip_rgb=True, extra_attr=[]) -> BCTraj:
        # fragment_length: None for self.fl, -1 for full traj
        if fragment_length is None:
            fragment_length = self.fragment_length

        # define the additional attributes to load
        embedding_attr = ["cam_{}_{}".format(c[0], a) for c in cams
                          for a in self.load_pre_embedding]
        encoding_attr = self.pre_encoding_attr
        # if we load the embeddings directly, we can skip the camera attributes
        skip_raw_cam_attr = (bool(embedding_attr)
                             or bool(encoding_attr)) and skip_rgb

        return self.scene_data._get_bc_traj(
            index, cams=cams, mask_type=self.config['mask_type'],
            sample_freq=self.config['sample_freq'], extra_attr=extra_attr,
            fragment_length=fragment_length, embedding_attr=embedding_attr,
            encoding_attr=encoding_attr, encoding_name=self.pre_encoding_name,
            encoder_name=self.encoder_name, skip_raw_cam_attr=skip_raw_cam_attr
            )

    def sample_bc(self, batch_size, cam=("wrist",), idx=None, skip_rgb=False):
        # legacy func for encoder init and viz

        if type(cam) is str:
            cam = (cam,)
        if idx is None:
            idx = self.scene_data.sample_traj_idx(batch_size=batch_size)
        return collate_bc([self._get_bc_traj(i, cams=cam, fragment_length=-1,
                                             skip_rgb=skip_rgb) for i in idx])

    def sample_data_point_with_object_labels(self, cam="wrist", traj_idx=None,
                                             img_idx=None):

        get_mask = False if self.config["mask_type"] in [
            None, MaskTypes.NONE] else True
        obs = self.scene_data.sample_observation(
            traj_idx=traj_idx, img_idx=img_idx,
            cam=cam, mask_type=self.config["mask_type"], raw_mask=False,
            labels=self.object_labels, collapse_labels=False, get_rgb=True,
            get_int=False, get_ext=False, get_depth=True, get_mask=get_mask,
            get_action=False, get_feedback=False, get_gripper_pose=False,
            get_object_pose=False, get_proprio_obs=False, get_wrist_pose=False)
        return obs.cam_rgb, obs.cam_d, obs.mask

    def sample_data_point_with_ground_truth(self, cam="wrist", traj_idx=None,
                                            img_idx=None):
        obs = self.scene_data.sample_observation(
            traj_idx=traj_idx, img_idx=img_idx,
            cam=cam, mask_type=MaskTypes.GT, raw_mask=False,
            labels=self.object_labels, collapse_labels=False, get_rgb=True,
            get_int=True, get_ext=True, get_depth=True, get_mask=True,
            get_action=False, get_feedback=False, get_gripper_pose=False,
            get_object_pose=True, get_proprio_obs=False, get_wrist_pose=False)

        return obs.cam_rgb, obs.cam_d, obs.mask, obs.cam_int, obs.cam_ext, \
            obs.object_pose

    def get_object_labels(self):
        if (labels := self.scene_data.object_labels) is not None:
            pass
        elif (type := self.config['mask_type']) is MaskTypes.GT:
            labels = self.scene_data.object_labels_gt
        elif type is MaskTypes.TSDF:
            labels = self.scene_data.object_labels_tsdf
        elif type is None or MaskTypes.NONE:
            labels = None
        else:
            raise ValueError("Could not get labels for type {}".format(type))

        if self.config.get('only_use_first_object_label'):
            logger.warning("Using hardcoded object_no instead of sample. "
                           "Is that intended?")
            labels = [labels[0]]  # HACK for Lid to always pick lid_label

        return labels

    def add_embedding(self, traj_idx, obs_idx, cam_name, emb_name, encoding,
                      encoder_name):
        return self.scene_data.add_embedding(
            traj_idx, obs_idx, cam_name, emb_name, encoding, encoder_name)

    def load_embedding(self, traj_idx, img_idx, cam, embedding_name):
        return self.scene_data.load_embedding(
            traj_idx, img_idx, cam, embedding_name, self.encoder_name)

    def load_embedding_batch(self, traj_idx, img_idx, cam, embedding_name):
        return self.scene_data.load_embedding_batch(
            traj_idx, img_idx, cam, embedding_name, self.encoder_name)

    def add_encoding(self, traj_idx, obs_idx, cam_name, attr_name, encoding,
                     encoder_name, encoding_name):
        return self.scene_data.add_encoding(
            traj_idx, obs_idx, cam_name, attr_name, encoding, encoder_name,
            encoding_name)

    def add_encoding_fig(self, traj_idx, obs_idx, cam_name, attr_name, fig,
                         encoder_name, encoding_name, bbox, channel=None):
        return self.scene_data.add_encoding_fig(
            traj_idx, obs_idx, cam_name, attr_name, fig, encoder_name,
            encoding_name, bbox, channel)

    def add_traj_attr(self, traj_idx, obs_idx, attr_name, value):
        return self.scene_data.add_traj_attr(traj_idx, obs_idx, attr_name,
                                             value)
