import bisect
import datetime
import json
import random
from copy import copy
from enum import Enum
from functools import cached_property

import jsonpickle
import numpy as np
import torch
import torchvision
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset

from dataset.dataclasses import (BCTraj, SceneDatapoint, SingleCamObservation,
                                 collate_singelcam_obs)
from dataset.trajectory import Trajectory
from utils.constants import METADATA_FILENAME  # TRAJECTORY_ATTRIBUTES,
from utils.constants import (FLAT_ATTRIBUTES, GENERIC_ATTRIBUTES, MaskTypes,
                             filter_list, get_cam_attributes, translate_names)
from utils.geometry import (compute_angle_between_poses,
                            compute_distance_between_poses)
from utils.misc import load_replay_memory
from utils.torch import invert_homogenous_transform

# from utils.torch import stack_trajs

NO_MASK_WARNING = \
    "SceneDataset does not contain masks. Did you forget to run TSDFFusion?"

TRAJECTORIES_DIRNAME = 'trajectories'
EMBEDDINGS_DIRNAME = 'embeddings'
ENCODINGS_DIRNAME = 'encodings'
RECONSTRUCTION_DIRNAME = 'fusion'
DATA_SUFFIX = ".dat"  # TODO: define somewhere central and also use in traj etc
IMG_SUFFIX = ".png"

SMO_PATHS = 'smo_paths'
SMO_ORDER = 'smo_order'
SMO_DATASETS = 'smo_data'

# TODO: just stashing these functions here. Structure!
img_to_tensor = torchvision.transforms.ToTensor()


def load_image(path, crop=None):
    # crop: left, right, top, bottom
    image = Image.open(path)
    tens = img_to_tensor(image)
    if crop is not None:
        l, r, t, b = crop
        tens = tens[:, t:b][:, :, l:r].contiguous()
    return tens


def load_tensor(path, crop=None):
    # crop: left, right, top, bottom
    tens = torch.load(path)
    if crop is not None:
        l, r, t, b = crop
        tens = tens[t:b][:, l:r].contiguous()
    return tens


def save_image(tens, path):
    return torchvision.utils.save_image(tens, path)


def save_tensor(tens, path):
    tens = tens.to('cpu') if tens is not None else tens
    return torch.save(tens, path)


def join_label_lists(lists):
    return sorted(list(set().union(*lists)))


class DirectoryNonEmptyError(Exception):
    pass


class MetadataFileNotExistsError(Exception):
    pass


class TODOUpdateError(Exception):
    pass


class SubSampleTypes(Enum):
    POSE = 1
    CONTENT = 2
    LENGTH = 3
    NONE = 4


class SceneDataset(Dataset):
    def __init__(self, camera_names=None,
                 image_size=(None, None), trim_left=None,
                 subsample_by_difference=True,
                 subsample_to_length=None, object_labels=None,
                 data_root=None, ground_truth_object_pose=None,
                 only_use_labels=None):

        self._debug = False

        self._data_root = data_root

        self.camera_names = camera_names

        image_size = image_size  # or (256, 256)
        self.image_height, self.image_width = image_size

        self.trim_left = trim_left

        self.subsample_by_difference = subsample_by_difference
        self.subsample_to_length = subsample_to_length

        self.object_labels = object_labels

        # NOTE: these are specific to the GT labels of RLBench.
        # Should generalize this.
        self._ignore_gt_labels = [10,
                                  31, 34, 35, 39,
                                  40, 41, 42, 43, 44, 45, 46, 48,
                                  52, 53, 54, 55, 56, 57,
                                  90, 92,
                                  255,
                                  16777215]


        self.ground_truth_object_pose = True  # ground_truth_object_pose

        self.shorten_cam_names = True

        if data_root.exists():
            self.initialize_existing_dir()
        else:
            self.initialize_new_dir()

        self.reset_current_traj()

    def update_camera_crop(self, crop_left):
        self.trim_left = crop_left
        self.image_width -= crop_left
        logger.info("Applied crop and updated camera dims to {}x{}",
                    self.image_width, self.image_height)


    @cached_property
    def _metadata_filename(self):
        return self._data_root / METADATA_FILENAME

    @cached_property
    def _trajectories_dir(self):
        return self._data_root / TRAJECTORIES_DIRNAME

    @cached_property
    def _embeddings_dir(self):
        return self._data_root / EMBEDDINGS_DIRNAME

    @cached_property
    def _encodings_dir(self):
        return self._data_root / ENCODINGS_DIRNAME

    @cached_property
    def _reconstruction_dir(self):
        return self._data_root / RECONSTRUCTION_DIRNAME

    def load_metadata(self):
        logger.info("  Initializing datasete using {}",
                    filename := self._metadata_filename)

        with open(filename) as f:
            data_dict = jsonpickle.decode(f.read())

        for k, v in data_dict.items():
            setattr(self, k, v)

    def write_metadata(self):
        logger.info("Dumping dataset metadata to {}",
                    filename := self._metadata_filename)
        data_dict = {k: v for k, v in vars(self).items()
                     if not k.startswith("_")}

        with open(filename, 'w') as f:
            f.write(jsonpickle.encode(data_dict))

    def initialize_existing_dir(self):
        try:
            self.load_metadata()
        except FileNotFoundError:
            raise MetadataFileNotExistsError(
                "The metadata file for the dataset could not be read. "
                "Expected to be located at {}".format(self._metadata_filename))

        self._paths = self._get_trajectory_paths()
        metadata = self._load_all_traj_metadata()
        self._len = self._get_no_trajectories()
        self._traj_lens = self._get_trajecory_lengths(metadata=metadata)

        if self.object_labels is None:  # skip if object_labels passed to init
            self.object_labels_gt = self._get_object_labels_gt(
                metadata=metadata)
            logger.info("  Extracted gt object labels {}".format(
                self.object_labels_gt))
            self.object_labels_tsdf = self._get_object_labels_tsdf(
                metadata=metadata)
            logger.info("  Extracted tsdf object labels {}".format(
                self.object_labels_tsdf))

        if self.object_labels is None and self.object_labels_gt is None and \
                self.object_labels_tsdf is None:
            logger.warning(NO_MASK_WARNING)

        self._setup_smo_sampling()

    def set_object_labels(self, object_labels):
        logger.info("Setting object labels to {}", object_labels)
        self.object_labels = object_labels

    def _setup_smo_sampling(self):
        if hasattr(self, SMO_PATHS):
            logger.info("Found SMO info in metadata. "
                        "Setting up self for smo sampling")

            smo_paths = getattr(self, SMO_PATHS).items()

            dataset_dict = {
                int(k): load_replay_memory(None, path=self._data_root / v)
                for k, v in smo_paths}

        else:
            dataset_dict = None

        setattr(self, SMO_DATASETS, dataset_dict)

        if not hasattr(self, SMO_ORDER):
            setattr(self, SMO_ORDER, None)

    def _get_object_labels_gt(self, metadata=None):
        if metadata is None:
            metadata = self._load_all_traj_metadata()
        raw_labels = join_label_lists(
            [m.get('object_label_gt', []) for m in metadata])

        return sorted(list(set(raw_labels) - set(self._ignore_gt_labels)))

    def _get_object_labels_tsdf(self, metadata=None):
        if metadata is None:
            metadata = self._load_all_traj_metadata()

        return join_label_lists(
            [m.get('object_label_tsdf', []) for m in metadata])

    def _get_no_trajectories(self):
        return len(self._paths)

    def _get_trajectory_paths(self):
        return sorted(
            [i for i in self._trajectories_dir.iterdir() if i.is_dir()])

    def _load_all_traj_metadata(self) -> list:
        return [self._load_traj_metadata(p) for p in self._paths]

    def _get_trajecory_lengths(self, metadata=None):
        if metadata is None:
            metadata = self._load_all_traj_metadata()

        return [m['len'] for m in metadata]

    @staticmethod
    def _load_traj_metadata(dir):
        with open(dir / METADATA_FILENAME) as f:
            return json.load(f)

    @staticmethod
    def _write_traj_metadata(dir, metadata):
        with open(dir / METADATA_FILENAME, 'w') as f:
            json.dump(metadata, f)

    def initialize_new_dir(self):
        self._data_root.mkdir()
        if any(self._data_root.iterdir()):
            raise DirectoryNonEmptyError(
                "Trying to initialize dataset in non-empty dir {}.".format(
                    self._data_root))
        self._len = 0
        self.write_metadata()

        self._trajectories_dir.mkdir()
        self._embeddings_dir.mkdir()
        self._encodings_dir.mkdir()

    def initialize_scene_reconstruction(self):
        path = self._reconstruction_dir
        exists = path.is_dir()
        if exists and any(path.iterdir()):
            logger.error("Fusion dir {} exists already and is not empty. "
                         "Please delete it manually and try again.", path)
            return None
        elif not exists:
            path.mkdir()
        return path

    def __len__(self):  # number of trajectories
        return self._len

    @cached_property
    def no_obs(self):
        return sum(self._traj_lens)

    @cached_property
    def _traj_idx_ends(self):
        return np.cumsum(self._traj_lens)

    def _index_split(self, index):
        """
        Split the sample index across all obs (generated by a dataloader) into
        trajectory and observation index.
        """
        traj_idx_ends = self._traj_idx_ends
        traj_idx = bisect.bisect(traj_idx_ends, index)
        traj_offset = 0 if traj_idx == 0 else traj_idx_ends[traj_idx - 1]
        obs_idx = index - traj_offset

        return traj_idx, obs_idx

    def reset_current_traj(self):
        self._current_trajectory = Trajectory(
            self.camera_names,
            self.subsample_by_difference, self.subsample_to_length)

    def add(self, obs, action, feedback):
        self._current_trajectory.add(obs, action, feedback)

    def generate_trajectory_dir_name(self):
        return datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    def save_current_traj(self, traj_suffix=""):
        traj_name = self.generate_trajectory_dir_name() + traj_suffix
        traj_path = self._trajectories_dir / traj_name
        traj_path.mkdir()
        with torch.no_grad():
            self._current_trajectory.save(traj_path)

    def __getitem__(self, idx):
        raise NotImplementedError("Implement in sub-classes!")

    def _get_traj_path(self, traj_idx, encoder_name=None, encoding_name=None):
        traj_path = self._paths[traj_idx]

        if encoder_name is None:
            return traj_path
        else:
            traj_name = traj_path.parts[-1]
            if encoding_name is None:
                return self._embeddings_dir / encoder_name / traj_name
            else:
                return self._encodings_dir / encoder_name / encoding_name / \
                    traj_name

    def _load_file(self, traj_idx, attribute, img_idx, encoder_name=None,
                   encoding_name=None):
        traj_path = self._get_traj_path(traj_idx, encoder_name=encoder_name,
                                        encoding_name=encoding_name)

        load_func = load_image if attribute.endswith("rgb") else load_tensor
        file_ending = ".png" if attribute.endswith("rgb") else ".dat"
        file_name = traj_path / attribute

        if attribute not in FLAT_ATTRIBUTES:
            file_name = file_name / str(img_idx)

        file_name_w_suffix = file_name.with_suffix(file_ending)

        if self._debug:
            logger.info("Loading file {}", file_name_w_suffix)

        if self.trim_left is not None and attribute.endswith(
                (("rgb", "_d", "_mask_gt", "_mask_tsdf"))):
            crop = [self.trim_left, None, None, None]
        else:
            crop = None

        return load_func(file_name_w_suffix, crop=crop)

    def _load_data_point(self, traj_idx, img_idx, *sample_attr,
                         encoder_name=None, embedding_attr=[],
                         encoding_name=None, encoding_attr=[], as_dict=False,
                         #  check_attribute=True
                         ):
        data = {}
        for k in sample_attr:
            # if check_attribute and k not in TRAJECTORY_ATTRIBUTES:
            #     raise ValueError("Unexpected attr. {} requested.".format(k))
            try:
                val = self._load_file(traj_idx, k, img_idx)
                # HACK -y: fix the principle point offset for cropped images
                if k.endswith('_int') and self.trim_left is not None:
                    val[0][2] = val[0][2] - self.trim_left
                data[k] = val
            except FileNotFoundError as e:
                logger.error("Could not find the requested file. "
                             "Did you request a non-set attribute? "
                             "Requested traj {}, attr {}, obs {}.",
                             traj_idx, k, img_idx)
                raise e
        for k in embedding_attr:
            try:
                data[k] = self._load_file(traj_idx, k, img_idx,
                                          encoder_name=encoder_name)
            except FileNotFoundError as e:
                logger.error("Could not find the requested file. "
                             "Did you request a non-set attribute? "
                             "Requested traj {}, attr {}, obs {}, enc {}",
                             traj_idx, k, img_idx, encoder_name)
                raise e
        for k in encoding_attr:
            try:
                data[k] = self._load_file(traj_idx, k, img_idx,
                                          encoder_name=encoder_name,
                                          encoding_name=encoding_name)
            except FileNotFoundError as e:
                logger.error(
                    "Could not find the requested file. Did you request a "
                    "non-set attribute? "
                    "Requested traj {}, attr {}, obs {}, enc {}, encoding {}",
                    traj_idx, k, img_idx, encoder_name, encoding_name)
                raise e
        if as_dict:
            return data
        else:
            return SceneDatapoint(**data)

    def sample_observation(self, traj_idx=None, img_idx=None, **kwargs):
        traj_idx = self.sample_traj_idx() if traj_idx is None else traj_idx
        img_idx = self.sample_img_idx(traj_idx) if img_idx is None else img_idx

        return self.get_observation(traj_idx=traj_idx, img_idx=img_idx,
                                    **kwargs)

    def get_observation(self, traj_idx=None, img_idx=None, cam=None,
                        mask_type=None, raw_mask=True,
                        collapse_labels=False, labels=None, get_rgb=True,
                        get_int=False, get_ext=False, get_depth=True,
                        get_mask=True, get_action=False, get_feedback=False,
                        get_gripper_pose=False, get_object_pose=False,
                        get_proprio_obs=False, get_wrist_pose=False,
                        get_object_poses=False,
                        ) -> SingleCamObservation:
        attributes = []

        cam_letter = cam[0] if self.shorten_cam_names else cam

        if get_rgb:
            attributes.append("cam_{}_rgb".format(cam_letter))
        if get_int:
            attributes.append("cam_{}_int".format(cam_letter))
        if get_ext:
            attributes.append("cam_{}_ext".format(cam_letter))
        if get_depth:
            attributes.append("cam_{}_d".format(cam_letter))
        if get_mask:
            if mask_type is None or mask_type.value is None:
                raise ValueError("Trying to get observation with mask, but "
                                 "set mask_type to None.")
            attributes.append("cam_{}_mask_{}".format(cam_letter,
                                                      mask_type.value))
        if get_action:
            attributes.append("action")
        if get_feedback:
            attributes.append("feedback")
        if get_gripper_pose:
            attributes.append("ee_pose")  # ("gripper_pose")
        if get_proprio_obs:
            attributes.append("proprio_obs")
        if get_wrist_pose:
            attributes.append("wrist_pose")
        if get_object_poses or get_object_pose:
            attributes.append("object_poses")

        data_dict = self._load_data_point(traj_idx, img_idx, *attributes,
                                          as_dict=True)
        # Get generic names, eg. replace cam_w_rgb by rgb.
        data_dict = translate_names(data_dict, cam)

        obs = SingleCamObservation(**data_dict)
        obs.mask = self.process_mask(obs.mask, object_labels=labels,
                                     raw_mask=raw_mask,
                                     collapse=collapse_labels)

        obs.frozen = True

        if get_object_pose:
            logger.warning(
                "get_object_pose is deprecated. Use get_object_poses instead."
                "The new way returns a dict of poses, not a stacked tensor.")
            obs.object_pose = torch.stack(
                [p for _, p in obs.object_poses.items()], dim=0)
            obs.object_pose = obs.object_pose.to(obs.cam_rgb.device)

            assert get_object_poses is False
            obs.object_poses = None

        return obs

    def get_gt_obs(self, traj_idx, img_idx, cam, object_labels):
        return self.get_observation(
            traj_idx=traj_idx, img_idx=img_idx,
            cam=cam, mask_type=MaskTypes.GT, raw_mask=False,
            labels=object_labels, collapse_labels=True, get_rgb=True,
            get_int=True, get_ext=True, get_depth=True, get_mask=True,
            get_action=False, get_feedback=False, get_gripper_pose=True,
            get_object_pose=True, get_proprio_obs=False, get_wrist_pose=False)

    def get_all_first_obs(self, cam):
        obs = [self.get_observation(
            traj_idx=t, img_idx=0, cam=cam, mask_type=MaskTypes.GT,
            raw_mask=True, collapse_labels=True, get_rgb=True,
            get_int=True, get_ext=True, get_depth=True, get_mask=True,
            get_action=False, get_feedback=False, get_gripper_pose=True,
            get_object_pose=True, get_proprio_obs=False, get_wrist_pose=False)
            for t in range(len(self))]

        return collate_singelcam_obs(obs)

    def sample_data_point_with_object_labels(self, cam="wrist", traj_idx=None,
                                             img_idx=None, get_mask=True,
                                             mask_type=MaskTypes.GT):

        obs = self.sample_observation(
            traj_idx=traj_idx, img_idx=img_idx,
            cam=cam, mask_type=mask_type, raw_mask=True,
            labels=self.object_labels, collapse_labels=False, get_rgb=True,
            get_int=False, get_ext=False, get_depth=True, get_mask=get_mask,
            get_action=False, get_feedback=False, get_gripper_pose=False,
            get_object_pose=False, get_proprio_obs=False, get_wrist_pose=False)

        return obs.cam_rgb, obs.cam_d, obs.mask

    def sample_data_point_with_ground_truth(self, cam="wrist", traj_idx=None,
                                            img_idx=None, get_mask=True,
                                            mask_type=MaskTypes.GT):
        obs = self.sample_observation(
            traj_idx=traj_idx, img_idx=img_idx,
            cam=cam, mask_type=mask_type, raw_mask=True,
            labels=self.object_labels, collapse_labels=False, get_rgb=True,
            get_int=True, get_ext=True, get_depth=True, get_mask=get_mask,
            get_action=False, get_feedback=False, get_gripper_pose=False,
            get_object_pose=True, get_proprio_obs=False, get_wrist_pose=False)

        return obs.cam_rgb, obs.cam_d, obs.mask, obs.cam_int, obs.cam_ext, \
            obs.object_pose

    def process_mask(self, mask, object_labels=1, raw_mask=False,
                     collapse=True):
        # gettig the raw mask is handy for TSNE etc
        if raw_mask or mask is None:
            pass
        elif object_labels == -1:  # take union of all object masks
            mask = torch.where(mask != 0, 1, 0).squeeze(-1).float()
        elif type(object_labels) is list:
            if collapse:  # Binary: filter the mask and set to 0/1.
                mask = torch.where(sum(mask == i for i in object_labels
                                       ).bool(), 1, 0).squeeze(-1).float()
            else:  # keep the label values, only filter
                mask = sum(torch.where(mask == i, i, 0)
                           for i in object_labels).float()
        else:
            raise DeprecationWarning(
                "Should not rely on 10er system for gt mask.")
            # HACK: objects are divded into parts with different labels, eg
            # MicroWave has body and door with labels 84, 87. So far, objects
            # seem to occupy a range of 10. To get a whole object we can thus
            # to integer dividion by 10.
            mask = torch.where(mask // 10 == object_labels // 10,
                               1, 0).squeeze(-1).float()

        return mask

    def sample_traj_idx(self, batch_size=None):
        if batch_size is None:
            return random.sample(range(len(self)), 1)[0]
        else:
            return random.sample(range(len(self)), batch_size)

    def sample_img_idx(self, traj_idx):
        return random.sample(range(self._traj_lens[traj_idx]), 1)[0]

    def get_img_idx_with_different_pose(self, traj_idx, pose_a,
                                        dist_threshold=0.2, angle_threshold=20,
                                        num_attempts=10, cam="wrist"):
        """
        Try to get an image with a different pose to the one passed in.
        This can get you a different timestep in case both cams are the same,
        or even the same one if the cams are different.
        If none can be found, return None.
        """

        cam_letter = cam[0] if self.shorten_cam_names else cam

        attribute = "cam_{}_ext".format(cam_letter)

        counter = 0

        while counter < num_attempts:
            img_idx = self.sample_img_idx(traj_idx)
            pose = self._load_data_point(traj_idx, img_idx, attribute,
                                         as_dict=True)[attribute]

            diff = compute_distance_between_poses(pose_a, pose)
            angle_diff = compute_angle_between_poses(pose_a, pose)
            if (diff > dist_threshold) or (angle_diff > angle_threshold):
                return img_idx

            counter += 1

        return None

    def get_img_idx_with_single_object_visible(self, traj_idx, labels,
                                               mask_type,
                                               num_attempts=100, cam="wrist"):
        """
        Try to get an image of a MO scene where only one object is visible.
        If none can be found, return None.
        """

        cam_letter = cam[0] if self.shorten_cam_names else cam

        mask_type_str = 'gt' if mask_type is MaskTypes.GT else 'tsdf'

        attribute = "cam_{}_mask_{}".format(cam_letter, mask_type_str)

        desired_elements = set(labels)
        desired_elements.add(0)

        counter = 0
        while counter < num_attempts:
            img_idx = self.sample_img_idx(traj_idx)
            mask = self._load_data_point(traj_idx, img_idx, attribute,
                                         as_dict=True)[attribute].long()

            if mask_type is MaskTypes.GT:
                ignore = sum(mask == i for i in self._ignore_gt_labels).bool()
                mask = torch.where(ignore, 0, mask)

            elements = set(mask.unique().numpy())

            if elements == desired_elements:
                return img_idx
            counter += 1

        return None

    def _get_bc_traj(self, traj_idx, cams=("wrist",), mask_type=None,
                     sample_freq=None, fragment_length=-1, extra_attr=[],
                     encoder_name=None, embedding_attr=[], encoding_name=None,
                     encoding_attr=[], skip_raw_cam_attr=False,
                     skip_generic_attributes=False):
        # define attributes to load
        generic = [] if skip_generic_attributes else copy(GENERIC_ATTRIBUTES)
        if not self.ground_truth_object_pose and not skip_generic_attributes:
            generic.remove("object_poses")
        sample_attr = generic + extra_attr
        for c in cams:  # always need mask, even for precomputed embedding
            cam_attributes = get_cam_attributes(c, mask_type=mask_type,
                                                depth_only=skip_raw_cam_attr)
            sample_attr.extend(cam_attributes)

        # define frames to load
        traj_len = self._traj_lens[traj_idx]
        # print(traj_len, self._get_traj_path(traj_idx))

        if fragment_length == -1:
            start, stop = 0, traj_len
        else:
            assert traj_len >= fragment_length
            # start = random.sample(range(traj_len - fragment_length), 1)[0]
            # make sampling of last fragment more likely
            start = random.sample(range(traj_len), 1)[0]
            start = min(start, traj_len - fragment_length)
            stop = start + fragment_length
        load_iter = iter(range(start, stop))

        # For a trajectory load flat attributes only at first timestep.
        datapoints = [self._load_data_point(
            traj_idx, next(load_iter), *sample_attr, encoder_name=encoder_name,
            embedding_attr=embedding_attr, encoding_name=encoding_name,
            encoding_attr=encoding_attr)]
        sample_attr = filter_list(sample_attr, FLAT_ATTRIBUTES)
        datapoints.extend([self._load_data_point(
            traj_idx, i, *sample_attr, encoder_name=encoder_name,
            embedding_attr=embedding_attr, encoding_name=encoding_name,
            encoding_attr=encoding_attr) for i in load_iter])

        return BCTraj(*datapoints, cams=cams, sample_freq=sample_freq,
                      mask_type=mask_type)

    def add_embedding(self, traj_idx, obs_idx, cam_name, emb_name, encoding,
                      encoder_name):
        # Need to catch Nones and have custom func to write those to disk?
        traj_path = self._paths[traj_idx]
        traj_name = traj_path.parts[-1]
        embed_path = self._embeddings_dir / encoder_name / traj_name

        cam_letter = cam_name[0]
        file = (embed_path / "cam_{}_{}".format(cam_letter, emb_name) /
                str(obs_idx)).with_suffix(DATA_SUFFIX)

        file.parents[0].mkdir(parents=True, exist_ok=True)

        save_tensor(encoding, file)

    def load_embedding(self, traj_idx, img_idx, cam, embedding_name,
                       encoder_name):
        cam_letter = cam[0] if self.shorten_cam_names else cam
        attr_name = "cam_" + cam_letter + "_" + embedding_name

        return self._load_data_point(
            traj_idx, img_idx, encoder_name=encoder_name,
            embedding_attr=[attr_name], as_dict=True)[attr_name]

    def load_embedding_batch(self, traj_idx, img_idx, cam, embedding_name,
                             encoder_name):
        if type(traj_idx) is list or type(embedding_name) is list:
            raise NotImplementedError
        return torch.stack(tuple(
            self.load_embedding(traj_idx, i, cam, embedding_name, encoder_name)
            for i in img_idx))

    def add_encoding(self, traj_idx, obs_idx, cam_name, attr_name, encoding,
                     encoder_name, encoding_name):
        traj_path = self._paths[traj_idx]
        traj_name = traj_path.parts[-1]

        attr_dir = attr_name if cam_name is None else \
            "cam_{}_{}".format(cam_name[0], attr_name)

        dir = self._encodings_dir / encoder_name / encoding_name / \
            traj_name / attr_dir

        file = (dir / str(obs_idx)).with_suffix(DATA_SUFFIX)

        file.parents[0].mkdir(parents=True, exist_ok=True)

        save_tensor(encoding, file)

    def add_encoding_fig(self, traj_idx, obs_idx, cam_name, attr_name, fig,
                         encoder_name, encoding_name, bbox, channel=None):
        traj_path = self._paths[traj_idx]
        traj_name = traj_path.parts[-1]

        attr_dir = "fig_cam_{}_{}".format(cam_name[0], attr_name)

        if channel is not None:
            attr_dir += "_" + str(channel)

        dir = self._encodings_dir / encoder_name / encoding_name / \
            traj_name / attr_dir

        file = (dir / str(obs_idx)).with_suffix(IMG_SUFFIX)

        file.parents[0].mkdir(parents=True, exist_ok=True)

        fig.savefig(file, transparent=True, bbox_inches=bbox)

    def add_traj_attr(self, traj_idx, obs_idx, attr_name, value):
        traj_path = self._paths[traj_idx]
        traj_name = traj_path.parts[-1]
        traj_path = self._trajectories_dir / traj_name

        file = (traj_path / attr_name / str(obs_idx)).with_suffix(DATA_SUFFIX)

        file.parents[0].mkdir(parents=True, exist_ok=True)

        save_tensor(value, file)

    def get_scene(self, traj_idx, cams=None, subsample_types=None):
        trajs = {}
        for c in cams:
            traj = self._get_bc_traj(
                traj_idx, (c, ), mask_type=None, sample_freq=None,
                fragment_length=-1, skip_generic_attributes=True)

            subsample = None if subsample_types is None else subsample_types[c]

            traj = self._subsample_trajectory(traj, subsample)

            trajs[c] = traj

        return trajs

    def _subsample_trajectory(self, traj, type):  # for dataclass
        if type is SubSampleTypes.POSE:
            indeces = Trajectory.get_idx_by_pose_difference_threshold_matrix(
                traj.cam_ext.numpy())
        elif type is SubSampleTypes.LENGTH:
            raise NotImplementedError("Need to pass target_length.")
            # indeces = Trajectory.get_idx_by_target_len(
            #     len(traj.cam_rgb), target_length)
        elif type is SubSampleTypes.CONTENT:
            ...  # TODO
        elif type is SubSampleTypes.NONE or type is None:
            indeces = list(range(len(traj.cam_ext)))

        else:
            raise ValueError("Unexpected subsample type {}".format(type))

        for field in dir(traj):
            if not field.startswith('__'):
                value = getattr(traj, field)
                if (is_tensor := torch.is_tensor(value)) and \
                        field not in FLAT_ATTRIBUTES:
                    subsampled = torch.index_select(
                        value, 0, torch.tensor(indeces))
                    setattr(traj, field, subsampled)
                elif is_tensor:
                    inflated = value.unsqueeze(0).repeat(
                        len(indeces), *[1 for _ in value.shape])
                    setattr(traj, field, inflated)
                elif value is None:
                    pass
                else:
                    raise TypeError(
                        "Encountered unexpected type {} in subsampling".format(
                            type(value)))

        return traj

    def add_tsdf_masks(self, traj_idx, cam_name, masks, labels):
        traj_path = self._paths[traj_idx]

        cam_letter = cam_name[0]
        mask_dir = traj_path / "cam_{}_mask_tsdf".format(cam_letter)
        mask_dir.mkdir(parents=False, exist_ok=True)

        for obs_idx, obs_mask in enumerate(masks):
            file = (mask_dir / str(obs_idx)).with_suffix(DATA_SUFFIX)

            save_tensor(obs_mask.clone(), file)

        traj_metadata = self._load_traj_metadata(traj_path)
        traj_metadata["object_label_tsdf"] = labels

        self._write_traj_metadata(traj_path, traj_metadata)

    @classmethod
    def join_datasets(self, *sets):
        if len(sets) == 1:
            return sets[0]
        else:
            raise NotImplementedError(
                "Didn't yet implement join for new dataset format.")
