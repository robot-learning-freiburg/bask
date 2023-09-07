import bisect
import datetime
import json
import random
from abc import abstractmethod
from collections.abc import Iterable
from copy import copy
from enum import Enum
from functools import cached_property
from pathlib import Path

import jsonpickle
import numpy as np
import torch
from loguru import logger
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from torch.utils.data import Dataset

from config import (ANGLE_DISTANCE_THRESHOLD, DATA_SUFFIX, IMG_SUFFIX,
                    LINEAR_DISTANCE_THRESHOLD, METADATA_FILENAME)
from dataset.trajectory import Trajectory
from utils.data_loading import load_image, load_tensor, save_tensor
from utils.geometry import (compute_angle_between_poses,
                            compute_distance_between_poses)
from utils.misc import load_replay_memory
from utils.observation import (ALL_CAMERA_ATTRIBUTES, GENERIC_ATTRIBUTES,
                               CameraOrder, MaskTypes, SceneObservation,
                               SingleCamObservation, SingleCamSceneObservation,
                               collate, dict_to_tensordict,
                               downsample_tensordict_by_idx,
                               downsample_to_target_freq, empty_batchsize,
                               get_cam_attributes,
                               get_idx_by_pose_difference_threshold_matrix,
                               get_idx_by_target_len, is_flat_attribute,
                               make_cam_attr_name)

NO_MASK_WARNING = \
    "SceneDataset does not contain masks. Did you forget to run TSDFFusion?"

TRAJECTORIES_DIRNAME = 'trajectories'
EMBEDDINGS_DIRNAME = 'embeddings'
ENCODINGS_DIRNAME = 'encodings'
RECONSTRUCTION_DIRNAME = 'fusion'

SMO_PATHS = 'smo_paths'
SMO_ORDER = 'smo_order'
SMO_DATASETS = 'smo_data'


def join_label_lists(lists):
    return sorted(list(set().union(*lists)))


class DirectoryNonEmptyError(Exception):
    pass


class MetadataFileNotExistsError(Exception):
    pass


class CannotCreateDatasetError(Exception):
    pass


class DepreceatedError(Exception):
    pass


class SubSampleTypes(Enum):
    POSE = 1
    CONTENT = 2
    LENGTH = 3
    NONE = 4


class SceneDataset(Dataset):
    def __init__(self, *,
                 allow_creation: bool = False,
                 camera_names: tuple[str] | None = None,
                 image_size: tuple[int, int] | tuple[None, None] = (None, None),
                 image_crop: tuple[int, int, int, int] | None = None,
                 subsample_by_difference: bool = True,
                 subsample_to_length: int | None = None,
                 object_labels: list[int] | None = None,
                 data_root: Path,
                 ground_truth_object_pose: bool = True):
        """
        Base dataset.

        Parameters
        ----------
        allow_creation: bool
            Allow the creation of a new dataset on disk. If False and the data
            directory cannot be found, an error is raised.
        camera_names : tuple[string] | None, optional
            Camera names in the order they should be returned.
            Will be overwritten by metadata loading if the dataset already
            exists.
        image_size : tuple[int] | tuple[None, None], optional
            Image size (height, width). Size is before cropping.
            Will be overwritten by metadata loading if the dataset already
            exists.
        image_crop : tuple[int] | None, optional
            Number of pixels to crop from the image (left, right, top, bottom).
        subsample_by_difference : bool, optional
            When recording a new trajectory, subsample it st the wrist pose
            difference betwen successive frames is at least a defined constant.
        subsample_to_length : int/None, optional
            When recording a new trajectory, subsample it to this length.
            No subsampling when None. Don't combine with subsample_by_difference.
            By default None
        object_labels : list[int], optional
            When sampling an observation with object masks, these are the
            labels included in the mask. Usually, this will be None and the
            labels are determined automatically from the available labels.
            But can be specified if only specific labels are desired.
            By default None.
        data_root : pathlib.Path
            The root directory of the dataset.
        ground_truth_object_pose : bool, optional
            Whether to return ground truth object poses. By default None

        Raises
        ------
        CannotCreateDatasetError
            The dat directory could not be found and allow_creation is False.
        """

        self._debug = False

        self._data_root = data_root

        self.camera_names = camera_names

        self.image_height, self.image_width = image_size

        self.image_crop = image_crop

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


        self.ground_truth_object_pose = ground_truth_object_pose

        self.shorten_cam_names = True

        if data_root.exists():
            self.initialize_existing_dir()
        elif allow_creation:
            self.initialize_new_dir()
        else:
            raise CannotCreateDatasetError(
                "Data directory does not exist and allow_creation is False."
                "Expected data to be located at {}".format(data_root))

        self.reset_current_traj()


    def update_camera_crop(self, image_crop: tuple[int, int, int, int] | None
                           ) -> None:
        """
        Sets the image crop and updates the image dimensions accordingly.

        Parameters
        ----------
        image_crop : tuple[int]
            Pixels to crop (left, right, top, bottom).
        """

        self.image_crop = image_crop

        if image_crop is not None:
            self.image_width -= image_crop[0] - image_crop[1] # type: ignore
            self.image_height -= image_crop[2] - image_crop[3]  # type: ignore

            child_sets = getattr(self, SMO_DATASETS) or {}
            for _, smo_set in child_sets.items():
                smo_set.update_camera_crop(image_crop)

            logger.info("Applied crop and updated camera dims to {}x{}",
                        self.image_width, self.image_height)


    @cached_property
    def _metadata_filename(self) -> Path:
        return self._data_root / METADATA_FILENAME

    @cached_property
    def _trajectories_dir(self) -> Path:
        return self._data_root / TRAJECTORIES_DIRNAME

    @cached_property
    def _embeddings_dir(self) -> Path:
        return self._data_root / EMBEDDINGS_DIRNAME

    @cached_property
    def _encodings_dir(self) -> Path:
        return self._data_root / ENCODINGS_DIRNAME

    @cached_property
    def _reconstruction_dir(self) -> Path:
        return self._data_root / RECONSTRUCTION_DIRNAME

    def load_metadata(self) -> None:
        logger.info("Initializing datasete using {}",
                    filename := self._metadata_filename)

        with open(filename) as f:
            data_dict = jsonpickle.decode(f.read())

        assert type(data_dict) == dict

        for k, v in data_dict.items():
            setattr(self, k, v)

    def write_metadata(self) -> None:
        logger.info("Dumping dataset metadata to {}",
                    filename := self._metadata_filename)
        data_dict = {k: v for k, v in vars(self).items()
                     if not k.startswith("_")}

        with open(filename, 'w') as f:
            f.write(jsonpickle.encode(data_dict))  # type: ignore

    def initialize_existing_dir(self) -> None:
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
            logger.info("Extracted gt object labels {}".format(
                self.object_labels_gt))
            self.object_labels_tsdf = self._get_object_labels_tsdf(
                metadata=metadata)
            logger.info("Extracted tsdf object labels {}".format(
                self.object_labels_tsdf))

        if self.object_labels is None and self.object_labels_gt is None and \
                self.object_labels_tsdf is None:
            logger.warning(NO_MASK_WARNING)

        self._setup_smo_sampling()

    def set_object_labels(self, object_labels: tuple[int]) -> None:
        logger.info("Setting object labels to {}", object_labels)
        self.object_labels = object_labels

    def _setup_smo_sampling(self) -> None:
        """
        Setup self for Synthethic Multi Object (SMO) sampling for dense
        correspondence sampling.

        Loads the single object datasets for SMO and adds them to
        self.SMO_DATASETS. If self.SMO_ORDER is not set, it is set to None.
        SMO_ORDER allows to specify how to layer the objects in SMO on top
        of each other.
        """
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

    def _get_object_labels_gt(self, metadata: Iterable[dict] | None = None
                              ) -> list[int]:
        if metadata is None:
            metadata = self._load_all_traj_metadata()
        raw_labels = join_label_lists(
            [m.get('object_label_gt', []) for m in metadata])

        return sorted(list(set(raw_labels) - set(self._ignore_gt_labels)))

    def _get_object_labels_tsdf(self, metadata: Iterable[dict] | None = None
                                ) -> list[int]:
        if metadata is None:
            metadata = self._load_all_traj_metadata()

        return join_label_lists(
            [m.get('object_label_tsdf', []) for m in metadata])

    def _get_no_trajectories(self) -> int:
        return len(self._paths)

    def _get_trajectory_paths(self) -> list[Path]:
        return sorted(
            [i for i in self._trajectories_dir.iterdir() if i.is_dir()])

    def _load_all_traj_metadata(self) -> list[dict]:
        return [self._load_traj_metadata(p) for p in self._paths]

    def _get_trajecory_lengths(self, metadata: Iterable[dict] | None = None
                               ) -> list[int]:
        if metadata is None:
            metadata = self._load_all_traj_metadata()

        return [m['len'] for m in metadata]

    @staticmethod
    def _load_traj_metadata(dir: Path) -> dict:
        with open(dir / METADATA_FILENAME) as f:
            return json.load(f)

    @staticmethod
    def _write_traj_metadata(dir, metadata: dict) -> None:
        with open(dir / METADATA_FILENAME, 'w') as f:
            json.dump(metadata, f)

    def initialize_new_dir(self) -> None:
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

    def initialize_scene_reconstruction(self) -> Path | None:
        path = self._reconstruction_dir
        exists = path.is_dir()
        if exists and any(path.iterdir()):
            logger.error("Fusion dir {} exists already and is not empty. "
                         "Please delete it manually and try again.", path)
            return None
        elif not exists:
            path.mkdir()

        return path

    def __len__(self) -> int:  # number of trajectories
        return self._len

    @cached_property
    def no_obs(self) -> int:
        return sum(self._traj_lens)

    @cached_property
    def _traj_idx_ends(self) -> np.ndarray:
        return np.cumsum(self._traj_lens)

    def _index_split(self, index: int) -> tuple[int, int]:
        """
        Split the sample index across all obs (generated by a dataloader) into
        trajectory and observation index.
        """
        traj_idx_ends = self._traj_idx_ends
        traj_idx = bisect.bisect(traj_idx_ends, index)
        traj_offset = 0 if traj_idx == 0 else traj_idx_ends[traj_idx - 1]
        obs_idx = index - traj_offset

        return traj_idx, obs_idx

    def reset_current_traj(self) -> None:
        self._current_trajectory = Trajectory(
            self.camera_names,   # type: ignore
            self.subsample_by_difference, self.subsample_to_length)

    def add_observation(self, obs: SceneObservation) -> None: # type: ignore
        self._current_trajectory.add(obs)

    def generate_trajectory_dir_name(self) -> str:
        return datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    def save_current_traj(self, traj_suffix: str = "") -> None:
        traj_name = self.generate_trajectory_dir_name() + traj_suffix
        traj_path = self._trajectories_dir / traj_name
        traj_path.mkdir()
        with torch.no_grad():
            self._current_trajectory.save(traj_path)

    @abstractmethod
    def __getitem__(self, idx: int):
        raise NotImplementedError("Implement in sub-classes!")

    def _get_traj_path(self, traj_idx: int, encoder_name: str | None = None,
                       encoding_name: str | None = None) -> Path:
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

    def _get_cam_attr_name(self, cam: str, attr: str) -> str:
        """
        Helper function to get the full name of an attribute for a given cam.
        Attributes on disk are named as cam_{cam}_{attr}
        """
        cam_letter = cam[0] if self.shorten_cam_names else cam

        return make_cam_attr_name(cam_letter, attr)

    def _load_file(self, traj_idx: int, attribute: str, img_idx: int,
                   encoder_name: str | None = None,
                   encoding_name: str | None = None) -> torch.Tensor:
        traj_path = self._get_traj_path(traj_idx, encoder_name=encoder_name,
                                        encoding_name=encoding_name)

        load_func = load_image if attribute.endswith("rgb") else load_tensor
        file_ending = ".png" if attribute.endswith("rgb") else ".dat"
        file_name = traj_path / attribute

        if not is_flat_attribute(attribute):
            file_name = file_name / str(img_idx)

        file_name_w_suffix = file_name.with_suffix(file_ending)

        if self._debug:
            logger.info("Loading file {}", file_name_w_suffix)

        if self.image_crop is not None and attribute.endswith(
                (("rgb", "_d", "_mask_gt", "_mask_tsdf"))):
            crop = self.image_crop
        else:
            crop = None

        return load_func(file_name_w_suffix, crop=crop)

    def _load_data_point(self, traj_idx: int, img_idx: int,
                         camera_order: tuple[str],
                         sample_attr: dict[str, str] | None = None,
                         camera_attr: dict[str, dict[str, str]] | None = None,
                         encoder_name: str | None = None,
                         embedding_attr: dict[str, dict[str, str]] | None = None,
                         encoding_name: str | None = None,
                         encoding_attr: dict[str, str] | None =None,
                         as_dict: bool = False, as_single_cam: bool = False
        ) -> dict | SingleCamSceneObservation | SceneObservation:  # type: ignore
        """
        Load a single data point from disk.

        Parameters
        ----------
        traj_idx : int
            Index of the trajectory.
        img_idx : int
            Index of the observation in the trajectory.
        sample_attr : dict
            Dict mapping generic attribute names to file names.
        camera_order : tuple[str]
            Tuple of camera names. Attached to the datapoint to ensure
            camera order is consistent with user expectations.
        camera_attr : dict
            Dict mapping camera names to dicts mapping attribute names to file
            names.
        encoder_name : str, optional
            Name of the encoder when loading pre-encoded attributes,
            by default None
        embedding_attr : dict[str, dict[str, str]], optional
            Dict mapping cameras to mapping embedding attributes to file names,
            by default empty.
        encoding_name : str, optional
            Name of a full kp-encoding, by default None
        encoding_attr : dict, optional
            Dict mapping the encoding name to its file name, by default empty.
        as_dict : bool, optional
            Whether to return the observation a nested dict or TensorClass,
            by default False
        as_single_cam : bool, optional
            Whether to return the observation as a SingleCamSeneObs or
            SceneObservation. SceneObservation is more general, as it can hold
            multi-cam data. SingleCamSceneObservation is more convenient when
            only a single camera is used, as attributes are tensors and not
            tuples of tensors.

        Returns
        -------
        dict/TensorClass
            The data point.

        Raises
        ------
        FileNotFoundError
            The requested attributed could not be found at the generated file
            location.
        """

        if sample_attr is None:
            sample_attr = {}

        if camera_attr is None:
            camera_attr = {}

        if embedding_attr is None:
            embedding_attr = {}

        if encoding_attr is None:
            encoding_attr = {}

        if as_single_cam:
            assert len(camera_order) == 1

        assert not (as_single_cam and as_dict)

        multicam_data = {}

        cam_data = {}  # to bind variable in case of empty camera_order

        for cam in camera_order:
            cam_data = {}

            if cam in camera_attr.keys():
                for k, n in camera_attr[cam].items():
                    try:
                        val = self._load_file(traj_idx, n, img_idx)
                        # HACK: fix the principle point offset for cropped imgs
                        if k == 'intr' and self.image_crop is not None:
                            val[0][2] = val[0][2] - self.image_crop[0]
                            val[1][2] = val[1][2] - self.image_crop[1]
                        cam_data[k] = val
                    except FileNotFoundError as e:
                        logger.error("Could not find the requested file. "
                                    "Did you request a non-set attribute? "
                                    "Requested traj {}, attr {}, obs {}.",
                                    traj_idx, n, img_idx)
                        raise e

            if cam in embedding_attr.keys():
                for k, n in embedding_attr[cam].items():
                    try:
                        cam_data[k] = self._load_file(traj_idx, n, img_idx,
                                                      encoder_name=encoder_name)
                    except FileNotFoundError as e:
                        logger.error("Could not find the requested file. "
                                    "Did you request a non-set attribute? "
                                    "Requested traj {}, attr {}, obs {}, enc {}",
                                    traj_idx, k, img_idx, encoder_name)
                        raise e

            if not as_dict and not as_single_cam:
                multicam_data[cam] = SingleCamObservation(
                    **cam_data, batch_size=empty_batchsize)
            elif as_dict:
                multicam_data[cam] = cam_data

        if as_single_cam:
            data = cam_data
        else:
            multicam_obs = multicam_data if as_dict else dict_to_tensordict(
                multicam_data | {'_order': CameraOrder._create(camera_order)})
            data = {'cameras': multicam_obs}

        for k, n in sample_attr.items():
            try:
                val = self._load_file(traj_idx, n, img_idx)
                if type(val) is dict:
                    val = dict_to_tensordict(val)
                data[k] = val   # type: ignore
            except FileNotFoundError as e:
                logger.error("Could not find the requested file. "
                             "Did you request a non-set attribute? "
                             "Requested traj {}, attr {}, obs {}.",
                             traj_idx, n, img_idx)
                raise e

        for k in encoding_attr:
            try:
                data[k] = self._load_file(traj_idx, k, img_idx,  # type: ignore
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
        elif as_single_cam:
            return SingleCamSceneObservation(**data, batch_size=empty_batchsize)
        else:
            return SceneObservation(**data, batch_size=empty_batchsize)

    def sample_observation(self, traj_idx: int | None = None,
                           img_idx: int | None = None, **kwargs):
        """
        Sample an observation from the dataset. Will NOT be uniform across all
        observations, as we first sample a trajectory and then an observation
        from it, and trajectories can have different lengths.

        Allows to sample only the trajectory index, only the image index, or
        both.

        Wraps get_observation and passes all kwargs to it.

        Parameters
        ----------
        traj_idx : int/None, optional
            If None, sample the trajectory index, by default None
        img_idx : int/None, optional
            If None, sample the observation index, by default None

        Returns
        -------
        _type_
            _description_
        """
        traj_idx = self.sample_traj_idx() if traj_idx is None else traj_idx  # type: ignore
        assert type(traj_idx) is int
        img_idx = self.sample_img_idx(traj_idx) if img_idx is None else img_idx

        return self.get_observation(traj_idx=traj_idx, img_idx=img_idx,
                                    **kwargs)

    def get_observation(self, *, traj_idx: int, img_idx: int, cam: str,
                        mask_type: MaskTypes | None = None,
                        raw_mask: bool = True,
                        collapse_labels: bool = False,
                        labels: Iterable[int] | None = None,
                        get_rgb: bool = True,
                        get_int: bool =False,
                        get_ext: bool = False,
                        get_depth: bool = True,
                        get_mask: bool = True,
                        get_action: bool = False,
                        get_feedback: bool = False,
                        get_object_poses: bool = False,
                        get_proprio_obs: bool = False,
                        get_gripper_pose: bool = False,
                        get_object_pose: bool = False,
                        get_wrist_pose: bool = False
                        ) -> SingleCamObservation:  # type: ignore
        """
        Get an observation from the dataset. Convenience wrapper around
        _load_data_point that allows to pick attributes via boolean flags
        instead of having to specify the attribute names.

        Returns either a SceneObservation or a SingleCamSceneObservation,
        depending on whether cam is a string or an iterable of strings.

        Parameters
        ----------
        traj_idx : int
            Trajectory index.
        img_idx : int
            Observation index.
        cam : str
            Camera name.
        mask_type : MaskType, optional
            Type of mask, ie. GT or TSDF, by default None
        raw_mask : bool, optional
            Whether to return the raw mask. See process_mask for details,
            by default True
        collapse_labels : bool, optional
            Whether to collapse mask labels. See process_mask for details,
            by default True
        labels : iterable[int], optional
            Which labels from the raw mask to set to one. See process_mask for
            details, by default None
        get_rgb : bool, optional
            Get the RGB observation, by default True
        get_int : bool, optional
            Get the camera intrinsics, by default False
        get_ext : bool, optional
            Get the camera extrinsics, by default False
        get_depth : bool, optional
            Get the depth image, by default True
        get_mask : bool, optional
            Get object masks, by default True
        get_action : bool, optional
            Get the action taken, by default False
        get_feedback : bool, optional
            Get the feedback received, by default False
        get_object_poses : bool, optional
            Get the ground-truth object poses, by default False
        get_proprio_obs : bool, optional
            Get the robot proprioception values, ie joint angles,
            by default False
        get_gripper_pose : bool, optional
            Get the gripper (EE) pose, by default False
        get_object_pose : bool, optional
            DEPRECEATED, by default False
        get_wrist_pose : bool, optional
            DEPRECEATED, by default False

        Returns
        -------
        SceneObservation/SingleCamSceneObservation
            The observation.

        Raises
        ------
        DepreceatedError
            When passing deprecated arguments, get_object_pose or
            get_wrist_pose.
        ValueError
            When requesting mask but passing None-mask_type.
        """

        if get_object_pose:
            raise DepreceatedError(
                "get_object_pose is deprecated. Use get_object_poses instead. "
                "The new way returns a dict of poses, not a stacked tensor.")
        if get_wrist_pose:
            raise DepreceatedError(
                "get_wrist_pose is deprecated. Use get_gripper_pose instead.")

        camera_attributes = {}

        # TODO: this is ugly. Make it nicer.
        cam_args = (get_rgb, get_int, get_ext, get_depth)
        cam_attr = ('rgb', 'intr', 'extr', 'depth')
        for arg, attr in zip(cam_args, cam_attr):
            if arg:
                file_suffix = ALL_CAMERA_ATTRIBUTES[attr]
                file_name = self._get_cam_attr_name(cam, file_suffix)
                camera_attributes[attr] = file_name

        if get_mask:
            if mask_type in [None, MaskTypes.NONE]:
                raise ValueError("Trying to get observation with mask, but "
                                 "set mask_type to None.")
            attr_name = 'mask_' + mask_type.value  # type: ignore
            file_suffix = ALL_CAMERA_ATTRIBUTES[attr_name]
            file_name = self._get_cam_attr_name(cam, file_suffix)
            camera_attributes['mask'] = file_name

        generic_attributes = {}

        gen_args = (get_action, get_feedback, get_gripper_pose,
                    get_proprio_obs, get_object_poses)
        gen_attr = ('action', 'feedback', 'ee_pose',
                    'proprio_obs', 'object_poses')
        for arg, attr in zip(gen_args, gen_attr):
            if arg:
                file_name = GENERIC_ATTRIBUTES[attr]
                generic_attributes[attr] = file_name

        obs = self._load_data_point(
            traj_idx, img_idx, camera_order=(cam, ),
            sample_attr=generic_attributes,
            camera_attr={cam: camera_attributes}, as_single_cam=True)

        obs.mask = self.process_mask(obs.mask, object_labels=labels,  # type: ignore
                                     raw_mask=raw_mask,
                                     collapse=collapse_labels)

        return obs

    def get_gt_obs(self, traj_idx: int, img_idx: int, cam: str,
                   object_labels: Iterable[int] | None = None
                   ) -> SingleCamObservation:  # type: ignore
        return self.get_observation(
            traj_idx=traj_idx, img_idx=img_idx,
            cam=cam, mask_type=MaskTypes.GT, raw_mask=False,
            labels=object_labels, collapse_labels=True, get_rgb=True,
            get_int=True, get_ext=True, get_depth=True, get_mask=True,
            get_action=False, get_feedback=False, get_gripper_pose=True,
            get_object_pose=True, get_proprio_obs=False, get_wrist_pose=False)

    def get_all_first_obs(self, cam: str) -> SingleCamObservation:  # type: ignore
        obs = [self.get_observation(
            traj_idx=t, img_idx=0, cam=cam, mask_type=MaskTypes.GT,
            raw_mask=True, collapse_labels=True, get_rgb=True,
            get_int=True, get_ext=True, get_depth=True, get_mask=True,
            get_action=False, get_feedback=False, get_gripper_pose=True,
            get_object_pose=True, get_proprio_obs=False, get_wrist_pose=False)
            for t in range(len(self))]

        return collate(obs)  # type: ignore

    def sample_data_point_with_object_labels(
            self, cam: str = "wrist", traj_idx: int | None =None,
            img_idx: int | None = None, get_mask: bool = True,
            mask_type=MaskTypes.GT) -> SingleCamObservation:  # type: ignore

        obs = self.sample_observation(
            traj_idx=traj_idx, img_idx=img_idx,
            cam=cam, mask_type=mask_type, raw_mask=True,
            labels=self.object_labels, collapse_labels=False, get_rgb=True,
            get_int=False, get_ext=False, get_depth=True, get_mask=get_mask,
            get_action=False, get_feedback=False, get_gripper_pose=False,
            get_object_pose=False, get_proprio_obs=False, get_wrist_pose=False)

        return obs

    def sample_data_point_with_ground_truth(
            self, cam: str = "wrist", traj_idx: int | None = None,
            img_idx: int | None = None, get_mask: bool = True,
            mask_type: MaskTypes | None = MaskTypes.GT
    ) -> SingleCamObservation:  # type: ignore

        obs = self.sample_observation(
            traj_idx=traj_idx, img_idx=img_idx,
            cam=cam, mask_type=mask_type, raw_mask=True,
            labels=self.object_labels, collapse_labels=False, get_rgb=True,
            get_int=True, get_ext=True, get_depth=True, get_mask=get_mask,
            get_action=False, get_feedback=False, get_gripper_pose=False,
            get_object_poses=True, get_proprio_obs=False, get_wrist_pose=False)

        return obs

    def process_mask(self, mask: torch.Tensor,
                     object_labels: int | Iterable[int] | None = 1,
                     raw_mask: bool = False, collapse: bool = True
                     ) -> torch.Tensor:
        """
        Process the loaded mask. Needed as masks can have different format
        depending on whether they are GT or TSDF. And masks are needed in
        different formats depending on the use case.

        Parameters
        ----------
        mask : torch.Tensor
            The raw stacked masks.
        object_labels : int/Iterable, optional
            If -1 return the union of all object masks as Boolean mask (0/1),
            if Iterable: only keep these labels - useful eg for DC sampling,
            by default 1, to trigger DeprecationWarning and force concious
            use of this function.
        raw_mask : bool, optional
            Whether to return the raw mask, by default False
        collapse : bool, optional
            When passing an Iterable to object_labels, whether to collapse the
            resulting mask to boolean values (0/1), by default True

        Returns
        -------
        torch.Tensor
            The processed masks.

        Raises
        ------
        DeprecationWarning
            When the combination of arguments does not fall into either:
            - raw_mask=True
            - object_labels=-1
            - object_labels is Iterable
            To force concious use of this function.
            Used a different mask format previously, and need to ensure that
            no undetected bugs arise from this change.
        """
        # gettig the raw mask is handy for TSNE etc
        if raw_mask or mask is None:
            pass
        elif object_labels == -1:  # take union of all object masks
            mask = torch.where(mask != 0, 1, 0).squeeze(-1).float()
        elif isinstance(object_labels, Iterable):
            if collapse:  # Binary: filter the mask and set to 0/1.
                mask = torch.where(
                    sum(mask == i for i in object_labels).bool(),  # type: ignore
                    1, 0).squeeze(-1).float()
            else:  # keep the label values, only filter
                mask = sum(torch.where(mask == i, i, 0)
                           for i in object_labels).float()  # type: ignore
        else:
            raise DeprecationWarning(
                "Unexpected combination of arguments. Still using 10er system?"
                )

        return mask

    def sample_traj_idx(self, batch_size: int | None = None) -> list[int] | int:
        if batch_size is None:
            return random.sample(range(len(self)), 1)[0]
        else:
            return random.sample(range(len(self)), batch_size)

    def sample_img_idx(self, traj_idx: int) -> int:
        return random.sample(range(self._traj_lens[traj_idx]), 1)[0]

    def get_img_idx_with_different_pose(
            self, traj_idx: int, pose_a: np.ndarray,
            dist_threshold: float = 0.2, angle_threshold: float = 20,
            num_attempts: int = 10, cam: str = "wrist") -> int | None:
        """
        Try to get an image with a different pose to the one passed in.
        This can get you a different timestep in case both cams are the same,
        or even the same one if the cams are different.
        If none can be found, return None.
        """

        cam_attributes = {cam: {'ext': self._get_cam_attr_name(cam, 'ext')}}

        counter = 0

        while counter < num_attempts:
            img_idx = self.sample_img_idx(traj_idx)
            obs = self._load_data_point(
                traj_idx, img_idx, camera_attr=cam_attributes,
                camera_order=(cam,), as_dict=True)

            pose = obs['cameras'][cam]['ext'].numpy()

            diff = compute_distance_between_poses(pose_a, pose)
            angle_diff = compute_angle_between_poses(pose_a, pose)
            if (diff > dist_threshold) or (angle_diff > angle_threshold):
                return img_idx

            counter += 1

        return None

    def get_img_idx_with_single_object_visible(
            self, traj_idx: int, labels: Iterable[int],
            mask_type: MaskTypes, num_attempts: int = 100, cam: str = "wrist"
            ) -> int | None:
        """
        Try to get an image of a MO scene where only one object is visible.
        If none can be found, return None.
        """

        raise NotImplemented("Needs updating to new data format.")

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

    def _get_bc_traj(self, traj_idx: int,
                     cams: tuple[str] = ("wrist",),
                     mask_type: MaskTypes | None = None,
                     sample_freq: int | None = None,
                     fragment_length: int = -1,
                     extra_attr: dict | None = None,
                     encoder_name: str | None = None,
                     embedding_attr: dict | None = None,
                     encoding_name: str | None = None,
                     encoding_attr: dict | None = None,
                     skip_rgb: bool = False,
                     skip_generic_attributes: bool = False
                     ) -> SceneObservation:  # type: ignore
        # define attributes to load
        generic = {} if skip_generic_attributes else copy(GENERIC_ATTRIBUTES)

        extra_attr = {} if extra_attr is None else extra_attr
        embedding_attr = {} if embedding_attr is None else embedding_attr
        encoding_attr = {} if encoding_attr is None else encoding_attr

        if not self.ground_truth_object_pose and not skip_generic_attributes:
            generic.pop("object_poses")

        sample_attr = generic | extra_attr

        camera_attr = {}

        for c in cams:  # always need mask, even for precomputed embedding
            cam_attributes = get_cam_attributes(
                c, mask_type=mask_type, skip_rgb=skip_rgb,
                shorten_cam_names=self.shorten_cam_names)
            camera_attr[c] = cam_attributes

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


        # Intrinsics are the only flat attribute (ie they don't change over
        # time). But the overhead from loading them multiple times should be
        # relatively small, so we don't bother with special handling anymore.
        # For a trajectory load flat attributes only at first timestep.
        # datapoints = [self._load_data_point(
        #     traj_idx, next(load_iter), *sample_attr, encoder_name=encoder_name,
        #     embedding_attr=embedding_attr, encoding_name=encoding_name,
        #     encoding_attr=encoding_attr)]
        # sample_attr = filter_list(sample_attr, FLAT_ATTRIBUTES)
        # datapoints.extend([self._load_data_point(
        #     traj_idx, i, *sample_attr, encoder_name=encoder_name,
        #     embedding_attr=embedding_attr, encoding_name=encoding_name,
        #     encoding_attr=encoding_attr) for i in load_iter])
        datapoints = [
            self._load_data_point(traj_idx, i, sample_attr=sample_attr,
                                  camera_order=cams,
                                  camera_attr=camera_attr,
                                  encoder_name=encoder_name,
                                  embedding_attr=embedding_attr,
                                  encoding_name=encoding_name,
                                  encoding_attr=encoding_attr)
            for i in load_iter]

        traj = torch.stack(datapoints)  # type: ignore

        # TODO: make source freq part of the metadata of the dataset
        if sample_freq is not None:
            traj = downsample_to_target_freq(traj, target_freq=sample_freq,
                                             source_freq=20)

        return traj

    def add_embedding(self, traj_idx: int , obs_idx: int, cam_name: str,
                      emb_name: str, encoding: torch.Tensor, encoder_name: str
                      ) -> None:
        traj_path = self._paths[traj_idx]
        traj_name = traj_path.parts[-1]
        embed_path = self._embeddings_dir / encoder_name / traj_name

        cam_letter = cam_name[0]
        file = (embed_path / "cam_{}_{}".format(cam_letter, emb_name) /
                str(obs_idx)).with_suffix(DATA_SUFFIX)

        file.parents[0].mkdir(parents=True, exist_ok=True)

        save_tensor(encoding, file)

    def load_embedding(self, traj_idx: int, img_idx: int, cam: str,
                       embedding_name: str, encoder_name: str) -> torch.Tensor:

        embedding_attr = {
            cam: {embedding_name: self._get_cam_attr_name(cam, embedding_name)}
        }

        obs = self._load_data_point(
            traj_idx, img_idx, encoder_name=encoder_name, camera_order=(cam,),
            embedding_attr=embedding_attr, as_single_cam=True)

        return getattr(obs, embedding_name)

    def load_embedding_batch(self, traj_idx: int, img_idx: Iterable[int],
                             cam: str, embedding_name: str, encoder_name: str
                             ) -> torch.Tensor:
        if type(traj_idx) is list or type(embedding_name) is list:
            raise NotImplementedError
        return torch.stack(tuple(
            self.load_embedding(traj_idx, i, cam, embedding_name, encoder_name)
            for i in img_idx))

    def add_encoding(self, traj_idx: int, obs_idx: int, cam_name: str,
                     attr_name: str, encoding: torch.Tensor, encoder_name: str,
                     encoding_name: str) -> None:
        traj_path = self._paths[traj_idx]
        traj_name = traj_path.parts[-1]

        attr_dir = attr_name if cam_name is None else \
            "cam_{}_{}".format(cam_name[0], attr_name)

        dir = self._encodings_dir / encoder_name / encoding_name / \
            traj_name / attr_dir

        file = (dir / str(obs_idx)).with_suffix(DATA_SUFFIX)

        file.parents[0].mkdir(parents=True, exist_ok=True)

        save_tensor(encoding, file)

    def add_encoding_fig(self, traj_idx: int, obs_idx: int, cam_name: str,
                         attr_name: str, fig: Figure, encoder_name: str,
                         encoding_name: str, bbox: Bbox,
                         channel: int | None = None) -> None:
        traj_path = self._paths[traj_idx]
        traj_name = traj_path.parts[-1]

        attr_dir = "fig_cam_{}_{}".format(cam_name[0], attr_name)

        if channel is not None:
            attr_dir += "_" + str(channel)

        dir = self._encodings_dir / encoder_name / encoding_name / \
            traj_name / attr_dir

        file = (dir / str(obs_idx)).with_suffix(IMG_SUFFIX)

        file.parents[0].mkdir(parents=True, exist_ok=True)

        fig.savefig(file, transparent=True, bbox_inches=bbox)  # type: ignore

    def add_traj_attr(self, traj_idx: int, obs_idx: int, attr_name: str,
                      value: torch.Tensor) -> None:
        traj_path = self._paths[traj_idx]
        traj_name = traj_path.parts[-1]
        traj_path = self._trajectories_dir / traj_name

        file = (traj_path / attr_name / str(obs_idx)).with_suffix(DATA_SUFFIX)

        file.parents[0].mkdir(parents=True, exist_ok=True)

        save_tensor(value, file)

    def get_scene(self, traj_idx: int, cams: list[str],
                  subsample_types: dict[str, SubSampleTypes] | None
                  ) -> dict[str, SingleCamObservation]:  # type: ignore
        """
        Convenience sampling function for TSDF fusion.
        """
        trajs = {}
        for c in cams:
            scene_traj = self._get_bc_traj(
                traj_idx, (c, ), mask_type=None, sample_freq=None,
                fragment_length=-1, skip_generic_attributes=True)

            cam_traj = scene_traj.cameras[c]  # extract SingleCamObs

            subsample = None if subsample_types is None else subsample_types[c]

            ss_cam_traj = self._subsample_trajectory(cam_traj, subsample)

            trajs[c] = ss_cam_traj

        return trajs

    def _subsample_trajectory(self, traj: SingleCamObservation,  # type: ignore
                              type: SubSampleTypes | None
                              ) -> SingleCamObservation:  # type: ignore
        if type is SubSampleTypes.POSE:
            ext = traj.extr.numpy()
            indeces = get_idx_by_pose_difference_threshold_matrix(
                ext, LINEAR_DISTANCE_THRESHOLD, ANGLE_DISTANCE_THRESHOLD)
        elif type is SubSampleTypes.LENGTH:
            raise NotImplementedError("Need to pass target_length.")
            # indeces = get_idx_by_target_len(
            #     len(traj.cam_rgb), target_length)
        elif type is SubSampleTypes.CONTENT:
            raise NotImplementedError
        elif type is SubSampleTypes.NONE or type is None:
            indeces = list(range(len(traj.extr)))
        else:
            raise ValueError("Unexpected subsample type {}".format(type))

        return downsample_tensordict_by_idx(traj, indeces)


    def add_tsdf_masks(self, traj_idx: int, cam_name: str, masks: torch.Tensor,
                       labels: list[int]) -> None:
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
    def join_datasets(cls, *sets):
        if len(sets) == 1:
            return sets[0]
        else:
            raise NotImplementedError(
                "Didn't yet implement join for new dataset format.")
