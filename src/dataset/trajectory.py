import json
import pathlib

import numpy as np
import torch

from config import (ANGLE_DISTANCE_THRESHOLD, DATA_SUFFIX, IMG_SUFFIX,
                    LINEAR_DISTANCE_THRESHOLD, METADATA_FILENAME)
from utils.data_loading import save_image, save_tensor
from utils.observation import (GENERIC_ATTRIBUTES, MaskTypes, SceneObservation,
                               downsample_traj_by_idx, get_cam_attributes,
                               get_idx_by_pose_difference_threshold,
                               get_idx_by_target_len, get_object_labels,
                               is_flat_attribute)


class Trajectory():
    def __init__(self, camera_names: tuple[str], subsample_by_difference: bool,
                 subsample_to_length: int | None) -> None:

        self.camera_names = camera_names

        self.subsample_by_difference = subsample_by_difference
        self.subsample_to_length = subsample_to_length

        if subsample_by_difference:
            assert 'wrist' in self.camera_names, "Using wrist for subsampling."

        self.reset()

    def reset(self) -> None:
        """
        Reset the trajectory to an empty list of observations.
        """
        self.observations = []

    def add(self, obs: SceneObservation) -> None:  # type: ignore
        """
        Append an observation to the trajectory.

        Parameters
        ----------
        obs : SceneObservation
            TensorClass holding the observation.
        """
        self.observations.append(obs)

    def save(self, dir: pathlib.Path) -> None:
        """
        Write the trajectory to disk.

        Parameters
        ----------
        dir : pathlib.Path
            The directory to write the trajectory to.
        """
        if self.subsample_by_difference:
            indeces = get_idx_by_pose_difference_threshold(
                [o.cameras['wrist'].extr for o in self.observations],
                LINEAR_DISTANCE_THRESHOLD, ANGLE_DISTANCE_THRESHOLD)
        elif self.subsample_to_length:
            indeces = get_idx_by_target_len(
                len(self.observations), self.subsample_to_length)
        else:
            indeces = list(range(len(self.observations)))

        object_label_gt = []

        obs_subsampled = downsample_traj_by_idx(self.observations, indeces)

        for cam in self.camera_names:
            attr_map = get_cam_attributes(cam, MaskTypes.GT)

            for attr, attr_dir_name in attr_map.items():
                attr_dir = dir / attr_dir_name

                save_func = save_image if attr == 'rgb' else save_tensor
                suffix = IMG_SUFFIX if attr == 'rgb' else DATA_SUFFIX

                if is_flat_attribute(attr):
                    value = getattr(obs_subsampled[0].cameras[cam], attr)
                    assert value is not None, "None intrinsics?"
                    file_path = attr_dir.with_suffix(suffix)
                    save_func(value, file_path)

                else:
                    attr_dir.mkdir()

                    for t in range(len(obs_subsampled)):
                        value = getattr(obs_subsampled[t].cameras[cam], attr)
                        if value is None:  # Only save non-None attributes
                            continue
                        file_path = (attr_dir / str(t)).with_suffix(suffix)
                        save_func(value, file_path)

            object_label_gt.append(get_object_labels(
                [obs.cameras[cam].mask for obs in obs_subsampled]
            ))

        for attr, attr_dir_name in GENERIC_ATTRIBUTES.items():
            attr_dir = dir / attr_dir_name
            attr_dir.mkdir()

            suffix = DATA_SUFFIX
            save_func = save_tensor

            for t in range(len(obs_subsampled)):
                value = getattr(obs_subsampled[t], attr)
                if value is None:  # Only save non-None attributes
                    continue
                file_path = (attr_dir / str(t)).with_suffix(suffix)
                save_func(value, file_path)

        object_label_gt = sorted(list(set().union(*object_label_gt)))
        object_label_gt = [] if object_label_gt == [None] else object_label_gt

        metadata = {
            "len": len(indeces),
            "object_label_gt": object_label_gt,
            }

        with open(dir / METADATA_FILENAME, 'w') as f:
            json.dump(metadata, f)

        self.reset()
