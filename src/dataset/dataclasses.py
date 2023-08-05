import dataclasses
from functools import partial

import numpy as np
import torch
from loguru import logger
from torch.nn.utils.rnn import pad_sequence

from utils.constants import (CAM_BASE_NAMES, FLAT_ATTRIBUTES,
                             GENERIC_ATTRIBUTES, OPTIONAL_PRECOMP_ATTRIBUTES,
                             TRAJECTORY_ATTRIBUTES, get_name_translation)


def downsample_to_target_freq(traj, target_freq=None, source_freq=None):
    source_freq = 20
    target_len = int(len(traj) * target_freq/source_freq)
    return downsample_traj_torch(traj, target_len)


def downsample_traj_torch(traj, target_len):
    if len(traj) == target_len:
        return traj
    elif len(traj) < target_len:
        raise ValueError("Traj shorter than target length.")
    else:
        indeces = np.linspace(start=0, stop=len(traj) - 1, num=target_len)
        indeces = np.round(indeces).astype(int)
    return traj.index_select(dim=0, index=torch.tensor(indeces))


def collate_dataclasses(datapoints, batch_class=None, pad=False, flat=False):
    # flat: for non-timeseries data -> need no time dim
    collated = {}
    for field in datapoints[0].__dataclass_fields__:
        # skip None masks
        value = getattr(datapoints[0], field)
        if value is None or type(value) in (int, float):
            collated[field] = value
        else:
            if pad and field not in FLAT_ATTRIBUTES:
                stacked = pad_sequence(tuple(getattr(d, field)
                                       for d in datapoints))
            else:
                stack_dim = 0 if field in FLAT_ATTRIBUTES or flat else 1
                vals = tuple(getattr(d, field) for d in datapoints)
                if type(value) is dict:
                    stacked = {
                        k: torch.stack(tuple(v[k] for v in vals),
                                       dim=stack_dim)
                        for k in value.keys()
                    }
                else:
                    stacked = torch.stack(vals, dim=stack_dim)
            collated[field] = stacked

    return batch_class(**collated)


def collate_custom_classes(datapoints, batch_class=None, pad=False):
    collated = {}
    for field in vars(datapoints[0]):
        # skip None masks
        value = getattr(datapoints[0], field)
        if value is None or type(value) in (int, float):
            collated[field] = value
        else:
            if pad and field not in FLAT_ATTRIBUTES:
                stacked = pad_sequence(tuple(getattr(d, field)
                                       for d in datapoints))
            else:
                stack_dim = 0 if field in FLAT_ATTRIBUTES else 1
                vals = tuple(getattr(d, field) for d in datapoints)
                if type(value) is dict:
                    assert field == "object_poses"
                    logger.warning("HACK in accesing object_poses.")
                    vals = tuple(v["obj_pose"] for v in vals)
                stacked = torch.stack(vals, dim=stack_dim)
            collated[field] = stacked

    return batch_class(**collated)


SceneDatapoint = dataclasses.make_dataclass(
    'SceneDatapoint',
    [(i, torch.tensor, dataclasses.field(default=None))
     for i in TRAJECTORY_ATTRIBUTES],
    frozen=True)

SceneBatch = dataclasses.make_dataclass(
    'SceneBatch',
    [(i, torch.tensor, dataclasses.field(default=None))
     for i in TRAJECTORY_ATTRIBUTES],
    frozen=True)

collate_scenepoints = partial(collate_dataclasses, batch_class=SceneBatch)


# BCBatch = dataclasses.make_dataclass(
#     'BCBatch',
#     [(i, torch.tensor, dataclasses.field(default=None))
#      for i in GENERIC_ATTRIBUTES + CAM_BASE_NAMES],
#     frozen=True)

# Need to create dataclass explicitely like this to avoid pickle fail.
@dataclasses.dataclass
class BCBatch:
    action: torch.tensor = None
    feedback: torch.tensor = None
    gripper_pose: torch.tensor = None
    object_pose: torch.tensor = None
    proprio_obs: torch.tensor = None
    wrist_pose: torch.tensor = None

    cam_rgb: torch.tensor = None
    cam_rgb2: torch.tensor = None

    cam_d: torch.tensor = None
    cam_d2: torch.tensor = None

    mask: torch.tensor = None
    mask2: torch.tensor = None

    cam_ext: torch.tensor = None
    cam_ext2: torch.tensor = None

    cam_int: torch.tensor = None
    cam_int2: torch.tensor = None

    cam_descriptor: torch.tensor = None
    cam_descriptor2: torch.tensor = None

    kp: torch.tensor = None
    gripper_state: torch.tensor = None

    def pin_memory(self):
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if torch.is_tensor(value):
                try:
                    setattr(self, field, value.pin_memory())
                except RuntimeError:
                    logger.warning("Could not pin field {}", field)
        return self

    def to(self, device, skip=None):
        if skip is None:
            skip = tuple()
        for field in self.__dataclass_fields__:
            try:
                if field not in skip:
                    value = getattr(self, field)
                    if torch.is_tensor(value):
                        setattr(self, field, value.to(device))
                    # logger.info("Moved {} to {}.", field, device)
            except RuntimeError as e:
                logger.error("Can't move field {} to device.", field)
                raise e
        return self

    def __len__(self):
        return len(self.proprio_obs)

    def __iter__(self):
        for idx in range(len(self)):
            vals = [v[idx] if (torch.is_tensor(v := getattr(self, f))
                               and f not in FLAT_ATTRIBUTES) else v
                    for f in BCData.__dataclass_fields__ if hasattr(self, f)]
            yield BCData(*vals)


@dataclasses.dataclass
class BCData:
    action: torch.tensor = None
    feedback: torch.tensor = None
    gripper_pose: torch.tensor = None
    object_pose: torch.tensor = None
    proprio_obs: torch.tensor = None
    wrist_pose: torch.tensor = None

    cam_rgb: torch.tensor = None
    cam_rgb2: torch.tensor = None

    cam_d: torch.tensor = None
    cam_d2: torch.tensor = None

    mask: torch.tensor = None
    mask2: torch.tensor = None

    cam_ext: torch.tensor = None
    cam_ext2: torch.tensor = None

    cam_int: torch.tensor = None
    cam_int2: torch.tensor = None

    cam_descriptor: torch.tensor = None
    cam_descriptor2: torch.tensor = None

    kp: torch.tensor = None
    gripper_state: torch.tensor = None

    def __len__(self):
        return len(self.proprio_obs)


collate_bc = partial(collate_custom_classes, batch_class=BCBatch)


class BCTraj:
    # TODO use mask_type!
    def __init__(self, *scene_datapoints, cams=None,
                 sample_freq=None, source_freq=20, mask_type=None):

        def _set_field(source, target):
            # skip fields that have None values, ie. were not loaded.
            value = getattr(scene_datapoints[0], source)
            # mask_gt, mask_tsdf both map to mask, so don't override
            if value is None and hasattr(self, target):
                pass
            elif value is None or source in FLAT_ATTRIBUTES:
                setattr(self, target, value)
            else:
                values = tuple(getattr(d, source) for d in scene_datapoints)
                if type(value) == dict:
                    stacked = {
                        k: torch.stack([v[k] for v in values], dim=0)
                        for k in value.keys()}
                    if sample_freq is not None and source not in FLAT_ATTRIBUTES:
                        stacked = {
                            k: downsample_to_target_freq(
                            v, target_freq=sample_freq,
                            source_freq=source_freq)
                            for k, v in stacked.items()}
                else:
                    stacked = torch.stack(values, dim=0)
                    if sample_freq is not None and source not in FLAT_ATTRIBUTES:
                        stacked = downsample_to_target_freq(
                            stacked, target_freq=sample_freq,
                            source_freq=source_freq)
                setattr(self, target, stacked)

        for field in GENERIC_ATTRIBUTES + OPTIONAL_PRECOMP_ATTRIBUTES["global"]:
            _set_field(field, field)

        for i, cam in enumerate(cams):
            suffix = str(i+1) if i > 0 else ""
            # TODO: get _desriptor attribute here as well.  Cannot just put it
            # into cam_attributes.
            for src, dst in get_name_translation(cam).items():
                _set_field(src, dst + suffix)


# @dataclass(frozen=True)
# class SceneDatapoint:
#     action: torch.tensor = None
#     feedback: torch.tensor = None
#     gripper_pose: torch.tensor = None
#     object_pose: torch.tensor = None
#     proprio_obs: torch.tensor = None
#     wrist_pose: torch.tensor = None

#     cam_l_rgb: torch.tensor = None
#     cam_r_rgb: torch.tensor = None
#     cam_w_rgb: torch.tensor = None
#     cam_o_rgb: torch.tensor = None

#     cam_l_d: torch.tensor = None
#     cam_r_d: torch.tensor = None
#     cam_w_d: torch.tensor = None
#     cam_o_d: torch.tensor = None

#     cam_l_mask_gt: torch.tensor = None
#     cam_r_mask_gt: torch.tensor = None
#     cam_w_mask_gt: torch.tensor = None
#     cam_o_mask_gt: torch.tensor = None

#     cam_l_mask_tsdf: torch.tensor = None
#     cam_r_mask_tsdf: torch.tensor = None
#     cam_w_mask_tsdf: torch.tensor = None
#     cam_o_mask_tsdf: torch.tensor = None

#     cam_r_ext: torch.tensor = None
#     cam_r_int: torch.tensor = None
#     cam_l_ext: torch.tensor = None
#     cam_l_int: torch.tensor = None
#     cam_w_ext: torch.tensor = None
#     cam_w_int: torch.tensor = None
#     cam_o_ext: torch.tensor = None
#     cam_o_int: torch.tensor = None


SingleCamObservation = dataclasses.make_dataclass(
    'SingleCamObservation',
    [(i, torch.tensor, dataclasses.field(default=None))
     for i in (CAM_BASE_NAMES + GENERIC_ATTRIBUTES)],
    frozen=False,
    namespace={'to': BCBatch.to})

SingleCamBatch = dataclasses.make_dataclass(
    'SingleCamObservation',
    [(i, torch.tensor, dataclasses.field(default=None))
     for i in (CAM_BASE_NAMES + GENERIC_ATTRIBUTES)],
    frozen=False,
    namespace={'to': BCBatch.to})

# TODO: can use the namespace kwarg to also generate BCBatch automatically.

collate_singelcam_obs = partial(
    collate_dataclasses, batch_class=SingleCamObservation, flat=True)
