from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger


def downsample_traj(traj, target_len):  # for tensor traj
    if len(traj) == target_len:
        return traj
    elif len(traj) < target_len:
        raise ValueError("Traj shorter than target length.")
    else:
        indeces = np.linspace(start=0, stop=len(traj) - 1, num=target_len)
        indeces = np.round(indeces).astype(int)
        return traj[indeces]


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


@dataclass
class CeilingObservation:
    camera_obs: np.ndarray
    proprio_obs: np.ndarray


@dataclass
class DCObservation:
    gripper_pose: np.ndarray
    proprio_obs: np.ndarray
    wrist_pose: np.ndarray

    cam_l_rgb: np.ndarray
    cam_r_rgb: np.ndarray
    cam_w_rgb: np.ndarray
    cam_o_rgb: np.ndarray

    cam_l_d: np.ndarray
    cam_r_d: np.ndarray
    cam_w_d: np.ndarray
    cam_o_d: np.ndarray

    # cam_l_pc: np.ndarray
    # cam_r_pc: np.ndarray
    # cam_w_pc: np.ndarray

    cam_r_ext: np.ndarray
    cam_r_int: np.ndarray
    cam_l_ext: np.ndarray
    cam_l_int: np.ndarray
    cam_w_ext: np.ndarray
    cam_w_int: np.ndarray
    cam_o_ext: np.ndarray
    cam_o_int: np.ndarray


@dataclass
class FrankaObservation:
    gripper_pose: np.ndarray
    proprio_obs: np.ndarray
    wrist_pose: np.ndarray

    cam_w_rgb: np.ndarray
    cam_o_rgb: np.ndarray

    cam_w_d: np.ndarray
    cam_o_d: np.ndarray

    cam_w_ext: np.ndarray
    cam_o_ext: np.ndarray

    cam_w_int: np.ndarray
    cam_o_int: np.ndarray

    cam_l_rgb = None
    cam_r_rgb = None
    cam_l_d = None
    cam_r_d = None
    cam_r_ext = None
    cam_r_int = None
    cam_l_ext = None
    cam_l_int = None

@dataclass
class FrankaObservationWristOnly:
    gripper_pose: np.ndarray
    proprio_obs: np.ndarray
    wrist_pose: np.ndarray

    cam_w_rgb: np.ndarray
    cam_w_d: np.ndarray
    cam_w_ext: np.ndarray
    cam_w_int: np.ndarray

    cam_o_rgb = None
    cam_o_d = None
    cam_o_ext = None
    cam_o_int = None

    cam_l_rgb = None
    cam_r_rgb = None
    cam_l_d = None
    cam_r_d = None
    cam_r_ext = None
    cam_r_int = None
    cam_l_ext = None
    cam_l_int = None

@dataclass
class GTObservation:
    gripper_pose: np.ndarray
    proprio_obs: np.ndarray
    wrist_pose: np.ndarray

    cam_l_rgb: np.ndarray
    cam_r_rgb: np.ndarray
    cam_w_rgb: np.ndarray
    cam_o_rgb: np.ndarray

    cam_l_d: np.ndarray
    cam_r_d: np.ndarray
    cam_w_d: np.ndarray
    cam_o_d: np.ndarray

    cam_l_mask: np.ndarray
    cam_r_mask: np.ndarray
    cam_w_mask: np.ndarray
    cam_o_mask: np.ndarray

    cam_r_ext: np.ndarray
    cam_r_int: np.ndarray
    cam_l_ext: np.ndarray
    cam_l_int: np.ndarray
    cam_w_ext: np.ndarray
    cam_w_int: np.ndarray
    cam_o_ext: np.ndarray
    cam_o_int: np.ndarray

    object_pose: np.ndarray
    gripper_state: np.ndarray = None

@dataclass
class MSObservation:
    ee_pose: np.ndarray
    proprio_obs: np.ndarray
    object_poses: dict

    cam_rgb: dict
    cam_depth: dict
    cam_mask: dict
    cam_ext: dict
    cam_int: dict

@dataclass
class MaskObservation:
    gripper_pose: np.ndarray
    proprio_obs: np.ndarray
    wrist_pose: np.ndarray

    cam_l_rgb: np.ndarray
    cam_r_rgb: np.ndarray
    cam_w_rgb: np.ndarray
    cam_o_rgb: np.ndarray

    cam_l_d: np.ndarray
    cam_r_d: np.ndarray
    cam_w_d: np.ndarray
    cam_o_d: np.ndarray

    cam_l_mask: np.ndarray
    cam_r_mask: np.ndarray
    cam_w_mask: np.ndarray
    cam_o_mask: np.ndarray

    cam_r_ext: np.ndarray
    cam_r_int: np.ndarray
    cam_l_ext: np.ndarray
    cam_l_int: np.ndarray
    cam_w_ext: np.ndarray
    cam_w_int: np.ndarray
    cam_o_ext: np.ndarray
    cam_o_int: np.ndarray


@dataclass
class CeilingData:
    camera_obs: np.ndarray
    proprio_obs: np.ndarray
    action: np.ndarray
    feedback: list


@dataclass
class BCData:
    cam_rgb: torch.tensor
    cam_d: torch.tensor
    cam_ext: torch.tensor
    cam_int: torch.tensor

    proprio_obs: torch.tensor
    action: torch.tensor
    feedback: torch.tensor

    mask: torch.tensor  # might need that for keypoints

    wrist_pose: torch.tensor
    object_pose: np.ndarray = None

    cam_rgb2: torch.tensor = None
    cam_d2: torch.tensor = None
    cam_ext2: torch.tensor = None
    cam_int2: torch.tensor = None

    mask2: torch.tensor = None

    def __len__(self):
        return len(self.proprio_obs)


@dataclass
class DCData:
    cam_l_rgb: torch.tensor
    cam_r_rgb: torch.tensor
    cam_w_rgb: torch.tensor
    cam_o_rgb: torch.tensor

    cam_l_d: torch.tensor
    cam_r_d: torch.tensor
    cam_w_d: torch.tensor
    cam_o_d: torch.tensor

    cam_r_int: torch.tensor
    cam_l_int: torch.tensor
    cam_w_int: torch.tensor
    cam_o_int: torch.tensor

    cam_r_ext: torch.tensor
    cam_l_ext: torch.tensor
    cam_w_ext: torch.tensor
    cam_o_ext: torch.tensor

    wrist_pose: torch.tensor

    proprio_obs: torch.tensor
    action: torch.tensor
    feedback: torch.tensor

    mask_l: torch.tensor
    mask_r: torch.tensor
    mask_w: torch.tensor
    mask_o: torch.tensor

    object_pose: torch.tensor

    def __len__(self):
        return len(self.proprio_obs)


class BCBatch:
    def __init__(self, *data, cam=None, sample_freq=None, source_freq=20):
        # NOTE: can init from BCData without cam or from DCData with cam.
        # Can also handle a list of cams.
        # Intrinsics are constant across the trajectory, thus we need to stack
        # them on the 0-th dimension, whereas all other fields are per timestep
        # and thus need to be stacked on the 1st dim so we can iterate over t.
        self.flat_fields = ["cam_r_int", "cam_l_int", "cam_w_int", "cam_o_int",
                            "cam_int", "cam_int2"]
        for field in data[0].__dataclass_fields__:
            # skip None masks
            value = getattr(data[0], field)
            if value is None or type(value) in (int, float):
                setattr(self, field, value)
            else:
                stack_dim = 0 if field in self.flat_fields else 1
                stacked = torch.stack(
                    tuple(getattr(d, field) for d in data), dim=stack_dim)
                if sample_freq is not None and field not in self.flat_fields:
                    stacked = downsample_to_target_freq(
                        stacked, target_freq=sample_freq,
                        source_freq=sample_freq)
                setattr(self, field, stacked)

        if cam is None:
            return

        aliases = {
            "overhead": {
                "cam_rgb": self.cam_o_rgb,
                "cam_d": self.cam_o_d,
                "cam_ext": self.cam_o_ext,
                "cam_int": self.cam_o_int,
                "mask": self.mask_o},
            "wrist": {
                "cam_rgb": self.cam_w_rgb,
                "cam_d": self.cam_w_d,
                "cam_ext": self.cam_w_ext,
                "cam_int": self.cam_w_int,
                "mask": self.mask_w}
             }

        if type(cam) == str:
            cam = [cam]

        self.no_cams = len(cam)

        for i, c in enumerate(cam):
            suffix = str(i+1) if i > 0 else ""
            for dst, src in aliases[c].items():
                setattr(self, dst + suffix, src)

        self.fields = ["proprio_obs", "action", "feedback", "object_pose",
                       "wrist_pose"]
        for i in range(self.no_cams):
            suffix = str(i+1) if i > 0 else ""
            for f in ["cam_rgb", "cam_d", "cam_ext", "cam_int", "mask"]:
                self.fields.append(f + suffix)

    def pin_memory(self):
        for field in self.fields:
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
        for field in self.fields:
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
                               and f not in self.flat_fields) else v
                    for f in BCData.__dataclass_fields__ if hasattr(self, f)]
            yield BCData(*vals)

    # def __iter__(self):
    #     return self.BatchIterator(self)


# @dataclass
# class DCBatch:
#     cam_l_rgb: torch.tensor
#     cam_r_rgb: torch.tensor
#     cam_w_rgb: torch.tensor
#
#     cam_l_d: torch.tensor
#     cam_r_d: torch.tensor
#     cam_w_d: torch.tensor
#
#     cam_r_int: torch.tensor
#     cam_l_int: torch.tensor
#     cam_w_int: torch.tensor
#
#     cam_r_ext: torch.tensor
#     cam_l_ext: torch.tensor
#     cam_w_ext: torch.tensor
#
#     wrist_pose: torch.tensor
#
#     proprio_obs: torch.tensor
#     action: torch.tensor
#     feedback: torch.tensor
#
#     mask_l: torch.tensor
#     mask_r: torch.tensor
#     mask_w: torch.tensor
#
#     def __init__(self, *dc_data):
#         # intrinsics are constant across the trajectory, thus we need to stack
#         # them on the 0-th dimension, whereas all other fields are per timestep
#         # and thus need to be stacked on the 1st dim so we can iterate over t.
#         self.flat_fields = ["cam_r_int", "cam_l_int", "cam_w_int", "cam_o_int"]
#         for field in DCData.__dataclass_fields__:
#             # skip None masks
#             value = getattr(dc_data[0], field)
#             if value is None or type(value) in (int, float):
#                 setattr(self, field, value)
#             else:
#                 stack_dim = 0 if field in self.flat_fields else 1
#                 stacked = torch.stack(
#                     tuple(getattr(d, field) for d in dc_data), dim=stack_dim)
#                 setattr(self, field, stacked)
#
#     def pin_memory(self):
#         for field in self.__dataclass_fields__:
#             value = getattr(self, field)
#             if torch.is_tensor(value):
#                 setattr(self, field, value.pin_memory())
#         return self
#
#     def to(self, device):
#         for field in self.__dataclass_fields__:
#             value = getattr(self, field)
#             if torch.is_tensor(value):
#                 setattr(self, field, value.to(device))
#         return self
#
#     def __len__(self):
#         return len(self.proprio_obs)
#
#     def __iter__(self):
#         # for field in self.__dataclass_fields__:
#         #     print("{}: {}".format(field, getattr(self, field).shape))
#         for idx in range(len(self)):
#             vals = [v[idx] if (torch.is_tensor(v := getattr(self, f))
#                                and f not in self.flat_fields) else v
#                     for f in self.__dataclass_fields__]
#             yield DCData(*vals)

    # def __iter__(self):
    #     return self.BatchIterator(self)

# class BatchIterator:
#     def __init__(self, batch):
#         self.__batch = batch
#         self.__index = 0
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         if self.__index >= len(self.__batch):
#             raise StopIteration
#
#         vals = [v[self.__index] if (v := getattr(self, f)) is not None else None
#                 for f in self.__dataclass_fields__]
#         data = DCData(*vals)
#         self.__index += 1
#         return data


@dataclass
class CeilingAdapter:
    cam_w_rgb: torch.tensor
    proprio_obs: torch.tensor

    def __init__(self, ceiling_obs, device):
        camera_obs_th = torch.tensor(
            ceiling_obs.camera_obs, dtype=torch.float32).unsqueeze(0)
        proprio_obs_th = torch.tensor(
            ceiling_obs.proprio_obs, dtype=torch.float32).unsqueeze(0)
        self.cam_w_rgb = camera_obs_th.to(device)
        self.proprio_obs = proprio_obs_th.to(device)


@dataclass
class DCEvalObs:
    cam_rgb: torch.tensor
    cam_d: torch.tensor
    cam_ext: torch.tensor
    cam_int: torch.tensor

    proprio_obs: torch.tensor
    wrist_pose: torch.tensor
    gripper_pose: torch.tensor = None

    cam_rgb2: torch.tensor = None
    cam_d2: torch.tensor = None
    cam_ext2: torch.tensor = None
    cam_int2: torch.tensor = None

    object_pose: torch.tensor = None
    gripper_state: torch.tensor = None

    def __init__(self, dc_obs, device, cam):
        if type(cam) == str:
            cam = [cam]

        self.no_cams = len(cam)

        for i, c in enumerate(cam):
            suffix = str(i+1) if i > 0 else ""

            if c == "wrist":
                rgb = dc_obs.cam_w_rgb
                depth = dc_obs.cam_w_d
                ext = dc_obs.cam_w_ext
                int = dc_obs.cam_w_int
            elif c == "overhead":
                rgb = dc_obs.cam_o_rgb
                depth = dc_obs.cam_o_d
                ext = dc_obs.cam_o_ext
                int = dc_obs.cam_o_int
            else:
                raise NotImplementedError(
                    "Didnt implement handling of cam {}.".format(cam))
            camera_obs_th = torch.tensor(rgb, dtype=torch.float32).unsqueeze(0)
            cam_d_th = torch.tensor(depth, dtype=torch.float32).unsqueeze(0)
            cam_ext_th = torch.tensor(ext, dtype=torch.float32).unsqueeze(0)
            cam_int_th = torch.tensor(int, dtype=torch.float32).unsqueeze(0)

            setattr(self, "cam_rgb" + suffix, camera_obs_th.to(device))
            setattr(self, "cam_d" + suffix, cam_d_th.to(device))
            setattr(self, "cam_ext" + suffix, cam_ext_th.to(device))
            setattr(self, "cam_int" + suffix, cam_int_th.to(device))

        proprio_obs_th = torch.tensor(
            dc_obs.proprio_obs, dtype=torch.float32).unsqueeze(0)
        setattr(self, "proprio_obs", proprio_obs_th.to(device))
        try:
            object_pose_th = torch.tensor(
                dc_obs.object_pose, dtype=torch.float32).unsqueeze(0).to(device)
        except AttributeError:
            object_pose_th = None
        setattr(self, "object_pose", object_pose_th)
        wrist_pose_th = torch.tensor(
                    dc_obs.wrist_pose, dtype=torch.float32).unsqueeze(0)
        setattr(self, "wrist_pose", wrist_pose_th.to(device))
        gripper_pose_th = torch.tensor(
            dc_obs.gripper_pose, dtype=torch.float32).unsqueeze(0)
        setattr(self, "gripper_pose", gripper_pose_th.to(device))

        if (gripper_state := dc_obs.gripper_state) is not None:
            gripper_state_th = torch.tensor(
                gripper_state, dtype=torch.float32).unsqueeze(0)
            setattr(self, "gripper_state", gripper_state_th.to(device))


@dataclass
class DCAdapter:
    cam_w_rgb: torch.tensor
    proprio_obs: torch.tensor
    action: torch.tensor
    feedback: torch.tensor

    cam_l_rgb = None
    cam_r_rgb = None

    cam_l_d = None
    cam_r_d = None
    cam_w_d = None

    cam_r_int = None
    cam_l_int = None
    cam_w_int = None

    cam_r_ext = None
    cam_l_ext = None
    cam_w_ext = None

    wrist_pose = None

    mask_l = None
    mask_r = None
    mask_w = None
    mask_w = None
