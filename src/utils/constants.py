from enum import Enum

# TODO: there's some mixed stuff in here. Re-organize, re-name!

METADATA_FILENAME = 'metadata.json'

LINEAR_DISTANCE_THRESHOLD = 0.03  # 3cm
ANGLE_DISTANCE_THRESHOLD = 10  # 10 degree

GENERIC_ATTRIBUTES = ["action", "feedback", "ee_pose", "object_poses",
                      "object_pose", "proprio_obs"]  # , "wrist_pose"]

CAM_BASE_NAMES = ["cam_rgb", "cam_d", "cam_ext", "cam_int", "mask"]
CAM_BASE_NAMES_FULL = CAM_BASE_NAMES + ["mask"]

WRIST_CAM_ATTRIBUTES = ["cam_w_rgb", "cam_w_d", "cam_w_ext", "cam_w_int",
                        "cam_w_mask_gt", "cam_w_mask_tsdf"]

OVERH_CAM_ATTRIBUTES = ["cam_o_rgb", "cam_o_d", "cam_o_ext",  "cam_o_int",
                        "cam_o_mask_gt", "cam_o_mask_tsdf"]

LEFT_CAM_ATTRIBUTES = ["cam_l_rgb", "cam_l_d", "cam_l_ext", "cam_l_int",
                       "cam_l_mask_gt", "mask_l_tsdf"]

RIGHT_CAM_ATTRIBUTES = ["cam_r_rgb", "cam_r_d", "cam_r_ext", "cam_r_int",
                        "cam_r_mask_gt", "cam_r_mask_tsdf"]

BASE_CAMERA_ATTRIBUTES = ["cam_b_rgb", "cam_b_d",
                          "cam_b_ext", "cam_b_int",
                          "cam_b_mask_gt",
                          "cam_b_mask_tsdf"]

GT_MASK_ATTRIBUTES = ["cam_w_mask_gt", "cam_o_mask_gt", "cam_r_mask_gt",
                      "cam_l_mask_gt"]
TSDF_MASK_ATTRIBUTES = ["cam_w_mask_tsdf", "cam_o_mask_tsdf",
                        "cam_r_mask_tsdf", "cam_l_mask_tsdf"]

OPTIONAL_PRECOMP_ATTRIBUTES = {"wrist": ["cam_w_descriptor"],
                               "overhead": ["cam_o_descriptor"],
                               "left": ["cam_l_descriptor"],
                               "right": ["cam_r_descriptor"],
                               "base": ["cam_b_descriptor"],
                               "global": ["kp", "gripper_state"]}

GENERIC_PRECOMP_ATTRIBUTES = ["cam_descriptor"]

TRAJECTORY_ATTRIBUTES = GENERIC_ATTRIBUTES + WRIST_CAM_ATTRIBUTES + \
    OVERH_CAM_ATTRIBUTES + LEFT_CAM_ATTRIBUTES + RIGHT_CAM_ATTRIBUTES + \
    sum(OPTIONAL_PRECOMP_ATTRIBUTES.values(), [])

FLAT_ATTRIBUTES = ["cam_r_int", "cam_l_int", "cam_w_int", "cam_o_int",
                   "cam_b_int", "cam_int", "cam_int2", "intrinsics",
                   "intrinsics2"]

ATTRIBUTE_MAP = {"wrist": WRIST_CAM_ATTRIBUTES,
                 "overhead": OVERH_CAM_ATTRIBUTES,
                 "left": LEFT_CAM_ATTRIBUTES,
                 "right": RIGHT_CAM_ATTRIBUTES,
                 "base": BASE_CAMERA_ATTRIBUTES,
                 }


class MaskTypes(Enum):
    NONE = None
    GT = 'gt'
    TSDF = 'tsdf'


class SampleTypes(Enum):
    DC = 1
    CAM_SINGLE = 2
    CAM_PAIR = 3


def filter_list(source, filter_elements):
    return [i for i in source if i not in filter_elements]


def get_cam_attributes(cam, mask_type, depth_only=False):
    if depth_only:
        if mask_type in [MaskTypes.NONE, None]:
            return []
        # TODO: using the construction in a few places. unify!
        return ["cam_" + cam[0] + "_d"]

    cam_attributes = ATTRIBUTE_MAP[cam]
    if mask_type in [MaskTypes.NONE, MaskTypes.TSDF, None]:
        cam_attributes = filter_list(cam_attributes, GT_MASK_ATTRIBUTES)
    if mask_type in [MaskTypes.NONE, MaskTypes.GT, None]:
        cam_attributes = filter_list(cam_attributes, TSDF_MASK_ATTRIBUTES)

    return cam_attributes


def get_name_translation(cam):
    return {k: v for k, v in zip(
        ATTRIBUTE_MAP[cam] + OPTIONAL_PRECOMP_ATTRIBUTES[cam],
        CAM_BASE_NAMES_FULL + GENERIC_PRECOMP_ATTRIBUTES)}


# TODO: this is ugly and inefficient
def translate_names(data_dict, cam):
    name_map = get_name_translation(cam)
    return {name_map[k] if k in name_map.keys()
            else k: v for k, v in data_dict.items()}
