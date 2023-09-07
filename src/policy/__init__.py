from enum import Enum

from loguru import logger

from policy.encoder import DiskReadEncoderPolicy, EncoderPolicy
from policy.manual import ManualPolicy

try:
    from policy.motion_planner import MotionPlannerPolicy
except ImportError:
    logger.error("Can't import MotionPlannerPolicy. Is mplib installed?")
    MotionPlannerPolicy = None

from policy.random import RandomPolicy
from policy.sphere import SpherePolicy

policy_switch = {
    "encoder": EncoderPolicy,
    "random": RandomPolicy,
    "sphere": SpherePolicy,
    "manual": ManualPolicy,
}

policy_names = list(policy_switch.keys())


class PolicyEnum(Enum):
    RANDOM = "random"
    MANUAL = "manual"
    SPHERE = "sphere"
    MOTION_PLANNER = "motion_planner"



def get_policy_class(policy_name, disk_read=False):
    if disk_read:
        return DiskReadEncoderPolicy
    else:
        return policy_switch[policy_name]
