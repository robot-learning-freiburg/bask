from policy.encoder import (EncoderPolicy, KPCompleteProcomputeEncoderPolicy,
                            PreEmbeddedEncoderPolicy)
from policy.manual import ManualPolicy
from policy.random import RandomPolicy
from policy.sphere import SpherePolicy

policy_switch = {
    "encoder": EncoderPolicy,
    "random": RandomPolicy,
    "sphere": SpherePolicy,
    "manual": ManualPolicy,
}


def get_policy_class(policy_name, pre_embd=False, pre_enc=False):
    if pre_enc:
        return KPCompleteProcomputeEncoderPolicy
    elif pre_embd:
        return PreEmbeddedEncoderPolicy
    else:
        return policy_switch[policy_name]
