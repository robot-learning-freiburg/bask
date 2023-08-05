from rlbench.tasks import (ArmScan, CloseMicrowave, PhoneBase, PhoneOnBase,
                           PhoneReceiver, PutRubbishInBin, TakeLidOffSaucepan)

task_switch = {
    "CloseMicrowave": CloseMicrowave,
    "TakeLidOffSaucepan": TakeLidOffSaucepan,

    "PhoneOnBase": PhoneOnBase,
    "PutRubbishInBin": PutRubbishInBin,
    "ArmScan": ArmScan,

    "PhoneBaseOnly": PhoneBase,
    "PhoneReceiverOnly": PhoneReceiver,
}

tasks = list(task_switch.keys())
