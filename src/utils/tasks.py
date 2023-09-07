from collections import defaultdict

from env import Environment

task_horizons = {
    "CloseMicrowave": 300,  # i.e. 15 seconds
    "TakeLidOffSaucepan": 300,
    "PhoneOnBase": 600,
    "PutRubbishInBin": 600,
}

task_horizons = defaultdict(lambda: 300, task_horizons)


def get_task_horizon(config: dict) -> int | None:
    if config["env_config"]["env"] is Environment.PANDA:
        return None
    else:
        return task_horizons[config["env_config"]["task"]]
