import numpy as np
import wandb
from loguru import logger


class KP_Dist_Tracker():
    def __init__(self) -> None:
        self.kp.kp_distances = {
            "gt_distance": [],
            "gt_distance2": [],
            "step_distance": [],
            "step_distance2": []
        }

    def add_episode(self):
        for k, l in self.kp_distances.items():
            l.append([])

    def add_step(self, info):
        for k, l in self.kp_distances.items():
            if (v := info.get(k, None)) is not None:
                l[-1].append(v)
            else:
                logger.warning(
                    "Key {} not found in info dict of kp_dist_tracker", k)

    def process_episode(self):
        for k, l in self.kp_distances.items():
            self.kp_distances[k][-1] = np.asarray(l[-1])

    def aggregate_and_save_episodes(self, save_path):
        self.aggregate_episodes()
        self.save_results(save_path)

    def aggregate_episodes(self):
        aggregated = {}

        for k, l in self.kp_distances.items():
            # stack and aggr over trajectories
            stacked = np.swapaxes(np.asarray(l), 0, 1)
            s = stacked.shape
            aggr = np.reshape(stacked, (s[0], s[1]*s[2]))

            aggregated[k] = np.asarray(
                [np.histogram(t) for t in aggr], dtype=object)

        return aggregated

    def save_results(self, save_path):
        save_path.mkdir(exist_ok=True)
        kp_dist_file_name = save_path / "kp_distances.npy"
        logger.info("Saving kp distances to file {}", kp_dist_file_name)
        np.save(kp_dist_file_name, self.kp_distances)
        artifact = wandb.Artifact('kp_distances', type='info-dict')
        artifact.add_file(kp_dist_file_name)
        wandb.run.log_artifact(artifact)
