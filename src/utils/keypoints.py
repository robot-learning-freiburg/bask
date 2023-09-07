import torch


def get_keypoint_distance(set1: torch.Tensor, set2: torch.Tensor
                          ) -> torch.Tensor:
    # pairwise distance needs both inputs to have shape (N, D), so flatten
    B, N_kp, d_kp = set1.shape
    set1 = set1.reshape((B * N_kp, d_kp))
    set2 = set2.reshape((B * N_kp, d_kp))

    distance = torch.nn.functional.pairwise_distance(set1, set2)
    distance = distance.reshape((B, N_kp))

    return distance
