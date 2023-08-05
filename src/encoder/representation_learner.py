import torch
import torch.nn as nn

from utils.select_gpu import device


class RepresentationLearner(nn.Module):

    sample_type = None  # Needs to be set in subclass.

    def __init__(self, config=None):
        super().__init__()

    def forward(self, batch, full_obs=None, **kwargs):
        """
        For visualization, the keypoints encoder also returns and info dict,
        containing among other things the latent embedding- hence the second
        return value. If needed, other encoders can do so, too. Check the
        forward method of the keypoints encoder for reference.
        """
        return self.encode(batch, full_obs=full_obs)

    def update_params(self, batch, **kwargs):
        """
        Different methods have different sample types. To unify the API, the
        batch is unpacked inside this function.
        Additionally, some methods (like BVAE) need additional information,
        hence the **kwargs.
        """
        pass

    def encode():
        pass

    def reconstruct(self, batch):
        pass

    @classmethod
    def get_latent_dim(self, config, **kwargs):
        pass

    def from_disk(self, chekpoint_path):
        self.load_state_dict(
            torch.load(chekpoint_path, map_location=device)
        )

    def to_disk(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)
