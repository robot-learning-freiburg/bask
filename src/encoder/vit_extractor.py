import torch
from loguru import logger

import encoder.representation_learner
from models.vit_extractor.extractor import ViTExtractor
from utils.constants import SampleTypes
from utils.select_gpu import device
from viz.operations import channel_back2front_batch


class VitFeatureEncoder(encoder.representation_learner.RepresentationLearner):

    # sample_type = None  # does not need pretraining
    sample_type = SampleTypes.DC  # for correspondence viz

    _config_keys = ["load_size", "layer", "facet", "bin",
                    "thresh", "vison_net", "stride"]

    def __init__(self, config=None):
        super().__init__(config=config)

        encoder_config = config["encoder"]
        self.config = config

        self.layer = encoder_config["layer"]
        self.facet = encoder_config["facet"]
        self.bin = encoder_config["bin"]

        self.extractor = ViTExtractor(
            encoder_config["vision_net"], encoder_config["stride"],
            device=device)

    def encode(self, camera_obs, full_obs=None):
        if hasattr(full_obs, "cam_rgb2") and full_obs.cam_rgb2 is not None:
            rgb = (camera_obs, full_obs.cam_rgb2)
        else:
            rgb = (camera_obs, )

        # TODO: this is for single img currently. Check if also works for batch
        enc = (self.compute_descriptor(img) for img in rgb)

        enc = torch.cat(enc, dim=-1)

        info = {}

        return enc, info

    def compute_descriptor(self, camera_obs):
        # camera_obs = camera_obs.to(device)
        prep = self.extractor.preprocess(camera_obs)

        descr = self.extractor.extract_descriptors(prep).squeeze(0)

        descr = descr.reshape(1, 32, 32, descr.shape[-1])  # TODO: make generic
        descr = channel_back2front_batch(descr)
        descr = torch.nn.functional.interpolate(
            input=descr, size=[256, 256],
            mode='bilinear', align_corners=True)

        return descr

    def generate_mask(self, camera_obs):
        prep = self.extractor.preprocess(camera_obs)

        saliency_map = self.extractor.extract_saliency_maps(prep)[0]
        fg_mask = saliency_map > self.config["encoder"]["thresh"]
        fg_mask = fg_mask.reshape(32, 32)

        return fg_mask

    def from_disk(self, *args, **kwargs):
        logger.info("This encoder needs no chekpoint loading.")
