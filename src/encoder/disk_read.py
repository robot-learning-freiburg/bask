import typing

import numpy as np
import torch
import torchvision
from loguru import logger

import encoder.representation_learner
from encoder.keypoints import KeypointsPredictor, PriorTypes
from encoder.keypoints_gt import GTKeypointsPredictor
from particle_filter.filter import ParticleFilter
from utils.select_gpu import device


class DiskReadEncoder(encoder.representation_learner.RepresentationLearner):
    def __init__(self, config=None, encoder_class=None,
                 attr_name='cam_descriptor'):
        super().__init__(config=config)

        self.sample_type = encoder_class.sample_type

        self.config = config
        self.encoder_config = config["encoder"]

        self.EncoderClass = encoder_class
        self.keypoints_emb = self.EncoderClass is KeypointsPredictor and \
            attr_name == 'cam_descriptor'

        if self.keypoints_emb:  # Ugly hotfix. TODO: fix properly
            attr_name = 'descriptor'

        self.embedding_name = attr_name

        self.only_use_first_emb = self.EncoderClass == GTKeypointsPredictor

        if self.only_use_first_emb:
            logger.info("Only using encoding of first camera. This is only "
                        "meant for the GT KP model.")

        if self.encoder_config.get('motion_model_noisy'):
            self.motion_blur = torchvision.transforms.GaussianBlur(
                kernel_size=self.encoder_config["motion_model_kernel"],
                sigma=self.encoder_config["motion_model_sigma"])
        else:
            self.motion_blur = None

        if self.keypoints_emb:
            if self.encoder_config.get("prior_type", PriorTypes.NONE) is \
                    PriorTypes.PARTICLE_FILTER:
                self.particle_filter = ParticleFilter(config)
                self.particle_filter_viz = None
            else:
                self.particle_filter = None
                self.particle_filter_viz = None

        self.add_gaussian_noise = self.config.get("training", {}).get(
            "add_gaussian_noise", False)

        self.reset_traj()

    def reset_traj(self):
        if self.keypoints_emb:
            return KeypointsPredictor.reset_traj(self)
        else:
            pass

    def forward(self, batch, full_obs=None):
        """
        Custom forward method to also return the latent embedding for viz.
        """
        return self.encode(batch, full_obs)

    def initialize_parameters_via_dataset(self, replay_memory):
        if self.keypoints_emb:
            # TODO: make these confable
            traj_idx = 1  # 0
            img_idx = 40  # 1  # 69
            cam = "wrist"  # "overhead"

            rgb, depth, mask = \
                replay_memory.sample_data_point_with_object_labels(
                  cam=cam, img_idx=img_idx, traj_idx=traj_idx)

            descriptor = replay_memory.load_embedding(
                img_idx=img_idx, traj_idx=traj_idx, cam=cam,
                embedding_name=self.embedding_name)

            # Usually, encoder upsamples back to full resolution. For pre-
            # computed that would waste disk-space, so upsample here.
            img_h, img_w = rgb.shape[-2:]
            if descriptor.shape[-1] != rgb.shape[-1]:
                logger.info("Upsampling descriptor to image size.")
                descriptor = torch.nn.functional.interpolate(
                    descriptor.unsqueeze(0), size=tuple((img_h, img_w)),
                    mode='bilinear', align_corners=True)

            object_labels = replay_memory.get_object_labels()
            n_keypoints_total = self.encoder_config["keypoints"]["n_sample"]
            manual_kp_selection = self.config.get("pretrain", None)[
                "training_config"].get('manual_kp_selection')

            if manual_kp_selection:
                traj_idx = replay_memory.scene_data.sample_traj_idx(1)
                preview_frames = replay_memory.sample_bc(
                    None, cam=(cam,), idx=traj_idx, skip_rgb=False
                    ).cam_rgb.squeeze(1)
                indeces = np.linspace(
                    start=0, stop=preview_frames.shape[0] - 1, num=20)
                indeces = np.round(indeces).astype(int)
                preview_frames = preview_frames.index_select(
                    dim=0, index=torch.tensor(indeces))

                preview_descr = replay_memory.load_embedding_batch(
                    traj_idx=traj_idx[0], img_idx=indeces, cam=cam,
                    embedding_name=self.embedding_name)

                if preview_descr.shape[-1] != rgb.shape[-1]:
                    preview_descr = torch.nn.functional.interpolate(
                        preview_descr, size=tuple((img_h, img_w)),
                        mode='bilinear', align_corners=True)
            else:
                preview_frames = None
                preview_descr = None

            self.ref_pixels_uv, self._reference_descriptor_vec = \
                KeypointsPredictor._select_reference_descriptors(
                    rgb, descriptor, mask, object_labels, n_keypoints_total,
                    manual_kp_selection, preview_frames, preview_descr)

            self.EncoderClass.config = self.encoder_config
            self.EncoderClass.image_height, self.EncoderClass.image_width = \
                self.encoder_config.get("image_size", (256, 256))
            self.EncoderClass.get_dc_dim()
            self.EncoderClass.setup_pixel_maps()
        else:
            # init_encoder = getattr(self.EncoderClass,
            #                        "initialize_parameters_via_dataset", None)
            # if callable(init_encoder):
            #     init_encoder(replay_memory)
            # else:
            #     logger.info("This policy does not use dataset initialization.")
            # NOTE: only kp-style encoders use this function. For gt-kp don't
            # need to call as ref pos are selected during encoding. So skip.
            logger.info("This policy does not use dataset initialization.")

    def encode(self, camera_obs, full_obs=None):
        if self.keypoints_emb:
            n_cams = 1 if full_obs.cam_rgb2 is None else 2

            if hasattr(full_obs, "cam_rgb2") and full_obs.cam_rgb2 is not None:
                rgb = (camera_obs, full_obs.cam_rgb2)
                depth = (full_obs.cam_d, full_obs.cam_d2)
                extr = (full_obs.cam_ext, full_obs.cam_ext2)
                intr = (full_obs.cam_int, full_obs.cam_int2)
                prior = (None, None)
                descriptor = (full_obs.cam_descriptor,
                              full_obs.cam_descriptor2)
            else:
                rgb = (camera_obs, )
                depth = (full_obs.cam_d, )
                extr = (full_obs.cam_ext, )
                intr = (full_obs.cam_int, )
                prior = (None, )
                descriptor = (full_obs.cam_descriptor, )

            if self.encoder_config.get("prior_type", PriorTypes.NONE) is \
                    PriorTypes.PARTICLE_FILTER:
                self.particle_filter.update(
                    rgb, depth, extr, intr, descriptor=descriptor,
                    ref_descriptor=self._reference_descriptor_vec,
                    gripper_pose=full_obs.gripper_pose)
                kp, info = self.particle_filter.estimate_state(extr, intr,
                                                               depth)

                if self.particle_filter_viz is not None:
                    self.particle_filter_viz.update(info['particles'],
                                                    info['weights'],
                                                    info['prediction'],
                                                    info['particles_2d'],
                                                    info['keypoints_2d'],
                                                    tuple(r.cpu() for r in rgb)
                                                    )
            else:
                prior = tuple(self.get_prior(
                    prior_sm=None if (l := self.last_post) is None else l[i],
                    prior_pos=None if (l := self.last_kp_raw_2d) is None
                    else l[i])
                    for i in range(n_cams))

                motion_model = tuple(
                    self.get_motion_model(
                        depth[i], intr[i], extr[i],
                        None if (l := self.last_d) is None else l[i],
                        None if (l := self.last_int) is None else l[i],
                        None if (l := self.last_ext) is None else l[i])
                    for i in range(n_cams))

                prior = tuple(self.apply_motion_model(
                    prior[i], motion_model[i]) for i in range(n_cams))

                overshadow = self.encoder_config.get("overshadow_keypoints")
                threshold = self.encoder_config.get("threshold_keypoint_dist")

                taper = self.encoder_config.get("taper_sm") or 1
                use_spatial_expectation = \
                    self.encoder_config["use_spatial_expectation"]
                projection = self.encoder_config["projection"]

                kp, info = KeypointsPredictor._encode(
                    rgb, depth, extr, intr, prior, descriptor=descriptor,
                    ref_descriptor=self._reference_descriptor_vec,
                    use_spatial_expectation=use_spatial_expectation,
                    taper=taper,
                    projection=projection, overshadow=overshadow,
                    threshold=threshold)

            self.last_kp_raw_2d = info["kp_raw_2d"]
            self.last_post = info["post"]
            self.last_d = depth
            self.last_int = intr
            self.last_ext = extr

            return kp, info
        else:
            # attr_name = "cam_" + self.embedding_name
            attr_name = self.embedding_name

            if hasattr(full_obs, attr_name + "2") and getattr(
                    full_obs, attr_name + "2") is not None:
                descriptor = (getattr(full_obs,  attr_name),
                              getattr(full_obs, attr_name + "2"))
            else:
                descriptor = (getattr(full_obs,  attr_name), )

            if self.only_use_first_emb:
                descriptor = descriptor[0]
            else:
                descriptor = torch.cat(descriptor, dim=-1)

            if self.add_gaussian_noise:
                descriptor = add_gaussian_noise(descriptor, 0.1, skip_z=False)
            return descriptor, {}

    def get_prior(self, prior_sm=None, prior_pos=None):
        if self.keypoints_emb:
            prior_type = self.encoder_config.get("prior_type", PriorTypes.NONE)

            if prior_type in [PriorTypes.POS_GAUSS, PriorTypes.POS_UNI]:
                try:
                    prior_var = self.encoder_config["prior_var"]
                except ValueError:
                    logger.error(
                        "When specifying positinal prior (gaus or uniform), "
                        "need to specify prior_var as well.")
            else:
                prior_var = None

            return KeypointsPredictor._get_prior(
                prior_sm, prior_pos, prior_type, prior_var=prior_var)
        else:
            raise NotImplementedError

    def get_motion_model(self, depth_a, intr_a, extr_a, depth_b, intr_b,
                         extr_b):
        if self.keypoints_emb:
            if depth_b is None or (self.encoder_config.get(
                "prior_type", PriorTypes.NONE) is PriorTypes.NONE) or \
                    not self.encoder_config.get("use_motion_model"):
                return None
            else:
                return KeypointsPredictor._get_motion_model(
                    depth_a, intr_a, extr_a, depth_b, intr_b, extr_b)
        else:
            raise NotImplementedError

    def apply_motion_model(self, *args, **kwargs):
        if self.keypoints_emb:
            return KeypointsPredictor.apply_motion_model(self, *args, **kwargs)
        else:
            raise NotImplementedError

    def _apply_motion_model(self, *args, **kwargs):
        if self.keypoints_emb:
            return KeypointsPredictor._apply_motion_model(*args, **kwargs)
        else:
            raise NotImplementedError

    def reconstruct_ref_descriptor_from_gt(self, replay_memory, ref_pixels_uv,
                                           ref_object_pose):
        assert self.keypoints_emb

        traj_idx = 1  # 0
        img_idx = 40  # 10
        cam = "wrist"

        rgb, depth, mask, intr, ext, object_pose = \
            replay_memory.sample_data_point_with_ground_truth(
                cam=cam, img_idx=img_idx, traj_idx=traj_idx)

        # ensure it's the same observation
        assert torch.equal(ref_object_pose, object_pose)

        descriptor = replay_memory.load_embedding(
            img_idx=img_idx, traj_idx=traj_idx, cam=cam,
            embedding_name=self.embedding_name)

        # Usually, encoder upsamples back to full resolution. For pre-
        # computed that would waste disk-space, so upsample here.
        img_h, img_w = rgb.shape[-2:]
        if descriptor.shape[-1] != rgb.shape[-1]:
            logger.info("Upsampling descriptor to image size.")
            descriptor = torch.nn.functional.interpolate(
                descriptor.unsqueeze(0), size=tuple((img_h, img_w)),
                mode='bilinear', align_corners=True)

        ref_pixels_flattened = \
            ref_pixels_uv[1] * mask.shape[1] + ref_pixels_uv[0]

        D = descriptor.shape[1]
        WxH = descriptor.shape[2] * descriptor.shape[3]

        # now view as D, H*W
        descriptor_image_tensor = descriptor.squeeze(
            0).contiguous().view(D, WxH)

        # now switch back to H*W, D
        descriptor_image_tensor = descriptor_image_tensor.permute(1, 0)

        # self.ref_descriptor_vec is Nref, D
        reference_descriptor_vec = torch.index_select(
            descriptor_image_tensor, 0, ref_pixels_flattened)

        return reference_descriptor_vec.to(device)


def add_gaussian_noise(coordinates: torch.tensor,
                       noise_scale: typing.Union[float, torch.tensor],
                       skip_z: bool = False):

    stacked_coords = torch.stack(torch.chunk(coordinates, 3, dim=-1), dim=-1)

    gauss = torch.distributions.normal.Normal(0, noise_scale)
    noise = gauss.sample(stacked_coords.shape).to(device)

    if skip_z:
        assert noise.shape[-1] == 3
        noise[..., 2] = 0

    augmented = stacked_coords + noise

    return torch.cat((augmented[..., 0], augmented[..., 1],augmented[..., 2]),
                     dim=-1)
