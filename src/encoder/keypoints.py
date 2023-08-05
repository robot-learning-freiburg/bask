from enum import Enum

import numpy as np
# import math
import torch
import torchvision
import tqdm
from loguru import logger

import dense_correspondence.loss.loss_composer as dc_loss_composer
import encoder.representation_learner
import models.keypoints.keypoints as keypoints
import models.keypoints.model_based_vision as model_based_vision
import wandb
from dense_correspondence.correspondence_finder import \
    random_sample_from_masked_image_torch
from dense_correspondence.loss.pixelwise_contrastive_loss import \
    PixelwiseContrastiveLoss
from particle_filter.filter import ParticleFilter
from utils.constants import SampleTypes
from utils.select_gpu import device
# from utils.debug import nan_hook, summarize_tensor
from utils.torch import append_depth_to_uv  # batched_project_onto_cam,
from utils.torch import (batched_pinhole_projection_image_to_world_coordinates,
                         batched_project_onto_cam, hard_pixels_to_3D_world,
                         heatmap_from_pos)
# from viz.image_series import vis_series
from viz.image_single import image_with_points_overlay_uv_list
from viz.operations import channel_front2back, channel_front2back_batch
# from viz.particle_filter import ParticleFilterViz
from viz.surface import depth_map_with_points_overlay_uv_list

KeypointsTypes = keypoints.KeypointsTypes


class PriorTypes(Enum):
    NONE = 1
    POS_GAUSS = 2
    POS_UNI = 3
    DISCRETE_FILTER = 4
    PARTICLE_FILTER = 5


class ProjectionTypes(Enum):
    NONE = 1
    UVD = 2
    LOCAL_SOFT = 3
    GLOBAL_SOFT = 4
    LOCAL_HARD = 5
    GLOBAL_HARD = 6
    EGO = 7  # for particle filter only
    EGO_STEREO = 8


class LRScheduleTypes(Enum):
    NONE = 1
    STEP = 2
    COSINE = 3
    COSINE_WR = 4


class KeypointsPredictor(encoder.representation_learner.RepresentationLearner):

    sample_type = SampleTypes.DC

    def __init__(self, config=None):  # , image_size=None):
        super().__init__(config=config)

        optional_keys = ["overshadow_keypoints", "threshold_keypoint_dist",
                         "prior_type", "taper_sm",
                         "use_motion_model", "motion_model_noisy"]

        optional_keys_training = ["manual_kp_selection", "lr_schedule"]

        encoder_config = config["encoder"]
        self.config = encoder_config

        for k in optional_keys:
            if k not in encoder_config:
                logger.warning(
                    "Key {} not in encoder config. Assuming False.", k)

        # if image_size is None:
        #     image_size = (256, 256)
        image_size = config["obs_config"]["image_dim"]

        KeypointsPredictor.image_height, KeypointsPredictor.image_width = \
            image_size

        self.get_dc_dim()

        self.descriptor_dimension = encoder_config["descriptor_dim"]
        self.keypoint_dimension = 2 if encoder_config["projection"] is \
            ProjectionTypes.NONE else 3
        self.use_spatial_expectation = encoder_config["use_spatial_expectation"]  # noqa 402

        self.model = keypoints.KeypointsModel(encoder_config)

        self.debug_kp_selection = config["training"]["debug"]

        if encoder_config.get('motion_model_noisy'):
            self.motion_blur = torchvision.transforms.GaussianBlur(
                kernel_size=encoder_config["motion_model_kernel"],
                sigma=encoder_config["motion_model_sigma"])
        else:
            self.motion_blur = None

        pretrain_config = config.get("pretrain", {}).get("training_config", {})

        if pretrain_config is not None:
            logger.info("Got pretrain config. Setting up DC-learning.")
            self.pretrain_config = pretrain_config
            for k in optional_keys_training:
                if k not in pretrain_config:
                    logger.warning(
                        "Key {} not in pretrain config. Assuming False.", k)

            self.loss = PixelwiseContrastiveLoss(
                image_shape=(self.image_height, self.image_width),
                config=pretrain_config["loss_function"])

            self.optimizer = torch.optim.Adam(
                self.model.parameters(), self.pretrain_config["lr"],
                weight_decay=self.pretrain_config['weight_decay'])

            schedule = pretrain_config.get("lr_schedule")
            if schedule is None or schedule is LRScheduleTypes.NONE:
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lambda _: 1)
            elif schedule is LRScheduleTypes.STEP:
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    pretrain_config["steps_between_learning_rate_decay"],
                    gamma=pretrain_config["learning_rate_decay"])
            elif schedule is LRScheduleTypes.COSINE:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=pretrain_config.get("T_max"))
            elif schedule is LRScheduleTypes.COSINE_WR:
                self.scheduler = \
                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                       self.optimizer,
                       T_0=pretrain_config.get("T_0"),
                       T_mult=pretrain_config.get("T_mult"))
            else:
                raise ValueError("Unexpected schedule {}".format(schedule))

        else:
            logger.info(
                "Got no pretrain config. Encoder is in policy-learning mode.")

        n_keypoints = self.get_no_keypoints()

        # Register keypoints as buffers, such that they will be saved with the
        # module. Then, we can use the same reference vectors at inference.
        self.register_buffer('ref_pixels_uv',
                             torch.Tensor(2, n_keypoints))
        self.register_buffer('_reference_descriptor_vec',
                             torch.Tensor(n_keypoints,
                                          self.descriptor_dimension))

        self.register_buffer('norm_mean', torch.Tensor(3))
        self.register_buffer('norm_std', torch.Tensor(3))

        self.setup_pixel_maps()

        if self.config.get("prior_type", PriorTypes.NONE) is \
                PriorTypes.PARTICLE_FILTER:
            self.particle_filter = ParticleFilter(config)
            self.particle_filter_viz = None
            # if self.debug_kp_selection:
            #     self.particle_filter_viz = ParticleFilterViz()
            #     self.particle_filter_viz.run()
        else:
            self.particle_filter = None
            self.particle_filter_viz = None

        self.reset_traj()

    def reset_traj(self):
        self.last_kp_raw_2d = None

        self.last_post = None

        self.last_d = None
        self.last_int = None
        self.last_ext = None

        if self.particle_filter is not None:
            self.particle_filter.reset()

        if self.particle_filter_viz is not None:
            self.particle_filter_viz.reset_traj()

    def get_no_keypoints(self):
        keypoint_type = self.config["keypoints"]["type"]

        if keypoint_type in [KeypointsTypes.SD, KeypointsTypes.ODS,
                             KeypointsTypes.WDS]:
            n_keypoints = self.config["keypoints"]["n_sample"]
        else:
            n_keypoints = self.config["keypoints"]["n_keep"]

        return n_keypoints

    @classmethod
    def get_dc_dim(cls):
        dim_mapping = {128: 32,
                       256: 32,
                       480: 60,
                       640: 80}
        cls.dc_height = dim_mapping[cls.image_height]
        cls.dc_width = dim_mapping[cls.image_width]

    def update_params(self, batch, dataset_size=None, batch_size=None,
                      **kwargs):
        loss, match_loss, masked_non_match_loss, \
            background_non_match_loss, blind_non_match_loss, \
            descriptor_distances = \
            self.process_batch(batch, batch_size=batch_size, train=True)

        training_metrics = {
            "train-loss": loss,
            "train-match_loss": match_loss,
            "train-masked_non_match_loss": masked_non_match_loss,
            "train-background_non_match_loss": background_non_match_loss,
            "train-blind_non_match_loss": blind_non_match_loss,
            "train-lr": self.scheduler.get_last_lr()[0],
            }
        descriptor_distances = {"train-" + k: wandb.Histogram(v) for k,
                                v in descriptor_distances.items()}
        training_metrics.update(descriptor_distances)

        return training_metrics

    def evaluate(self, batch, batch_size=None, **kwargs):
        loss, match_loss, masked_non_match_loss, \
            background_non_match_loss, blind_non_match_loss, \
            descriptor_distances = \
            self.process_batch(batch, batch_size=batch_size, train=False)

        eval_metrics = {
            "eval-loss": loss,
            "eval-match_loss": match_loss,
            "eval-masked_non_match_loss": masked_non_match_loss,
            "eval-background_non_match_loss": background_non_match_loss,
            "eval-blind_non_match_loss": blind_non_match_loss
            }
        descriptor_distances = {"eval-" + k: wandb.Histogram(v) for k,
                                v in descriptor_distances.items()}
        eval_metrics.update(descriptor_distances)
        return eval_metrics

    def process_batch(self, batch, batch_size=None, train=False):

        list_loss = []
        list_match_loss = []
        list_masked_non_match_loss = []
        list_background_non_match_loss = []
        list_blind_non_match_loss = []

        if train:
            batch_loss = 0
            self.optimizer.zero_grad()

        for i, data in enumerate(batch):
            match_type, \
                img_a, img_b, \
                matches_a, matches_b, \
                masked_non_matches_a, masked_non_matches_b, \
                background_non_matches_a, background_non_matches_b, \
                blind_non_matches_a, blind_non_matches_b, \
                metadata = data

            if match_type == -1:
                tqdm.tqdm.write("empty data. continuing")
                continue

            matches_a = matches_a.to(device)
            matches_b = matches_b.to(device)
            masked_non_matches_a = masked_non_matches_a.to(device)
            masked_non_matches_b = masked_non_matches_b.to(device)

            background_non_matches_a = background_non_matches_a.to(device)
            background_non_matches_b = background_non_matches_b.to(device)

            blind_non_matches_a = blind_non_matches_a.to(device)
            blind_non_matches_b = blind_non_matches_b.to(device)

            # run both images through the network
            image_a_pred = self.compute_descriptor(img_a)
            # reshape from (N, D, H, W) to (N, H*W, D)
            image_a_pred = self.process_network_output(
                image_a_pred, 1)  # batch_size)

            image_b_pred = self.compute_descriptor(img_b)
            image_b_pred = self.process_network_output(
                image_b_pred, 1)  # batch_size)

            loss, match_loss, masked_non_match_loss, \
                background_non_match_loss, blind_non_match_loss, \
                descriptor_distances = dc_loss_composer.get_loss(
                    self.loss, match_type,
                    image_a_pred, image_b_pred,
                    matches_a, matches_b,
                    masked_non_matches_a, masked_non_matches_b,
                    background_non_matches_a, background_non_matches_b,
                    blind_non_matches_a, blind_non_matches_b)

            list_loss.append(loss.detach().clone())
            list_match_loss.append(match_loss.detach())
            list_masked_non_match_loss.append(masked_non_match_loss.detach())
            list_background_non_match_loss.append(
                background_non_match_loss.detach())
            list_blind_non_match_loss.append(blind_non_match_loss.detach())

            if train:
                loss /= batch_size
                loss.backward()
                batch_loss += loss

        if train:
            self.optimizer.step()
            self.scheduler.step()

        return torch.mean(torch.stack(list_loss)), \
            torch.mean(torch.stack(list_match_loss)), \
            torch.mean(torch.stack(list_masked_non_match_loss)), \
            torch.mean(torch.stack(list_background_non_match_loss)), \
            torch.mean(torch.stack(list_blind_non_match_loss)), \
            descriptor_distances

    def process_network_output(self, image_pred, N):
        """
        Processes the network output into a new shape

        :param image_pred: output of the network img.shape =
        [N,descriptor_dim, H , W]
        :type image_pred: torch.Tensor
        :param N: batch size
        :type N: int
        :return: same as input, new shape is [N, W*H, descriptor_dim]
        :rtype:
        """

        W = self.image_width
        H = self.image_height
        image_pred = image_pred.view(N, self.descriptor_dimension, W * H)
        image_pred = image_pred.permute(0, 2, 1)
        return image_pred

    def encode(self, camera_obs, full_obs=None):
        # TODO: can easily get rid of the separate camera_obs?
        # TODO: already setup _encoder to work with tuples, ie arbitary number
        # of cameras. Might wanna restructure this here as well.

        if hasattr(full_obs, "cam_rgb2"):
            n_cams = 1 if full_obs.cam_rgb2 is None else 2
        else:
            n_cams = 1

        # TODO: set to None in constructor to remove hasattr check.
        if hasattr(full_obs, "cam_rgb2") and full_obs.cam_rgb2 is not None:
            rgb = (camera_obs, full_obs.cam_rgb2)
            depth = (full_obs.cam_d, full_obs.cam_d2)
            extr = (full_obs.cam_ext, full_obs.cam_ext2)
            intr = (full_obs.cam_int, full_obs.cam_int2)
        else:
            rgb = (camera_obs, )
            depth = (full_obs.cam_d, )
            extr = (full_obs.cam_ext, )
            intr = (full_obs.cam_int, )

        descriptor = tuple(self.compute_descriptor_batch(r, upscale=False)
                           for r in rgb)

        if self.config.get("prior_type", PriorTypes.NONE) is \
                PriorTypes.PARTICLE_FILTER:
            self.particle_filter.update(
                rgb, depth, extr, intr, descriptor=descriptor,
                ref_descriptor=self._reference_descriptor_vec,
                gripper_pose=full_obs.gripper_pose)
            kp, info = self.particle_filter.estimate_state(extr, intr, depth)
            # print(torch.stack(torch.chunk(kp, 3, dim=-1), dim=-1))

            if self.particle_filter_viz is not None:
                self.particle_filter_viz.update(info['particles'],
                                                info['weights'],
                                                info['prediction'],
                                                info['particles_2d'],
                                                info['keypoints_2d'],
                                                tuple(r.cpu() for r in rgb))

        else:
            prior = tuple(self.get_prior(
                prior_sm=None if (l := self.last_post) is None else l[i],
                prior_pos=None if (l := self.last_kp_raw_2d) is None else l[i])
                        for i in range(n_cams))

            motion_model = tuple(
                self.get_motion_model(
                    depth[i], intr[i], extr[i],
                    None if (l := self.last_d) is None else l[i],
                    None if (l := self.last_int) is None else l[i],
                    None if (l := self.last_ext) is None else l[i])
                for i in range(n_cams))

            # if motion_model is not None:
            #     vis_series(prior[0].cpu(), channeled=False,
            #     file_name="prior")

            prior = tuple(self.apply_motion_model(prior[i], motion_model[i])
                          for i in range(n_cams))

            # if motion_model is not None:
            #     vis_series(prior[0].cpu(), channeled=False,
            #     file_name="prior_mm")

            overshadow = self.config.get("overshadow_keypoints")
            threshold = self.config.get("threshold_keypoint_dist")

            taper = self.config.get("taper_sm") or 1
            use_spatial_expectation = self.config["use_spatial_expectation"]
            projection = self.config["projection"]

            kp, info = self._encode(
                rgb, depth, extr, intr, prior, descriptor=descriptor,
                ref_descriptor=self._reference_descriptor_vec,
                use_spatial_expectation=use_spatial_expectation, taper=taper,
                projection=projection, overshadow=overshadow,
                threshold=threshold)

        self.last_kp_raw_2d = info["kp_raw_2d"]
        self.last_post = info["post"]
        self.last_d = depth
        self.last_int = intr
        self.last_ext = extr

        # print(kp)

        return kp, info

    @classmethod
    def _encode(cls, rgb, depth, extr, intr, prior, descriptor, ref_descriptor,
                use_spatial_expectation=True, taper=1, projection=None,
                overshadow=False, threshold=None):
        # All args are tuples, besides the kwargs.
        # the value to which overshadowed and super-threshold are set. -1
        # corresponds to 0 in pixel-space. Should be out of (0, img_size) to
        # avoid confusion? TODO! Eg set st it will end up being -1 in pixel
        ZERO_VAL = torch.tensor(-1, dtype=torch.float32,
                                device=descriptor[0].device)

        keypoint_dimension = 2 if projection is ProjectionTypes.NONE else 3

        n_cams = len(rgb)

        kwargs = {
            "ref_descriptor": ref_descriptor,
            "taper": taper,
            "use_spatial_expectation": use_spatial_expectation,
            "projection": projection
        }

        kp, distance, kp_raw_2d, prior, sm, post = tuple(zip(
            *(cls.compute_keypoints(r, d, e, i, desc, p, **kwargs)
              for r, d, e, i, desc, p in
              zip(rgb, depth, extr, intr, descriptor, prior))))

        if overshadow and n_cams > 1:
            dist_per_cam = torch.stack(distance)
            best_cam = torch.argmin(dist_per_cam, dim=0)
            # repeat to fit size of kp-tensor which has x_comps, y_comps
            best_cam = best_cam.repeat(1, keypoint_dimension)

            kp = tuple(torch.where(best_cam == i, k, ZERO_VAL)
                       for i, k in enumerate(kp))

        kp = torch.cat(kp, dim=-1)

        if threshold:
            # repeat to fit size of kp-tensor which has x_comps, y_comps
            # dist_per_cam = distance.chunk(n_cams, dim=-1)
            # expanded_dist = torch.cat(
            #     tuple(d.repeat(1, keypoint_dimension)
            #           for d in dist_per_cam), dim=-1)
            expanded_dist = torch.cat(
                tuple(d.repeat(1, keypoint_dimension)
                      for d in distance), dim=-1)
            kp = torch.where(expanded_dist > threshold, ZERO_VAL, kp)

        info = {
            "descriptor": descriptor,
            "distance": distance,
            "kp_raw_2d": kp_raw_2d,
            "depth": depth,
            "prior": prior,
            "sm": sm,
            "post": post
        }

        return kp, info

    @classmethod
    def get_descriptor_distance(cls, descriptor, keypoints, ref_descriptor):
        img_height, img_width = descriptor.shape[-2:]

        # map from [-1, 1] to descriptor size
        kp_x, kp_y = keypoints.chunk(2, dim=-1)
        kp_x = ((kp_x + 1) * (img_width - 1)/2).long()
        kp_y = ((kp_y + 1) * (img_height - 1)/2).long()

        # extract per keypoint descriptor
        B, N_kp = kp_y.shape
        batch_indeces = [i for i in range(B) for _ in range(N_kp)]
        desc_per_kp = channel_front2back_batch(
            descriptor)[batch_indeces, kp_y.flatten(), kp_x.flatten(), :]
        desc_per_kp = desc_per_kp.reshape(B, N_kp, -1)

        # pairwise distance needs both inputs to have shape (N, D), so repeat
        # the reference vector and flatten, afterwards unflatten
        B, N_kp, d_kp = desc_per_kp.shape
        desc_per_kp_flat = desc_per_kp.reshape((B * N_kp, d_kp))
        ref_vec_flat = ref_descriptor.unsqueeze(
            0).repeat(B, 1, 1).reshape((B * N_kp, d_kp))
        distance = torch.nn.functional.pairwise_distance(
            desc_per_kp_flat, ref_vec_flat).reshape(B, N_kp)

        return distance

    @classmethod
    def compute_keypoints(cls, camera_obs, depth, extrinsics, intrinsics,
                          descriptor, prior=None, ref_descriptor=None,
                          taper=1, use_spatial_expectation=False,
                          projection=None):

        sm = cls.softmax_of_reference_descriptors(
            descriptor, ref_descriptor, taper=taper)

        post = prior * sm if prior is not None else sm
        # When correspondence is (almost) zero across the image, the tensor
        # degenerates (becomes zeros, hence nan after renomalization below).
        # Fix by adding small epsilon.
        # post += 1e-10
        # # normalize to sum to one
        post /= torch.sum(post, dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)
        # vis_series(post[0].cpu(), channeled=False, file_name="post")
        # exit()

        # print(" ------------------------------------------")
        # print(prior)
        # print(sm)
        # print(post)

        # when using exp and eg Gaussian prior, the first frame has no prior
        # and thus is too noisy, so use mode there instead.
        # But does not matter for sm as Bayes filter.
        if use_spatial_expectation:  # and prior is not None:
            kp = cls.get_spatial_expectation(post)
        else:
            kp = cls.get_mode(post)

        # TODO: does not work properly anymore when we manipulate the sm after
        # compute descriptor distances for metric tracking
        # distance = cls.get_descriptor_distance(descriptor, kp, ref_descriptor)
        distance = None

        # from viz.activation_map import activation_map
        # from viz.image_series import vis_series
        # vis_series(camera_obs.cpu())
        # activation_map(sm, keypoints=kp)
        # exit()

        kp_raw_2d = kp

        if projection == ProjectionTypes.NONE:
            pass
        elif projection == ProjectionTypes.EGO:
            raise ValueError("Ego projection makes no sense for vanilla kp, "
                              "only for GT or particle filter models.")
        elif projection == ProjectionTypes.UVD:
            kp = append_depth_to_uv(
                    kp, depth, cls.image_width - 1, cls.image_height - 1)
        else:
            if projection in [ProjectionTypes.LOCAL_HARD,
                              ProjectionTypes.LOCAL_SOFT]:
                # create identity extrinsics
                extrinsics = torch.zeros_like(extrinsics)
                extrinsics[:, range(4), range(4)] = 1
            if projection in [ProjectionTypes.LOCAL_SOFT,
                              ProjectionTypes.GLOBAL_SOFT]:
                kp = model_based_vision.soft_pixels_to_3D_world(
                    kp, post, depth, extrinsics, intrinsics,
                    cls.image_width - 1, cls.image_height - 1)
            else:
                kp = hard_pixels_to_3D_world(
                    kp, depth, extrinsics, intrinsics,
                    cls.image_width - 1, cls.image_height - 1)

        return kp, distance, kp_raw_2d, prior, sm, post

    def forward(self, batch, full_obs=None):
        """
        Custom forward method to also return the latent embedding for viz.
        """
        return self.encode(batch, full_obs)

    def compute_descriptor(self, camera_obs):
        # print("desc", camera_obs.min(), camera_obs.max())
        camera_obs = camera_obs.to(device)
        return self.model.compute_descriptors(camera_obs.unsqueeze(0))

    def compute_descriptor_batch(self, camera_obs, upscale=True):
        # print("desc_batch", camera_obs.min(), camera_obs.max())
        camera_obs = camera_obs.to(device)
        # camera_obs = channel_back2front_batch(camera_obs)
        return self.model.compute_descriptors(camera_obs, upscale=upscale)

    def reconstruct(self, batch):
        pass

    @classmethod
    def get_latent_dim(self, config, n_cams=1, image_dim=None):
        projection = config["projection"]

        keypoint_dimension = 2 if projection is ProjectionTypes.NONE else 3

        if config.get("prior_type", PriorTypes.NONE) is \
                PriorTypes.PARTICLE_FILTER and config.get(
                    "projection") is ProjectionTypes.EGO:
            n_obs = 1
        else:
            n_obs = n_cams

        if config.get("prior_type", PriorTypes.NONE) is \
                PriorTypes.PARTICLE_FILTER and config.get("return_spread"):
            keypoint_dimension += 1

        # TODO: when not using SD should be n_keep
        return keypoint_dimension * config["keypoints"]["n_sample"] * n_obs

    def from_disk(self, chekpoint_path, ignore=None):
        if ignore is None:
            ignore = tuple()

        state_dict = torch.load(chekpoint_path, map_location='cpu')
        state_dict = {k: v for k, v in state_dict.items() if k not in ignore}

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys: {}".format(missing))
        if unexpected:
            logger.warning("Unexpected keys: {}".format(unexpected))
        self = self.to(device)

        self.set_model_image_normalization()

    def initialize_image_normalization(self, replay_memory, cam="wrist"):
        mean, std = replay_memory.estimate_image_mean_and_std(
            self.pretrain_config["no_samples_normalization"],
            cam=cam)
        self.norm_mean = torch.Tensor(mean)
        self.norm_std = torch.Tensor(std)

        self.set_model_image_normalization()

    def set_model_image_normalization(self):
        self.model.setup_image_normalization(self.norm_mean.cpu().numpy(),
                                             self.norm_std.cpu().numpy())

    def initialize_parameters_via_dataset(self, replay_memory):
        self.select_reference_descriptors(replay_memory)

    def select_reference_descriptors(self, replay_memory, traj_idx=0, img_idx=0,
                                     object_labels=None, cam="wrist"):
        # rgb, depth, mask, intr, ext = \
        #     replay_memory.sample_data_point_with_cam_matrices(
        #         cam="cam, img_idx=69, traj_idx=0)
        # traj_idx = 1  # 0
        # img_idx = 40  # 10

        rgb, depth, mask = \
            replay_memory.sample_data_point_with_object_labels(
                cam=cam, img_idx=img_idx, traj_idx=traj_idx)

        descriptor = self.compute_descriptor(rgb).detach()
        object_labels = object_labels or replay_memory.get_object_labels()

        n_keypoints_total = self.config["keypoints"]["n_sample"]
        manual_kp_selection = self.pretrain_config.get('manual_kp_selection')

        if manual_kp_selection:
            n_prev_frames = 20
            # TODO: make generic in cam
            preview_frames = replay_memory.sample_bc(
                1, cam=(cam,)).cam_rgb.squeeze(1)
            indeces = np.linspace(
                start=0, stop=preview_frames.shape[0] - 1, num=n_prev_frames)
            indeces = np.round(indeces).astype(int)
            preview_frames = preview_frames.index_select(
                dim=0, index=torch.tensor(indeces))

            preview_descr = self.compute_descriptor_batch(
                preview_frames).detach()
        else:
            preview_frames = None
            preview_descr = None

        self.ref_pixels_uv, self._reference_descriptor_vec = \
            self._select_reference_descriptors(
                rgb, descriptor, mask, object_labels, n_keypoints_total,
                manual_kp_selection, preview_frames, preview_descr)

        try:
            if self.config["keypoints"]["type"] is KeypointsTypes.SD:
                self._reference_descriptor_vec.requires_grad = False
            elif self.config["keypoints"]["type"] is KeypointsTypes.ODS:
                self._reference_descriptor_vec.requires_grad = True
            else:
                raise NotImplementedError(
                    "Only implemented SD, ODS so far, not {}.".format(
                        self.config["keypoints"]["type"]))
        except RuntimeError:
            logger.warning(
                "Can't set require_grad of reference descriptors as they are"
                " non-leaf vars. Thus, E2E learning must be active and "
                "requires_grad is {}",
                self._reference_descriptor_vec.requires_grad)

        if self.debug_kp_selection:
            if self.keypoint_dimension == 2:
                image_with_points_overlay_uv_list(channel_front2back(
                    rgb.cpu()),
                    (self.ref_pixels_uv[0].numpy(),
                     self.ref_pixels_uv[1].numpy()),
                    mask=mask)
                # descriptor_image_np = descriptor_image_tensor.cpu().numpy()
                # plt.imshow(descriptor_image_np)
                # plt.show()
            elif self.keypoint_dimension == 3:
                depth_map_with_points_overlay_uv_list(
                    depth.cpu().numpy(),
                    (self.ref_pixels_uv[0].numpy(),
                     self.ref_pixels_uv[1].numpy()),
                    mask=mask.cpu().numpy())
            else:
                raise ValueError("No viz for {}d keypoints.".format(
                    self.keypoint_dimension))

        # print(self._reference_descriptor_vec.requires_grad)
        # print(self.ref_pixels_uv.requires_grad)

    @classmethod
    def _select_reference_descriptors(
            cls, rgb, descriptor, mask, object_labels, n_keypoints_total,
            manual_kp_selection, preview_frames=None, preview_descr=None):

        if manual_kp_selection:
            ref_pixels_uv, reference_descriptor_vec = \
                cls.manual_keypoints(
                    channel_front2back(rgb), descriptor, mask, object_labels,
                    n_keypoints_total,
                    preview_rgb=preview_frames, preview_descr=preview_descr)

        else:
            ref_pixels_uv, reference_descriptor_vec = \
                cls.sample_keypoints(rgb, descriptor, mask, object_labels,
                                     n_keypoints_total)

        return torch.stack(ref_pixels_uv),  reference_descriptor_vec

    @classmethod
    def sample_keypoints(cls, rgb, descriptor, mask, object_labels,
                         n_keypoints_total):
        logger.info("Sampling keypoints.")
        descriptor = descriptor.cpu()
        # NOTE: number of keypoints should be divisible by no labels.
        keypoints_per_label = int(n_keypoints_total/len(object_labels))
        ref_pixels = []
        for label in object_labels:
            label_mask = torch.where(mask == label, 1.0, 0.0)
            ref_pixels.append(random_sample_from_masked_image_torch(
                label_mask, keypoints_per_label))  # tuple of (u's, v's)
        ref_pixels_uv = (
            torch.cat(tuple(r[0] for r in ref_pixels)),
            torch.cat(tuple(r[1] for r in ref_pixels))
        )

        ref_pixels_flattened = \
            ref_pixels_uv[1] * mask.shape[1] + ref_pixels_uv[0]

        ref_pixels_flattened = ref_pixels_flattened

        # print(descriptor_image_tensor.shape, "should be 1, D, H, W")

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

        return ref_pixels_uv, reference_descriptor_vec.to(device)

    @classmethod
    def manual_keypoints(cls, rgb, descriptor, mask, object_labels,
                         n_keypoints, preview_rgb=None, preview_descr=None):
        # In this modul CV2 is imported, which causes RLBench to crash if its
        # loaded before the env is created, so import here instead.
        from viz.keypoint_selector import KeypointSelector

        descriptor = descriptor.cpu()

        logger.info("Please select {} keypoints.", n_keypoints)
        kp_selector = KeypointSelector(
            rgb, descriptor, mask, n_keypoints,
            preview_rgb=preview_rgb, preview_descr=preview_descr)
        ref_pixels = kp_selector.run()

        ref_pixels_uv = (
            torch.tensor(tuple(r[0] for r in ref_pixels)),
            torch.tensor(tuple(r[1] for r in ref_pixels))
        )

        ref_pixels_flattened = \
            ref_pixels_uv[1] * rgb.shape[1] + ref_pixels_uv[0]

        ref_pixels_flattened = ref_pixels_flattened

        # print(descriptor_image_tensor.shape, "should be 1, D, H, W")

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

        del kp_selector

        return ref_pixels_uv, reference_descriptor_vec.to(device)

    def reconstruct_ref_descriptor_from_gt(self, replay_memory, ref_pixels_uv,
                                           ref_object_pose, traj_idx=0,
                                           img_idx=0, cam="wrist"):
        # traj_idx = 1  # 0
        # img_idx = 40  # 10 # 10
        rgb, depth, mask, intr, ext, object_pose = \
            replay_memory.sample_data_point_with_ground_truth(
                cam=cam, img_idx=img_idx, traj_idx=traj_idx)

        object_pose = object_pose.to(device)
        rgb = rgb.to(device)

        # ensure it's the same observation
        assert torch.equal(ref_object_pose, object_pose)

        descriptor = self.compute_descriptor(rgb).detach().cpu()

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

    @classmethod
    def softmax_of_reference_descriptors(cls, descriptor_images,
                                         ref_descriptor, taper=1):
        N, D, H, W = descriptor_images.shape
        Nref, Dref = ref_descriptor.shape

        neg_squared_norm_diffs = \
            cls.compute_reference_descriptor_distances(
                descriptor_images, ref_descriptor, taper=taper)

        neg_squared_norm_diffs_flat = neg_squared_norm_diffs.view(
            N, Nref, H*W)  # 1, nm, H*W
        # print(neg_squared_norm_diffs_flat.shape, "should be N, Nref, H*W")
        # neg_squared_norm_diffs_flat /= math.sqrt(D)

        softmax = torch.nn.Softmax(dim=2)
        softmax_activations = softmax(neg_squared_norm_diffs_flat).view(
            N, Nref, H, W)  # N, Nref, H, W
        # print(softmax_activations.shape, "should be N, Nref, H, W")

        return softmax_activations

    @classmethod
    def compute_reference_descriptor_distances(cls, descriptor_images,
                                               ref_descriptor, taper=1):
        N, D, H, W = descriptor_images.shape
        # print("N, D, H, W", N, D, H, W)
        Nref, Dref = ref_descriptor.shape
        # print("Nref, Dref", Nref, Dref)
        assert Dref == D

        descriptor_images = descriptor_images.permute(0, 2, 3, 1)  # N, H, W, D
        descriptor_images = descriptor_images.unsqueeze(3)      # N, H, W, 1, D

        # print(descriptor_images.shape, "should be N, H, W, 1, D")
        descriptor_images = descriptor_images.expand(N, H, W, Nref, D)
        # print(descriptor_images.shape, "should be N, H, W, Nref, D")

        deltas = descriptor_images - ref_descriptor
        # print(deltas.shape, "should also be N, H, W, Nref, D?")

        deltas *= taper

        neg_squared_norm_diffs = -1.0 * \
            torch.sum(torch.pow(deltas, 2), dim=4)  # N, H, W, Nref
        # print(neg_squared_norm_diffs.shape, "should be N, H, W, Nref")

        # spatial softmax
        neg_squared_norm_diffs = neg_squared_norm_diffs.permute(
            0, 3, 1, 2)   # N, Nref, H, W
        # print(neg_squared_norm_diffs.shape, "should be N, Nref, H, W")

        return neg_squared_norm_diffs

    @classmethod
    def get_spatial_expectation(cls, softmax_activations):
        # softmax_attentions shape is N, Nref, H, W
        # print(softmax_activations.shape)
        expected_x = torch.sum(softmax_activations*cls.pos_x, dim=(2, 3))
        # print(expected_x.shape, "expected_x.shape")

        expected_y = torch.sum(softmax_activations*cls.pos_y, dim=(2, 3))
        # print(expected_y.shape, "expected_y.shape")

        stacked_2d_features = torch.cat((expected_x, expected_y), 1)
        # print(stacked_2d_features.shape, "should be N, 2*Nref")

        return stacked_2d_features

    @classmethod
    def get_mode(cls, softmax_activations):
        # need argmax over two last dimensions, so join them first
        s = softmax_activations.shape
        sm_flat = softmax_activations.view(s[0], s[1], -1)
        modes_flat = torch.argmax(sm_flat, dim=2)

        # reshape back to 2D. Note that the new dim is in the front for now.
        modes_2d = modes_flat.unsqueeze(0).repeat((2, 1, 1))

        # get H, W from the flat indeces
        modes_2d[1] = modes_2d[1] // cls.dc_width
        modes_2d[0] = modes_2d[0] % cls.dc_width

        # map from [0, img_size] to [-1, 1] to match pixel_map from spatial exp
        modes_2d = modes_2d.float()
        modes_2d[1] = modes_2d[1] / (cls.dc_height - 1) * 2 - 1
        modes_2d[0] = modes_2d[0] / (cls.dc_width - 1) * 2 - 1

        # move new dim into the middle and flatten to get (N, 2*Nref)
        stacked_2d_features = torch.cat((modes_2d[0], modes_2d[1]), 1)
        stacked_2d_features = modes_2d.permute((1, 0, 2))
        stacked_2d_features = stacked_2d_features.reshape(s[0], -1)

        return stacked_2d_features

    @classmethod
    def setup_pixel_maps(cls):
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., cls.dc_width),
            np.linspace(-1., 1., cls.dc_height)
        )

        cls.pos_x = torch.from_numpy(pos_x).float().to(device)
        cls.pos_y = torch.from_numpy(pos_y).float().to(device)

    def get_prior(self, prior_sm=None, prior_pos=None):
        prior_type = self.config.get("prior_type", PriorTypes.NONE)

        if prior_type in [PriorTypes.POS_GAUSS, PriorTypes.POS_UNI]:
            try:
                prior_var = self.config["prior_var"]
            except ValueError:
                logger.error("When specifying positinal prior (gaus or uniform"
                            "), need to specify prior_var as well.")
        else:
            prior_var = None

        return self._get_prior(prior_sm, prior_pos, prior_type,
                               prior_var=prior_var)

    @staticmethod
    def _get_prior(prior_sm, prior_pos, prior_type, prior_var=None):
        if prior_type is PriorTypes.NONE or prior_pos is None:
            # First frame of trajectory. Use reference pixel positions and
            # correct for camera movement.
            # prior_pos = batched_project_onto_cam(
            #     self.ref_pixel_world, depth, intrinsics, extrinsics)
            # prior_pos = prior_pos/128 - 1  # TODO: make generic
            # Can't account for object pose -> just use no prior here.
            prior = None

        elif prior_type in [PriorTypes.POS_GAUSS, PriorTypes.POS_UNI]:
            use_gaussian = prior_type is PriorTypes.POS_GAUSS
            assert prior_var is not None
            # Non-frist frame. Use last frame's kp positions as prior.
            # shape is (B, N*2), chunk x,y components and create new dim.
            prior_pos = torch.stack(prior_pos.chunk(2, dim=-1), dim=-1)
            # sm: (B,N,H,W), prior_pos: (B,N,2)
            B, N = prior_pos.shape[0:2]
            # TODO: make var config instead of hard-coded.
            var = torch.tensor([prior_var], device=prior_pos.device)
            var = var.unsqueeze(0).unsqueeze(0).repeat(B, N, 1)
            prior = heatmap_from_pos(prior_pos, var, use_gaussian=use_gaussian)

        elif prior_type is PriorTypes.DISCRETE_FILTER:
            prior = prior_sm

        elif prior_type is PriorTypes.PARTICLE_FILTER:
            prior = None  # can't use the same simple matrix multiplication

        else:
            raise ValueError("Unexpected prior type {}".format(prior_type))

        # vis_series(sm[0].cpu(), channeled=False, file_name="sm")
        # vis_series(prior[0].cpu(), channeled=False, file_name="prior")

        return prior

    def get_motion_model(self, depth_a, intr_a, extr_a, depth_b, intr_b,
                         extr_b):
        """
        Simple wrapper first checking wether the motion model is needed.
        """
        if depth_b is None or (self.config.get("prior_type", PriorTypes.NONE)
                               is PriorTypes.NONE) or not self.config.get(
                                   "use_motion_model"):
            return None

        return self._get_motion_model(
            depth_a, intr_a, extr_a, depth_b, intr_b, extr_b)

    @staticmethod
    def _get_motion_model(depth_a, intr_a, extr_a, depth_b, intr_b, extr_b):
        """
        Computes the new position of all pixels of img_b in img_a. Doing so
        in this way (and not the other way around), directly gives us pixel
        coordinates for all positions in descriptor_b to read from for descr_a.
        Makes things a bit smoother as the mapping might neither be injective,
        nor surjective.
        Instead of interpolating between pixels, just round to neareas pixel
        position (currently done in apply_motion_model).
        In case of a pixel position outside the view frustrum (eg. img_b has a
        wider perspective than img_a), just take the nearest value inside the
        frustrum, ie. pad descriptor_b with the outer-most values if needed.
        closest value

        Parameters
        ----------
        depth_a : Tensor (B, H, W)
        intr_a : Tensor (B, 3, 3)
        extr_a : Tensor (B, 4, 4)
        depth_b : Tensor (B, H, W)
        intr_b : Tensor (B, 3, 3)
        extr_b : Tensor (B, 4, 4)

        Returns
        -------
        Tensor (B, H, W, 2)
        """
        B, H, W = depth_a.shape

        # create pixel coordinates
        px_u = torch.arange(0, W, device=depth_a.device)
        px_v = torch.arange(0, H, device=depth_a.device)
        # (B, H*W, 2)
        # cartesian product varies first dim first, so need to swap dims as
        # u,v coordinates are 'right, down'
        px_vu = torch.cartesian_prod(px_u, px_v).unsqueeze(0).repeat(B, 1, 1)

        # project pixel coordinates of current img into pixel space of last cam
        world = batched_pinhole_projection_image_to_world_coordinates(
            px_vu[..., 1], px_vu[..., 0], depth_a.reshape(B, H*W),
            intr_a, extr_a)
        cam_b = batched_project_onto_cam(world, depth_b, intr_b, extr_b,
                                         clip=False)
        cam_b = cam_b.reshape((B, H, W, 2))

        # moved rounding and clamping to apply_motion_model
        # cam_b = torch.round(cam_b)
        # cam_b[..., 0] = torch.clamp(cam_b[..., 0], 0, W - 1)
        # cam_b[..., 1] = torch.clamp(cam_b[..., 1], 0, H - 1)

        return cam_b

    def apply_motion_model(self, descriptor, motion_model):

        """
        Simple wrapper suppling the motion_blur func of self.
        """

        return self._apply_motion_model(descriptor, motion_model,
                                        blur_func=self.motion_blur)

    @staticmethod
    def _apply_motion_model(descriptor, motion_model, blur_func=None):
        """
        Apply the motion model to the descriptor image, i.e. for each pixel,
        set the value of the returned image to the value of the pixel in the
        image specified by the motion model.

        The motion model is in pixel space, so subsample to descriptor size
        first.

        Parameters
        ----------
        descriptor :Tensor (B, N, H, W)
        motion_model : Tensor (B, H', W', 2)

        Returns
        -------
        Tensor (B, N, H, W)
        """
        if motion_model is None:
            return descriptor

        B, N, H, W = descriptor.shape
        _, H2, W2, _ = motion_model.shape

        # downsample motion model and map to new range
        motion_model = torch.movedim(motion_model, 3, 1)
        motion_model = torch.nn.functional.interpolate(
            motion_model, size=(H, W), mode='bilinear', align_corners=True)
        motion_model = torch.movedim(motion_model, 1, 3)
        motion_model[..., 0] = torch.clamp(
            torch.round(motion_model[..., 0] / W2 * W), 0, W - 1)
        motion_model[..., 1] = torch.clamp(
            torch.round(motion_model[..., 1] / H2 * H), 0, H - 1)

        # add extra kp-dim and flatten
        motion_model = motion_model.unsqueeze(1).repeat(1, N, 1, 1, 1)
        mm_flat = motion_model.reshape((B*N*H*W, 2)).long()

        batch_indeces = [i for i in range(B) for _ in range(N*H*W)]
        kp_indeces = [i for _ in range(B) for i in range(N)
                      for _ in range(H*W)]
        # motion model is in uv coordinates, so 'swap' dim order
        new_img_flat = descriptor[batch_indeces, kp_indeces,
                                  mm_flat[..., 1], mm_flat[..., 0]]

        new_img = new_img_flat.reshape((B, N, H, W))

        new_img_wo_blur = new_img

        if blur_func is not None:
            new_img = blur_func(new_img)

        assert not torch.equal(new_img_wo_blur, new_img)

        # re-normalize
        new_img /= torch.sum(new_img, dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)

        return new_img
