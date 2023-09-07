from enum import Enum
from typing import Iterable

import numpy as np
import torch
import torchvision
import tqdm
from loguru import logger

import dense_correspondence.loss.loss_composer as dc_loss_composer
import encoder.representation_learner
import models.keypoints.keypoints as keypoints
import models.keypoints.model_based_vision as model_based_vision
import wandb
from dataset.bc import BCDataset
from dense_correspondence.correspondence_finder import \
    random_sample_from_masked_image_torch
from dense_correspondence.loss.pixelwise_contrastive_loss import \
    PixelwiseContrastiveLoss
from filter.discrete_filter import DiscreteFilter
from filter.particle_filter import ParticleFilter
# from utils.debug import nan_hook, summarize_tensor
from utils.logging import indent_func_log, log_constructor
from utils.misc import get_and_log_failure as get_conf
from utils.observation import SampleTypes, tensor_dict_equal
from utils.select_gpu import device
from utils.torch import (append_depth_to_uv, hard_pixels_to_3D_world,
                         heatmap_from_pos)
# from viz.image_series import vis_series
from viz.image_single import image_with_points_overlay_uv_list
from viz.operations import channel_front2back, channel_front2back_batch
# from viz.particle_filter import ParticleFilterViz
from viz.surface import depth_map_with_points_overlay_uv_list

KeypointsTypes = keypoints.KeypointsTypes


class PriorTypes(Enum):
    NONE = 1
    DISCRETE_FILTER = 2
    PARTICLE_FILTER = 3


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


class KeypointModelModes(Enum):
    ...


class KeypointsPredictor(encoder.representation_learner.RepresentationLearner):

    sample_type = SampleTypes.DC

    encoding_name = 'kp'

    @log_constructor
    def __init__(self, config: dict):
        # TODO: set disk_read_embedding and disk_read_keypoints, organize into
        # subfuncs and only execute what is needed depending on the two flags.
        super().__init__(config=config)

        self.disk_read_keypoints = get_conf(config["obs_config"],
                                            "disk_read_keypoints", False)

        if self.disk_read_keypoints:
            logger.info("Reading precomputed keypoints from disk.")

        encoder_config = config["encoder"]
        self.config = encoder_config

        self.configure_dc_maps(config)

        # Required keys
        self.descriptor_dimension = encoder_config["descriptor_dim"]

        # Optional keys
        self.noise_scale = get_conf(encoder_config, "noise_scale", None)
        self.prior_type = get_conf(encoder_config, "prior_type",
                                   PriorTypes.NONE)
        self.overshadow = get_conf(encoder_config, "overshadow_keypoints",
                                   False)
        self.threshold = get_conf(encoder_config, "threshold_keypoint_dist",
                                  False)
        self.taper = get_conf(encoder_config, "taper_sm", 1)
        self.use_spatial_expectation = get_conf(
            encoder_config, "use_spatial_expectation", False)
        self.debug_kp_selection = get_conf(config["training"], "debug", False)

        self.projection = get_conf(encoder_config, "projection", False)

        self.keypoint_dimension = 2 if self.projection is ProjectionTypes.NONE\
             else 3

        self.model = keypoints.KeypointsModel(encoder_config)

        pretrain_config = config.get("pretrain", {}).get("training_config", {})

        if pretrain_config is not None:
            logger.info("Got pretrain config. Setting up DC-learning.")
            self.pretrain_config = pretrain_config

            self.manual_kp_selection = get_conf(
                pretrain_config, "manual_kp_selection", False)
            self.lr_schedule = get_conf(pretrain_config, "lr_schedule", None)

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

        # Register references as buffers, such that they will be saved with the
        # module. Then, we can use the same reference vectors at inference.
        n_keypoints = self.get_no_keypoints()
        self.register_buffer('ref_pixels_uv',
                             torch.Tensor(2, n_keypoints))
        self.register_buffer('_reference_descriptor_vec',
                             torch.Tensor(n_keypoints,
                                          self.descriptor_dimension))

        self.register_buffer('norm_mean', torch.Tensor(3))
        self.register_buffer('norm_std', torch.Tensor(3))

        if self.prior_type is PriorTypes.PARTICLE_FILTER:
            self.filter = ParticleFilter(config)
            self.filter_viz = None
            # if self.debug_kp_selection:
            #     self.filter_viz = ParticleFilterViz()
            #     self.filter_viz.run()
        elif self.prior_type is PriorTypes.DISCRETE_FILTER:
            self.filter = DiscreteFilter(config)
            self.filter_viz = None
        else:
            self.filter = None
            self.filter_viz = None

        self.reset_episode()

    def configure_dc_maps(self, config: dict):
        self.image_height, self.image_width = config["obs_config"]["image_dim"]
        self.dc_height, self.dc_width = self.get_dc_dim()
        self.setup_pixel_maps()

        # self.config = config

    def get_dc_dim(self):
        dim_mapping = {128: 32,
                       256: 32,
                       360: 45,
                       480: 60,
                       640: 80}

        return dim_mapping[self.image_height], dim_mapping[self.image_width]

    def setup_pixel_maps(self):
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.dc_width),
            np.linspace(-1., 1., self.dc_height)
        )

        self.pos_x = torch.from_numpy(pos_x).float().to(device)
        self.pos_y = torch.from_numpy(pos_y).float().to(device)


    def reset_episode(self):
        if self.filter is not None:
            self.filter.reset()
        if self.filter_viz is not None:
            self.filter_viz.reset_episode()

    def get_no_keypoints(self):
        keypoint_type = self.config["keypoints"]["type"]

        if keypoint_type in [KeypointsTypes.SD, KeypointsTypes.ODS,
                             KeypointsTypes.WDS]:
            n_keypoints = self.config["keypoints"]["n_sample"]
        else:
            n_keypoints = self.config["keypoints"]["n_keep"]

        return n_keypoints

    def update_params(self, batch, dataset_size=None, batch_size=None,
                      **kwargs):
        loss, match_loss, masked_non_match_loss, \
            background_non_match_loss, blind_non_match_loss, \
            descriptor_distances = \
            self.process_batch(batch, batch_size=batch_size, train=True)

        if loss is None:
            return None

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

        if loss is None:
            return None

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

        if not list_loss:  # empty list
            return [None] * 6

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

    def encode(self, batch):

        if self.disk_read_keypoints:
            return getattr(batch, self.encoding_name), {}

        camera_obs = batch.camera_obs

        rgb = tuple((o.rgb for o in camera_obs))
        depth = tuple((o.depth for o in camera_obs))
        extr = tuple((o.extr for o in camera_obs))
        intr = tuple((o.intr for o in camera_obs))

        n_cams = len(rgb)

        if self.disk_read_embedding:
           descriptor = tuple((o.descriptor for o in camera_obs))
        else:
            descriptor = tuple(self.compute_descriptor_batch(r, upscale=False)
                            for r in rgb)


        if self.prior_type is PriorTypes.PARTICLE_FILTER:
            self.filter.update(
                rgb, depth, extr, intr, descriptor=descriptor,
                ref_descriptor=self._reference_descriptor_vec,
                gripper_pose=batch.ee_pose)
            kp, info = self.filter.estimate_state(extr, intr, depth)

            if self.filter_viz is not None:
                self.filter_viz.update(info['particles'],
                                       info['weights'],
                                       info['prediction'],
                                       info['particles_2d'],
                                       info['keypoints_2d'],
                                       tuple(r.cpu() for r in rgb))
        else:
            if self.prior_type is PriorTypes.DISCRETE_FILTER:
                prior = self.filter.get_prior(rgb, depth, extr, intr)
            else:
                prior = tuple((None for _ in range(n_cams)))

            kp, info = self._encode(
                rgb, depth, extr, intr, prior, descriptor=descriptor,
                ref_descriptor=self._reference_descriptor_vec,
                use_spatial_expectation=self.use_spatial_expectation,
                projection=self.projection, overshadow=self.overshadow,
                threshold=self.threshold, taper=self.taper)

        if self.noise_scale is not None:
            kp = self.add_gaussian_noise(kp, self.noise_scale, skip_z=False)

        return kp, info

    def _encode(self, rgb, depth, extr, intr, prior, descriptor, ref_descriptor,
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
            *(self.compute_keypoints(r, d, e, i, desc, p, **kwargs)
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

    def compute_keypoints(self, camera_obs, depth, extrinsics, intrinsics,
                          descriptor, prior=None, ref_descriptor=None,
                          taper=1, use_spatial_expectation=False,
                          projection=None):

        sm = self.softmax_of_reference_descriptors(
            descriptor, ref_descriptor, taper=taper)

        post = prior * sm if prior is not None else sm
        # When correspondence is (almost) zero across the image, the tensor
        # degenerates (becomes zeros, hence nan after renomalization below).
        # Fix by adding small epsilon.
        # post += 1e-10
        # # normalize to sum to one
        post /= torch.sum(post, dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)

        if use_spatial_expectation:
            kp = self.get_spatial_expectation(post)
        else:
            kp = self.get_mode(post)

        distance = None  # TODO: does not work properly anymore -> removed

        kp_raw_2d = kp

        if projection == ProjectionTypes.NONE:
            pass
        elif projection == ProjectionTypes.EGO:
            raise ValueError("Ego projection makes no sense for vanilla kp, "
                              "only for GT or particle filter models.")
        elif projection == ProjectionTypes.UVD:
            kp = append_depth_to_uv(
                kp, depth, self.image_width - 1, self.image_height - 1)
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
                    self.image_width - 1, self.image_height - 1)
            else:
                kp = hard_pixels_to_3D_world(
                    kp, depth, extrinsics, intrinsics,
                    self.image_width - 1, self.image_height - 1)

        return kp, distance, kp_raw_2d, prior, sm, post

    def forward(self, batch):
        """
        Custom forward method to also return the latent embedding for viz.
        """
        return self.encode(batch)

    def compute_descriptor(self, camera_obs):
        camera_obs = camera_obs.to(device)
        return self.model.compute_descriptors(camera_obs.unsqueeze(0))

    def compute_descriptor_batch(self, camera_obs, upscale=True):
        camera_obs = camera_obs.to(device)
        return self.model.compute_descriptors(camera_obs, upscale=upscale)

    def reconstruct(self, batch):
        pass

    @staticmethod
    def get_latent_dim(config: dict, n_cams: int = 1, image_dim=None):
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

    @indent_func_log
    def from_disk(self, chekpoint_path, ignore=None):
        logger.info("Loading encoder checkpoint from {}", chekpoint_path)
        if ignore is None:
            ignore = tuple()

        state_dict = torch.load(chekpoint_path, map_location='cpu')
        state_dict = {k: v for k, v in state_dict.items() if k not in ignore}

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing and set(ignore) != set(missing):
            logger.warning("Missing keys: {}".format(missing))
        if unexpected:
            logger.warning("Unexpected keys: {}".format(unexpected))
        self = self.to(device)

        self.set_model_image_normalization()

    def initialize_image_normalization(self, replay_memory, cam=("wrist",)):
        mean, std = replay_memory.estimate_image_mean_and_std(
            self.pretrain_config["no_samples_normalization"], cam=cam)

        self.norm_mean = torch.Tensor(mean)
        self.norm_std = torch.Tensor(std)

        self.set_model_image_normalization()

    def set_model_image_normalization(self):
        self.model.setup_image_normalization(self.norm_mean.cpu().numpy(),
                                             self.norm_std.cpu().numpy())

    @indent_func_log
    def initialize_parameters_via_dataset(self, replay_memory, cam):
        self.select_reference_descriptors(replay_memory, cam=cam)

    def select_reference_descriptors(
            self, dataset: BCDataset, traj_idx: int = 0, img_idx: int = 0,
            object_labels: Iterable[int] | None = None, cam: str = "wrist"
            ) -> None:
        """
        Select reference descriptors from one observation in the replay memory.
        Depending on config, randomly sampled or manually selected.

        Usually, for policy learning the used object labels are extracted
        automatically from the dataset.
        But for other purposes, it is possible to pass specific object labels
        as well.

        Parameters
        ----------
        dataset : BCDataset
            The dataset from which to sample the reference descriptors.
        traj_idx : int, optional
            The trajectory index, by default 0
        img_idx : int, optional
            The observation index, by default 0
        object_labels : Iterable[int] | None, optional
            Object labels to sample from, by default None
        cam : str, optional
            The name of the camera to sample from, by default "wrist"

        Raises
        ------
        NotImplementedError
            For not implemented keypoint types.
        ValueError
            For invalid keypoint dimensions. Supported: 2, 3.
        """

        ref_obs = dataset.sample_data_point_with_object_labels(
                cam=cam, img_idx=img_idx, traj_idx=traj_idx)

        object_labels = object_labels or dataset.get_object_labels()

        n_keypoints_total = self.config["keypoints"]["n_sample"]
        manual_kp_selection = self.pretrain_config.get('manual_kp_selection')

        if manual_kp_selection:
            preview_obs = dataset._get_bc_traj(
                traj_idx, cams=(cam,), fragment_length=-1, force_skip_rgb=False
            )

            indeces = np.linspace(0, stop=preview_obs.shape[0] - 1, num=20)
            indeces = np.round(indeces).astype(int)
            preview_obs = preview_obs.get_sub_tensordict(torch.tensor(indeces))

            preview_frames = preview_obs.camera_obs[0].rgb
            preview_descr = preview_obs.camera_obs[0].descriptor

        else:
            preview_frames = None
            preview_descr = None

        if self.disk_read_embedding:
            ref_descriptor = dataset.load_embedding(
                img_idx=img_idx, traj_idx=traj_idx, cam=cam,
                embedding_name=self.embedding_name).unsqueeze(0)

            # Usually, encoder upsamples back to full resolution. For pre-
            # computed that would waste disk-space, so upsample here.
            img_h, img_w = ref_obs.rgb.shape[-2:]
            if ref_descriptor.shape[-1] != ref_obs.rgb.shape[-1]:
                logger.info("Upsampling descriptor to image size.")
                ref_descriptor = torch.nn.functional.interpolate(
                    ref_descriptor, size=tuple((img_h, img_w)),
                    mode='bilinear', align_corners=True)

                if preview_descr is not None and \
                        preview_descr.shape[-1] != ref_obs.rgb.shape[-1]:
                    preview_descr = torch.nn.functional.interpolate(
                        preview_descr, size=tuple((img_h, img_w)),
                        mode='bilinear', align_corners=True)
        else:
            ref_descriptor = self.compute_descriptor(ref_obs.rgb).detach()

        self.ref_pixels_uv, self._reference_descriptor_vec = \
            self._select_reference_descriptors(
                ref_obs.rgb, ref_descriptor, ref_obs.mask, object_labels,
                n_keypoints_total, manual_kp_selection, preview_frames,
                preview_descr)

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
                    ref_obs.rgb.cpu()),
                    (self.ref_pixels_uv[0].numpy(),
                     self.ref_pixels_uv[1].numpy()),
                    mask=ref_obs.mask)
            elif self.keypoint_dimension == 3:
                depth_map_with_points_overlay_uv_list(
                    ref_obs.depth.cpu().numpy(),
                    (self.ref_pixels_uv[0].numpy(),
                     self.ref_pixels_uv[1].numpy()),
                    mask=ref_obs.mask.cpu().numpy())
            else:
                raise ValueError("No viz for {}d keypoints.".format(
                    self.keypoint_dimension))

    @classmethod
    def _select_reference_descriptors(
            cls, rgb, descriptor, mask, object_labels, n_keypoints_total,
            manual_kp_selection, preview_frames=None, preview_descr=None,
            object_order=None):

        if manual_kp_selection:
            ref_pixels_uv, reference_descriptor_vec = \
                cls.manual_keypoints(
                    channel_front2back(rgb), descriptor, mask, object_labels,
                    n_keypoints_total,
                    preview_rgb=preview_frames, preview_descr=preview_descr,
                    object_order=object_order)

        else:
            ref_pixels_uv, reference_descriptor_vec = \
                cls.sample_keypoints(rgb, descriptor, mask, object_labels,
                                     n_keypoints_total)

        return torch.stack(ref_pixels_uv),  reference_descriptor_vec

    @staticmethod
    def sample_keypoints(rgb, descriptor, mask, object_labels,
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

    @staticmethod
    def manual_keypoints(rgb, descriptor, mask, object_labels, n_keypoints,
                         preview_rgb=None, preview_descr=None,
                         object_order=None):
        # In this modul CV2 is imported, which causes RLBench to crash if its
        # loaded before the env is created, so import here instead.
        from viz.keypoint_selector import KeypointSelector

        descriptor = descriptor.cpu()

        logger.info("Please select {} keypoints.", n_keypoints)
        if object_order is not None:
            logger.info("Make sure to follow the object order {}.",
                        object_order)
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
                                           ref_object_poses, traj_idx=0,
                                           img_idx=0, cam="wrist"):

        ref_obs = replay_memory.sample_data_point_with_ground_truth(
                cam=cam, img_idx=img_idx, traj_idx=traj_idx)

        rgb = ref_obs.rgb.to(device)
        mask = ref_obs.mask.to(device)
        object_poses = ref_obs.object_poses.to(device)

        # ensure it's the same observation
        assert tensor_dict_equal(ref_object_poses, object_poses)

        if self.disk_read_embedding:
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
        else:
            descriptor = self.compute_descriptor(rgb).detach().cpu()

        ref_pixels_flat = ref_pixels_uv[1] * mask.shape[1] + ref_pixels_uv[0]

        D = descriptor.shape[1]
        WxH = descriptor.shape[2] * descriptor.shape[3]

        # view as D, H*W
        descriptor_image_tensor = descriptor.squeeze(
            0).contiguous().view(D, WxH)

        # switch back to H*W, D
        descriptor_image_tensor = descriptor_image_tensor.permute(1, 0)

        # self.ref_descriptor_vec is Nref, D
        reference_descriptor_vec = torch.index_select(
            descriptor_image_tensor, 0, ref_pixels_flat)

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

    @staticmethod
    def compute_reference_descriptor_distances(descriptor_images,
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

    def get_spatial_expectation(self, softmax_activations):
        # softmax_attentions shape is N, Nref, H, W
        # print(softmax_activations.shape)
        expected_x = torch.sum(softmax_activations*self.pos_x, dim=(2, 3))
        # print(expected_x.shape, "expected_x.shape")

        expected_y = torch.sum(softmax_activations*self.pos_y, dim=(2, 3))
        # print(expected_y.shape, "expected_y.shape")

        stacked_2d_features = torch.cat((expected_x, expected_y), 1)
        # print(stacked_2d_features.shape, "should be N, 2*Nref")

        return stacked_2d_features

    def get_mode(self, softmax_activations):
        # need argmax over two last dimensions, so join them first
        s = softmax_activations.shape
        sm_flat = softmax_activations.view(s[0], s[1], -1)
        modes_flat = torch.argmax(sm_flat, dim=2)

        # reshape back to 2D. Note that the new dim is in the front for now.
        modes_2d = modes_flat.unsqueeze(0).repeat((2, 1, 1))

        # get H, W from the flat indeces
        modes_2d[1] = modes_2d[1] // self.dc_width
        modes_2d[0] = modes_2d[0] % self.dc_width

        # map from [0, img_size] to [-1, 1] to match pixel_map from spatial exp
        modes_2d = modes_2d.float()
        modes_2d[1] = modes_2d[1] / (self.dc_height - 1) * 2 - 1
        modes_2d[0] = modes_2d[0] / (self.dc_width - 1) * 2 - 1

        # move new dim into the middle and flatten to get (N, 2*Nref)
        stacked_2d_features = torch.cat((modes_2d[0], modes_2d[1]), 1)
        stacked_2d_features = modes_2d.permute((1, 0, 2))
        stacked_2d_features = stacked_2d_features.reshape(s[0], -1)

        return stacked_2d_features
