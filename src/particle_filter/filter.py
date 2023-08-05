import typing

import matplotlib.pyplot as plt
import torch
from loguru import logger

import encoder.keypoints
from utils.select_gpu import device
from utils.torch import batchwise_project_onto_cam  # flipped_logistic,
from utils.torch import gaussian_cdf, noisy_pixel_coordinates_to_world


class ParticleFilter:
    def __init__(self, config: dict, ref_pixel_world=None) -> None:
        self.batch_size = None
        self.n_keypoints = None

        self.cam_width = config["obs_config"]["image_dim"][1]

        self.weights = None
        self.coordinates = None

        self.config = config

        self.debug = config["training"]["debug"]
        self.running_eval = False

        self.projection_type = config["encoder"]["projection"]

        self.n_particles = 250  # TODO: conf

        self.resampling_threshold = 0.1 * self.n_particles  # TODO: conf

        self.descriptor_distance_for_outside_pixels = config["encoder"].get(
            "descriptor_distance_for_outside_pixels", 0.5)

        self.use_oh_likelihood_only = False

        self.depth_likelihood_for_outside_pixels = 0.5

        self.noise_scale = config["encoder"].get("filter_noise_scale", 0.009)
        self.use_gripper_motion = config["encoder"].get("use_gripper_motion")
        self.gripper_motion_prob = config["encoder"].get("gripper_motion_prob",
                                                         0.5)

        self.clip_projection = config["encoder"].get("clip_projection")

        # The consistency model can help single unobserved keypoints from
        # diverging and thusly fix outliers. However, it is also prone to
        # enforcing consensus. Eg. when multiple kps diverge, they pull the
        # rest with them. And it reduces particle diversity.
        self.use_consistency_model = config["encoder"].get(
          "use_consistency_model")
        self.consistency_alpha = 128  # 128 # 16  # 0.5  # 0.125

        self.use_depth_consistency = False  # True
        self.depth_consistency_alpha = 256
        assert not self.use_consistency_model or not self.use_depth_consistency

        # self.depth_model_growth_rate = 90
        self.depth_model_eps = 0.1  # 0.25
        self.depth_model_sigma_rel = 0.02  # 0.005
        self.depth_model_sigma_abs = 0.001
        self.taper = 4
        self.taper_initial_sample = True
        self.extra_init_taper = 4  # NOTE: 2 on rubbish
        self.base_descr_measurement_model = lambda x: torch.exp(self.taper*x)

        self.ref_pixel_world = ref_pixel_world

        self.refine_initialization = False  # config["encoder"].get("refine_init")
        self.resample_after_refine = True
        self.refine_consistency_alpha = 1
        # self.use_occlusion_in_depth_consistency = config["encoder"].get(
        #     "use_occlusion_in_depth_consistency")

        self.background_margin = config["pretrain"]["training_config"][
            "loss_function"]["M_background"]
        assert self.background_margin == 0.75
        self.reconsctruct_same_objectness = not config["encoder"].get(
            "predefined_same_objectness")

        self.dbg_init = False
        # NOTE: this should default to False and be configed to True for the
        # wrist-only runs. But when I started them, this was hardcoded to True,
        # and I need to be False for the Gaussian runs. So fix when eval.
        self.sample_from_each_obs = config["encoder"]["sample_from_each_obs"]
        self.dbg_obs_sample = False

        self.return_spread = config["encoder"].get("return_spread", False)

        if self.dbg_init:
            self.fig, self.axes = plt.subplots(3, 2, figsize=(19.2, 14.4))
            self.fig2, self.axes2 = plt.subplots(4, 4, figsize=(19.2, 14.4))
            for i in range(2):
                sps = self.axes[1][i].get_subplotspec()
                self.axes[1][i].remove()
                self.axes[1][i] = self.fig.add_subplot(sps, projection="3d")
        else:
            self.axes = None

        if self.dbg_obs_sample:
            self.re_fig, self.re_axes = plt.subplots(4, 4, figsize=(19.2, 14.4))
        else:
            self.re_axes = None

    def reset(self) -> None:
        self.batch_size = None
        self.n_keypoints = None

        self.weights = None
        self.coordinates = None

        self.last_gripper_pose = None

        self.running_sum_of_squared_differences = None
        self.running_mean = None
        self.last_coordinate_mean = None
        self.t = 0

        self.last_projected_depth = None
        self.last_projected_depth_diff = None

        self.same_object = None

    def sample_particles(self, depth: tuple, extr: tuple, intr: tuple,
                         diffs: tuple, ref_descriptor: torch.Tensor) -> None:

        # self.coordinates = uniform_sample(self.batch_size, self.n_keypoints,
        #                                   self.n_particles)

        if self.taper_initial_sample:
            diffs = tuple(d*self.taper*self.extra_init_taper for d in diffs)

        self.cam_idx = cam_idx = len(depth) - 1  # NOTE: use 0 for wrist-only
        self.coordinates = sample_from_obs((depth[cam_idx],), (extr[cam_idx],),
                                           (intr[cam_idx],), (diffs[cam_idx],),
                                           self.n_particles,
                                           self.depth_model_sigma_rel,
                                           self.depth_model_sigma_abs,
                                           self.axes
                                           )

        # Reconstructing same-objectness from ref descriptor distances.
        # NOTE: if not good enough, can still just set the first/second
        # half to True/False for the first/second half of the set.
        if self.reconsctruct_same_objectness:
            self.reference_similarity = get_pairwise_distance(ref_descriptor)
            self.same_object = \
                self.reference_similarity < self.background_margin
        else:
            K = self.n_keypoints
            O = int(K/2)
            self.same_object = torch.zeros((K, K), device=device).bool()
            self.same_object[:O, :O] = True
            self.same_object[O:, O:] = True
        self.same_object = self.same_object.fill_diagonal_(
            False)  # don't use self-similarity

        self.set_uniform_weights()

        if self.refine_initialization:
            self.weights *= refine_with_prior(self.coordinates,
                                              self.ref_pixel_world.to(device),
                                              self.refine_consistency_alpha)

            # TODO: put in resample method (and use below in update as well)
            if self.resample_after_refine:
                B = self.weights.shape[0]
                P = self.n_particles
                K = self.n_keypoints

                # particle_indeces = stratified_resample(self.weights)
                particle_indeces = systematic_resample(self.weights)

                particle_indeces = particle_indeces.flatten()

                batch_indeces = [i for i in range(B) for _ in range(K*P)]
                kp_indeces = [i for _ in range(B) for i in range(K)
                              for _ in range(P)]

                new_coords = self.coordinates[batch_indeces,
                                              kp_indeces, particle_indeces]
                new_coords = new_coords.reshape((B, K, P, 3))
                self.coordinates = new_coords
                self.set_uniform_weights()

        if self.dbg_init:
            wc = self.coordinates.cpu()
            mean = self.get_weighted_mean().cpu()
            ax = self.axes[1][1]
            ax.scatter(wc[0, 0, :, 0], wc[0, 0, :, 1], wc[0, 0, :, 2],
                       color='r')
            ax.scatter(mean[0, 0, 0], mean[0, 0, 1], mean[0, 0, 2],
                       color='b')

            heatmap, keypoints_2d = project_particles_to_camera(
                self.coordinates, self.weights, mean.to(device), depth, extr,
                intr)

            for i in range(1): # 2):
                kp_idx = 0
                self.axes[2][i].imshow(depth[i].cpu().squeeze(0))
                self.axes[2][i].imshow(heatmap[i][0, kp_idx], alpha=0.5)
                self.axes[2][i].scatter(keypoints_2d[i][0, :, 0],
                                        keypoints_2d[i][0, :, 1], color='r')

            for i in range(16):
                ax = self.axes2.flat[i]
                # self.axes2.flat[i].imshow(depth[self.cam_idx].cpu().squeeze(0))
                self.axes2.flat[i].imshow(heatmap[self.cam_idx][0, i])
                self.axes2.flat[i].imshow(
                    heatmap[self.cam_idx][0, i], alpha=0.5)
                self.axes2.flat[i].scatter(keypoints_2d[self.cam_idx][0, i, 0],
                                           keypoints_2d[self.cam_idx][0, i, 1], color='r')

            plt.show()

    def set_uniform_weights(self, mask: torch.tensor = None) -> None:
        weights = normalize(torch.ones(
            (1, 1, self.n_particles), device=device))
        weights = weights.repeat((self.batch_size, self.n_keypoints, 1))
        # , names=('B', 'K', 'P')))

        if mask is None:
            self.weights = weights
        else:
            self.weights = torch.where(mask, weights, self.weights)

    def update(self, rgb: tuple, depth: tuple, extr: tuple, intr: tuple,
               descriptor: tuple, ref_descriptor: torch.Tensor,
               gripper_pose: torch.Tensor = None) -> None:
        KeypointsPredictor = encoder.keypoints.KeypointsPredictor

        diffs = tuple(
            KeypointsPredictor.compute_reference_descriptor_distances(
                d, ref_descriptor)  # .refine_names('B', 'K', 'H', 'W')
            for d in descriptor)

        if self.batch_size is None:  # on first step of new trajectory
            self.batch_size = extr[0].shape[0]
            self.n_keypoints = ref_descriptor.shape[0]
            self.sample_particles(depth, extr, intr, diffs, ref_descriptor)

        else:
            self.coordinates = self.predict_motion(extr, intr, gripper_pose)

            P = self.n_particles
            K = self.n_keypoints
            _, H, W = depth[0].shape

            # Add K, P dims to extr, intr and flatten for projection
            b_extr = tuple(
                e.unsqueeze(1).unsqueeze(1).repeat(1, K, P, 1, 1).reshape(
                -1, 4, 4) for e in extr)
            b_intr = tuple(
                i.unsqueeze(1).unsqueeze(1).repeat(1, K, P, 1, 1).reshape(
                -1, 3, 3) for i in intr)

            CLIP_VALUE = -1

            particle_vu, proj_depth = project_onto_image(
                self.coordinates, depth, b_extr, b_intr, clip_value=CLIP_VALUE)

            # either coordinate is outside the view frustrum
            px_mask = tuple(
                (p == CLIP_VALUE).sum(dim=-1) > 0 for p in particle_vu)

            B, _, _ = self.weights.shape
            _, _, H2, W2 = diffs[0].shape

            # scale particle_vu from img (eg [0, 256]) to descriptor eg [0, 32]
            particle_vu = tuple(torch.stack(
                (torch.clamp(torch.round(p[..., 0] / W * W2), 0, W2 - 1),
                 torch.clamp(torch.round(p[..., 1] / H * H2), 0, H2 - 1)),
                dim=-1) for p in particle_vu)

            pvu_flat = tuple(p.reshape((B*K*P, 2)).long() for p in particle_vu)
            batch_indeces = [i for i in range(B) for _ in range(K*P)]
            kp_indeces = [i for _ in range(B) for i in range(K)
                          for _ in range(P)]

            # select the difference from the map using the calculated px coords
            particle_diffs = tuple(
                d[batch_indeces, kp_indeces, p[..., 1], p[..., 0]].reshape(
                    B, K, P) for p, d in zip(pvu_flat, diffs))

            # select the corresponding depth values - need to downsample first
            ds_depth = tuple(torch.nn.functional.interpolate(
                d.unsqueeze(0), size=(H2, W2), mode='bilinear',
                align_corners=True).squeeze(0)
                for d in depth)

            proj_depth = tuple(p.reshape(B, K, P) for p in proj_depth)

            hypothesis_depth = tuple(
                d[batch_indeces, p[..., 1], p[..., 0]].reshape(B, K, P)
                for p, d in zip(pvu_flat, ds_depth))

            self.weights *= self.measurement_model(particle_diffs, px_mask,
                                                   hypothesis_depth,
                                                   proj_depth)

            if self.use_consistency_model:
                self.weights *= self.query_consistency_model()

            self.weights = normalize(self.weights)

            neff = get_neff(self.weights)

            # if self.debug:
            #     logger.info('Particle filter neff {}', neff)

            resample_mask = neff < self.resampling_threshold

            if resample_mask.any():
                with logger.contextualize(filter=False):
                    logger.info('  Resampling for {}/{} batch-KP pairs.',
                                resample_mask.sum(),
                                self.batch_size * self.n_keypoints)
                # particle_indeces = stratified_resample(self.weights)
                particle_indeces = systematic_resample(self.weights)

                particle_indeces = particle_indeces.flatten()

                batch_indeces = [i for i in range(B) for _ in range(K*P)]
                kp_indeces = [i for _ in range(B) for i in range(K)
                              for _ in range(P)]

                new_coords = self.coordinates[batch_indeces,
                                              kp_indeces, particle_indeces]
                new_coords = new_coords.reshape((B, K, P, 3))

                # new_coords = torch.where(
                #     random_mask.unsqueeze(-1), obs_samples, new_coords)
                resample_mask = resample_mask.unsqueeze(-1).unsqueeze(-1)
                self.coordinates = torch.where(resample_mask, new_coords,
                                               self.coordinates)

                if self.sample_from_each_obs:
                    diffs = tuple(d*self.taper*64 for d in diffs)
                    obs_samples = sample_from_obs(
                        (depth[self.cam_idx],), (extr[self.cam_idx],),
                        (intr[self.cam_idx],), (diffs[self.cam_idx],),
                        self.n_particles,
                        self.depth_model_sigma_rel,
                        self.depth_model_sigma_abs,
                        re_axes=self.re_axes
                        )

                    random_mask = torch.rand(obs_samples.shape[:-1], device=device) < 0.05
                    self.coordinates = torch.where(
                        random_mask.unsqueeze(-1), obs_samples,
                        self.coordinates)

                self.set_uniform_weights(resample_mask.squeeze(-1))

        self.weights = normalize(self.weights)

        self.last_gripper_pose = gripper_pose

    def measurement_model(self, neg_diffs, px_mask, hypothesis_depth,
                          proj_depth):
        B, K, P = neg_diffs[0].shape
        device = neg_diffs[0].device

        descr_likelihood = tuple(self.base_descr_measurement_model(d)
                                 for d in neg_diffs)

        depth_likelihood_model = tuple(torch.distributions.normal.Normal(
            p, torch.clamp(p*self.depth_model_sigma_rel, min=0)
            + self.depth_model_sigma_abs)
            for p in proj_depth)

        # subtracting the logprob of the mean is equivalent to dividing by its
        # prob, ie normalizes the probs to [0,1]
        depth_likelihood = tuple(
            torch.exp(d.log_prob(h) - d.log_prob(p))
            for d, h, p in zip(depth_likelihood_model, hypothesis_depth,
                               proj_depth))

        # compute the occlusion likelihood from measured and predicted depth
        occlusion_likelihood = tuple(self.occlusion_model(d, p)
                                     for d, p in zip(hypothesis_depth,
                                                     proj_depth))

        # cannot estimate occlusion likelihood for outside pixels, so set to 0
        one_val = torch.tensor(1, dtype=torch.float32, device=device)

        occlusion_likelihood = tuple(
            torch.where(m, one_val, p)
            for m, p in zip(px_mask, occlusion_likelihood))

        # Filter outside values from descriptor distances
        if type(self.descriptor_distance_for_outside_pixels) is tuple:
            outside_descr_dist_val = tuple(
                torch.tensor(-d, dtype=torch.float32, device=device)
                for d in self.descriptor_distance_for_outside_pixels)
        else:
            outside_descr_dist_val = tuple(torch.tensor(
                -self.descriptor_distance_for_outside_pixels, dtype=torch.float32,
                device=device)
                for _ in range(len(occlusion_likelihood)))

        occlusion_descr_likelihood = tuple(
            self.base_descr_measurement_model(o)
            for o in outside_descr_dist_val)

        # Filter outside values for depth likelihood
        outside_depth_likelihood = torch.tensor(
            self.depth_likelihood_for_outside_pixels, dtype=torch.float32,
            device=device)

        occlusion_depth_likelihood = tuple(
            outside_depth_likelihood
            for _ in range(len(occlusion_likelihood)))

        if self.use_depth_consistency:
            cons_likelihood = self.query_depth_consistency(
                    proj_depth, occlusion_likelihood)
        else:
            cons_likelihood = [1, 1]

        obs_likelihood = tuple(
            (1 - o) * c * d + o * oc * od * dc for c, d, o, oc, od, dc in zip(
                descr_likelihood, depth_likelihood, occlusion_likelihood,
                occlusion_descr_likelihood, occlusion_depth_likelihood,
                cons_likelihood))

        # TODO: make this generic for arbitraty number of cameras
        if self.use_oh_likelihood_only:
            joint_likelihood = obs_likelihood[1]
        elif len(obs_likelihood) == 1:
            joint_likelihood = obs_likelihood[0]
        elif len(obs_likelihood) == 2:
            joint_likelihood = obs_likelihood[0] * obs_likelihood[1]
        else:
            raise NotImplementedError

        return joint_likelihood

    def occlusion_model(self, measurement, hypothesis):
        # return flipped_logistic(measurement,
        #                         hypothesis - self.depth_model_growth_rate,
        #                         self.depth_model_growth_rate)
        occ = 1 - gaussian_cdf(
            measurement, hypothesis - self.depth_model_eps,
            self.depth_model_sigma_rel*hypothesis + self.depth_model_sigma_abs)

        one_tensor = torch.tensor(1, dtype=torch.float, device=device)

        return torch.where(measurement == 0, one_tensor, occ)

    def get_weighted_mean(self):
        weights = self.weights
        # weights = torch.pow(self.weights, self.taper)
        weights = weights / weights.sum(dim=2, keepdim=True)
        weighted_coords = self.coordinates * weights.unsqueeze(-1)
        mean = weighted_coords.sum(dim=-2)

        return mean

    def estimate_state(self, extr, intr, depth):
        mean = self.get_weighted_mean()

        var = torch.mean(torch.var(self.coordinates, dim=-2), dim=-1)

        self.last_coordinate_mean = mean

        world_coordinates = mean.detach().cpu()

        if self.use_consistency_model:
            self.update_consistency_model(mean)

        if self.running_eval or self.debug:
            heatmap, keypoints_2d = project_particles_to_camera(
                self.coordinates, self.weights, mean, depth, extr, intr)
        else:
            heatmap = (None, None)
            keypoints_2d = (None, None)

        if self.projection_type in (
                    encoder.keypoints.ProjectionTypes.EGO,
                    encoder.keypoints.ProjectionTypes.EGO_STEREO) or \
                self.use_depth_consistency:
            K, P, _ = mean.shape
            _, H, W = depth[0].shape

            b_extr = tuple(
                e.unsqueeze(1).unsqueeze(1).repeat(1, K, P, 1, 1).reshape(
                    -1, 4, 4) for e in extr)
            b_intr = tuple(
                i.unsqueeze(1).unsqueeze(1).repeat(1, K, P, 1, 1).reshape(
                    -1, 3, 3) for i in intr)
            if self.projection_type is encoder.keypoints.ProjectionTypes.EGO:
                b_extr = (b_extr[0],)
                b_intr = (b_intr[0],)
                depth = (depth[0],)
            px_mean, m_depth = project_onto_image(mean.unsqueeze(0), depth,
                                                  b_extr, b_intr,
                                                  clip=self.clip_projection)

            px_mean = (m.squeeze(0)*2/self.cam_width - 1 for m in px_mean)
            m_depth = tuple(m.unsqueeze(0).unsqueeze(-1) for m in m_depth)

            if self.projection_type in (
                    encoder.keypoints.ProjectionTypes.EGO,
                    encoder.keypoints.ProjectionTypes.EGO_STEREO):
                mean = tuple(torch.cat((m, d), dim=2)
                             for m, d in zip(px_mean, m_depth))
            else:
                mean = (mean, )
            if self.use_depth_consistency:
                self.update_depth_consistency(m_depth)

        else:
            mean = (mean, )

        kp_dim = 3

        # permute before flatten, st x features are stacked onto y, z features.
        # needed for consitency with vanilla 3d projection
        mean = tuple(m.permute(0, 2, 1).reshape(
            self.batch_size, self.n_keypoints*kp_dim) for m in mean)
        mean = torch.cat(mean, dim=1)

        if self.return_spread:
            mean = torch.cat((mean, var), dim=1)

        info = {
            'kp_raw_2d': (None, None),
            "prior": (None, None),
            "sm": (None, None),
            "post": (None, None),
            "particles_2d": heatmap,
            "keypoints_2d": keypoints_2d,
            'particles': self.coordinates.detach().cpu(),
            'weights': self.weights.detach().cpu(),
            'prediction': mean.detach().cpu(),
            'particle_var': var.detach().cpu(),
            'world_coordinates': world_coordinates,
        }

        return mean, info

    def predict_motion(self, extr, intr, gripper_pose):
        coordinates = self.coordinates

        if self.last_gripper_pose is not None and self.use_gripper_motion:
            gripper_delta = gripper_pose[:, :3] - self.last_gripper_pose[:, :3]
            coordinates = apply_motion_randomly(coordinates, gripper_delta,
                                                self.gripper_motion_prob)

        return add_gaussian_noise(coordinates, self.noise_scale)

    def update_consistency_model(self, mean):
        if self.t == 0:
            self.running_sum_of_squared_differences = torch.zeros(
                (self.batch_size, self.n_keypoints, self.n_keypoints),
                device=device)
            self.running_mean = torch.zeros(
                (self.batch_size, self.n_keypoints, self.n_keypoints),
                device=device)

        self.t += 1

        B, K, D = mean.shape

        # compute pairwise distance matrix of kp means
        # mean[:, :, None].expand(B, K, K, D) is equivalent
        # to mean.unsqueeze(2).repeat(1, 1, K, 1), but faster
        dist = torch.functional.F.pairwise_distance(
            mean[:, :, None].expand(B, K, K, D).reshape((-1, D)),
            mean[:, None].expand(B, K, K, D).reshape(-1, D)).reshape(B, K, K)

        # update running variance of pariwise distances via Welford's algorithm
        diff = dist - self.running_mean
        self.running_mean = self.running_mean + diff / self.t
        new_diff = dist - self.running_mean
        self.running_sum_of_squared_differences += diff * new_diff

    def update_depth_consistency(self, kp_depth):
        kp_depth = tuple(k.squeeze(2) for k in kp_depth)

        B, K = kp_depth[0].shape

        # diff = torch.functional.F.pairwise_distance(
        #     kp_depth[None].expand(B, K, K).reshape((-1, D)),
        #     kp_depth[:, None].expand(B, K, K).reshape(-1, D)).reshape(B, K, K)
        diff = tuple(k[:, :, None].expand(B, K, K) - k[:, None].expand(B, K, K)
                for k in kp_depth)

        self.last_projected_depth = kp_depth
        self.last_projected_depth_diff = diff
        self.t += 1

    def query_consistency_model(self, bressel_correction=True):

        if self.t < 10:
            return 1  # so far no cross trajectory consistency model

        B, K, D = self.last_coordinate_mean.shape
        P = self.n_particles

        last_kp_distance = torch.functional.F.pairwise_distance(
            self.last_coordinate_mean[:, :, None].expand(
                B, K, K, D).reshape(-1, D),
            self.last_coordinate_mean[:, None].expand(
                B, K, K, D).reshape(-1, D)
        ).reshape(B, K, K)[:, :, :, None].expand(B, K, K, P)

        # compute pairwise distance between kp location and ALL particles
        current_dist = torch.functional.F.pairwise_distance(
            self.last_coordinate_mean[:, :, None].expand(B, K, K, D)[
                :, :, :, None].expand(B, K, K, P, D).reshape(-1, D),
            self.coordinates[:, None].expand(B, K, K, P, D).reshape(-1, D)
            ).reshape(B, K, K, P)

        if bressel_correction:
            if self.t == 1:
                return 1
            var = self.running_sum_of_squared_differences / (self.t - 1)
        else:
            var = self.running_sum_of_squared_differences / self.t

        # weight distance by historically obseved variance in kp distance
        var_u = var[:, :, :, None].expand(B, K, K, P)

        # cons_likelihood = torch.exp(self.consistency_alpha * torch.sum(
        #     -torch.abs(last_kp_distance - current_dist) /
        #     (torch.sqrt(var_u) + 1e-7),
        #     dim=1))

        same_object = self.same_object.unsqueeze(0).unsqueeze(3).repeat(
            1, 1, 1, P)

        cons_likelihood = torch.exp(self.consistency_alpha * torch.sum(
            torch.clamp(torch.sqrt(var_u) - same_object * torch.abs(
                last_kp_distance - current_dist), max=0), dim=1))

        # print(torch.sqrt(torch.mean(
        #     torch.abs(last_kp_distance - current_dist)**2 /
        #     (var_u + 1e-5),
        #     dim=1)))

        # return 1

        # print(cons_likelihood)

        return cons_likelihood

    def query_depth_consistency(self, particle_depth: tuple,
                                occlusion_likelihood: tuple):
        if self.t == 0:  # TODO: try larger t
            return 1

        B, K, P = particle_depth[0].shape

        # TODO: last_projected_depth_diff should also have diagonal set to 0?
        # I think currently values are slightly larger than 0 bcs numeric stuff
        # NOTE: no, should be fine as same_object has False on diagonal

        current_diff = tuple(
            l[:, :, None].expand(B, K, K)[
                :, :, :, None].expand(B, K, K, P) - \
            p[:, None].expand(B, K, K, P) for l, p in zip(
                self.last_projected_depth, particle_depth))

        # set distance to own particles to zero
        current_diff = tuple(c * (1 -
            torch.eye(K, device=current_diff[0].device)[
                None, :, :, None].expand(B, K, K, P)) for c in current_diff)

        # print("****")
        # print(current_diff)

        same_object = self.same_object[None].expand(
            B, K, K)[:, :, :, None].expand(B, K, K, P)
        # NOTE: can also deactivate the same-object filter. In that case, the
        # other object also helps to buffer the depth consistency, but might
        # get more noisy itself.
        # same_object = 1

        # print(current_diff - self.last_projected_depth_diff[
        #     :, :, :, None].expand(B, K, K, P))

        if self.weights is None:
            weights = 1
        else:
            # weights = torch.pow(self.weights, self.taper)
            weights = self.weights
            weights = weights / weights.sum(dim=2, keepdim=True)
        weighted_coords = tuple(o * weights for o in occlusion_likelihood)
        kp_occlusion = tuple(w.mean(dim=2) for w in weighted_coords)
        # kp_occlusion = kp_occlusion / kp_occlusion.sum(dim=1, keepdim=True)


        kp_occlusion = tuple(k[:, :, None].expand(
            B, K, K)[:, :, :, None].expand(B, K, K, P) for k in kp_occlusion)

        # cons_likelihood = torch.exp(- self.depth_consistency_alpha * torch.sum(
        #     same_object * (1 - occlusion_likelihood[:, None].expand(
        #         B, K, K, P)) * torch.abs(
        #         current_diff - self.last_projected_depth_diff[
        #             :, :, :, None].expand(B, K, K, P)), dim=1))

        cons_likelihood = tuple(
            torch.exp(- self.depth_consistency_alpha * torch.sum(
            same_object * k * torch.abs(
                c - l[:, :, :, None].expand(B, K, K, P)), dim=1))
            for k, c, l in zip(kp_occlusion, current_diff,
                               self.last_projected_depth_diff))

        # if self.use_occlusion_in_depth_consistency:
        #     cons_likelihood = cons_likelihood * occlusion_likelihood + (
        #         1 - occlusion_likelihood)

        # print(cons_likelihood)

        return cons_likelihood


def project_particles_to_camera(coordinates: torch.tensor,
                                weights: torch.tensor, mean: torch.tensor,
                                depth: tuple, extr: tuple, intr: tuple
                                ) -> tuple:

    n_cams = len(extr)
    B, K, P, _ = coordinates.shape
    _, H, W = depth[0].shape

    b_extr = tuple(e.unsqueeze(1).unsqueeze(1).repeat(1, K, P, 1, 1).reshape(
        -1, 4, 4) for e in extr)
    b_intr = tuple(i.unsqueeze(1).unsqueeze(1).repeat(1, K, P, 1, 1).reshape(
        -1, 3, 3) for i in intr)

    CLIP_VALUE = -1

    particle_vu, _ = project_onto_image(coordinates, depth, b_extr, b_intr,
                                        clip_value=CLIP_VALUE)

    # either coordinate is outside the view frustrum
    px_mask = tuple((p == CLIP_VALUE).sum(dim=-1) > 0 for p in particle_vu)
    px_mask = tuple(p.flatten() for p in px_mask)

    heatmap = tuple(torch.zeros((B*K*P, H, W), device=device)
                    for _ in range(n_cams))

    pvu_flat = tuple(p.reshape((B*K*P, 2)).long() for p in particle_vu)
    weights_flat = weights.reshape((B*K*P))

    for c in range(n_cams):
        for i in range(B*K*P):
            p = pvu_flat[c]
            if not px_mask[c][i]:
                heatmap[c][i][p[i, 1],  p[i, 0]] += weights_flat[i]

    heatmap = tuple(h.reshape(B, K, P, H, W) for h in heatmap)
    heatmap = tuple(h.sum(dim=2).cpu() for h in heatmap)

    # Project mean to 2D keypoint, too.
    b_extr = tuple(e.unsqueeze(1).unsqueeze(1).repeat(1, K, 1, 1, 1).reshape(
        -1, 4, 4) for e in extr)
    b_intr = tuple(i.unsqueeze(1).unsqueeze(1).repeat(1, K, 1, 1, 1).reshape(
        -1, 3, 3) for i in intr)

    mean = mean.unsqueeze(2)

    keypoint, _ = project_onto_image(mean, depth, b_extr, b_intr,
                                     clip_value=CLIP_VALUE)  # , clip=False
    keypoint = tuple(k.reshape(B, K, 2).cpu() for k in keypoint)

    return heatmap, keypoint


def get_neff(weights: torch.tensor):
    n_eff = 1. / weights.square().sum(dim=-1)

    return n_eff


def normalize(weights: torch.tensor, eps=1e-30):
    weights += eps
    normalized = weights / weights.sum(keepdim=True, dim=-1)

    return normalized


def project_onto_image(coord_3d: torch.tensor, depth: tuple, extr: tuple,
                       intr: tuple, clip_value: int = -1, clip: bool = True,
                       drop_names: bool = True):
    names = coord_3d.names
    B, K, P, _ = coord_3d.shape

    pc = coord_3d.rename(None).reshape(-1, 3)
    cam, proj_depth = tuple(zip(*(
        batchwise_project_onto_cam(pc, d, i, e, clip_value=clip_value,
                                   clip=clip)
        for d, i, e in zip(depth, intr, extr))))

    if drop_names:
        cam = tuple(c.reshape(B, K, P, 2) for c in cam)
    else:
        cam = tuple(c.reshape(B, K, P, 2).refine_names(*names) for c in cam)

    return cam, proj_depth


def named_repeat(named_tensor: torch.tensor, repeats: tuple) -> torch.tensor:
    names = named_tensor.names
    repeated = named_tensor.rename(None).repeat(*repeats)

    return repeated.refine_names(*names)


def uniform_sample(batch_size, n_keypoints, n_particles, x_min=-1, x_max=1,
                   y_min=-1, y_max=1, z_min=0.75, z_max=1.75):
    x = torch.rand((batch_size, n_keypoints, n_particles))
    y = torch.rand((batch_size, n_keypoints, n_particles))
    z = torch.rand((batch_size, n_keypoints, n_particles))

    # scale standard particle distribution to scene size
    x = x * (x_max - x_min) - x_min
    y = y * (y_max - y_min) - y_min
    z = z + (z_max - z_min) * z_min

    return torch.stack((x, y, z), dim=-1).to(device)
    # .refine_names('B', 'K', 'P', 'D')  # Batch, keypoint, particle, dim


def sample_from_obs(depth: tuple, extr: tuple, intr: tuple, diffs: tuple,
                    samples_total: int, depth_model_sigma_rel: float,
                    depth_model_sigma_abs: float, axes=None, re_axes=None
                    ) -> torch.tensor:

    n_cams = len(depth)
    samples_per_cam = int(samples_total / n_cams)

    _, H, W = depth[0].shape
    B, K, H2, W2 = diffs[0].shape

    diffs_flat = tuple(d.view(B, K, H2*W2) for d in diffs)
    softmax = torch.nn.Softmax(dim=2)
    softmax_activations = tuple(softmax(d) for d in diffs_flat)

    distr = tuple(torch.distributions.categorical.Categorical(probs=a)
                  for a in softmax_activations)
    samples = tuple(d.sample((samples_per_cam,)).to(device) for d in distr)

    # samples have shape n_samples, B, K and are in [0, H2*W2], fix order,
    # expand uv and map to [-1, 1]. Shape will be B, K, n_samples, 2.
    samples = tuple(s.movedim(0, 2) for s in samples)

    samples = tuple(
        torch.stack((s % W2 * (W / W2),                                    # x
                     torch.div(s, W2, rounding_mode="floor") * (H / H2)),  # y
                    dim=-1)
        for s in samples)

    if axes is not None:
        axes[0][0].imshow(depth[0][0].cpu())
        # axes[0][1].imshow(depth[1][0].cpu())
        axes[0][0].scatter(
            samples[0].cpu()[0, 0, :, 0], samples[0].cpu()[0, 0, :, 1],
            color='r')
        # axes[0][1].scatter(
        #     samples[1].cpu()[0, 0, :, 0], samples[1].cpu()[0, 0, :, 1],
        #     color='r')

    if re_axes is not None:
        for i in range(K):
            re_axes.flat[i].imshow(diffs[0][0][i].cpu(), interpolation='none')
            re_axes.flat[i].scatter(
                samples[0].cpu()[0, i, :, 0], samples[0].cpu()[0, i, :, 1],
                color='r')


    # project into scene
    depth = tuple(d.unsqueeze(1).repeat(1, K, 1, 1).reshape(B*K, H, W)
                  for d in depth)
    extr = tuple(e.unsqueeze(1).repeat(1, K, 1, 1).reshape(B*K, 4, 4)
                 for e in extr)
    intr = tuple(i.unsqueeze(1).repeat(1, K, 1, 1).reshape(B*K, 3, 3)
                 for i in intr)

    samples = tuple(s.reshape(B * K, samples_per_cam, 2).long()
                    for s in samples)
    world_coordinates = tuple(
        noisy_pixel_coordinates_to_world(s, d, e, i, depth_model_sigma_rel,
                                         depth_model_sigma_abs)
        for s, d, e, i in zip(samples, depth, extr, intr))
    world_coordinates = tuple(c.reshape(B, K, samples_per_cam, 3)
                              for c in world_coordinates)

    world_coordinates = torch.cat(world_coordinates, dim=2)

    if axes is not None:
        wc = world_coordinates.cpu()
        axes[1][0].scatter(
            wc[0, 0, :, 0], wc[0, 0, :, 1], wc[0, 0, :, 2], color='r')
        axes[1][0].scatter(
            wc[0, 0, :, 0].mean(), wc[0, 0, :, 1].mean(),
            wc[0, 0, :, 2].mean(), color='b')

    return world_coordinates
    # .refine_names('B', 'K', 'P', 'D')  # Batch, keypoint, particle, dim


def refine_with_prior(coordinates: torch.tensor, ref_coordinates: torch.tensor,
                      consistency_alpha: float):
    B, K, P, D = coordinates.shape

    ref_coordinates = ref_coordinates.unsqueeze(0).repeat(B, 1, 1)

    ref_dist = torch.functional.F.pairwise_distance(
        ref_coordinates[:, :, None].expand(1, K, K, D).reshape((-1, D)),
        ref_coordinates[:, None].expand(1, K, K, D).reshape(-1, D)
        ).reshape(1, K, K)

    current_mean = coordinates.mean(dim=2)

    current_dist = torch.functional.F.pairwise_distance(
        current_mean[:, :, None].expand(B, K, K, D)[
            :, :, :, None].expand(B, K, K, P, D).reshape(-1, D),
        coordinates[:, :, None].expand(B, K, K, P, D).reshape(-1, D)
    ).reshape(B, K, K, P)

    cons_likelihood = torch.exp(-consistency_alpha * torch.sum(
        torch.abs(ref_dist.unsqueeze(3) - current_dist), dim=1))

    return cons_likelihood


def add_gaussian_noise(coordinates: torch.tensor,
                       noise_scale: typing.Union[float, torch.tensor],
                       skip_z: bool = False):
    gauss = torch.distributions.normal.Normal(0, noise_scale)
    noise = gauss.sample(coordinates.shape).to(device)

    return coordinates + noise


def apply_motion_randomly(coordinates: torch.tensor, delta: torch.tensor,
                          prob: float):

    random_mask = torch.rand(coordinates.shape[:-1], device=device) < prob

    return coordinates + random_mask.unsqueeze(3) * delta


def get_pairwise_distance(descriptors: torch.tensor):
    K, D = descriptors.shape

    # compute pairwise distance matrix of kp means
    # mean[:, :, None].expand(B, K, K, D) is equivalent
    # to mean.unsqueeze(2).repeat(1, 1, K, 1), but faster
    dist = torch.functional.F.pairwise_distance(
        descriptors[:, None].expand(K, K, D).reshape((-1, D)),
        descriptors[None].expand(K, K, D).reshape(-1, D)).reshape(K, K)

    dist.fill_diagonal_(0)

    return dist


# https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
# Systematic sampling does an excellent job of ensuring we sample from all
# parts of the particle space while ensuring larger weights are proportionality
# resampled more often. Stratified resampling is not quite as uniform as
# systematic resampling, but it is a bit better at ensuring the higher weights
# get resampled more.
# As I have relatively few particles and a expect a relatively sparse
# hypothesis space, I go with stratified resampling for now.
# If that does not work, I might revisit systematic resampling.

def stratified_resample(weights: torch.tensor) -> torch.tensor:
    n_particles = weights.shape[-1]
    positions = (torch.rand_like(weights) +
                 torch.arange(n_particles, device=device)
                 ) / n_particles

    indeces = torch.empty_like(positions, dtype=torch.double)
    cumulative_sum = torch.cumsum(weights, dim=-1)

    for p in range(n_particles):
        p_smaller = positions[..., p].unsqueeze(-1) < cumulative_sum
        idx = torch.arange(p_smaller.shape[-1], 0, -1, device=device)
        first_smaller_idx = torch.argmax(p_smaller * idx, dim=-1)

        indeces[..., p] = first_smaller_idx

    return indeces.long()

    # Non-batched version
    # Adapted from: https://github.com/rlabbe/filterpy/blob/master/filterpy
    # positions = (torch.rand((n_particles, )) +
    #              torch.arange(n_particles)) / n_particles

    # indeces = torch.empty_like(positions, dtype=torch.int)
    # cumulative_sum = torch.cumsum(weights)

    # i, j = 0, 0

    # while i < n_particles:
    #     if positions[i] < cumulative_sum[j]:
    #         indeces[i] = j
    #         i += 1
    #     else:
    #         j += 1

    # return indeces


def systematic_resample(weights: torch.tensor) -> torch.tensor:
    n_particles = weights.shape[-1]

    positions = (
        torch.rand_like(weights[:, :, 0]).unsqueeze(-1).repeat(
            1, 1, n_particles) + torch.arange(
                n_particles, device=device).unsqueeze(0).unsqueeze(0).repeat(
                    weights.shape[0], weights.shape[1], 1)
                 ) / n_particles

    indeces = torch.empty_like(positions, dtype=torch.double)
    cumulative_sum = torch.cumsum(weights, dim=-1)

    for p in range(n_particles):
        p_smaller = positions[..., p].unsqueeze(-1) < cumulative_sum
        idx = torch.arange(p_smaller.shape[-1], 0, -1, device=device)
        first_smaller_idx = torch.argmax(p_smaller * idx, dim=-1)

        indeces[..., p] = first_smaller_idx

    return indeces.long()
