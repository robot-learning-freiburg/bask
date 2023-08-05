import numpy as np
import torch
from loguru import logger

import encoder.keypoints
import encoder.representation_learner
import models.keypoints.keypoints as keypoints
import models.keypoints.model_based_vision as model_based_vision
from utils.constants import SampleTypes
# from utils.debug import nan_hook, summarize_tensor
from utils.select_gpu import device
from utils.torch import (append_depth_to_uv, batched_project_onto_cam,
                         batched_rigid_transform, conjugate_quat, eye_like,
                         hard_pixels_to_3D_world, quaternion_multiply,
                         quaternion_to_matrix)
# from utils.torch import conjugate_quat_realfirst as conjugate_quat
# from utils.torch import quaternion_multiply_realfirst as quaternion_multiply
# from utils.torch import quaternion_to_matrix_realfirst as quaternion_to_matrix
from viz.image_single import image_with_points_overlay_uv_list
from viz.surface import depth_map_with_points_overlay_uv_list, scatter3d

KeypointsTypes = keypoints.KeypointsTypes
ProjectionTypes = encoder.keypoints.ProjectionTypes


class GTKeypointsPredictor(encoder.keypoints.KeypointsPredictor):

    sample_type = SampleTypes.DC

    def __init__(self, config=None):
        encoder_config = config["encoder"]
        self.config = encoder_config
        self.pretrain_config = config.get("pretrain", {}).get(
            "training_config", {})

        encoder.representation_learner.RepresentationLearner.__init__(
            self, config=config)

        self.n_keypoints = self.get_no_keypoints()

        self.only_use_first_emb = True  # with 3D or ego proj only need one cam

        if "image_size" in config and config["image_size"] is not None:
            image_size = config["image_size"]
        else:
            image_size = (256, 256)

        self.image_height, self.image_width = image_size
        self.descriptor_dimension = encoder_config["descriptor_dim"]
        self.keypoint_dimension = 2 if encoder_config["projection"] is \
            ProjectionTypes.NONE else 3

        self.debug_kp_selection = encoder_config["debug"]

        self.register_buffer('ref_pixels_uv',
                             torch.Tensor(2, self.n_keypoints))
        self.register_buffer('ref_pixel_world',
                             torch.Tensor(self.n_keypoints, 3))

        # TODO: make generic in n objects.
        # NOTE: Still matters? Gets overwritten in kp selection anyway? Might
        # have only been a problem because the kp selection wasnt stored.
        n_obj = 2 # 2
        self.register_buffer('ref_object_pose', torch.Tensor(n_obj, 7))
        self.register_buffer('ref_depth', torch.Tensor(self.image_height,
                                                       self.image_width))
        self.register_buffer('ref_int', torch.Tensor(3, 3))
        self.register_buffer('ref_ext', torch.Tensor(4, 4))

    def encode(self, camera_obs, full_obs=None):
        ZERO_VAL = torch.tensor(-1, dtype=torch.float32,
                                device=camera_obs.device)

        if full_obs is None:
            raise ValueError("Need full obs for this encoder.")

        depth = full_obs.cam_d
        extrinsics = full_obs.cam_ext
        intrinsics = full_obs.cam_int
        object_pose = full_obs.object_pose

        kp, descriptor, distance, kp_raw_2d = self.compute_keypoints(
            camera_obs, depth, extrinsics, intrinsics, object_pose)

        if not self.only_use_first_emb and full_obs.cam_rgb2 is not None:
            kp2, descriptor2, distance2, kp_raw_2d2 = self.compute_keypoints(
                full_obs.cam_rgb2, full_obs.cam_d2, full_obs.cam_ext2,
                full_obs.cam_int2, object_pose)

            if self.config.get("overshadow_keypoints"):
                dist_per_cam = torch.stack((distance, distance2))
                best_cam = torch.argmin(dist_per_cam, dim=0)
                # repeat to fit size of kp-tensor which has x_comps, y_comps
                best_cam = best_cam.repeat(1, self.keypoint_dimension)

                kp = torch.where(best_cam == 0, kp, ZERO_VAL)
                kp2 = torch.where(best_cam == 1, kp2, ZERO_VAL)

            kp = torch.cat((kp, kp2), dim=-1)
            kp_raw_2d = (kp_raw_2d, kp_raw_2d2)
            descriptor = (descriptor, descriptor2)
            distance = (distance, distance2)
            depth = (depth, full_obs.cam_d2)

        if threshold := self.config.get("threshold_keypoint_dist"):
            # repeat to fit size of kp-tensor which has x_comps, y_comps
            n_cams = 1 if full_obs.cam_rgb2 is None else 2
            dist_per_cam = distance.chunk(n_cams, dim=-1)
            expanded_dist = torch.cat(
                tuple(d.repeat(1, self.keypoint_dimension)
                      for d in dist_per_cam), dim=-1)
            kp = torch.where(expanded_dist > threshold, ZERO_VAL, kp)

        info = {"descriptor": descriptor,
                "distance": distance,
                "kp_raw_2d": kp_raw_2d,
                "depth": depth,
                }

        return kp, info

    def compute_keypoints(self, camera_obs, depth, extrinsics, intrinsics,
                          cur_object_pose,
                          ref_pixel_world=None, ref_object_pose=None,
                          projection=None):
        # Can pass other reference pixel, pose than the GT one saved at kp
        # sampling. If None provided, will use those.
        if ref_pixel_world is None:
            ref_pixel_world = self.ref_pixel_world
        if ref_object_pose is None:
            ref_object_pose = self.ref_object_pose

        # intrinsics[:, 0, 0] = -intrinsics[:, 0, 0]
        # intrinsics[:, 1, 1] = -intrinsics[:, 1, 1]

        B = camera_obs.shape[0]
        n_objs = ref_object_pose.shape[0]

        kp_x = []
        kp_y = []
        kp_z = []
        kp_world = []

        ref_pixel_world = ref_pixel_world.chunk(n_objs, dim=0)

        projection = self.config["projection"]

        # Iterate over objects as their relative poses can change.
        for i in range(n_objs):
            # Move them by the pose difference of the object between the ref
            # pose and current pose (poses change between trajectories).
            ref_shift = ref_object_pose[i, :3]
            ref_rot = ref_object_pose[i, 3:7]
            cur_shift, cur_rot = cur_object_pose[:, i, :3], \
                cur_object_pose[:, i, 3:7]

            quat_diff = quaternion_multiply(
                conjugate_quat(ref_rot).repeat(B, 1), cur_rot)
            rel_rot_matrix = quaternion_to_matrix(quat_diff)

            move_back = eye_like(extrinsics)
            move_back[:, :3, 3] = - ref_shift
            rel_rot = eye_like(extrinsics)
            rel_rot[:, :3, :3] = rel_rot_matrix
            move_forth = eye_like(extrinsics)
            move_forth[:, :3, 3] = cur_shift
            ref_to_cur = torch.matmul(move_forth,
                                      torch.matmul(rel_rot, move_back))

            cur_pixel_world = batched_rigid_transform(
                ref_pixel_world[i], ref_to_cur)

            clip = projection != ProjectionTypes.EGO

            # Project the new points onto the current camera
            cur_pixel_cam, cur_pixel_depth = batched_project_onto_cam(
                cur_pixel_world, depth, intrinsics, extrinsics,
                clip=clip, get_depth=True)

            kp_world.append(cur_pixel_world)

            kp_x.append(cur_pixel_cam[:, :, 0])
            kp_y.append(cur_pixel_cam[:, :, 1])
            kp_z.append(cur_pixel_depth)

        kp = torch.cat((torch.cat(kp_x, dim=-1),
                        torch.cat(kp_y, dim=-1)), dim=-1).to(device)
        kp = kp/128 - 1  # TODO: normalize keypoints earlier in process?
        # TODO: generalize for other cam resolutions

        kp_z = torch.cat(kp_z, dim=-1).to(device)

        kp_world = torch.cat(kp_world, dim=1).to(device)

        distance = torch.zeros(B, self.n_keypoints)

        kp_raw_2d = kp

        if projection == ProjectionTypes.NONE:
            pass
        elif projection in (ProjectionTypes.UVD, ProjectionTypes.EGO):
            kp = torch.cat((kp, kp_z), dim=-1)
        elif projection == ProjectionTypes.GLOBAL_HARD:
            kp = kp_world.permute(0, 2, 1).reshape((B, -1))
        elif projection == ProjectionTypes.LOCAL_HARD:
            # create identity extrinsics
            extrinsics = torch.zeros_like(extrinsics)
            extrinsics[:, range(4), range(4)] = 1
            kp = hard_pixels_to_3D_world(
                kp, kp_z, extrinsics, intrinsics,
                self.image_width - 1, self.image_height - 1)
        else:
            raise NotImplementedError

        return kp, torch.zeros_like(camera_obs), distance, kp_raw_2d

    def initialize_parameters_via_dataset(self, replay_memory):
        self.select_reference_descriptors(replay_memory)

    def select_reference_descriptors(self, replay_memory, traj_idx=0, img_idx=0,
                                     object_labels=None, cam="wrist"):
        # traj_idx = 1  # 0
        # img_idx = 40 # 10  # for microwave: 25
        rgb, depth, mask, intr, ext, object_pose = \
            replay_memory.sample_data_point_with_ground_truth(
                cam=cam, img_idx=img_idx, traj_idx=traj_idx)

        self.ref_depth = depth.to(device)
        self.ref_object_pose = object_pose.to(device)
        self.ref_int = intr.to(device)
        self.ref_ext = ext.to(device)

        # dummy tensor
        descriptor = \
            torch.zeros((1, self.descriptor_dimension,
                         self.image_height, self.image_width))

        object_labels = object_labels or replay_memory.object_labels

        n_keypoints_total = self.config["keypoints"]["n_sample"]
        manual_kp_selection = self.pretrain_config.get('manual_kp_selection')

        if manual_kp_selection:
            n_prev_frames = 20
            preview_frames = replay_memory.sample_bc(
                1, cam=(cam,)).cam_rgb.squeeze(1)
            indeces = np.linspace(
                start=0, stop=preview_frames.shape[0] - 1, num=n_prev_frames)
            indeces = np.round(indeces).astype(int)
            preview_frames = preview_frames.index_select(
                dim=0, index=torch.tensor(indeces))

            preview_descr = \
                torch.zeros((n_prev_frames, self.descriptor_dimension,
                             self.image_height, self.image_width))
        else:
            preview_frames = None
            preview_descr = None

        # ref_pixels_uv, reference_descriptor_vec = \
        #     self.sample_keypoints(rgb, descriptor, mask,
        #                           object_labels, n_keypoints_total)

        self.ref_pixels_uv, self._reference_descriptor_vec = \
            self._select_reference_descriptors(
                rgb, descriptor, mask, object_labels, n_keypoints_total,
                manual_kp_selection, preview_frames, preview_descr)

        ref_pixels_stacked = torch.cat((self.ref_pixels_uv[0],
                                        self.ref_pixels_uv[1]), dim=-1)

        # Map the reference pixel to world coordinates.
        ref_pixel_world = model_based_vision.raw_pixels_to_3D_world(
            ref_pixels_stacked.unsqueeze(0), self.ref_depth.unsqueeze(0),
            self.ref_ext.unsqueeze(0),
            self.ref_int.unsqueeze(0))

        # Shape is (1, 3*k). Reshape to (k, 3)
        self.ref_pixel_world = torch.stack(
            ref_pixel_world.squeeze(0).chunk(3, dim=-1), dim=1).to(device)

        if self.debug_kp_selection:
            if self.keypoint_dimension == 2:
                image_with_points_overlay_uv_list(
                    rgb.cpu(),
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
                    mask=mask.cpu().numpy(), object_labels=object_labels,
                    # object_poses=self.ref_object_pose.cpu().numpy()
                    )
                local_ref = model_based_vision.raw_pixels_to_camera_frame(
                    ref_pixels_stacked.unsqueeze(0), self.ref_depth.unsqueeze(0),
                    self.ref_int.unsqueeze(0))
                # u_indeces = torch.arange(0, 256)
                # v_indeces = torch.arange(0, 256)
                # depth_indeces =
                # local_depth = model_based_vision.raw_pixels_to_camera_frame(
                #     depth_indeces, self.ref_depth.unsqueeze(0),
                #     self.ref_int.unsqueeze(0))
                scatter3d(local_ref[0].cpu().numpy())
                scatter3d(self.ref_pixel_world.cpu().numpy())
            else:
                raise ValueError("No viz for {}d keypoints.".format(
                    self.keypoint_dimension))

    def from_disk(self, chekpoint_path, force_read=False):
        if not force_read:
            logger.info(
                "  GT Keypoints encoder does not need snapshot loading."
                "Skipping.")
        else:
            logger.info(
                "  Force-reading the GT Keypoints encoder from disk. "
                "Should only be needed to preserve the reference selection "
                "in bc if you used embed_trajectories.")

            state_dict = torch.load(chekpoint_path, map_location='cpu')
            missing, unexpected = self.load_state_dict(state_dict,
                                                       strict=False)
            if missing:
                logger.warning("Missing keys: {}".format(missing))
            if unexpected:
                logger.warning("Unexpected keys: {}".format(unexpected))
            self = self.to(device)

    @classmethod
    def get_latent_dim(self, config, n_cams=1, image_dim=None):
        return encoder.keypoints.KeypointsPredictor.get_latent_dim(config,
                                                                   n_cams=1)
