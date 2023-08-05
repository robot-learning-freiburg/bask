import hashlib
from functools import lru_cache

import numpy as np
import torch
from loguru import logger

from utils.quat_realfirst import (conjugate_quat, modulo_rotation_angle,
                                  quaternion_raw_multiply, rotate_quat_y180)
from utils.torch import (axis_angle_to_matrix, axis_angle_to_quaternion,
                         batched_block_diag, cat,
                         get_b_from_homogenous_transforms,
                         get_R_from_homogenous_transforms, hom_to_shift_quat,
                         homogenous_transform_from_rot_shift,
                         identity_quaternions, invert_homogenous_transform,
                         list_or_tensor, list_or_tensor_mult_args,
                         quarter_rot_angle, quaternion_is_unit)
from utils.torch import quaternion_to_matrix_realfirst as quaternion_to_matrix
from utils.torch import stack, standardize_quaternion, to_numpy
from viz.quaternion import plot_quat_components


@list_or_tensor
def rotate_frames_180degrees(quats, rot_idx=None, skip_idx=None, axis='y'):
    """
    Rotate a set of quaternions (representing frame orientation) by 180 degrees
    around the specified axis.

    Parameters
    ----------
    quats : torch.Tensor
        Quaternions of shape (n_frames, ..., 4).
    axis : str, optional
        Axis to rotate around, by default 'y'.
    rot_idx : list[int]/None, optional
        Indices of the frames to rotate, by default None.
    skip_idx : list[int]/None, optional
        Indices of the frames to skip, by default None.

    Either rot_idx or skip_idx must be be a list of indices, the other must be
    None.

    Returns
    -------
    torch.Tensor
        Rotated quaternions.
    """
    if axis != 'y':
        raise NotImplementedError

    shape = quats.shape[0]

    all_frames = list(range(shape))

    if rot_idx is not None:
        assert skip_idx is None
        skip_idx = [i for i in all_frames if i not in rot_idx]

    elif skip_idx is not None:
        rot_idx = [i for i in all_frames if i not in skip_idx]

    to_keep = quats[skip_idx]
    to_rotate = quats[rot_idx]

    rotated = rotate_quat_y180(to_rotate)

    res = []
    k, r = 0, 0
    k_max, r_max = len(to_keep), len(rotated)
    for i in range(shape):
        if k < k_max and i == skip_idx[k]:
            res.append(to_keep[k])
            k += 1
        elif r <= r_max:
            assert i == rot_idx[r]
            res.append(rotated[r])
            r += 1
        else:
            raise ValueError

    return torch.stack(res)


def configurable_rotate_frames(quats, enforce_z_down, enforce_z_up,
                               with_init_ee_pose, with_world_frame):
    """
    Homogenize the frame orientation based on the given parameters.
    If enforce_z_down is True, the z-axis of the frames will point down,
    if enforce_z_up is True, the z-axis of the frames will point up.
    If neither, do nothing.

    Parameters
    ----------
    quats : torch.Tensor/tuple[torch.Tensor]
        Quaternions of shape (n_frames, ..., 4).
    enforce_z_down : bool
        Enforce that the z-axis of the frames is pointing down.
    enforce_z_up : bool
        Enforce that the z-axis of the frames is pointing up.
    with_world_frame : bool
        Wether the first frame is the world frame.
    with_init_ee_pose : bool
        Wether the the ee_pose_frame is included (first frame if without
        world frame else second frame).

    Returns
    -------
    torch.Tensor/tuple[torch.Tensor]
        Rotated quaternions.
    """
    ee_idx = 1 if with_world_frame else 0

    if enforce_z_down:  # rotate all frames, but ee_pose_frame
        assert not enforce_z_up
        rot_idx = None
        skip_idx = [ee_idx] if with_init_ee_pose else []
    elif enforce_z_up:  # only rotate ee_pose_frame
        rot_idx = [ee_idx] if with_init_ee_pose else []
        skip_idx = None
    else:
        return quats

    return rotate_frames_180degrees(quats, rot_idx, skip_idx)


class Demos():
    """
    Convenience class to store and access demonstrations for gmm models.
    """

    def __init__(self, trajectories, add_init_ee_pose_as_frame=True,
                 add_world_frame=True, meta_data=None,
                 enforce_z_down=False, enforce_z_up=True,
                 modulo_object_z_rotation=True):
        """
        Extract information from a list of BCTrajectories.

        Parameters
        ----------
        trajectories : list[BCTrajectory]
            List of trajectories.
        add_init_ee_pose_as_frame : bool, optional
            Add the initial EE pose as a frame, by default True.
        add_world_frame : bool, optional
            Add a world frame, by default True.
        meta_data : dict, optional
            Meta data for efficient hashing of the class, by default None.
        enforce_z_down : bool, optional
            Enforce that the z-axis of the frames is pointing down, by default
            True.
        enforce_z_up : bool, optional
            Enforce that the z-axis of the frames is pointing up, by default

        enforce_z_down and enforce_z_up can be used to homogenize frame
        orientation as the EE frame points down, while object and world frames
        point up.
        """
        assert not (enforce_z_down and enforce_z_up)

        self.meta_data = {} if meta_data is None else meta_data
        self.meta_data['add_init_ee_pose_as_frame'] = add_init_ee_pose_as_frame
        self.meta_data['add_world_frame'] = add_world_frame
        self.meta_data['enforce_z_down'] = enforce_z_down
        self.meta_data['enforce_z_up'] = enforce_z_up

        # Add the EE frame as the first frame. As this will be the obervation
        # we remove it later. However, it needs the same transformations, so
        # we add it here temporarily.
        if type(trajectories[0].object_poses) is torch.Tensor:
            logger.warning(
                "Legacy support for non-named objects. Auto-naming frames.")
            for t in trajectories:
                t.object_poses = {
                    f'frame_{i}': v for i, v in enumerate(
                    t.object_poses.swapdims(0, 1))}

        frame_poses = tuple(
            torch.stack([o.ee_pose] + [v for _, v in o.object_poses.items()])
            for o in trajectories)

        if modulo_object_z_rotation:
            frame_poses = modulo_rotation_angle(
                frame_poses, quarter_rot_angle, 2, skip_first=True)

        self.n_trajs = len(frame_poses)

        self.world2frames = []
        self.world2frames_velocities = []
        self.frames2world = []
        self.frames2world_velocities = []
        self.ee_poses = []
        self.ee_poses_raw = tuple(o.ee_pose for o in trajectories)
        self.ee_quats = tuple(o[..., 3:] for o in self.ee_poses_raw)

        frame_quats = []
        if add_world_frame:
            frame_quats.append(
                tuple(identity_quaternions(o[0:1, :, 0].shape)
                      for o in frame_poses))
        if add_init_ee_pose_as_frame:
            frame_quats.append(tuple(
                o[0].unsqueeze(0).unsqueeze(0).repeat(1, o.shape[0], 1)
                for o in self.ee_quats))
        frame_quats.append(tuple(o[1:, :, 3:] for o in frame_poses))
        self.frame_quats = tuple(torch.cat(o) for o in zip(*frame_quats))

        self.frame_quats = configurable_rotate_frames(
            self.frame_quats, enforce_z_down, enforce_z_up,
            add_init_ee_pose_as_frame, add_world_frame)

        # Convert the reference frames and EE pose into homogeneous transforms.
        for i in range(self.n_trajs):
            frame_poses_i = frame_poses[i]
            n_frames, n_steps, len_quat = frame_poses_i.shape
            assert len_quat == 7

            frame_poses_i_b = frame_poses_i[:, :, :3]  # position
            frame_poses_i_q = frame_poses_i[:, :, 3:]  # quaternion

            # First frame in frame_poses_i_q is the EE pose, see line 165ff.
            # So, need to skip it and rotate the rest. Can achieve this by
            # setting with_init_ee_pose to True and with_world_frame to False.
            frame_poses_i_q = configurable_rotate_frames(
                frame_poses_i_q, enforce_z_down, enforce_z_up,
                True, False)

            assert quaternion_is_unit(frame_poses_i_q)
            frame_poses_i_q = standardize_quaternion(frame_poses_i_q)
            # assert quaternion_is_standard(frame_poses_i_q)

            f_b = frame_poses_i_b.reshape(-1, 3)
            f_A = torch.Tensor(
                quaternion_to_matrix(frame_poses_i_q.reshape(-1, 4)))

            world2frame = get_frame_transform_flat(f_A, f_b).reshape(
                n_frames, n_steps, 4, 4)
            world2frame_vel = get_frame_transform_flat(
                f_A, torch.zeros_like(f_b)).reshape(n_frames, n_steps, 4, 4)
            frame2world = get_frame_transform_flat(
                f_A, f_b, invert=False).reshape(n_frames, n_steps, 4, 4)
            frame2world_vel = get_frame_transform_flat(
                f_A, torch.zeros_like(f_b), invert=False).reshape(
                    n_frames, n_steps, 4, 4)

            # Pop out the EE pose
            ee2world = frame2world[0, :, :, :].clone()
            self.ee_poses.append(ee2world)

            # Add world frame and or initial EE pose as frame
            id_frame = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(
                1, n_steps, 1, 1)
            ee_frame = ee2world[0].unsqueeze(0).unsqueeze(0).repeat(
                1, n_steps, 1, 1)
            ee_frame_vel = ee_frame.clone()
            ee_frame_vel[:, :, :3, 3] = 0  # Zero frame velocity

            list_world2frames = []
            list_frames2world = []
            list_world2frames_vel = []
            list_frames2world_vel = []
            if add_world_frame:
                list_world2frames.append(id_frame.clone())
                list_frames2world.append(id_frame.clone())
                list_world2frames_vel.append(id_frame.clone())
                list_frames2world_vel.append(id_frame.clone())
            if add_init_ee_pose_as_frame:
                list_world2frames.append(invert_homogenous_transform(ee_frame))
                list_frames2world.append(ee_frame)
                list_world2frames_vel.append(invert_homogenous_transform(ee_frame_vel))
                list_frames2world_vel.append(ee_frame_vel)

            list_frames2world.append(frame2world[1:, :, :, :])
            list_world2frames.append(world2frame[1:, :, :, :])
            list_frames2world_vel.append(frame2world_vel[1:, :, :, :])
            list_world2frames_vel.append(world2frame_vel[1:, :, :, :])

            frame2world = torch.cat(list_frames2world, dim=0)
            world2frame = torch.cat(list_world2frames, dim=0)
            frame2world_vel = torch.cat(list_frames2world_vel, dim=0)
            world2frame_vel = torch.cat(list_world2frames_vel, dim=0)

            self.world2frames.append(world2frame)
            self.frames2world.append(frame2world)
            self.world2frames_velocities.append(world2frame_vel)
            self.frames2world_velocities.append(frame2world_vel)

        self.world2frames = tuple(self.world2frames)
        self.world2frames_velocities = tuple(self.world2frames_velocities)
        self.frames2world = tuple(self.frames2world)
        self.frames2world_velocities = tuple(self.frames2world_velocities)
        self.ee_poses = tuple(self.ee_poses)

        self.frame_names = []
        if add_world_frame:
            self.frame_names.append('world')
        if add_init_ee_pose_as_frame:
            self.frame_names.append('ee_init')
        self.frame_names += [k for k, _ in trajectories[0].object_poses.items()]
        self.frame_names = tuple(self.frame_names)

        self.min_traj_len = min(
            [self.world2frames[j][0].shape[0] for j in range(self.n_trajs)])
        self.mean_traj_len = int(np.mean(
            [self.world2frames[j][0].shape[0] for j in range(self.n_trajs)]))

        actions = [o.action for o in trajectories]
        actions_hom = []
        actions_quats = []

        # Actions are EE-delta, EE-rotation (axis-angle) and gripper action.
        # Convert rotation into the homogeneous transforms as well to simplify
        # projections into local frames.
        self.gripper_actions = tuple(a[:, 6] for a in actions)
        for i in range(self.n_trajs):
            n_steps, len_action = actions[i].shape
            assert len_action == 7
            a_A = axis_angle_to_matrix(actions[i][:, 3:6].reshape(-1, 3))
            a_b = actions[i][:, :3].reshape(-1, 3)
            actions_hom.append(get_frame_transform_flat(
                a_A, a_b, invert=False).reshape(n_steps, 4, 4))

            actions_quats.append(
                axis_angle_to_quaternion(actions[i][:, 3:6].reshape(-1, 3)))

        self.ee_actions = tuple(actions_hom)
        self.ee_actions_quats = tuple(actions_quats)

        if not add_world_frame:
            n_frames -= 1
        if add_init_ee_pose_as_frame:
            n_frames += 1

        self.n_frames = n_frames

        self.subsample_to_common_length()

    def _subsample(self, trajectories, indeces=None, dim=1):
        """
        Subsample a list of trajectories to a common length using the given
        indeces.
        If no indeces provided, defaults to the indeces in self._ss_idx,
        which are computed in subsample_to_common_length.

        Parameters
        ----------
        trajectories : list[torch.Tensor] or torch.Tensor
            The list of trajectories.
            Shape: (n_trajectories, n_frames, n_steps, ...)
        indeces : list[Iterable[int]], optional
            Subsampling indeces per trajectory.
            Shape: (n_trajectories, n_obserbations/ss_len)
        dim : int, optional
            The dimension along which to subsample.

        Returns
        -------
        torch.Tensor
            Stacked, subsampled trajectories.
        """

        if indeces is None:
            indeces = self._ss_idx

        # return torch.stack(
        #     [t[:, indeces[i], ...] for i, t in enumerate(trajectories)])

        subsampled = [
            torch.index_select(t, dim, indeces[i])
            for i, t in enumerate(trajectories)]

        return torch.stack(subsampled)

    def subsample_to_common_length(self, use_min=False):
        """
        Subsample the trajectories to a common length.
        This is needed for reprduction approach - I think.
        """
        target_len = self.min_traj_len if use_min else self.mean_traj_len
        self.ss_len = target_len

        logger.info('Subsampling to length {} using strategy {}-length.',
                    target_len, 'min' if use_min else 'mean')

        indeces = [
            get_idx_by_target_len(self.world2frames[i][1].shape[0], target_len)
            for i in range(self.n_trajs)]

        self._ss_idx = indeces

        self.stacked_world2frames = self._subsample(self.world2frames, indeces)
        self.stacked_world2frame_velocities = \
            self._subsample(self.world2frames_velocities, indeces)
        self.stacked_ee_actions = self._subsample(
            self.ee_actions, indeces, dim=0)
        self.stacked_ee_poses = self._subsample(self.ee_poses, indeces, dim=0)

        self.stacked_ee_quats = self._subsample(self.ee_quats, indeces, dim=0)
        self.stacked_frame_quats = self._subsample(
            self.frame_quats, indeces, dim=1)

        self.stacked_ee_actions_quats = self._subsample(
            self.ee_actions_quats, indeces, dim=0)

    @property
    def _world2frames_fixed(self):
        """
        Get world2frames transform for fixed coordinate frames.
        """
        return torch.stack([w2f[:, 0:1, :, :] for w2f in self.world2frames])

    @property
    def _frames2world_fixed(self):
        """
        Get frames2world transform for fixed coordinate frames.
        """
        return torch.stack([f2w[:, 0:1, :, :] for f2w in self.frames2world])

    @property
    def _world2frames_velocities_fixed(self):
        return torch.stack(
            [w2f[:, 0:1, :, :] for w2f in self.world2frames_velocities])

    @property
    def _frames2world_velocities_fixed(self):
        return torch.stack(
            [f2w[:, 0:1, :, :] for f2w in self.frames2world_velocities])

    @property
    def _frame_origins_fixed(self):
        """
        Get the origin of the fixed frames. Ie the frame2world transform.
        As homogenous transform.
        """
        return self._frames2world_fixed
        # w2fs = self._world2frames_fixed
        # shape = w2fs.shape
        # return invert_homogenous_transform(w2fs.reshape(-1, 4, 4)).reshape(
        #     *shape)

    @property
    def _frame_origins_fixed_wquats(self):
        """
        Get the origin of the fixed frames. Ie the frame2world transform.
        As position + quaternion.
        """

        frame_pos = self._frame_origins_fixed[..., 0:3, 3].squeeze(2)
        frame_quats = self._frame_quats2world_fixed.squeeze(2)

        return torch.cat([frame_pos, frame_quats], dim=2)

    @property
    def _frame_quats2world_fixed(self):
        return torch.stack([f2w[:, 0:1, :] for f2w in self.frame_quats])

    @property
    def _frame_quats2world(self):
        return self.frame_quats

    @property
    def _world_quats2frame(self):
        return conjugate_quat(self._frame_quats2world)

    @property
    def _world_quats2frame_fixed(self):
        return conjugate_quat(self._frame_quats2world_fixed)

    @property
    def _frame_quats2world_velocities(self):
        logger.info("Assuming zero frame velocity. Should be fixed.")
        return self._frame_quats2world

    @property
    def _world_quats2frame_velocities(self):
        return conjugate_quat(self._frame_quats2world_velocities)

    @property
    def _frame_quats2world_velocities_fixed(self):
        return torch.stack(
            [f2w[:, 0:1, :] for f2w in self._frame_quats2world_velocities])

    @property
    def _world_quats2frame_velocities_fixed(self):
        return conjugate_quat(self._frame_quats2world_velocities_fixed)

    @lru_cache
    def get_raw_traj(self, add_time_dim=False, add_action_dim=False,
                     position_only=False):
        """
        Get the raw trajectory. Ie the ee pose in the world frame.
        """
        obs = self.ee_poses_raw

        if position_only:
            obs = tuple([o[..., 0:3] for o in obs])

        if add_action_dim:
            action_pos = tuple([a[..., 0:3, 3] for a in self.ee_actions])

            if position_only:
                obs = tuple([torch.cat([o, p], dim=1) for o, p in zip(
                    obs, action_pos)])
            else:
                action_quat = self.ee_actions_quats

                obs = tuple([torch.cat([o, p, q], dim=1) for o, p, q in zip(
                    obs, action_pos, action_quat)])

        if add_time_dim:
            obs = add_time_dimension(obs)

        return obs

    @lru_cache
    def get_obs_per_frame(self, subsampled=False, fixed_frames=False,
                          as_quaternion=False, skip_quat_dim=None):
        """
        Project the EE pose into all coordinate frames.

        Parameters
        ----------
        subsampled : bool, optional
            If true, returns the trajectories subsampled to the same length.
            By default False. The ss strategy depends on the args passed to
            subsample_to_common_length.
        fixed_frame : bool, optional
            Wether to use the fixed coordinate frames, by default False.
        as_quaternion : bool, optional
            If true, returns the rotation as a quaternion, by default False.
        skip_quat_dim : int, optional
            If not None and as_quaternion is True, pops the given dimension of
            the quaternion.
        NOTE: the quaternion conversion is not tested for non-stacked/ss trajs.
        Probabaly makes problems.

        Returns
        -------
        torch.Tensor or list[torch.Tensor]
            The projected coordinate frames per trajectory and frame.
        """

        @lru_cache
        def _get_obs_per_frame(self, subsampled=False, fixed_frames=False):
            transforms = \
                self._world2frames_fixed if fixed_frames else self.world2frames

            if subsampled and not fixed_frames:
                transforms = self._subsample(transforms)

            poses = self.stacked_ee_poses if subsampled else self.ee_poses

            return get_obs_per_frame(transforms, poses)

        obs = _get_obs_per_frame(self, subsampled, fixed_frames)

        if as_quaternion:
            # obs = hom_to_shift_quat(obs, skip_quat_dim=skip_quat_dim,
            #                         prefer_positives=True)
            pos = get_b_from_homogenous_transforms(obs)
            rot = self.get_quat_obs_per_frame(subsampled=subsampled,
                fixed_frames=fixed_frames, skip_quat_dim=skip_quat_dim)

            # HACK: for obs, poses I have inconsistens dim orders.
            # In get_obs_per_frame this is fixed by a final permute.
            # Can't apply the same directly to get_quat_obs_per_frame because
            # it uses the list_or_tensor decorator. Need to properly fix this.
            if type(rot) in (list, tuple):
                rot = [r.permute(1, 0, 2) for r in rot]
            elif type(rot) is tuple:
                rot = tuple(r.permute(1, 0, 2) for r in rot)
            else:
                rot = rot.permute(0, 2, 1, 3)

            obs = cat(pos, rot, dim=-1)

        return obs

    @lru_cache
    def get_action_per_frame(self, subsampled=False, fixed_frames=False,
                             as_quaternion=True, skip_quat_dim=0):
        """
        Project the EE action (pose delta) into all coordinate frames.

        Parameters
        ----------
        subsampled : bool, optional
            If true, returns the trajectories subsampled to the same length.
            By default False. The ss strategy depends on the args passed to
            subsample_to_common_length.
        fixed_frame : bool, optional
            Wether to use the fixed coordinate frames, by default False.
        as_quaternion : bool, optional
            If true, returns the rotation as a quaternion, by default False.
        skip_quat_dim : int, optional
            If not None and as_quaternion is True, pops the given dimension of
            the quaternion.
        NOTE: the quaternion conversion is not tested for non-staked/ss trajs.
        Probabaly makes problems.

        Returns
        -------
        torch.Tensor or list[torch.Tensor]
            The projected coordinate frames per trajectory and frame.
        """

        @lru_cache
        def _get_action_per_frame(self, subsampled=False, fixed_frames=False):
            transforms = \
                self._world2frames_velocities_fixed if fixed_frames else \
                self.world2frames_velocities

            if subsampled and not fixed_frames:
                transforms = self._subsample(transforms)

            actions = self.stacked_ee_actions if subsampled else \
                self.ee_actions

            return get_obs_per_frame(transforms, actions)

        actions = _get_action_per_frame(self, subsampled, fixed_frames)

        if as_quaternion:
            # actions = hom_to_shift_quat(actions, skip_quat_dim=skip_quat_dim)
            pos = get_b_from_homogenous_transforms(actions)
            rot = self.get_quat_action_per_frame(subsampled=subsampled,
                fixed_frames=fixed_frames, skip_quat_dim=skip_quat_dim)

            # HACK (see function above)
            if type(rot) in (list, tuple):
                rot = [r.permute(1, 0, 2) for r in rot]
            elif type(rot) is tuple:
                rot = tuple(r.permute(1, 0, 2) for r in rot)
            else:
                rot = rot.permute(0, 2, 1, 3)

            actions = cat(pos, rot, dim=-1)

        return actions

    @lru_cache
    def get_quat_obs_per_frame(self, subsampled=False, fixed_frames=False,
                               skip_quat_dim=None):
        """
        Get the EE rotation in all coordinate frames - as a quaternion.
        Bypasses the conversion to homogenous transforms, thus preventing
        possible discontinuities.

        Parameters
        ----------
        subsampled : bool, optional
            If true, returns the trajectories subsampled to the same length.
            By default False. The ss strategy depends on the args passed to
            subsample_to_common_length.
        fixed_frame : bool, optional
            Wether to use the fixed coordinate frames, by default False.

        Returns
        -------
        torch.Tensor or list[torch.Tensor]
            Analog to get_obs_per_frame, but rotation only.
        """

        transforms = self._world_quats2frame_fixed if fixed_frames \
            else self._world_quats2frame

        if subsampled and not fixed_frames:
            transforms = self._subsample(transforms)

        poses = self.stacked_ee_quats if subsampled else self.ee_quats

        quats = get_quat_per_frame(transforms, poses)

        if skip_quat_dim is not None:
            quats = skip_quat_dim(quats, dim=skip_quat_dim)

        return quats

    @lru_cache
    def get_quat_action_per_frame(self, subsampled=False, fixed_frames=False,
                                  skip_quat_dim=None):
        """
        Get the EE rotation action in all coordinate frames - as a quaternion.

        Parameters
        ----------
        subsampled : bool, optional
            If true, returns the trajectories subsampled to the same length.
            By default False. The ss strategy depends on the args passed to
            subsample_to_common_length.
        fixed_frame : bool, optional
            Wether to use the fixed coordinate frames, by default False.

        Returns
        -------
        torch.Tensor or list[torch.Tensor]
            Analog to get_action_per_frame, but rotation only.
        """

        transforms = self._world_quats2frame_velocities_fixed if fixed_frames \
            else self._world_quats2frame_velocities

        if subsampled and not fixed_frames:
            transforms = self._subsample(transforms)

        actions = self.stacked_ee_actions_quats if subsampled else \
            self.ee_actions_quats

        quats = get_quat_per_frame(transforms, actions)

        if skip_quat_dim is not None:
            quats = skip_quat_dim(quats, dim=skip_quat_dim)

        return quats

    @lru_cache
    def get_x_per_frame(self, subsampled=True, fixed_frames=False, flat=False,
                        pos_only=False, as_quaternion=True, skip_quat_dim=0):
        """
        Get the position of the EE in all coordinate frames.
        Same as get_obs_per_frame, but only returns the position, not the full
        homogeneous transform.

        Parameters
        ----------
        subsampled : bool, optional
            If true, returns the trajectories subsampled to the same length.
            By default True.
        fixed_frames : bool, optional
            If true, uses the fixed coordinate frames. By default False.
        flat : bool, optional
            If True, flattens the output over the frames. By default False.
        pos_only : bool, optional
            If True, only returns the position, not the full transform.
            By default False.
        as_quaternion : bool, optional
            If true, returns the rotation as a quaternion, by default True.
        skip_quat_dim : int, optional
            If not None and as_quaternion is True, pops the given dimension of
            the quaternion. By default 0.

        Returns
        -------
        torch.Tensor or list[torch.Tensor]
        """

        @lru_cache  # Nested function to cache result independtly of flat-arg.
        def _get_x_per_frame(self, subsampled=True, fixed_frames=False):
            if pos_only:
                obs = self.get_obs_per_frame(subsampled, fixed_frames,
                                             False, skip_quat_dim)
                obs = get_b_from_homogenous_transforms(obs)
            else:
                obs = self.get_obs_per_frame(subsampled, fixed_frames,
                                             as_quaternion, skip_quat_dim)

            return obs

        x = _get_x_per_frame(self, subsampled, fixed_frames)

        if flat:
            return x.reshape(self.n_trajs, self.ss_len, -1)
        else:
            return x

    @lru_cache
    def get_dx_per_frame(self, subsampled=True, fixed_frames=False, flat=False,
                         pos_only=False, as_quaternion=True, skip_quat_dim=0):
        """
        Get the position delta of the EE in all coordinate frames.
        Same as get_action_per_frame, but only returns the position, not the
        full homogeneous transform.

        Parameters
        ----------
        subsampled : bool, optional
            If true, returns the trajectories subsampled to the same length.
            By default True.
        fixed_frames : bool, optional
            If true, uses the fixed coordinate frames. By default False.
        flat : bool, optional
            If True, flattens the output over the frames. By default False.
        pos_only : bool, optional
            If True, only returns the position, not the full transform.
            By default False.
        as_quaternion : bool, optional
            If true, returns the rotation as a quaternion, by default True.
        skip_quat_dim : int, optional
            If not None and as_quaternion is True, pops the given dimension of
            the quaternion. By default 0.

        Returns
        -------
        torch.Tensor or list[torch.Tensor]
        """

        @lru_cache  # Nested function to cache result independtly of flat-arg.
        def _get_dx_per_frame(self, subsampled=True, fixed_frames=False):
            if pos_only:
                actions = self.get_action_per_frame(subsampled, fixed_frames,
                                                    False, skip_quat_dim)
                actions = get_b_from_homogenous_transforms(actions)
            else:
                actions = self.get_action_per_frame(
                    subsampled, fixed_frames, as_quaternion, skip_quat_dim)

            return actions

        dx = _get_dx_per_frame(self, subsampled, fixed_frames)

        if flat:
            return dx.reshape(self.n_trajs, self.ss_len, -1)
        else:
            return dx

    @lru_cache
    def get_per_frame_data(self, subsampled=True, fixed_frames=False,
                           flat=False, numpy=False, pos_only=False,
                           as_quaternion=True, skip_quat_dim=0,
                           add_time_dim=False, add_action_dim=False):
        """
        Get the stacked position and position delta of the EE in all
        coordinate frames.
        """

        @lru_cache
        def _get_per_frame_data(self, subsampled=True, fixed_frames=False,
                                add_time_dim=False, add_action_dim=False):
            x = self.get_x_per_frame(subsampled, fixed_frames, False,
                                     pos_only, as_quaternion, skip_quat_dim)

            if add_action_dim:
                dx = self.get_dx_per_frame(subsampled, fixed_frames, False,
                                           pos_only, as_quaternion,
                                           skip_quat_dim)

                if subsampled:
                    obs = torch.cat((x, dx), dim=3)
                else:
                    obs = tuple(torch.cat((i, j), dim=2)
                                 for i, j in zip(x, dx))
            else:
                obs = x

            return obs

        obs = _get_per_frame_data(self, subsampled, fixed_frames,
                                  add_time_dim, add_action_dim)

        if flat:
            if subsampled:
                obs = obs.reshape(self.n_trajs, self.ss_len, -1)
            else:
                obs = tuple(i.reshape(i.shape[0], -1) for i in obs)

            if add_time_dim:
                obs = add_time_dimension(obs)

        else:
            if add_time_dim:
                logger.warning('add_time_dim is ignored when not flat.')

        if numpy:
            return to_numpy(obs)
        else:
            return obs

    @lru_cache
    def get_f_hom_per_frame_xdx(self, subsampled=True, fixed_frames=False,
                                add_time_dim=False, add_action_dim=False,
                                numpy=False):
        """
        Get the homogenous frames2world transforms per trajectory, frame and
        time, stacked for position + velocity.

        Parameters
        ----------
        subsampled : bool, optional
            Subsample in time to common length, by default True
        fixed_frames : bool, optional
            Use fixed coordinate frames per trajectory, by default False
        add_time_dim : bool, optional
            Add a time dimension with identity transform, by default False
        add_action_dim : bool, optional
            Generate transform for action dim as well, by default False
        numpy : bool, optional
            Convert the resulting tensor(s) to numpy, by default False

        Returns
        -------
        torch.Tensor or list[torch.Tensor] or np.ndarray or list[np.ndarray]
            The transforms. Result is a list of tensors/ndarrays if not
            subsampled to common length. Otherwise a single tensor/ndarray.
            Shape: (n_trajs, n_frames, n_observations, 7,7)
        """

        @lru_cache
        def _get_f_hom_xdx(self, subsampled=True, fixed_frames=False,
                           add_time_dim=False, add_action_dim=False):
            f_A = self.get_f_A_xdx(subsampled, fixed_frames, add_time_dim,
                                   add_action_dim, False)
            f_b = self.get_f_b_xdx(subsampled, fixed_frames, add_time_dim,
                                   add_action_dim, False)

            n_dim = f_b[0].shape[-1]
            n_frames = self.n_frames

            if type(f_A) is tuple:
                A_flat = [f.reshape(-1, n_dim, n_dim) for f in f_A]
                b_flat = [f.reshape(-1, n_dim) for f in f_b]
                hom = tuple(
                    homogenous_transform_from_rot_shift(A, b).reshape(
                        n_frames, -1, n_dim + 1, n_dim + 1)
                    for A, b in zip(A_flat, b_flat))
                return hom
            else:
                traj_len = 1 if fixed_frames else self.ss_len
                n_trajs = self.n_trajs

                A_flat = f_A.reshape(-1, n_dim, n_dim)
                b_flat = f_b.reshape(-1, n_dim)
                hom = homogenous_transform_from_rot_shift(
                    A_flat, b_flat).reshape(
                    n_trajs, n_frames, traj_len, n_dim + 1, n_dim + 1)
                return hom

        f_hom = _get_f_hom_xdx(self, subsampled, fixed_frames, add_time_dim,
                               add_action_dim)

        if numpy:
            return to_numpy(f_hom)
        else:
            return f_hom

    @lru_cache
    def get_f_A_xdx(self, subsampled=True, fixed_frames=False,
                    add_time_dim=False, add_action_dim=True, numpy=False):
        """
        Get the rotation part of the frames2world transform, stacked for
        position + velocity.
        """

        @lru_cache
        def _get_f_A_xdx(self, subsampled=True, fixed_frames=False,
                        add_time_dim=False, add_action_dim=True):
            f_hom = self._frames2world_fixed if fixed_frames \
                else self.frames2world

            f_A = get_R_from_homogenous_transforms(f_hom)

            if add_action_dim:
                df_hom = self._frames2world_velocities_fixed if fixed_frames \
                    else self.frames2world_velocities

                df_A = get_R_from_homogenous_transforms(df_hom)

                # pos and vel trans should be the same, so just take one + kron
                if type(f_A) is tuple:
                    for f, df in zip(f_A, df_A):
                        assert torch.equal(f, df)
                    f_A = tuple(torch.kron(torch.eye(2), f) for f in f_A)

                    if subsampled:
                        f_A = torch.stack(
                            [f[:, i, :, :] for f, i in zip(f_A, self._ss_idx)])
                else:
                    assert torch.equal(f_A, df_A)
                    f_A = torch.kron(torch.eye(2), f_A)

                    assert not subsampled, "Subsam not needed for fixed frames"

            if add_time_dim:
                if type(f_A) is tuple:
                    t_As = [
                        torch.ones_like(f[:, :, 0, 0]).unsqueeze(
                            -1).unsqueeze(-1) for f in f_A]
                    f_A = tuple(batched_block_diag(t, f) for t, f in zip(
                        t_As, f_A))
                else:
                    t_A = torch.ones_like(
                        f_A[:, :, :, 0, 0]).unsqueeze(-1).unsqueeze(-1)

                    f_A = batched_block_diag(t_A, f_A)

            return f_A

        f_A = _get_f_A_xdx(self, subsampled, fixed_frames, add_time_dim,
                           add_action_dim)

        if numpy:
            return to_numpy(f_A)
        else:
            return f_A

    @lru_cache
    def get_f_b_xdx(self, subsampled=True, fixed_frames=False,
                    add_time_dim=False, add_action_dim=True, numpy=False):
        """
        Get the translation part of the frames2world transform, stacked for
        position + velocity.
        """

        @lru_cache
        def _get_f_b_xdx(self, subsampled=True, fixed_frames=False,
                         add_time_dim=False, add_action_dim=True):
            f_hom = self._frames2world_fixed if fixed_frames \
                else self.frames2world

            f_b = get_b_from_homogenous_transforms(f_hom)

            if add_action_dim:
                df_hom = self._frames2world_velocities_fixed if fixed_frames \
                    else self.frames2world_velocities

                df_b = get_b_from_homogenous_transforms(df_hom)

                # Velocity transform should be zero. Sanity check and stack.
                # NOTE: for static frames this is unnecessary overhead. But
                # needed for extension to dynamic frames.
                if type(f_b) is tuple:
                    for df in df_b:
                        assert df.sum().data == 0
                    f_b = tuple(
                        torch.cat((f, df), dim=2) for f, df in zip(f_b, df_b))

                    if subsampled:
                        f_b = torch.stack(
                            [f[:, i, :] for f, i in zip(f_b, self._ss_idx)])
                else:
                    f_b = torch.cat((f_b, df_b), dim=3)

                    assert not subsampled, "Subsam not needed for fixed frames"

            if add_time_dim:
                if type(f_b) is tuple:
                    t_bs = [
                        torch.zeros_like(f[:, :, 0]).unsqueeze(-1)
                                         for f in f_b]
                    f_b = tuple(cat((t, f), dim=-1) for t, f in zip(t_bs, f_b))
                else:
                    t_b = torch.zeros_like(f_b[:, :, :, 0]).unsqueeze(-1)

                    f_b = torch.cat((t_b, f_b), dim=-1)

            return f_b

        f_b = _get_f_b_xdx(self, subsampled, fixed_frames, add_time_dim,
                           add_action_dim)

        if numpy:
            return to_numpy(f_b)
        else:
            return f_b

    @lru_cache
    def get_f_quat_per_frame_xdx(self, subsampled=True, fixed_frames=False,
                                 numpy=False):
        """
        Get the quaternion part of the frames2world transform, stacked for
        position + velocity.

        NOTE: this function bypasses the conversion to homogenous transforms
        thus ensuring that the quaternions are continuous.
        """

        @lru_cache
        def _get_f_quat_xdx(self, subsampled=True, fixed_frames=False):
            f_quat = self._frame_quats2world_fixed if fixed_frames \
                else self._frame_quats2world

            df_quat = self._frame_quats2world_velocities_fixed if fixed_frames \
                else self._frame_quats2world_velocities

            fdf_quat = stack(f_quat, df_quat, dim=-1)

            if fixed_frames:
                assert not subsampled, "Subsamp. not needed for fixed frames"
            elif subsampled:
                self._ss_stack(fdf_quat, dim=1)

            return fdf_quat

        f_quat = _get_f_quat_xdx(self, subsampled, fixed_frames)

        if numpy:
            return to_numpy(f_quat)
        else:
            return f_quat

    def _ss_stack(self, lot, ss_dim=1, stack_dim=0, idx=None):
        """
        Subsample and stack a list of tensors.

        Parameters
        ----------
        lot : list[torch.Tensor]
            The data to subsample and stack.
        ss_dim : int, optional
            The dimension along which to subsample, by default 1
        stack_dim : int, optional
            The dimension on which to stack, by default 0
        idx : list[int], optional
            The subsampling indices, by default uses self._ss_idx

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        NotImplementedError
            _description_
        """
        if idx is None:
            idx = self._ss_idx

        if type(lot) in (list, tuple):
            # return torch.stack(
            #     [t[:, i, :, :] for t, i in zip(lot, idx)])
            return torch.stack(
                [torch.index_select(t, ss_dim, i) for t, i in zip(lot, idx)],
                 stack_dim)
        else:
            raise NotImplementedError(
                "Subsampling-stacking not implemented for type {type(x)}")

    @lru_cache
    def get_gmr_data(self, use_ss=False, fix_frames=False,
                     position_only=False, skip_quaternion_dim=None,
                     add_time_dim=False, add_action_dim=False):
        """
        Helper function to get the data prepared for GMR.

        Parameters
        ----------
        use_ss : bool
            Whether to use subsampled data, by default False
        fix_frames : bool
            Whether to use fixed frame positions per trajectory, by default
            False
        position_only : bool
            Wether to return only the position part of the data or pos + rot,
            by default False
        skip_quaternion_dim : int or None
            If not None, the dimension of the quaternion to remove, by default
            None

        Returns
        -------
        np.ndarray or list[np.ndarray] x 3
            State + action data, frame translations, frame quaternions.
        """
        frame_trans = self.get_f_hom_per_frame_xdx(
            subsampled=use_ss, fixed_frames=fix_frames,
            add_time_dim=add_time_dim, add_action_dim=add_action_dim,
            numpy=True)
        frame_quat = self.get_f_quat_per_frame_xdx(
            subsampled=use_ss, fixed_frames=fix_frames, numpy=True)

        xdx = self.get_per_frame_data(
            subsampled=use_ss, fixed_frames=fix_frames, flat=True,
            numpy=True, pos_only=position_only,
            skip_quat_dim=skip_quaternion_dim,
            add_time_dim=add_time_dim, add_action_dim=add_action_dim)

        return xdx, frame_trans, frame_quat

    def key(self):
        return self.__key()

    def __key(self):
        return self.meta_data

    def __hash__(self):
        hash = hashlib.sha1(repr(sorted(self.__key().items())).encode('utf-8'))

        return int(hash.hexdigest(), 16)

    def __eq__(self, other):
        if isinstance(other, Demos):
            return self.__key() == other.__key()
        return NotImplemented


def get_frame_transform(A, b, invert=True):
    """
    Get the frame transform from the given rotation and shift.
    When invert is True, returns world2frame, else frame2world.
    """
    assert len(A.shape) == 3
    assert A.shape[:-2] == b.shape[:-1]
    n_frames, n_steps, _ = b.shape
    A_flat = A.reshape(-1, 3, 3)
    b_flat = b.reshape(-1, 3)

    frame_transform = get_frame_transform_flat(A_flat, b_flat, invert=invert)

    frame_transform = frame_transform.reshape(n_frames, n_steps, 4, 4)

    return frame_transform


def get_frame_transform_flat(A, b, invert=True):
    frame_transform = homogenous_transform_from_rot_shift(A, b)
    if invert:
        frame_transform = invert_homogenous_transform(frame_transform)

    return frame_transform


def get_obs_per_frame(frame_transform, ee_poses):
    """
    Transform the observations from world to a given reference frame.
    Both frame_transform and EE_poses are given as homogeneous transforms.
    """
    if type(ee_poses) in (list, tuple):
        x_per_frame = []
        for i, (ft, eep) in enumerate(zip(frame_transform, ee_poses)):
            n_steps, _, _ = eep.shape
            n_frames, _, _, _ = ft.shape

            if ft.shape[1] == 1:  # static frame case, repeat
                ft = ft.repeat(1, n_steps, 1, 1)
            # Add frame dimension
            eep = eep.unsqueeze(0).repeat(n_frames, 1, 1, 1)

            x_per_frame.append(
                (ft.reshape(-1, 4, 4) @ eep.reshape(-1, 4, 4)).reshape(
                    n_frames, n_steps, 4, 4).permute(1, 0, 2, 3))

        if type(ee_poses) == tuple:
            x_per_frame = tuple(x_per_frame)

        return x_per_frame

    else:
        n_trajs, n_steps, _, _ = ee_poses.shape
        _, n_frames, _, _, _ = frame_transform.shape

        if frame_transform.shape[2] == 1:  # static frame case, repeat
           frame_transform = frame_transform.repeat(1, 1, n_steps, 1, 1)
        # Add frame dimension
        ee_poses = ee_poses.unsqueeze(1).repeat(1, n_frames, 1, 1, 1)

        x_per_frame = (frame_transform.reshape(-1, 4, 4) @ ee_poses.reshape(
            -1, 4, 4)).reshape(n_trajs, n_frames, n_steps, 4, 4)

        return x_per_frame.permute(0, 2, 1, 3, 4)


@list_or_tensor_mult_args
def get_quat_per_frame(frame_quats, ee_quats):
    """
    Analog to get_obs_per_frame, but for quaternions.
    """
    trans_shape = frame_quats.shape
    ee_shape = ee_quats.shape

    # The trajectory dim is part of both (if we stacked/subsampled) or none.
    # So, we can (almost) ignore it.
    w_traj_dim = True if len(trans_shape) == 4 else False

    if w_traj_dim:
        assert len(ee_shape) == 3
        trans_shape = trans_shape[1:]
        ee_shape = ee_shape[1:]

    n_frames, n_steps_trans, _ = trans_shape
    n_steps_ee, _ = ee_shape

    if n_steps_trans == 1:  # static frame case, repeat
        repeats = (1, 1, n_steps_ee, 1) if w_traj_dim \
            else (1, n_steps_ee, 1)  # repeats depend on trajectory dim
        frame_quats = frame_quats.repeat(*repeats)

    # Add frame dimension
    repeats = (1,  n_frames, 1, 1) if w_traj_dim else (n_frames, 1, 1)
    ee_quats = ee_quats.unsqueeze(-3).repeat(*repeats)

    return quaternion_raw_multiply(frame_quats, ee_quats)


# Copied from dataset.trajectory.py, but changed the supersampling strategy
# for shorter trajs. Analogous to subsampling now, instead of padding with
# the last frame. And using torch instead of numpy.


def get_idx_by_target_len(traj_len, target_len):
    indeces = torch.linspace(start=0, end=traj_len - 1, steps=target_len)
    indeces = torch.round(indeces).int()
    return indeces


@list_or_tensor
def add_time_dimension(lot):
    # Assumes tensors have 2 or 3 dimensions and that time is the third last
    # dimension. Ie (n_traj), n_steps, n_dim.
    # NOTE: dropped the n_frame dimension, as adding time per frame does not
    # make sense.
    n_time_steps = lot.shape[-2]

    time = torch.linspace(0, 1, n_time_steps).unsqueeze(-1)

    if len(lot.shape) == 3:
        time = time.unsqueeze(0).repeat(lot.shape[0], 1, 1)

    return torch.cat([time, lot], dim=-1)
