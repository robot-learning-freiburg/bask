import torch

from utils.torch import list_or_tensor

y180_quaternion = torch.tensor([0, 0, 1, 0], dtype=torch.float32)


@list_or_tensor
def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


@list_or_tensor
def conjugate_quat(quaternion):
    assert quaternion.shape[-1] == 4
    conj = torch.clone(quaternion)
    # Both lines are equivalent, though the second one preserves the sign of
    # the real part.
    # conj[..., 0] = - conj[..., 0]
    conj[..., 1:4] = - conj[..., 1:4]
    return conj


def rotate_quat_y180(quaternion):
    return quaternion_raw_multiply(y180_quaternion, quaternion)


@list_or_tensor
def quaternion_to_axis_and_angle_batched(q):
    """
    Convert quaternion representation to axis-angle representation.

    Args:
        q: Tensor of shape (..., 4) representing the quaternion.

    Returns:
        axis: Tensor of shape (..., 3) representing the axis of rotation.
        angle: Tensor of shape (..., ) representing the angle of rotation in
               radians.
    """
    if torch.abs(torch.norm(q, dim=-1) - 1).max() > 2e-3:
        raise ValueError("Input quaternions must be normalized.")
    angle = 2 * torch.arccos(q[..., 0])
    axis = q[..., 1:] / torch.sqrt(1 - q[..., 0]**2).unsqueeze(-1)
    return axis, angle

@list_or_tensor
def axis_angle_to_quaternion_batched(axis, angle):
    """
    Convert axis-angle representation to quaternion representation.

    Args:
        axis: Tensor of shape (..., 3) representing the axis of rotation.
        angle: Tensor of shape (..., ) representing the angle of rotation in
               radians.

    Returns:
        Tensor of shape (..., 4) representing the quaternion.
    """
    norm = torch.norm(axis, dim=-1, keepdim=True)
    normed_axis = axis / norm
    half_angle = angle / 2
    qw = torch.cos(half_angle)
    qxyz = torch.sin(half_angle).unsqueeze(-1) * normed_axis
    return torch.cat([qw.unsqueeze(-1), qxyz], dim=-1)


# @list_or_tensor
def modulo_rotation_angle(pose, mod_angle, dim=2, skip_first=True,
                          ensure_positive_rot=True, eps=0.02):
    return tuple(
        _modulo_rotation_angle(p, mod_angle, dim, skip_first, ensure_positive_rot,
                                 eps, sub_one=i==8) #i==16)
        for i, p in enumerate(pose))


def _modulo_rotation_angle(pose, mod_angle, dim=2, skip_first=True,
                          ensure_positive_rot=True, eps=0.02, sub_one=False):
    """
    Modulo the rotation angle of a pose around a given dim (axis).

    Args:
        pose: Tensor of shape (..., 7) representing the pose.
        mod_angle: The angle in radians to modulo the rotation angle by.
        dim: The axis to modulo the rotation angle around. Default: 2 (z-axis).
        skip_first: Whether to skip the rotation of the first frame (second
                    dimension). Used to skip the EE pose prepended to the frame
                    poses. Default: True.

    Returns:
        The pose with the rotation angle moduloed.
    """
    quaternion = pose[..., 3:]
    # if ensure_positive_rot:
    #     ref = quaternion[0]
    #     ref_axis, ref_angle = quaternion_to_axis_and_angle_batched(ref)
    #     ref_aa = ref_axis * ref_angle.unsqueeze(-1)
    if skip_first:
        quaternion = quaternion[1:]
    axis, angle = quaternion_to_axis_and_angle_batched(quaternion)
    aa = axis * angle.unsqueeze(-1)
    aa_mod = torch.remainder(aa[..., dim], mod_angle)
    if ensure_positive_rot:
        # print("=====================================")
        # print(aa[:, -1, dim])
        # print(aa_mod[0, -1])
        # print(ref_aa[-1, dim])
        # print(ref_aa[-1, dim] > aa_mod[-1, dim])
        # print(aa_mod[0, -1], mod_angle / 2 - eps, aa_mod[0, -1] - mod_angle / 2)
        # print(aa_mod[0, 0])
        aa_mod +=  torch.where(aa_mod < mod_angle / 2 + eps,
            torch.ones_like(aa_mod) * mod_angle, torch.zeros_like(aa_mod))
        if sub_one:
            aa_mod -= mod_angle
    aa[..., dim] = aa_mod
    angle_mod = torch.norm(aa, dim=-1)
    axis_mod = aa / angle_mod.unsqueeze(-1)
    quaternion_mod = axis_angle_to_quaternion_batched(axis_mod, angle_mod)
    if skip_first:
        pose[1:, :, 3:] = quaternion_mod
    else:
        pose[..., 3:] = quaternion_mod
    return pose


def mod_quat(quaternion, mod_angle, dim=2):
    axis, angle = quaternion_to_axis_and_angle_batched(quaternion)
    aa = axis * angle.unsqueeze(-1)
    aa[..., dim] = torch.round(aa[..., dim] / mod_angle) * mod_angle
    angle_mod = torch.norm(aa, dim=-1)
    axis_mod = aa / angle_mod.unsqueeze(-1)
    quaternion_mod = axis_angle_to_quaternion_batched(axis_mod, angle_mod)

    return quaternion_mod
