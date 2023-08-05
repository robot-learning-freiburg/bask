import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation

identity_quaternion_np = np.array([1, 0, 0, 0])

# NOTE: reusing some functions from https://github.com/facebookresearch/pytorch3d


def quaternion_to_axis_angle(quaternions):
    norms = np.linalg.norm(quaternions[..., 1:], axis=-1, keepdims=True)
    half_angles = np.arctan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = np.abs(angles) < eps
    sin_half_angles_over_angles = np.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        np.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def quaternion_to_axis_and_angle(q):
    q = np.array(q)
    if np.abs(np.linalg.norm(q) - 1) > 2e-3:
        raise ValueError("Input quaternion must be normalized.")
    angle = 2 * np.arccos(q[0])
    axis = q[1:] / np.sqrt(1 - q[0]**2)
    return axis, angle


def euler_to_quaternion(euler_angle):
    roll, pitch, yaw = euler_angle
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    quaternion = [qx, qy, qz, qw]

    return quaternion


def quaternion_to_euler(quaternion):
    qx, qy, qz, qw = quaternion
    roll = np.arctan2(2*(qw*qx + qy*qz), 1-2*(qx**2+qy**2))
    pitch = np.arcsin(2*(qw*qy - qz*qx))
    yaw = np.arctan2(2*(qw*qz + qx*qy), 1-2*(qy**2-qz**2))
    return np.array([roll, pitch, yaw])


def compute_angle_between_quaternions(q, r):
    """
    Computes the angle between two quaternions.

    theta = arccos(2 * <q1, q2>^2 - 1)

    See https://math.stackexchange.com/questions/90081/quaternion-distance
    :param q: numpy array in form [w,x,y,z]. As long as both q,r are consistent
              it doesn't matter
    :type q:
    :param r:
    :type r:
    :return: angle between the quaternions, in radians
    :rtype:
    """

    theta = 2*np.arccos(2 * np.dot(q, r)**2 - 1)
    return theta


def compute_distance_between_poses(pose_a, pose_b):
    """
    Computes the linear difference between pose_a and pose_b
    :param pose_a: 4 x 4 homogeneous transform
    :type pose_a:
    :param pose_b:
    :type pose_b:
    :return: Distance between translation component of the poses
    :rtype:
    """

    pos_a = pose_a[0:3, 3]
    pos_b = pose_b[0:3, 3]

    return np.linalg.norm(pos_a - pos_b)


def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                      [m01+m10,     m11-m00-m22, 0.0,         0.0],
                      [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                      [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def compute_angle_between_poses(pose_a, pose_b):
    """
    Computes the angle distance in radians between two homogenous transforms
    :param pose_a: 4 x 4 homogeneous transform
    :type pose_a:
    :param pose_b:
    :type pose_b:
    :return: Angle between poses in radians
    :rtype:
    """

    quat_a = quaternion_from_matrix(pose_a)
    quat_b = quaternion_from_matrix(pose_b)

    return compute_angle_between_quaternions(quat_a, quat_b)


def conjugate_quat(quaternion):
    assert len(quaternion) == 4
    conj = np.copy(quaternion)
    conj[-1] = - conj[-1]

    return conj


def quaternion_multiply(quaternion0, quaternion1):
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array([
        x1*w0 + y1*z0 - z1*y0 + w1*x0,
        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
        x1*y0 - y1*x0 + z1*w0 + w1*z0,
        -x1*x0 - y1*y0 - z1*z0 + w1*w0], dtype=np.float64)


def euler_to_matrix(euler_angle):
    roll, pitch, yaw = euler_angle

    matrix = np.zeros((4, 4))
    matrix[:3, :3] = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    matrix[3, 3] = 1

    return matrix


def matrix_to_euler(matrix):
    return Rotation.from_matrix(matrix).as_euler('xyz')


def normalize_quaternion(quat):
    norm = np.linalg.norm(quat)

    return quat / norm


def quaternion_to_matrix(quaternion):
    return Rotation.from_quat(quaternion).as_matrix()


def homogenous_transform_from_rot_shift(rot, shift):
    if len(rot.shape) == 3:
        B = rot.shape[0]
        matrix = np.zeros((B, 4, 4))
        matrix[:, :3, :3] = rot
        matrix[:, :3, 3] = shift
        matrix[:, 3, 3] = 1
    else:
        matrix = np.zeros((4, 4))
        matrix[:3, :3] = rot
        matrix[:3, 3] = shift
        matrix[3, 3] = 1

    return matrix


def invert_homogenous_transform(matrix):
    rot = matrix[:3, :3]
    shift = matrix[:3, 3]

    rot_inv = torch.transpose(rot, -1, -2)
    shift_inv = torch.matmul(-rot_inv, shift)

    inverse = torch.zeros_like(matrix)
    inverse[:3, :3] = rot_inv
    inverse[:3, 3] = shift_inv
    inverse[3, 3] = 1

    return inverse


def np_invert_homogenous_transform(matrix):
    rot = matrix[:3, :3]
    shift = matrix[:3, 3]

    rot_inv = np.transpose(rot)
    shift_inv = np.matmul(-rot_inv, shift)

    inverse = np.zeros_like(matrix)
    inverse[:3, :3] = rot_inv
    inverse[:3, 3] = shift_inv
    inverse[3, 3] = 1

    return inverse


def torch_np_wrapper(func, tensor, device):
    return torch.from_numpy(func(tensor.cpu())).to(device)


def arccos_star(rho):
    if type(rho) is not np.ndarray:
        # Check rho
        if abs(rho) > 1:
            # Check error:
            if (abs(rho) - 1 > 1e-6):
                print('arcos_star: abs(rho) > 1+1e-6:'.format(abs(rho)-1))

            # Fix error
            rho = 1*np.sign(rho)

        # Single mode:
        if (-1.0 <= rho and rho < 0.0):
            return np.arccos(rho) - np.pi
        else:
            return np.arccos(rho)
    else:
        # Batch mode:
        rho = np.array([rho])

        ones = np.ones(rho.shape)
        rho = np.max(np.vstack((rho, -1*ones)), axis=0)
        rho = np.min(np.vstack((rho, 1*ones)), axis=0)

        acos_rho = np.zeros(rho.shape)
        sl1 = np.ix_((-1.0 <= rho)*(rho < 0.0) == 1)
        sl2 = np.ix_((1.0 > rho)*(rho >= 0.0) == 1)

        acos_rho[sl1] = np.arccos(rho[sl1]) - np.pi
        acos_rho[sl2] = np.arccos(rho[sl2])

        return acos_rho


def quat_log_e(q, reg=1e-6):
    if abs(q[0]- 1.0) > reg:
        return arccos_star(q[0]) * (q[1:]/np.linalg.norm(q[1:]))
    else:
        return np.zeros(3)

def log_e(dp):
    assert len(dp.shape) == 1
    assert dp.shape[0] % 7 == 0

    n_frames = dp.shape[0] // 7

    out = []
    # split into per frame, then process the quaternion parts
    for i in range(n_frames):
        frame = dp[i*7:(i+1)*7]
        out.append(frame[:3])
        out.append(quat_log_e(frame[3:7]))

    out = np.concatenate(out, axis=0)

    return out

def quaternion_diff(q1, q2):
    return quaternion_multiply(q1, conjugate_quat(q2))
