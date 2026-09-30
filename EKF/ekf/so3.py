"""SO(3) helpers in NumPy.  Rotation vectors are (..., 3), matrices (..., 3, 3)."""
import numpy as np

# body FRD <-> FLU and world NED <-> NWU: a 180 deg rotation about x (det +1), so
# it is an exact change of basis, not a reflection.  The same T the IMU loader uses.
T_FLIP = np.diag([1.0, -1.0, -1.0])


def skew(v):
    v = np.asarray(v, dtype=float)
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -v[..., 2], v[..., 1]
    K[..., 1, 0], K[..., 1, 2] = v[..., 2], -v[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -v[..., 1], v[..., 0]
    return K


def exp(phi):
    """Rodrigues, exact, smooth at 0."""
    phi = np.asarray(phi, dtype=float)
    th = np.linalg.norm(phi, axis=-1)[..., None, None]
    K = skew(phi)
    small = th < 1e-8
    ths = np.where(small, 1.0, th)
    a = np.where(small, 1.0 - th ** 2 / 6.0, np.sin(ths) / ths)
    b = np.where(small, 0.5 - th ** 2 / 24.0, (1.0 - np.cos(ths)) / ths ** 2)
    return np.eye(3) + a * K + b * (K @ K)


def log(R):
    """Inverse of exp for angles in [0, pi)."""
    R = np.asarray(R, dtype=float)
    c = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(c)
    w = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0],
                  R[..., 1, 0] - R[..., 0, 1]], axis=-1)
    small = th < 1e-6
    s = np.where(small, 1.0, np.sin(np.where(small, 1.0, th)))
    k = np.where(small, 0.5 + th ** 2 / 12.0, th / (2.0 * s))
    return w * k[..., None]


def orthonormalize(R):
    u, _, vt = np.linalg.svd(R)
    return u @ vt


def euler_zyx(yaw, pitch, roll):
    """Intrinsic ZYX (yaw-pitch-roll) -> matrix, body -> world."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    R = np.empty(np.shape(yaw) + (3, 3))
    R[..., 0, 0] = cy * cp
    R[..., 0, 1] = cy * sp * sr - sy * cr
    R[..., 0, 2] = cy * sp * cr + sy * sr
    R[..., 1, 0] = sy * cp
    R[..., 1, 1] = sy * sp * sr + cy * cr
    R[..., 1, 2] = sy * sp * cr - cy * sr
    R[..., 2, 0] = -sp
    R[..., 2, 1] = cp * sr
    R[..., 2, 2] = cp * cr
    return R


def frd_to_flu(v):
    """Body FRD vector -> body FLU (VO output -> IMU convention)."""
    return np.asarray(v, dtype=float) * np.array([1.0, -1.0, -1.0])
