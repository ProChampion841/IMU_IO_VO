"""Turn a rotation into a network input feature.

Why not feed Euler angles
-------------------------
The obvious thing -- concatenate ``(roll, pitch, yaw)`` onto the IMU channels --
is wrong for three separate reasons:

1. **Yaw wraps.**  ``+179.9 deg`` and ``-179.9 deg`` are the same heading but sit
   at opposite ends of the input range, so a smooth function of the input cannot
   be a smooth function of the attitude.
2. **Yaw is arbitrary.**  Nothing about an IMU error depends on which compass
   direction the aircraft happens to be pointing.  Handing the network an
   absolute heading invites it to memorise "flight 105_21 flew north-east", which
   is a per-flight identifier, not a physical feature.
3. **On this corpus yaw is the one channel that is actually broken.**  Measured
   against the GPS-aided nav solution, the second (MTI/magnetometer) attitude
   agrees to ~1.4 deg RMS in roll and ~1.8 deg in pitch, but its heading offset
   drifts by tens of degrees *within a single flight* (median within-flight
   circular std 11.5 deg; six flights exceed 50 deg).  See
   ``tools/verify_mti_attitude.py``.

What we feed instead
--------------------
``g_body = R.Inv() @ [0, 0, 1]``: the world "up" direction expressed in the body
frame -- a unit 3-vector.  It is exactly the part of the attitude that a
strapdown accelerometer actually cares about (it is the direction gravity is
projected along), it is continuous everywhere, and it is **invariant to yaw**, so
all three objections above vanish at once.  On this corpus that invariance is
worth a great deal: the MTI's g_body agrees with the ground-truth g_body to a
median 2.0 deg (p95 5.9 deg) *despite* the heading being unusable.

``"gravity_sincos"`` appends ``sin/cos`` of roll and pitch (4 more channels).
Those are redundant with ``g_body`` in the information-theoretic sense -- they
are a fixed function of it -- but they are a different *parameterisation*: the
network gets the angles pre-normalised rather than having to learn the
normalisation ``sin(roll) = g_y / sqrt(g_y^2 + g_z^2)`` itself.

Conventions
-----------
Body is FLU, world is NWU (z up), matching everything the UAV loader publishes.
For an intrinsic ``ZYX`` (yaw-pitch-roll) rotation ``R`` from body to world,

    g_body = R^T @ e_z = ( -sin(p),  sin(r) cos(p),  cos(r) cos(p) )

so roll and pitch can be read straight back out of ``g_body`` with no ``arcsin``
or ``arctan`` at all -- which is what ``attitude_feature`` does, both because it
is cheaper and because it avoids the gradient blow-up of ``arcsin`` near +-1 g.
"""

import pypose as pp
import torch

#: number of extra input channels contributed by each mode
FEATURE_DIMS = {"none": 0, "gravity": 3, "gravity_sincos": 7}

#: valid values for the ``att_input`` config key
MODES = tuple(FEATURE_DIMS.keys())


def attitude_feature_dim(mode="gravity"):
    """Channels added to the network input by ``mode``.  0, 3 or 7."""
    mode = "none" if mode is None else str(mode)
    if mode not in FEATURE_DIMS:
        raise ValueError(
            "att_input must be one of %s, got %r" % (list(MODES), mode)
        )
    return FEATURE_DIMS[mode]


def input_dim(mode="gravity", base=6):
    """Total width of the network input: ``base`` (acc+gyro) plus the attitude."""
    return base + attitude_feature_dim(mode)


def _batch_shape(rot):
    """Leading shape of a rotation: ``lshape`` for a pypose LieTensor, else all
    but the last axis of a raw quaternion tensor."""
    ls = getattr(rot, "lshape", None)
    return tuple(ls) if ls is not None else tuple(rot.shape[:-1])


def gravity_direction(rot):
    """``g_body = R.Inv() @ [0, 0, 1]`` -- world up, in the body frame.

    Args:
        rot: ``pypose.SO3`` (or anything supporting ``.Inv()`` and ``@``) with
            leading shape ``(...,)``, e.g. ``(B, T)``.

    Returns:
        Tensor ``(..., 3)``, unit norm.
    """
    up = torch.zeros(_batch_shape(rot) + (3,), dtype=rot.dtype, device=rot.device)
    up[..., 2] = 1.0
    return rot.Inv() @ up


def attitude_feature(rot, mode="gravity", eps=1e-6):
    """Attitude input feature for HybridNet.

    Args:
        rot: ``pypose.SO3`` of shape ``(B, T)`` (body -> world, FLU -> NWU).
        mode: ``"none"`` -> ``(B, T, 0)``;
              ``"gravity"`` -> ``(B, T, 3)`` = ``g_body``;
              ``"gravity_sincos"`` -> ``(B, T, 7)`` =
              ``[g_body, sin r, cos r, sin p, cos p]``.
        eps: guard for the gimbal-lock case ``|pitch| -> 90 deg``, where roll is
            undefined; there ``sin r, cos r`` are driven to ``0, 1``.

    Returns:
        Tensor ``(B, T, C)`` with ``C == attitude_feature_dim(mode)``, in the
        dtype of ``rot``.
    """
    mode = "none" if mode is None else str(mode)
    if mode not in FEATURE_DIMS:
        raise ValueError(
            "att_input must be one of %s, got %r" % (list(MODES), mode)
        )

    if mode == "none":
        return torch.zeros(_batch_shape(rot) + (0,), dtype=rot.dtype, device=rot.device)

    return gravity_feature(gravity_direction(rot), mode, eps)


def gravity_feature(g, mode="gravity", eps=1e-6):
    """Attitude feature from an already-computed ``g_body`` (..., 3).

    The rotation-free half of :func:`attitude_feature`.  It exists so an exported
    graph (tools/export_onnx.py) can take ``g_body`` as a plain tensor input -- a
    pypose LieTensor cannot cross the ONNX boundary -- and still build EXACTLY the
    same channels the network was trained on.
    """
    mode = "none" if mode is None else str(mode)
    if mode == "none":
        return g[..., :0]
    if mode == "gravity":
        return g

    # g = (-sin p, sin r cos p, cos r cos p) with cos p >= 0 for p in [-90, 90]
    gx, gy, gz = g[..., 0], g[..., 1], g[..., 2]
    cos_p = torch.sqrt(torch.clamp(gy * gy + gz * gz, min=0.0))
    sin_p = -gx
    safe = torch.clamp(cos_p, min=eps)
    sin_r = gy / safe
    cos_r = gz / safe
    # at gimbal lock roll is not observable; fall back to the identity roll
    lock = (cos_p < eps).to(g.dtype)
    sin_r = sin_r * (1.0 - lock)
    cos_r = cos_r * (1.0 - lock) + lock
    return torch.stack([gx, gy, gz, sin_r, cos_r, sin_p, cos_p], dim=-1)


def pad_rotation(rot, init_rot, pad_len):
    """Front-pad a rotation sequence so it lines up with a padded acc/gyro.

    ``collate_fcs["padding9"]`` prepends ``pad_len = 9`` synthetic frames to
    ``acc`` and ``gyro`` but leaves ``rot`` / ``mti_rot`` at ``window_size`` --
    so an attitude feature concatenated onto the raw IMU channels *before* the
    CNN is 9 frames short and will either crash or, worse, silently misalign the
    attitude by 9 samples.

    The pad ``padding_collate`` invents is ``init_rot.Inv() @ [0,0,g]``: it
    pretends the aircraft sat at its initial attitude for those 9 frames.  The
    exactly consistent attitude pad is therefore ``init_rot`` repeated
    ``pad_len`` times, which is what this returns.

    Args:
        rot: ``(B, T)`` rotation (``data["rot"]`` or ``data["mti_rot"]``).
        init_rot: ``(B, 1)`` initial rotation (``init_state["rot"]`` /
            ``init_state["mti_rot"]``).  ``(B, T')`` is also accepted -- only
            the first frame is used, which is what ``SeqDataset`` hands over.
        pad_len: frames to prepend; use ``self.interval`` (9).

    Returns:
        ``(B, T + pad_len)`` rotation of the same type as ``rot``.
    """
    if pad_len <= 0:
        return rot
    head = init_rot[:, :1].tensor()
    cat = torch.cat([head.expand(-1, pad_len, -1), rot.tensor()], dim=1)
    return pp.LieTensor(cat, ltype=rot.ltype)


def select_attitude(data, init_state=None, source="gt", warn=True):
    """Pick the rotation HybridNet should build its attitude feature from.

    ``source="mti"`` uses ``data["mti_rot"]`` when the batch carries it and falls
    back to ``data["rot"]`` (the GPS-aided nav solution) with a warning when it
    does not.  That fallback is LEAKAGE -- it substitutes the attitude the labels
    came from -- and since the UAV loader always publishes an MTi channel it
    should never fire; treat the warning as a bug report, not as information.

    Note:
        The returned rotation is ``window_size`` long, while ``data["acc"]`` is
        ``window_size + pad_len`` under the ``padding9`` collate.  Run it through
        :func:`pad_rotation` before concatenating the feature onto acc/gyro.

    Returns:
        ``(rot, used_source)`` where ``used_source`` is ``"gt"`` or ``"mti"``.
    """
    source = "gt" if source is None else str(source)
    if source not in ("gt", "mti"):
        raise ValueError("att_source must be 'gt' or 'mti', got %r" % (source,))
    if source == "mti":
        if data.get("mti_rot", None) is not None:
            return data["mti_rot"], "mti"
        if warn and not getattr(select_attitude, "_warned", False):
            select_attitude._warned = True
            print(
                "[attitude] WARNING: att_source='mti' but this dataset publishes "
                "no 'mti_rot'; falling back to the ground-truth rotation 'rot'. "
                "Only the UAV loader has an MTI channel."
            )
    return data["rot"], "gt"
