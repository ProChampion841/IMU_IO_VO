"""Turn run settings into a frontend - one place, for every tool.

The trainer builds a frontend from its ``argparse`` namespace; the evaluators
rebuild one from a checkpoint's saved ``args`` dict. Two hand-written copies of
that mapping is how a setting gets honoured in training and silently dropped
in evaluation (a gated frontend scored ungated, say), so both call
:func:`build_frontend` here instead.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from .vision_mamba_vo import VisionMambaFlowFrontend


def _getter(settings: Any):
    if isinstance(settings, Mapping):
        return settings.get
    return lambda key, default=None: getattr(settings, key, default)


def resolve_velocity_mode(args: Any) -> str:
    """``--velocity-mode``, with ``auto`` resolved by the frontend."""

    get = _getter(args)
    mode = str(get("velocity_mode", "auto") or "auto")
    if mode == "auto":
        return "geometric_residual" if get("frontend", "mamba_correlation") == "planar" else "heads"
    return mode


def frontend_class(name: str):
    """The frontend class a ``--frontend`` name builds."""

    from .planar_frontend import PlanarFlowFrontend

    classes = {"mamba_correlation": VisionMambaFlowFrontend, "planar": PlanarFlowFrontend}
    if name not in classes:
        raise ValueError(f"unknown frontend {name!r}; choose from {sorted(classes)}")
    return classes[name]


def build_frontend(
    settings: Any,
    *,
    camera_from_body: Optional[Sequence[Sequence[float]]] = None,
    prior_velocity: Optional[Sequence[float]] = None,
    dropout: Optional[float] = None,
) -> VisionMambaFlowFrontend:
    """One frontend from a trainer ``argparse`` namespace OR a checkpoint's
    saved ``args`` dict - the single place both the trainer and the evaluators
    turn settings into a module, so the two cannot build different things.

    Missing keys fall back to the values a checkpoint written before they
    existed can only have used (the same convention FINGERPRINT_DEFAULTS
    documents): the original frontend, ungated, grayscale.
    """

    get = _getter(settings)
    kind = str(get("frontend", "mamba_correlation") or "mamba_correlation")
    common = dict(
        input_channels=3 if bool(get("color", False)) else 1,
        visual_dim=int(get("visual_dim", 64)),
        d_model=int(get("stem_dim", 64)),
        depth=int(get("stem_depth", 2)),
        patch_size=int(get("patch_size", 8)),
        image_size=tuple(int(v) for v in get("image_size", (576, 1024))),
        context_grid=tuple(int(v) for v in get("context_grid", (12, 16))),
        token_grid=int(get("token_grid", 6)),
        correlation_radius=int(get("correlation_radius", 4)),
        dropout=float(get("dropout", 0.1) if dropout is None else dropout),
        min_pool_weight=float(get("min_pool_weight", 1e-4)),
        max_cell_entropy=float(get("max_cell_entropy", 1.0)),
        min_cell_confidence=float(get("min_cell_confidence", 0.0)),
        min_score_margin=float(get("min_score_margin", 0.0)),
        reject_boundary_peaks=bool(get("reject_boundary_peaks", False)),
        min_reliable_cell_fraction=float(get("min_reliable_cell_fraction", 0.0)),
    )
    if kind == "planar":
        if camera_from_body is None:
            raise ValueError(
                "--frontend planar needs the camera mounting: pass --camera-mounting "
                "(top_forward / right_forward / left_forward / bottom_forward, or the "
                "matrix tools/estimate_camera_mounting.py prints), or put "
                "mounting.camera_from_body in the calibration file"
            )
        from .planar_frontend import PlanarFlowFrontend

        return PlanarFlowFrontend(
            camera_from_body=camera_from_body,
            prior_velocity_body=(
                (20.0, 0.0, 0.0) if prior_velocity is None else tuple(float(v) for v in prior_velocity)
            ),
            coarse_factor=int(get("coarse_factor", 4)),
            coarse_radius=int(get("coarse_radius", 6)),
            coarse_highpass=int(get("coarse_highpass", 5)),
            fine_highpass=int(get("fine_highpass", 9)),
            fine_iterations=int(get("fine_iterations", 2)),
            huber_cells=float(get("huber_cells", 1.0)),
            altitude_constraint=float(get("altitude_constraint", 300.0)),
            learn_mounting=not bool(get("no_learn_mounting", False)),
            max_speed_m_s=float(get("max_geometric_speed", 80.0)),
            min_fit_cells=float(get("min_fit_cells", 8.0)),
            **common,
        )
    return VisionMambaFlowFrontend(
        rotation_mode=str(get("rotation_mode", "constant")),
        **common,
    )


__all__ = ["build_frontend", "frontend_class", "resolve_velocity_mode"]
