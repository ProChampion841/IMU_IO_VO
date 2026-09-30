"""Velocity error broken down by flight condition.

One RMSE over a whole split averages together straight-and-level cruise, where
a camera looking down sees a clean translation, and a 30-degree banked turn,
where the rotation removed before matching is ten times the translation left
behind. A model can be excellent at the first, poor at the second, and report
a number that describes neither. This splits the error by the conditions that
change what the camera sees:

* ``turn_rate_deg_s`` - |yaw rate|: how much rotation had to be removed;
* ``bank_deg`` - |roll|: how tilted the ground is under the camera;
* ``altitude_m`` - the metric scale of every measurement;
* ``speed_m_s`` - the reference ground speed: how far the ground moves.

Fixed bin edges for the angles (so runs compare bin for bin) and quantile
edges for altitude and speed (so every bin is populated whatever the flight).
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

#: Fixed edges where the physics has natural breakpoints; quantiles elsewhere.
DEFAULT_EDGES: Dict[str, Optional[Sequence[float]]] = {
    "turn_rate_deg_s": (0.0, 2.0, 5.0, 10.0, 20.0, float("inf")),
    "bank_deg": (0.0, 5.0, 15.0, 30.0, float("inf")),
    "altitude_m": None,
    "speed_m_s": None,
}


def flight_conditions(
    aiding: np.ndarray, log_altitude: np.ndarray, target_velocity: np.ndarray
) -> Dict[str, np.ndarray]:
    """Per-tick conditions from the arrays a VO dataset already holds.

    ``aiding`` is the dataset's aiding matrix (``VO_AIDING_CHANNELS`` order:
    sin/cos roll, sin/cos pitch, centred log altitude, p, q, r, dt) - its
    body rates are the attitude-derived ones the model reads; ``log_altitude``
    the RAW log altitude.
    """

    aiding = np.asarray(aiding, dtype=np.float64)
    roll = np.degrees(np.arctan2(aiding[:, 0], aiding[:, 1]))
    return {
        "turn_rate_deg_s": np.abs(np.degrees(aiding[:, 7])),
        "bank_deg": np.abs(roll),
        "altitude_m": np.exp(np.asarray(log_altitude, dtype=np.float64)),
        "speed_m_s": np.linalg.norm(np.asarray(target_velocity, dtype=np.float64), axis=-1),
    }


def _edges(values: np.ndarray, fixed: Optional[Sequence[float]], quantiles: int) -> np.ndarray:
    if fixed is not None:
        return np.asarray(fixed, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.asarray([0.0, float("inf")])
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, quantiles + 1)))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def stratified_errors(
    predicted: np.ndarray,
    target: np.ndarray,
    conditions: Mapping[str, np.ndarray],
    *,
    edges: Optional[Mapping[str, Optional[Sequence[float]]]] = None,
    quantiles: int = 4,
    min_ticks: int = 20,
) -> Dict[str, List[Dict[str, float]]]:
    """RMSE (3-axis magnitude and per axis) per condition bin.

    ``predicted``/``target`` are ``(T, 3)`` with NaN on unscored ticks (the
    horizon series' convention), ``conditions`` maps a name to a ``(T,)``
    array. A bin with fewer than ``min_ticks`` scored ticks is reported with
    its count and NaN errors rather than a number built from a handful.
    """

    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    scored = np.isfinite(predicted).all(axis=-1) & np.isfinite(target).all(axis=-1)
    residual = predicted - target
    chosen = dict(DEFAULT_EDGES)
    if edges is not None:
        chosen.update(edges)
    report: Dict[str, List[Dict[str, float]]] = {}
    for name, values in conditions.items():
        values = np.asarray(values, dtype=np.float64)
        bins = _edges(values[scored], chosen.get(name), quantiles)
        rows = []
        for low, high in zip(bins[:-1], bins[1:]):
            inside = scored & (values >= low) & (values < high)
            count = int(inside.sum())
            row: Dict[str, float] = {"low": float(low), "high": float(high), "ticks": count}
            if count >= min_ticks:
                chunk = residual[inside]
                row["vel_rmse"] = float(np.sqrt((chunk ** 2).sum(axis=-1).mean()))
                for axis, label in enumerate("xyz"):
                    row[f"vel_rmse_{label}"] = float(np.sqrt((chunk[:, axis] ** 2).mean()))
            else:
                row["vel_rmse"] = float("nan")
                for label in "xyz":
                    row[f"vel_rmse_{label}"] = float("nan")
            rows.append(row)
        report[name] = rows
    return report


def format_stratified(report: Mapping[str, Sequence[Mapping[str, float]]]) -> str:
    lines = []
    for name, rows in report.items():
        lines.append(f"  by {name}:")
        for row in rows:
            high = "inf" if not np.isfinite(row["high"]) else f"{row['high']:.1f}"
            lines.append(
                f"    [{row['low']:7.1f}, {high:>7}) ticks {int(row['ticks']):7d}  "
                f"rmse {row['vel_rmse']:6.3f}  "
                f"x {row['vel_rmse_x']:6.3f} y {row['vel_rmse_y']:6.3f} z {row['vel_rmse_z']:6.3f}"
            )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_EDGES",
    "flight_conditions",
    "format_stratified",
    "stratified_errors",
]
