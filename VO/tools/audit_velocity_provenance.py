#!/usr/bin/env python3
"""Audit whether fixed-wing velocity labels are independent ground truth.

The report is deliberately diagnostic rather than authoritative. Statistical
patterns can identify a likely onboard navigation solution, but only logger or
firmware documentation can establish which sensors produced a label.

The JSON output contains aggregate statistics only: no paths, timestamps,
samples, or trajectories, so it is suitable for sharing with a training log.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

TARGET_COLUMNS = ("GPSNavVnX", "GPSNavVnY", "GPSNavVnZ")
IMU_COLUMNS = ("GyroX", "GyroY", "GyroZ", "AcclX", "AcclY", "AcclZ")
EULER_DEG_COLUMNS = ("EulX", "EulY", "EulZ")
EULER_RAD_COLUMNS = ("GPSNavEulX", "GPSNavEulY", "GPSNavEulZ")


def _first_indices(headers: Sequence[str], names: Iterable[str]) -> Dict[str, int]:
    """Select the first instance of every column, preserving duplicate headers."""

    result: Dict[str, int] = {}
    for name in names:
        if name in headers:
            result[name] = headers.index(name)
    return result


def read_numeric_columns(
    csv_path: Path, names: Sequence[str]
) -> Tuple[Dict[str, np.ndarray], int]:
    """Read requested finite numeric columns using first-header semantics."""

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            headers = next(reader)
        except StopIteration as exc:
            raise ValueError("Telemetry CSV is empty") from exc
        indices = _first_indices(headers, names)
        missing = [name for name in names if name not in indices]
        if missing:
            raise ValueError(f"Telemetry is missing required columns: {missing}")
        values = {name: [] for name in names}
        for row_number, row in enumerate(reader, start=2):
            for name in names:
                try:
                    value = float(row[indices[name]])
                except (IndexError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid {name} value at CSV row {row_number}"
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite {name} value at CSV row {row_number}")
                values[name].append(value)
    if len(next(iter(values.values()))) < 4:
        raise ValueError("Telemetry needs at least four rows")
    duplicate_count = len(headers) - len(set(headers))
    return {
        name: np.asarray(column, dtype=np.float64) for name, column in values.items()
    }, duplicate_count


def _longest_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask.tolist():
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _spectral_summary(values: np.ndarray, sample_rate_hz: float) -> Dict[str, float]:
    centered = values - values.mean()
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    frequencies = np.fft.rfftfreq(centered.size, d=1.0 / sample_rate_hz)
    if spectrum.size:
        spectrum[0] = 0.0
    total = float(spectrum.sum())
    if total <= np.finfo(np.float64).eps:
        return {
            "spectral_centroid_hz": 0.0,
            "energy_fraction_above_10_hz": 0.0,
        }
    return {
        "spectral_centroid_hz": float(np.sum(frequencies * spectrum) / total),
        "energy_fraction_above_10_hz": float(
            spectrum[frequencies >= 10.0].sum() / total
        ),
    }


def change_statistics(
    values: np.ndarray, sample_rate_hz: float, repeat_tolerance: float
) -> Dict[str, object]:
    delta = np.diff(values)
    exact_repeat = delta == 0.0
    tolerance_repeat = np.abs(delta) <= repeat_tolerance
    changed = ~tolerance_repeat
    return {
        "count": int(values.size),
        "exact_repeat_fraction": float(exact_repeat.mean()),
        "tolerance_repeat_fraction": float(tolerance_repeat.mean()),
        "longest_exact_repeat_run_samples": _longest_run(exact_repeat),
        "longest_tolerance_repeat_run_samples": _longest_run(tolerance_repeat),
        "effective_change_rate_hz": float(changed.mean() * sample_rate_hz),
        "delta_rms_per_sample": float(np.sqrt(np.mean(delta * delta))),
        "delta_abs_p95_per_sample": float(np.quantile(np.abs(delta), 0.95)),
        **_spectral_summary(values, sample_rate_hz),
    }


def quantization_fingerprint(values: np.ndarray) -> Dict[str, object]:
    """Find strong coarse grids while treating a missing grid as inconclusive."""

    finite = values[np.isfinite(values)]
    serialized_scale = 1_000_000
    serialized = np.rint(finite * serialized_scale).astype(np.int64)
    nonzero_delta = np.abs(np.diff(serialized))
    nonzero_delta = nonzero_delta[nonzero_delta > 0]
    gcd_units = 0
    for value in nonzero_delta.tolist():
        gcd_units = math.gcd(gcd_units, int(value))
        if gcd_units == 1:
            break
    candidate_alignment: Dict[str, float] = {}
    coarsest_supported: float | None = None
    for quantum in (1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001):
        scaled = finite / quantum
        fraction = float((np.abs(scaled - np.rint(scaled)) <= 1e-4).mean())
        candidate_alignment[f"{quantum:.6g}"] = fraction
        if coarsest_supported is None and fraction >= 0.999:
            coarsest_supported = quantum
    least_digit = np.abs(serialized) % 10
    return {
        "serialized_resolution_assumed": 1.0 / serialized_scale,
        "delta_gcd_at_serialized_resolution": float(gcd_units / serialized_scale),
        "least_significant_decimal_histogram": {
            str(digit): int((least_digit == digit).sum()) for digit in range(10)
        },
        "candidate_grid_alignment_fraction": candidate_alignment,
        "coarsest_grid_with_99_9pct_alignment": coarsest_supported,
        "interpretation": (
            "A coarse grid is strong evidence for a quantized source. No coarse grid "
            "is only weak evidence because logging/interpolation can densify GNSS data."
        ),
    }


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 3 or right.size != left.size:
        return None
    if left.std() < 1e-12 or right.std() < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def best_lag_correlation(
    left: np.ndarray,
    right: np.ndarray,
    sample_rate_hz: float,
    max_lag_s: float,
) -> Dict[str, float | None]:
    max_lag = max(0, int(round(max_lag_s * sample_rate_hz)))
    best_corr: float | None = None
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = left[-lag:], right[:lag]
        elif lag > 0:
            a, b = left[:-lag], right[lag:]
        else:
            a, b = left, right
        correlation = _safe_correlation(a, b)
        if correlation is not None and (
            best_corr is None or abs(correlation) > abs(best_corr)
        ):
            best_corr = correlation
            best_lag = lag
    return {
        "zero_lag": _safe_correlation(left, right),
        "best_correlation": best_corr,
        # Positive means the right-hand signal is shifted later than the left.
        "best_lag_s": float(best_lag / sample_rate_hz),
    }


def euler_equivalence(columns: Mapping[str, np.ndarray]) -> Dict[str, object]:
    axes: Dict[str, object] = {}
    all_residuals = []
    for axis, degrees_name, radians_name in zip(
        "xyz", EULER_DEG_COLUMNS, EULER_RAD_COLUMNS
    ):
        converted = np.deg2rad(columns[degrees_name])
        residual = converted - columns[radians_name]
        all_residuals.append(residual)
        axes[axis] = {
            "rms_rad": float(np.sqrt(np.mean(residual * residual))),
            "max_abs_rad": float(np.abs(residual).max()),
            "correlation": _safe_correlation(converted, columns[radians_name]),
        }
    concatenated = np.concatenate(all_residuals)
    return {
        "axes": axes,
        "rms_rad": float(np.sqrt(np.mean(concatenated * concatenated))),
        "max_abs_rad": float(np.abs(concatenated).max()),
        "equivalent_within_1e_4_rad": bool(np.abs(concatenated).max() <= 1e-4),
    }


def build_report(
    columns: Mapping[str, np.ndarray],
    *,
    duplicate_header_count: int,
    time_column: str,
    declared_provenance: str = "unknown",
    source_id: str | None = None,
    repeat_tolerance: float = 1e-9,
    max_lag_s: float = 0.5,
) -> Dict[str, object]:
    times = columns[time_column]
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError("Telemetry time must be strictly increasing")
    sample_rate_hz = float(1.0 / np.median(dt))

    target_change = {
        name: change_statistics(columns[name], sample_rate_hz, repeat_tolerance)
        for name in TARGET_COLUMNS
    }
    quantization = {
        name: quantization_fingerprint(columns[name]) for name in TARGET_COLUMNS
    }
    derivative = {name: np.gradient(columns[name], times) for name in TARGET_COLUMNS}
    imu_correlation: Dict[str, object] = {}
    for target_name in TARGET_COLUMNS:
        imu_correlation[target_name] = {
            imu_name: best_lag_correlation(
                derivative[target_name], columns[imu_name], sample_rate_hz, max_lag_s
            )
            for imu_name in IMU_COLUMNS
        }

    equivalence = euler_equivalence(columns)
    horizontal_speed = np.linalg.norm(
        np.column_stack((columns[TARGET_COLUMNS[0]], columns[TARGET_COLUMNS[1]])),
        axis=1,
    )
    airspeed = columns["AirSpeed"]
    airspeed_comparison = {
        "horizontal_speed_rms": float(np.sqrt(np.mean(horizontal_speed**2))),
        "airspeed_rms": float(np.sqrt(np.mean(airspeed**2))),
        "rms_difference": float(np.sqrt(np.mean((horizontal_speed - airspeed) ** 2))),
        "correlation": _safe_correlation(horizontal_speed, airspeed),
        "warning": "airspeed is not ground speed in wind; target XY axes are unverified",
    }

    mean_change_rate = float(
        np.mean(
            [
                float(target_change[name]["effective_change_rate_hz"])
                for name in TARGET_COLUMNS
            ]
        )
    )
    statistical_likelihood = bool(
        equivalence["equivalent_within_1e_4_rad"]
        and mean_change_rate >= 0.5 * sample_rate_hz
    )
    if declared_provenance == "independent":
        conclusion = "independent"
    elif declared_provenance == "navigation_solution" or statistical_likelihood:
        conclusion = "likely_navigation_solution"
    else:
        conclusion = "unknown"

    return {
        "schema_version": 1,
        "status": "diagnostic_not_proof",
        "row_count": int(times.size),
        "duplicate_header_count": int(duplicate_header_count),
        "timing": {
            "sample_rate_hz": sample_rate_hz,
            "duration_s": float(times[-1] - times[0]),
            "dt_median_s": float(np.median(dt)),
            "dt_p99_s": float(np.quantile(dt, 0.99)),
        },
        "target_change": target_change,
        "target_quantization_fingerprint": quantization,
        "target_derivative_to_imu_correlation": imu_correlation,
        "euler_unit_equivalence": equivalence,
        "airspeed_comparison": airspeed_comparison,
        "provenance": {
            "declared": declared_provenance,
            "source_id": source_id,
            "statistical_likelihood_of_navigation_solution": statistical_likelihood,
            "statistical_rule": (
                "Euler-unit duplication plus target updates on at least half of IMU ticks"
            ),
            "diagnostic_only_not_in_rule": (
                "target/IMU derivative correlation and quantization fingerprint; either "
                "can also occur with an independent physical target or logger resampling"
            ),
            "conclusion": conclusion,
            "independent_ground_truth_allowed": conclusion == "independent",
            "warning": (
                "Only logger/firmware documentation or an independent sensor manifest "
                "can establish target provenance."
            ),
        },
    }


def _atomic_json_dump(report: Mapping[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--csv-name", default="flight.csv")
    parser.add_argument("--time-column", default="Time")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/velocity_provenance.json")
    )
    parser.add_argument("--repeat-tolerance", type=float, default=1e-9)
    parser.add_argument("--max-lag-s", type=float, default=0.5)
    parser.add_argument(
        "--declared-provenance",
        choices=("unknown", "independent", "navigation_solution"),
        default="unknown",
        help="Use a documented declaration only; statistics cannot select independent.",
    )
    parser.add_argument(
        "--source-id",
        default=None,
        help="Non-path identifier for the logger/firmware document supporting a declaration.",
    )
    args = parser.parse_args()
    if args.repeat_tolerance < 0 or args.max_lag_s < 0:
        parser.error("repeat-tolerance and max-lag-s cannot be negative")
    if args.declared_provenance != "unknown" and not args.source_id:
        parser.error("--source-id is required for a documented provenance declaration")
    return args


def main() -> int:
    args = parse_args()
    csv_path = args.dataset.expanduser().resolve() / args.csv_name
    if not csv_path.is_file():
        raise SystemExit(f"Telemetry CSV does not exist: {csv_path}")
    requested = (
        args.time_column,
        *TARGET_COLUMNS,
        *IMU_COLUMNS,
        *EULER_DEG_COLUMNS,
        *EULER_RAD_COLUMNS,
        "AirSpeed",
    )
    columns, duplicate_count = read_numeric_columns(csv_path, requested)
    report = build_report(
        columns,
        duplicate_header_count=duplicate_count,
        time_column=args.time_column,
        declared_provenance=args.declared_provenance,
        source_id=args.source_id,
        repeat_tolerance=args.repeat_tolerance,
        max_lag_s=args.max_lag_s,
    )
    _atomic_json_dump(report, args.output)
    print(json.dumps(report["provenance"], indent=2, sort_keys=True))
    print(f"Wrote aggregate provenance audit to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
