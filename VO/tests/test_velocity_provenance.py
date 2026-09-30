import numpy as np

from tools.audit_velocity_provenance import (
    EULER_DEG_COLUMNS,
    EULER_RAD_COLUMNS,
    build_report,
    quantization_fingerprint,
)


def _columns(row_count=400):
    times = np.arange(row_count, dtype=np.float64) * 0.01
    heading = np.linspace(0.0, 90.0, row_count)
    euler = {
        "EulX": 5.0 * np.sin(times),
        "EulY": 2.0 * np.cos(times),
        "EulZ": heading,
    }
    columns = {"Time": times, **euler}
    columns.update(
        {
            radians_name: np.deg2rad(columns[degrees_name])
            for degrees_name, radians_name in zip(EULER_DEG_COLUMNS, EULER_RAD_COLUMNS)
        }
    )
    columns.update(
        {
            "GPSNavVnX": 12.0 * np.cos(np.deg2rad(heading)),
            "GPSNavVnY": 12.0 * np.sin(np.deg2rad(heading)),
            "GPSNavVnZ": 0.2 * np.sin(0.5 * times),
            "GyroX": np.gradient(columns["EulX"], times),
            "GyroY": np.gradient(columns["EulY"], times),
            "GyroZ": np.gradient(np.deg2rad(heading), times),
            "AcclX": np.sin(times),
            "AcclY": np.cos(times),
            "AcclZ": -np.ones(row_count),
            "AirSpeed": np.full(row_count, 12.0),
        }
    )
    return columns


def test_provenance_statistics_never_call_unknown_data_independent():
    report = build_report(
        _columns(),
        duplicate_header_count=1,
        time_column="Time",
    )

    assert report["euler_unit_equivalence"]["equivalent_within_1e_4_rad"]
    assert report["provenance"]["conclusion"] == "likely_navigation_solution"
    assert not report["provenance"]["independent_ground_truth_allowed"]
    assert report["duplicate_header_count"] == 1


def test_documented_independent_declaration_requires_external_source_upstream():
    report = build_report(
        _columns(),
        duplicate_header_count=0,
        time_column="Time",
        declared_provenance="independent",
        source_id="logger-spec-v3",
    )

    assert report["provenance"]["conclusion"] == "independent"
    assert report["provenance"]["source_id"] == "logger-spec-v3"


def test_quantization_fingerprint_detects_centimeter_grid():
    values = np.asarray([1.00, 1.01, 1.04, 1.07, 1.08], dtype=np.float64)
    fingerprint = quantization_fingerprint(values)

    assert fingerprint["coarsest_grid_with_99_9pct_alignment"] == 0.01
    assert fingerprint["delta_gcd_at_serialized_resolution"] == 0.01
