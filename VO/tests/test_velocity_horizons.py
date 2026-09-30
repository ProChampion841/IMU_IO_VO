"""Long-leg velocity scoring: chunking, state resets, and the VO resume guard.

The point of a horizon metric is that it can disagree with the windowed one.
These tests pin the two mechanisms that make that possible and would fail
silently if broken: state genuinely carries across a streamed block, and state
genuinely resets at the start of a leg. A bug in either direction still
produces a plausible-looking table.
"""

from __future__ import annotations

from pathlib import Path

import argparse

import numpy as np
import pytest
import torch

from tools.train_fixedwing_vo import (
    FINGERPRINT_DEFAULTS,
    epoch_checkpoints,
    fingerprint_differences,
    prune_epoch_checkpoints,
    resume_fingerprint,
)
from vio.data.fixedwing_vo import visual_age_seconds
from vio.models.velocity_horizons import (
    SpanTokens,
    _PositionStats,
    prefix_leg,
    format_horizon_table,
    horizon_csv_row,
    horizon_label,
    horizon_metric_names,
    parse_horizon_minutes,
    scatter_span_tokens,
    stream_horizon_metrics,
    ticks_per_horizon,
    whole_span_series,
)
from vio.models.vision_mamba_vo import AIDING_INPUT_DIM, VisionMambaVO

TICKS = 2000
RATE_HZ = 100.0


def build_model(seed: int = 0) -> VisionMambaVO:
    torch.manual_seed(seed)
    model = VisionMambaVO(visual_dim=8, aiding_dim=16, fusion_dim=16)
    # The heads ship zero-initialised so an untrained model predicts straight
    # and level. That also makes every output identical regardless of the
    # hidden state, which would let a broken reset pass every test below.
    for head in (model.direction_head, model.log_rate_head, model.log_variance_head):
        torch.nn.init.normal_(head.weight, std=0.2)
    return model.eval()


def build_inputs(ticks: int = TICKS, seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    aiding = torch.randn(1, ticks, AIDING_INPUT_DIM, generator=generator)
    token = torch.randn(1, ticks, 8, generator=generator)
    present = (torch.rand(1, ticks, 1, generator=generator) > 0.8).float()
    quality = torch.randn(1, ticks, 1, generator=generator)
    return aiding, token, present, quality


# ---------------------------------------------------------------------------
# naming and chunking
# ---------------------------------------------------------------------------


def test_horizon_labels_and_columns():
    assert horizon_label(5) == "h5m"
    assert horizon_label(0.5) == "h0.5m"
    # Both the typical error and the worst one, per horizon: an RMS over a
    # ten-minute leg can hide a single very bad second completely.
    assert horizon_metric_names((1,)) == (
        "vel_rmse_h1m",
        "vel_max_error_h1m",
        "vel_max_error_time_s_h1m",
        "vel_dir_rmse_h1m",
        "vel_dir_max_error_h1m",
        "vel_dir_max_error_time_s_h1m",
        "diag_boundary_hit_fraction_h1m",
        "diag_mean_usable_confidence_h1m",
        "diag_mean_entropy_h1m",
        "diag_mean_entropy_normalized_h1m",
        "diag_occupied_fraction_h1m",
        # The reliability gate's own accounting. Separate from the four above
        # because those are means over DIFFERENT populations - entropy and
        # confidence cover every cell, occupancy only the cells that passed the
        # weight threshold - so none of them says how many cells were both
        # measured and trustworthy, which is what the gate is tuned on.
        "diag_reliable_cell_fraction_h1m",
        "diag_pair_reliable_fraction_h1m",
        # Which gate did the rejecting. These overlap - a cell can fail two
        # tests - so they do not sum to the total rejected; the question they
        # answer is which threshold is carrying the filtering.
        "diag_rejected_low_confidence_h1m",
        "diag_rejected_high_entropy_h1m",
        "diag_rejected_low_score_margin_h1m",
        "diag_rejected_boundary_peak_h1m",
        # The learned softmax temperature. The gate deliberately does not read
        # it; this column is how a reader sees it drift from the fixed
        # reference the thresholds were chosen against.
        "diag_correlation_temperature_h1m",
    )
    # Two horizons produce the same block twice, in order. Counted off the
    # one-horizon block rather than hardcoded, so adding a diagnostic changes
    # the column list above (which IS the contract) without also failing here
    # on an arithmetic restatement of it.
    block = len(horizon_metric_names((1,)))
    assert len(horizon_metric_names((1, 5))) == 2 * block
    # position=True adds a fixed five columns to the block, once.
    assert len(horizon_metric_names((1,), position=True)) == block + 5


def whole_span_run(
    ticks: int = TICKS,
    rotation: bool = False,
    dropouts: bool = False,
    warmup: int = 20,
):
    """The whole span as ONE continuous leg - what the plots are drawn from."""

    model = build_model()
    intervals = np.full(ticks, 1.0 / RATE_HZ)
    if dropouts:
        # A logger that drops samples: most gaps are nominal, a few are long.
        # The MEDIAN gap stays at the nominal one while the MEAN rises, which
        # is exactly the clock that used to make the whole-span leg overshoot
        # the span it was cut from.
        intervals[::20] *= 3.0
    times = np.concatenate(([0.0], np.cumsum(intervals[:-1])))
    aiding = np.random.default_rng(2).standard_normal((ticks, AIDING_INPUT_DIM))
    altitude = np.full(ticks, np.log(120.0))
    target = np.stack(
        (np.full(ticks, 20.0), 5.0 * np.sin(times * 0.4), np.full(ticks, 2.0)), axis=1
    )
    tick = np.arange(0, ticks, 5, dtype=np.int64)
    tokens = SpanTokens(
        tick=tick,
        token=torch.zeros(tick.size, 8),
        quality=torch.zeros(tick.size, 1),
        visual_dim=8,
    )
    return whole_span_series(
        model,
        aiding=aiding,
        log_altitude=altitude,
        target_velocity=target,
        times_s=times,
        span=(0, ticks),
        tokens=tokens,
        deployment_latency_s=0.35,
        device=torch.device("cpu"),
        warmup_ticks=warmup,
        block_ticks=250,
        baseline=target.mean(axis=0),
        rotation_body_to_ned=(
            np.tile(np.eye(3), (ticks, 1, 1)) if rotation else None
        ),
    )


def test_whole_span_run_is_a_single_unbroken_leg():
    entry = whole_span_run(rotation=True)
    assert entry["fits"] is True
    # One leg covering (almost) the whole span - the trailing partial tick or
    # two can be dropped by the leg-length rounding, nothing more.
    assert entry["ticks"] >= TICKS - 2
    assert entry["series"]["vel_error_m_s"].shape[0] == 1


def test_whole_span_run_survives_a_clock_with_dropouts():
    """The regression behind "only the summary figure was written".

    On a jittery clock the span's duration divided by the MEDIAN interval comes
    out longer than the span itself, so the one leg did not fit, no leg was
    scored, and the trajectory and error figures - which are drawn from this
    run's series - were skipped without an error. The summary figure reads the
    horizon tables instead, so it still appeared, and the failure looked like a
    plotting bug rather than a leg that was never cut.
    """

    entry = whole_span_run(rotation=True, dropouts=True)
    assert "skipped" not in entry
    assert entry["fits"] is True
    # The leg is the span exactly - not a count re-derived from the clock.
    assert entry["ticks"] == TICKS
    assert entry["series"]["vel_error_m_s"].shape == (1, TICKS)
    assert "trajectory_predicted_ned" in entry["series"]

    # And the derivation this replaces really would have overshot: proof that
    # the test's clock reproduces the reported failure rather than passing for
    # an unrelated reason.
    times = np.concatenate(
        ([0.0], np.cumsum(np.where(np.arange(TICKS - 1) % 20 == 0, 3.0, 1.0) / RATE_HZ))
    )
    minutes = float(times[TICKS - 1] - times[0]) / 60.0
    assert ticks_per_horizon(times, (0, TICKS), minutes) > TICKS


def test_the_time_of_the_worst_error_is_reported_and_locatable():
    """A maximum with no timestamp cannot be gone and looked at."""

    entry = whole_span_run(rotation=True)
    interval = entry["tick_interval_s"]
    for stem in ("vel_max_error", "vel_dir_max_error", "pos_error_max"):
        seconds = entry[f"{stem}_time_s"]
        assert 0.0 <= seconds <= entry["ticks"] * interval
        # The absolute telemetry index has to agree with the leg-relative one.
        assert entry[f"{stem}_flight_tick"] == entry[f"{stem}_tick"]

    # And the reported peak must be the actual peak of the reported curve.
    curve = entry["series"]["vel_error_m_s"][0]
    assert entry["vel_max_error"] == pytest.approx(np.nanmax(curve))
    assert entry["vel_max_error_tick"] == int(np.nanargmax(curve))


def test_parse_sorts_deduplicates_and_rejects_nonsense():
    assert parse_horizon_minutes("10, 1,5 ,1") == (1.0, 5.0, 10.0)
    assert parse_horizon_minutes("") == ()
    assert parse_horizon_minutes(None) == ()
    with pytest.raises(ValueError):
        parse_horizon_minutes("0")
    with pytest.raises(ValueError):
        parse_horizon_minutes("-5")
    with pytest.raises(ValueError):
        parse_horizon_minutes("ten")


def test_tick_count_follows_the_measured_clock_not_the_nominal_rate():
    # A logger that actually ran at 50 Hz must give half as many ticks per
    # minute, without anything being told what rate to expect.
    times = np.arange(0, 6000) * 0.02
    assert ticks_per_horizon(times, (0, times.size), 1.0) == 3000


def test_a_horizon_is_the_first_leg_only_and_is_never_shortened():
    # A horizon is the FIRST H minutes of the split, not the split cut into
    # repeated H-minute pieces: there is no trailing remainder to drop and no
    # second cold start averaged in with the first.
    assert prefix_leg((0, 2500), 1000) == (0, 1000)
    assert prefix_leg((0, 5000), 1000) == (0, 1000)
    # A horizon that does not fit is skipped, never quietly truncated.
    assert prefix_leg((0, 900), 1000) is None
    # Exactly-fitting is not "does not fit".
    assert prefix_leg((0, 1000), 1000) == (0, 1000)


def test_the_leg_starts_at_the_span_and_stays_inside_it():
    leg = prefix_leg((300, 4000), 700)
    assert leg == (300, 1000)
    start, end = leg
    assert start >= 300 and end <= 4000

    with pytest.raises(ValueError):
        prefix_leg((0, 1000), 0)


# ---------------------------------------------------------------------------
# the two mechanisms
# ---------------------------------------------------------------------------


def test_streaming_in_blocks_equals_one_pass():
    """Carrying state must be exact, or a long leg is not one run at all."""

    model = build_model()
    aiding, token, present, quality = build_inputs(ticks=120)
    times = torch.arange(120, dtype=torch.float64) / RATE_HZ
    age, _ = visual_age_seconds(present, times, deployment_latency_s=0.35)
    log_altitude = aiding[..., 4]
    with torch.no_grad():
        whole = model(
            aiding, token, present, age, visual_quality=quality, log_altitude=log_altitude
        )
        state = None
        pieces = []
        for begin in range(0, 120, 17):  # deliberately not a divisor
            stop = min(begin + 17, 120)
            block, state = model.forward_stream(
                aiding[:, begin:stop],
                token[:, begin:stop],
                present[:, begin:stop],
                age[:, begin:stop],
                visual_quality=quality[:, begin:stop],
                log_altitude=log_altitude[:, begin:stop],
                state=state,
            )
            pieces.append(block["predicted_velocity"])
    streamed = torch.cat(pieces, dim=1)
    assert torch.allclose(whole["predicted_velocity"], streamed, atol=1e-5)


def test_a_reset_actually_discards_the_past():
    """A leg that starts fresh must not equal one that carried state into it."""

    model = build_model()
    aiding, token, present, quality = build_inputs(ticks=120)
    times = torch.arange(120, dtype=torch.float64) / RATE_HZ
    log_altitude = aiding[..., 4]
    with torch.no_grad():
        whole_age, _ = visual_age_seconds(present, times, deployment_latency_s=0.35)
        carried = model(
            aiding, token, present, whole_age,
            visual_quality=quality, log_altitude=log_altitude,
        )
        # A real reset forgets staleness along with everything else, so the
        # age fed here is recomputed fresh from the reset slice (carry=None)
        # - not sliced from `carried`'s age field, which would smuggle
        # pre-reset history back in through the one channel this test exists
        # to show is gone.
        reset_age, _ = visual_age_seconds(
            present[:, 40:], times[40:], deployment_latency_s=0.35
        )
        fresh, _ = model.forward_stream(
            aiding[:, 40:],
            token[:, 40:],
            present[:, 40:],
            reset_age,
            visual_quality=quality[:, 40:],
            log_altitude=log_altitude[:, 40:],
            state=None,
        )
    difference = (
        fresh["predicted_velocity"] - carried["predicted_velocity"][:, 40:]
    ).abs()
    # Only the first ticks after the reset can differ; that they differ AT ALL
    # is the property under test.
    assert float(difference.max()) > 1e-4


def test_block_size_does_not_change_the_result():
    model = build_model()
    ticks = TICKS
    times = np.arange(ticks) / RATE_HZ
    aiding = np.random.default_rng(0).standard_normal((ticks, AIDING_INPUT_DIM))
    altitude = np.full(ticks, np.log(120.0))
    target = np.tile(np.array([20.0, 1.0, 2.0]), (ticks, 1))
    tick = np.arange(0, ticks, 5, dtype=np.int64)
    tokens = SpanTokens(
        tick=tick,
        token=torch.zeros(tick.size, 8),
        quality=torch.zeros(tick.size, 1),
        visual_dim=8,
    )
    common = dict(
        aiding=aiding,
        log_altitude=altitude,
        target_velocity=target,
        times_s=times,
        span=(0, ticks),
        tokens=tokens,
        deployment_latency_s=0.35,
        horizons_minutes=(0.1,),
        device=torch.device("cpu"),
        warmup_ticks=20,
    )
    coarse = stream_horizon_metrics(model, block_ticks=10_000, **common)
    fine = stream_horizon_metrics(model, block_ticks=37, **common)
    assert coarse["h0.1m"]["vel_rmse"] == pytest.approx(
        fine["h0.1m"]["vel_rmse"], rel=1e-5
    )


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def horizon_run(
    horizons,
    ticks: int = TICKS,
    warmup: int = 20,
    rotation: bool = False,
    collect_series: bool = False,
):
    model = build_model()
    times = np.arange(ticks) / RATE_HZ
    aiding = np.random.default_rng(2).standard_normal((ticks, AIDING_INPUT_DIM))
    altitude = np.full(ticks, np.log(120.0))
    # The lateral axis has to vary, or the constant-mean baseline is exactly
    # right and its RMSE is zero - which would make the skill comparison
    # meaningless rather than merely hard.
    target = np.stack(
        (
            np.full(ticks, 20.0),
            5.0 * np.sin(times * 0.4),
            np.full(ticks, 2.0),
        ),
        axis=1,
    )
    tick = np.arange(0, ticks, 5, dtype=np.int64)
    tokens = SpanTokens(
        tick=tick,
        token=torch.zeros(tick.size, 8),
        quality=torch.zeros(tick.size, 1),
        visual_dim=8,
    )
    body_to_ned = np.tile(np.eye(3), (ticks, 1, 1)) if rotation else None
    return stream_horizon_metrics(
        model,
        aiding=aiding,
        log_altitude=altitude,
        target_velocity=target,
        times_s=times,
        span=(0, ticks),
        tokens=tokens,
        deployment_latency_s=0.35,
        horizons_minutes=horizons,
        device=torch.device("cpu"),
        warmup_ticks=warmup,
        block_ticks=250,
        baseline=target.mean(axis=0),
        rotation_body_to_ned=body_to_ned,
        collect_series=collect_series,
    )


def test_a_horizon_that_does_not_fit_is_reported_not_dropped():
    results = horizon_run((0.1, 5.0))
    assert results["h5m"]["fits"] is False
    assert "needs 5 min" in results["h5m"]["skipped"]
    # It still occupies its columns, so a CSV row keeps a fixed width.
    row = horizon_csv_row(results, prefix="val_")
    assert np.isnan(row["val_vel_rmse_h5m"])
    assert np.isnan(row["val_vel_max_error_h5m"])
    assert not np.isnan(row["val_vel_rmse_h0.1m"])
    assert not np.isnan(row["val_vel_max_error_h0.1m"])
    assert "skipped" in format_horizon_table(results)


def test_max_error_is_reported_alongside_rmse():
    """An RMS over a long leg can hide the one second that went badly wrong."""

    entry = horizon_run((0.1,))["h0.1m"]
    # A maximum can never be below the RMS it is drawn from, and here it is
    # strictly above - which is the whole reason for reporting it.
    assert entry["vel_max_error"] > entry["vel_rmse"]
    assert entry["vel_dir_max_error"] > entry["vel_dir_rmse"]
    # The maximum is located, not just reported: within the leg and inside it.
    assert 0 <= entry["vel_max_error_tick"] < entry["ticks"]
    assert 0 <= entry["vel_dir_max_error_tick"] < entry["ticks"]
    assert "vel_max" in format_horizon_table({"h0.1m": entry})


def test_warmup_is_excluded_from_the_leg():
    ticks_per_leg = int(0.1 * 60 * RATE_HZ)
    lenient = horizon_run((0.1,), warmup=0)["h0.1m"]
    strict = horizon_run((0.1,), warmup=100)["h0.1m"]
    # One leg per horizon, so the scored count is the leg minus its warm-up.
    assert lenient["scored_ticks"] == ticks_per_leg
    assert strict["scored_ticks"] == ticks_per_leg - 100


def test_a_horizon_reports_one_leg_so_rmse_is_unambiguous():
    """A horizon is one prefix, so there is exactly one RMS to report.

    The retired chunked scoring cut a split into repeated legs and had to
    report mean-of-legs, pooled and worst-leg separately, because those three
    genuinely differ. A single leg collapses them, and reporting one number
    removes the chance of comparing a mean against a pooled value.
    """

    entry = horizon_run((0.1,))["h0.1m"]
    assert entry["fits"] is True
    assert entry["ticks"] == int(0.1 * 60 * RATE_HZ)
    assert entry["vel_rmse"] > 0.0
    assert entry["baseline_vel_rmse"] > 0.0
    # The superseded multi-leg keys must be gone, not silently still emitted.
    for retired in ("chunks", "per_chunk", "ticks_per_chunk",
                    "vel_rmse_pooled", "vel_rmse_worst_leg"):
        assert retired not in entry


def test_tokens_land_on_their_own_leg_only():
    tick = np.array([5, 999, 1000, 1500], dtype=np.int64)
    tokens = SpanTokens(
        tick=tick,
        token=torch.arange(tick.size * 4, dtype=torch.float32).reshape(tick.size, 4),
        quality=torch.ones(tick.size, 1),
        visual_dim=4,
    )
    _, _, present = scatter_span_tokens(
        tokens, [(0, 1000), (1000, 2000)], 1000, device=torch.device("cpu")
    )
    assert present[0].sum() == 2  # ticks 5 and 999
    assert present[1].sum() == 2  # ticks 1000 and 1500
    assert present[0, 5] == 1 and present[1, 500] == 1


# ---------------------------------------------------------------------------
# dead-reckoning position error
# ---------------------------------------------------------------------------


def straight_leg(count: int = 2, ticks: int = 6000, dt: float = 0.01):
    """Level flight due north at 20 m/s, identity attitude."""

    times = torch.arange(ticks, dtype=torch.float64).unsqueeze(0).repeat(count, 1) * dt
    rotation = torch.eye(3).view(1, 1, 3, 3).repeat(count, ticks, 1, 1)
    target = torch.zeros(count, ticks, 3)
    target[:, :, 0] = 20.0
    return times, rotation, target


def test_position_error_is_the_integral_of_the_velocity_error():
    times, rotation, target = straight_leg()
    predicted = target.clone()
    predicted[0, :, 0] += 1.0  # 1 m/s too fast
    predicted[1, :, 0] += 2.0  # 2 m/s too fast
    mask = torch.ones(*target.shape[:2])

    stats = _PositionStats(2, torch.device("cpu"))
    stats.update(predicted, target, rotation, times, mask)
    legs = stats.per_leg()
    # 5999 intervals of 0.01 s = 59.99 s of integration.
    assert legs["pos_error_final"] == pytest.approx([59.99, 119.98])
    assert legs["path_length_m"] == pytest.approx([1199.8, 1199.8])
    assert legs["pos_drift_percent"] == pytest.approx([5.0, 10.0])


def test_position_does_not_depend_on_the_block_size():
    """The integral is carried across blocks, so subdividing must not move it."""

    times, rotation, target = straight_leg(count=1)
    predicted = target.clone()
    predicted[:, :, 1] += 0.5
    mask = torch.ones(*target.shape[:2])
    finals = []
    for block in (target.shape[1], 1000, 137):
        stats = _PositionStats(1, torch.device("cpu"))
        for begin in range(0, target.shape[1], block):
            stop = begin + block
            stats.update(
                predicted[:, begin:stop],
                target[:, begin:stop],
                rotation[:, begin:stop],
                times[:, begin:stop],
                mask[:, begin:stop],
            )
        finals.append(float(stats.per_leg()["pos_error_final"][0]))
    assert finals[1] == pytest.approx(finals[0])
    assert finals[2] == pytest.approx(finals[0])


def test_the_error_is_rotated_at_every_tick_not_once():
    """A body-frame error flown round a full circle has to integrate away.

    If the rotation were applied once, or not at all, a constant body-frame
    error would accumulate linearly instead of closing the loop - which is the
    difference between a drift figure that means something and one that does
    not.
    """

    ticks, dt = 6000, 0.01
    seconds = np.arange(ticks) * dt
    yaw = 2 * np.pi * seconds / (ticks * dt)  # exactly one turn over the leg
    rotation = np.zeros((ticks, 3, 3))
    rotation[:, 2, 2] = 1.0
    rotation[:, 0, 0] = np.cos(yaw)
    rotation[:, 0, 1] = -np.sin(yaw)
    rotation[:, 1, 0] = np.sin(yaw)
    rotation[:, 1, 1] = np.cos(yaw)

    times = torch.from_numpy(seconds).unsqueeze(0)
    body_to_ned = torch.from_numpy(rotation).float().unsqueeze(0)
    target = torch.zeros(1, ticks, 3)
    target[:, :, 0] = 20.0
    predicted = target.clone()
    predicted[:, :, 0] += 1.0
    mask = torch.ones(1, ticks)

    stats = _PositionStats(1, torch.device("cpu"))
    stats.update(predicted, target, body_to_ned, times, mask)
    legs = stats.per_leg()
    # Back where it started, but 19 m away at the far side of the circle: the
    # radius is 1 m/s over an angular rate of 2*pi/60 rad/s.
    assert float(legs["pos_error_final"][0]) < 0.05
    assert float(legs["pos_error_max"][0]) == pytest.approx(
        2.0 / (2 * np.pi / (ticks * dt)), rel=1e-3
    )


def test_position_starts_at_the_end_of_the_warmup():
    times, rotation, target = straight_leg(count=1)
    predicted = target.clone()
    predicted[:, :, 0] += 1.0
    mask = torch.ones(*target.shape[:2])
    mask[:, :2000] = 0.0
    stats = _PositionStats(1, torch.device("cpu"))
    stats.update(predicted, target, rotation, times, mask)
    # 3999 intervals remain, not 5999: the leg's origin is where it goes live.
    assert float(stats.per_leg()["pos_error_final"][0]) == pytest.approx(39.99)


def test_a_non_finite_tick_voids_its_leg_rather_than_resuming():
    """An integral cannot skip a hole; the distance after one is undefined."""

    times, rotation, target = straight_leg(count=2, ticks=100)
    predicted = target.clone()
    predicted[:, :, 0] += 1.0
    predicted[1, 50, 0] = float("nan")
    mask = torch.ones(*target.shape[:2])
    stats = _PositionStats(2, torch.device("cpu"))
    stats.update(predicted, target, rotation, times, mask)
    legs = stats.per_leg()
    assert np.isfinite(legs["pos_error_final"][0])
    assert np.isnan(legs["pos_error_final"][1])


def test_position_columns_appear_only_when_a_rotation_is_supplied():
    velocity_only = horizon_metric_names((5,))
    with_position = horizon_metric_names((5,), position=True)
    assert "pos_error_final_h5m" not in velocity_only
    assert with_position[: len(velocity_only)] == velocity_only
    assert "pos_error_final_h5m" in with_position
    assert "pos_drift_percent_h5m" in with_position


def test_horizon_run_reports_position_end_to_end():
    results = horizon_run((0.1,), rotation=True)
    entry = results["h0.1m"]
    assert entry["pos_error_final"] > 0.0
    assert entry["pos_error_max"] >= entry["pos_error_final"]
    assert "pos_final" in format_horizon_table(results)
    row = horizon_csv_row(results, prefix="val_")
    assert "val_pos_error_final_h0.1m" in row
    # Without a rotation there is nothing to integrate, and no column claims
    # there is.
    plain = horizon_csv_row(horizon_run((0.1,)), prefix="val_")
    assert not any("pos_" in name for name in plain)


# ---------------------------------------------------------------------------
# per-tick series, for the plots
# ---------------------------------------------------------------------------


def test_series_is_absent_unless_requested():
    results = horizon_run((0.1,), rotation=True, collect_series=False)
    assert "series" not in results["h0.1m"]


def test_series_matches_the_scalar_summary_it_was_drawn_from():
    """The plotted curve and the printed number have to agree with each other."""

    # Series come from the whole-span run: stream_horizon_metrics deliberately
    # drops them, so the horizon table stays a table of scalars.
    entry = whole_span_run(rotation=True)
    series = entry["series"]
    legs = 1

    assert series["vel_error_m_s"].shape == (legs, entry["ticks"])
    assert series["time_since_start_s"].shape == (entry["ticks"],)
    # RMS of the per-tick curve, ignoring the NaN warm-up, must reproduce the
    # per-leg RMSE the scalar table already reports.
    per_leg_rmse = np.sqrt(np.nanmean(np.square(series["vel_error_m_s"]), axis=1))
    assert per_leg_rmse[0] == pytest.approx(entry["vel_rmse"], rel=1e-6)

    predicted = series["trajectory_predicted_ned"]
    reference = series["trajectory_reference_ned"]
    drift = np.linalg.norm(predicted - reference, axis=-1)
    assert drift[0, -1] == pytest.approx(entry["pos_error_final"], rel=1e-6)
    # The two agree everywhere AFTER the warm-up. Before it, "pos_error_m" is
    # masked to NaN like every other per-tick curve - nothing has been scored
    # yet - while the trajectory itself correctly sits at the origin, since it
    # has not started integrating; the two conventions differ on purpose and
    # this checks both halves.
    warmup = entry["warmup_ticks"]
    assert np.allclose(drift[:, warmup:], series["pos_error_m"][:, warmup:])
    assert np.isnan(series["pos_error_m"][:, :warmup]).all()
    assert np.allclose(predicted[:, :warmup], 0.0)
    assert np.allclose(reference[:, :warmup], 0.0)
    # Every trajectory starts at its own leg's origin.
    assert np.allclose(predicted[:, 0], 0.0)
    assert np.allclose(reference[:, 0], 0.0)


def test_warmup_ticks_are_nan_in_the_series_not_zero():
    series = whole_span_run(rotation=True, warmup=30)["series"]
    assert np.isnan(series["vel_error_m_s"][:, :30]).all()
    assert np.isfinite(series["vel_error_m_s"][:, 30:]).all()


def test_series_without_rotation_carries_no_position_keys():
    series = whole_span_run(rotation=False)["series"]
    assert "vel_error_m_s" in series
    assert "pos_error_m" not in series
    assert "trajectory_predicted_ned" not in series


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------


def test_split_figures_are_written_from_a_whole_span_run(tmp_path):
    from vio.utils.horizon_plots import save_split_plots

    whole = whole_span_run(rotation=True)
    results = horizon_run((0.1,), rotation=True)
    written = save_split_plots(tmp_path, "validation", whole, results)
    names = sorted(path.name for path in written)
    assert names == [
        "validation_errors.png",
        "validation_summary_vs_horizon.png",
        "validation_trajectory.png",
        "validation_velocity.png",
    ]
    assert all(path.stat().st_size > 0 for path in written)


FIGURE_NAMES = [
    "validation_errors.png",
    "validation_summary_vs_horizon.png",
    "validation_trajectory.png",
    "validation_velocity.png",
]


def test_every_figure_is_written_even_with_nothing_to_draw(tmp_path):
    """All four files, always.

    A missing file reads as a plotting fault and carries no information. When
    there is nothing to draw the figure is still written, saying why - which
    is the same fact, delivered where the reader is already looking.
    """

    from vio.utils.horizon_plots import save_split_plots

    # No reference attitude: no path to integrate, and every horizon too long
    # for the span, so not one of the four has real data behind it.
    whole = whole_span_run(rotation=False)
    results = horizon_run((5.0, 10.0), rotation=False)
    written = save_split_plots(tmp_path, "validation", whole, results)
    assert sorted(path.name for path in written) == FIGURE_NAMES
    assert all(path.stat().st_size > 0 for path in written)


def test_summary_plot_uses_only_fitted_horizons(tmp_path):
    from vio.utils.horizon_plots import save_horizon_summary_plot

    results = horizon_run((0.1, 5.0), rotation=True, collect_series=True)
    path = save_horizon_summary_plot(tmp_path, "validation", results)
    assert path is not None and path.is_file()

    # Every horizon skipped -> still a file, carrying the reasons.
    all_skipped = horizon_run((5.0, 10.0), rotation=True, collect_series=True)
    written = save_horizon_summary_plot(tmp_path, "validation", all_skipped)
    assert written is not None and written.stat().st_size > 0


# ---------------------------------------------------------------------------
# the resume guard
# ---------------------------------------------------------------------------


class FakeSource:
    attitude_columns = ("NavEulX", "NavEulY", "NavEulZ")
    altitude_column = "RelativeAlt"
    times_s = np.arange(1000) / RATE_HZ


class FakeNormalizer:
    def as_dict(self):
        return {"log_altitude_mean": 4.6, "delta_time_scale": 0.01}


def fake_args(**overrides) -> argparse.Namespace:
    values = dict(
        dataset="dataset",
        csv_name="flight.csv",
        calibration=None,
        image_size=(576, 1024),
        frame_gap=1,
        deployment_latency_s=0.35,
        # Preprocessing knobs the fingerprint reads. They must appear here with
        # the CLI's own defaults, or these tests stop exercising the resume
        # path the trainer actually takes.
        max_frame_gap_s=None,
        image_time_offset=0.0,
        image_time_offset_file=None,
        lever_arm=None,
        window_length=600,
        stride=300,
        warmup=20,
        max_visual_events=120,
        disable_visual_input=False,
        visual_dim=64,
        stem_dim=64,
        stem_depth=2,
        patch_size=8,
        context_grid=(12, 16),
        token_grid=6,
        correlation_radius=4,
        aiding_dim=64,
        fusion_dim=96,
        rotation_mode="field",
        ablate_body_rate=False,
        ablate_visual_age=False,
        # The reliability gate, at the CLI's off-defaults - same reason as the
        # preprocessing knobs above: the fingerprint reads them, so leaving
        # them out here would stop these tests exercising the real resume path.
        min_pool_weight=1e-4,
        max_cell_entropy=1.0,
        min_cell_confidence=0.0,
        min_score_margin=0.0,
        reject_boundary_peaks=False,
        min_reliable_cell_fraction=0.0,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def fingerprint(**overrides):
    return resume_fingerprint(
        fake_args(**overrides),
        FakeSource(),
        FakeNormalizer(),
        {"train": (0, 700), "validation": (700, 850), "test": (850, 1000)},
    )


def test_an_identical_command_resumes():
    assert fingerprint_differences(fingerprint(), fingerprint()) == []


def test_a_changed_contract_is_refused_by_name():
    differences = fingerprint_differences(fingerprint(), fingerprint(frame_gap=3))
    assert differences == ["contract.frame_gap: 1 -> 3"]


def test_a_changed_model_shape_is_refused_by_name():
    differences = fingerprint_differences(fingerprint(), fingerprint(fusion_dim=128))
    assert differences == ["model.fusion_dim: 96 -> 128"]


def test_optimisation_knobs_are_not_fingerprinted():
    """Changing --epochs or --learning-rate on a resume is legitimate."""

    parts = fingerprint()
    flat = {key for group in parts.values() for key in group}
    assert not flat & {"epochs", "learning_rate", "batch_size", "num_workers"}


def test_a_checkpoint_without_a_fingerprint_is_refused():
    differences = fingerprint_differences(None, fingerprint())
    assert len(differences) == 1 and "no fingerprint" in differences[0]


def test_every_reliability_gate_setting_is_fingerprinted():
    """A resume that changes the gate must be refused, not silently accepted.

    The gate changes WHICH correlation cells count and whether a pair is
    delivered at all, while every tensor shape stays identical and
    ``frontend_id`` does not move - so load_state_dict cannot catch it and
    neither can any other guard. Without this the optimizer state from an
    ungated run would continue, unremarked, on a gated frontend.
    """

    for name, changed in (
        ("min_pool_weight", 0.05),
        ("max_cell_entropy", 0.6),
        ("min_cell_confidence", 0.1),
        ("min_score_margin", 0.02),
        ("reject_boundary_peaks", True),
        ("min_reliable_cell_fraction", 0.25),
    ):
        differences = fingerprint_differences(
            fingerprint(), fingerprint(**{name: changed})
        )
        assert len(differences) == 1, (name, differences)
        assert differences[0].startswith(f"model.{name}: "), (name, differences)


def test_a_pre_gate_checkpoint_still_resumes():
    """A checkpoint written before the gate existed must not be refused.

    Its absent keys provably mean "ungated" - the gate did not exist - so
    FINGERPRINT_DEFAULTS supplies the off-values rather than failing on a key
    the run could not have carried. This is the opposite of frontend_id, whose
    absence proves nothing and is therefore deliberately NOT defaulted.
    """

    current = fingerprint()
    pre_gate = {
        group: {
            key: value for key, value in values.items()
            if key not in (
                "min_pool_weight", "max_cell_entropy", "min_cell_confidence",
                "min_score_margin", "reject_boundary_peaks",
                "min_reliable_cell_fraction",
            )
        }
        for group, values in current.items()
    }
    assert fingerprint_differences(pre_gate, current) == []


# ---------------------------------------------------------------------------
# per-epoch checkpoints
# ---------------------------------------------------------------------------


def test_epoch_checkpoints_sort_by_epoch_not_by_name(tmp_path):
    for epoch in (2, 10, 1):
        (tmp_path / f"epoch_{epoch:04d}.pt").write_bytes(b"x")
    assert [p.name for p in epoch_checkpoints(tmp_path)] == [
        "epoch_0001.pt",
        "epoch_0002.pt",
        "epoch_0010.pt",
    ]


def test_pruning_keeps_the_newest_and_touches_nothing_else(tmp_path):
    for epoch in range(1, 6):
        (tmp_path / f"epoch_{epoch:04d}.pt").write_bytes(b"x")
    # Files this trainer did not write must survive whatever the setting.
    (tmp_path / "best.pt").write_bytes(b"x")
    (tmp_path / "last.pt").write_bytes(b"x")
    (tmp_path / "epoch_notanumber.pt").write_bytes(b"x")

    prune_epoch_checkpoints(tmp_path, 2)
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == [
        "best.pt",
        "epoch_0004.pt",
        "epoch_0005.pt",
        "epoch_notanumber.pt",
        "last.pt",
    ]


def test_keep_last_zero_deletes_nothing(tmp_path):
    """The default must never remove a checkpoint the user did not ask it to."""

    for epoch in range(1, 4):
        (tmp_path / f"epoch_{epoch:04d}.pt").write_bytes(b"x")
    prune_epoch_checkpoints(tmp_path, 0)
    assert len(epoch_checkpoints(tmp_path)) == 3
    prune_epoch_checkpoints(tmp_path, -1)
    assert len(epoch_checkpoints(tmp_path)) == 3


def test_the_frontend_algorithm_is_fingerprinted_by_name():
    """Shapes are not identity: v2 and v3 fit each other and differ in meaning.

    The pooling weight moved from raw confidence to usable confidence without
    changing a single tensor dimension, so load_state_dict(strict=True) accepts
    a v2 checkpoint into a v3 build and would score it under a measurement it
    was never trained with. Only the name catches that.
    """

    import copy

    from vio.models.vision_mamba_vo import VisionMambaFlowFrontend

    current = str(VisionMambaFlowFrontend.frontend_id)
    reference = fingerprint()
    assert reference["model"]["frontend_id"] == current

    # A checkpoint from an older algorithm is refused by name.
    stale = copy.deepcopy(reference)
    stale["model"]["frontend_id"] = "vision_mamba_correlation_v2"
    assert fingerprint_differences(stale, reference) == [
        f"model.frontend_id: 'vision_mamba_correlation_v2' -> {current!r}"
    ]

    # And so is one predating the key. It CANNOT be shown to have used the
    # current algorithm, so it must not be waved through by a default.
    assert "frontend_id" not in FINGERPRINT_DEFAULTS
    absent = copy.deepcopy(reference)
    del absent["model"]["frontend_id"]
    assert fingerprint_differences(absent, reference) == [
        f"model.frontend_id: '<absent>' -> {current!r}"
    ]


def test_skipped_horizons_keep_their_place_in_the_table():
    """Order is the requested order, whether a horizon fitted or not.

    A skip is recorded before the encoding pass and a fit only after it, so
    plain insertion order puts every skipped horizon above every fitted one -
    h40m above h0.5m. The table exists to be read down a column as the horizon
    grows, and that ordering destroys it.
    """

    # The CLI sorts before it gets here, so this is what a real run looks like:
    # one skip in the middle of the fitted ones, staying in its place.
    results = horizon_run((0.1, 0.2, 5.0))
    assert list(results) == ["h0.1m", "h0.2m", "h5m"]
    assert results["h5m"]["fits"] is False
    assert results["h0.1m"]["fits"] is True

    # And the rendered table follows the same order. Take the first token of
    # each line and keep the ones that name a horizon, so the header row and
    # the baseline block below it cannot be mistaken for table rows.
    wanted = set(results)
    ordered = []
    for line in format_horizon_table(results).splitlines():
        token = line.split()[0] if line.split() else ""
        if token in wanted and token not in ordered:
            ordered.append(token)
    assert ordered == ["h0.1m", "h0.2m", "h5m"]

    # The contract is the REQUESTED order, not a sort applied here: a caller
    # that asks out of order gets its own order back, and parse_horizon_minutes
    # is the single place that sorting happens.
    assert list(horizon_run((0.2, 5.0, 0.1))) == ["h0.2m", "h5m", "h0.1m"]
    assert parse_horizon_minutes("5,0.1,0.2") == (0.1, 0.2, 5.0)


# ---------------------------------------------------------------------------
# multi-GPU launch
# ---------------------------------------------------------------------------


def test_a_gpu_list_is_recognised_and_a_single_device_is_not():
    from tools.train_fixedwing_vo import parse_gpu_list

    assert parse_gpu_list("0,1,2,3,4,5,6,7,8,9") == list(range(10))
    assert parse_gpu_list("cuda:0,cuda:1") == [0, 1]
    assert parse_gpu_list("0, 1 ,2") == [0, 1, 2]
    for single in ("cuda", "cuda:0", "cpu", "3", ""):
        assert parse_gpu_list(single) is None, single
    with pytest.raises(ValueError):
        parse_gpu_list("0,1,1")


def test_a_gpu_list_becomes_a_torchrun_job_with_one_rank_per_card(monkeypatch):
    """The list must become DDP, not DataParallel and not ten ranks on cuda:0.

    setup_distributed takes the card from LOCAL_RANK, so a list of GPUs is only
    meaningful once there is one process per card. This checks the translation
    without needing the cards: what matters is nproc_per_node, the visible-device
    mask, and that the list is rewritten to a plain "cuda" for the child - which
    would otherwise re-enter the same branch and fork forever.
    """

    import torch

    from tools import train_fixedwing_vo as train

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 10)
    captured = {}

    def fake_call(command, env=None):
        captured["command"] = command
        captured["env"] = env
        return 0

    monkeypatch.setattr(train.subprocess, "call", fake_call)
    code = train.relaunch_under_torchrun(
        ["--dataset", "d", "--device", "0,1,2,3,4,5,6,7,8,9", "--epochs", "3"],
        "0,1,2,3,4,5,6,7,8,9",
    )
    assert code == 0
    command = captured["command"]
    assert "--nproc_per_node=10" in command
    assert "--standalone" in command
    assert "torch.distributed.run" in command
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7,8,9"
    # The child must not see the list again.
    assert "0,1,2,3,4,5,6,7,8,9" not in command
    assert command[command.index("--device") + 1] == "cuda"
    # Everything else is passed through untouched.
    assert command[command.index("--epochs") + 1] == "3"


def test_naming_a_card_the_machine_lacks_fails_before_any_work(monkeypatch):
    import torch

    from tools import train_fixedwing_vo as train

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(SystemExit) as caught:
        train.relaunch_under_torchrun(["--device", "0,1,7"], "0,1,7")
    assert "7" in str(caught.value)


def test_the_selection_metric_is_recorded_and_defaults_to_all_three_axes():
    """best.pt must say which number chose it.

    With no early stopping and thousands of epochs, best.pt is the minimum of a
    noisy validation statistic over very many draws. Selecting on one axis
    picks the luckiest epoch on that axis while the other two may have got
    worse, so the default is the 3-axis RMSE - and whichever was used has to be
    written down, or a later comparison is against an unknown quantity.
    """

    from tools import train_fixedwing_vo as train

    parser = train.build_parser()
    assert parser.get_default("select_on") == "vel_rmse"
    # There is no early stopping, so --epochs IS the run length and the default
    # has to be a trained model rather than a smoke test.
    assert parser.get_default("epochs") >= 1000

    choices = {
        action.dest: action.choices
        for action in parser._actions if action.dest == "select_on"
    }["select_on"]
    assert "vel_rmse_y" in choices and "loss" in choices


def test_a_key_never_holds_a_metric_it_does_not_name():
    """val_vel_rmse_y must be the y axis or nothing at all.

    It was written unconditionally from whatever score selected the checkpoint,
    so once --select-on existed it would have named one metric and held
    another. A missing number is recoverable; a mislabelled one is not.
    """

    from tools import train_fixedwing_vo as train

    import math
    from pathlib import Path

    source = Path(train.__file__).read_text(encoding="utf-8")
    assert '"val_vel_rmse_y": score if args.select_on == "vel_rmse_y"' in source
    assert '"select_on": str(args.select_on)' in source
    assert math.isnan(float("nan"))
