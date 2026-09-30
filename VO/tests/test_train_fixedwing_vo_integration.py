"""Integration coverage for the actual trainer/evaluator entry points.

``vio.data.image_pairs`` and ``vio.models.vision_mamba_vo`` are each unit
tested in isolation, and those tests pass even when the SCRIPTS that wire them
together do not: a regression here previously had ``tools/train_fixedwing_vo.py``
and ``tools/evaluate_velocity_horizons.py`` hand the frontend the camera
matrix at NATIVE (calibration) resolution while the images the frontend
actually receives are already resized to the working resolution by
``VisualPairSource`` - so the frontend's own native/working rescale, which
infers "native size" from the image tensor it is handed, silently became a
no-op and the raw native fx/fy/cx/cy were used as if they already belonged to
the working image. At a typical 1920x1080 native / 576x1024 working size that
overstates focal length by ~1.875x and misplaces the principal point by
hundreds of working pixels - corrupting both the bearing scale and the
per-cell rotational field's coordinate grid.

Nothing in ``test_image_pairs.py`` or ``test_vision_mamba_vo.py`` could catch
this: both exercise the pieces directly, already wired correctly by hand. This
module runs ``tools/train_fixedwing_vo.py``'s real ``main()`` end to end on a
tiny synthetic flight and inspects what actually reaches the frontend.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from vio.data.fixedwing_vo import (
    ChronologicalWindowSampler,
    build_vo_dataset,
    mask_age_carry,
)
from vio.data.image_pairs import resize_camera_matrix
from vio.models.vision_mamba_vo import (
    VisionMambaFlowFrontend,
    VisionMambaVO,
    detach_stream_state,
    mask_stream_state,
)

import tools.evaluate_velocity_horizons as evaluate_velocity_horizons
import tools.train_fixedwing_vo as train_fixedwing_vo

# Deliberately not the same aspect-preserving value on both axes as any
# default, and deliberately NOT equal to the working size below - the bug is
# invisible whenever native and working happen to coincide.
NATIVE_HEIGHT, NATIVE_WIDTH = 256, 384
WORKING_HEIGHT, WORKING_WIDTH = 64, 96
TICKS = 600
IMU_RATE_HZ = 100.0
IMAGE_RATE_HZ = 20.0


def _write_flight_csv(path: Path, *, ticks: int = TICKS, forward_speed: float = 25.0) -> None:
    columns = [
        "Time", "NavEulX", "NavEulY", "NavEulZ", "relativeAlt",
        "GPSNavVnX", "GPSNavVnY", "GPSNavVnZ",
        "GPSNavEulX", "GPSNavEulY", "GPSNavEulZ",
    ]
    rows = []
    dt = 1.0 / IMU_RATE_HZ
    for index in range(ticks):
        time_s = index * dt
        # A small yaw ramp: enough spread to pass the "attitude is not
        # constant" guard and to give body_rate a nonzero yaw component,
        # without being large enough to matter for anything this test checks.
        yaw = 0.02 * time_s
        rows.append([
            time_s, 0.0, 0.0, yaw, 100.0,
            forward_speed, 0.0, 0.0,
            0.0, 0.0, yaw,
        ])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def _write_images(folder: Path, *, ticks: int = TICKS) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    base = np.arange(NATIVE_HEIGHT * NATIVE_WIDTH, dtype=np.uint8).reshape(
        NATIVE_HEIGHT, NATIVE_WIDTH
    )
    duration_ms = int(ticks / IMU_RATE_HZ * 1000)
    step_ms = int(1000.0 / IMAGE_RATE_HZ)
    for timestamp_ms in range(0, duration_ms, step_ms):
        shifted = np.roll(base, timestamp_ms // step_ms, axis=1)
        Image.fromarray(shifted).save(folder / f"{timestamp_ms}.jpg")


def _write_calibration(path: Path) -> None:
    payload = {
        "camera": {
            "fx": 200.0, "fy": 200.0,
            "cx": NATIVE_WIDTH / 2.0, "cy": NATIVE_HEIGHT / 2.0,
            "width": NATIVE_WIDTH, "height": NATIVE_HEIGHT,
            "distortion": [],
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture(scope="module")
def flight(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("vo_trainer_integration")
    _write_flight_csv(root / "flight.csv")
    _write_images(root / "images")
    _write_calibration(root / "calibration.json")
    return root


def _train_argv(flight: Path, run_dir: Path, **overrides) -> list:
    """The minimal argv a tiny CPU training run needs on ``flight``.

    Shared by the trainer and evaluator integration tests so both exercise
    identical model/window geometry; ``overrides`` replaces or adds flags by
    name (e.g. ``rotation_mode="constant"`` -> ``--rotation-mode constant``).
    A value of ``True``/``False`` produces a bare store-true flag (present or
    omitted); any other value is stringified as its argument.
    """

    args = {
        "dataset": str(flight),
        "calibration": str(flight / "calibration.json"),
        "window-length": "40", "stride": "40", "warmup": "5",
        "max-visual-events": "16",
        "visual-dim": "8", "stem-dim": "8", "stem-depth": "1",
        "patch-size": "8", "token-grid": "4",
        "correlation-radius": "2",
        "aiding-dim": "8", "fusion-dim": "8", "dropout": "0.0",
        "epochs": "1", "batch-size": "1", "num-workers": "0",
        "device": "cpu", "run-dir": str(run_dir),
    }
    args.update({key.replace("_", "-"): value for key, value in overrides.items()})
    argv = ["--no-progress", "--image-size", str(WORKING_HEIGHT), str(WORKING_WIDTH),
            "--context-grid", "4", "6"]
    for name, value in args.items():
        if isinstance(value, bool):
            if value:
                argv.append(f"--{name}")
        else:
            argv += [f"--{name}", str(value)]
    return argv


def test_trainer_hands_the_frontend_a_working_resolution_camera_matrix(
    flight, tmp_path, monkeypatch
):
    """Regression: the raw calibration matrix must never reach the frontend.

    Spies on ``VisionMambaFlowFrontend.forward`` to record the camera matrix
    ``tools/train_fixedwing_vo.py`` actually hands it, then runs the real
    ``main()`` for one tiny epoch on a synthetic flight whose native
    resolution differs from the working resolution - the exact condition the
    bug needed to manifest.
    """

    captured: dict = {}
    original_forward = VisionMambaFlowFrontend.forward

    def spying_forward(self, image0, image1, **kwargs):
        if "camera_matrix" not in captured and kwargs.get("camera_matrix") is not None:
            captured["camera_matrix"] = kwargs["camera_matrix"].detach().clone().numpy()
            captured["frontend_native_size"] = (int(image0.shape[2]), int(image0.shape[3]))
        return original_forward(self, image0, image1, **kwargs)

    monkeypatch.setattr(VisionMambaFlowFrontend, "forward", spying_forward)

    run_dir = tmp_path / "run"
    assert train_fixedwing_vo.main(_train_argv(flight, run_dir)) == 0
    assert "camera_matrix" in captured, "the frontend was never called with a camera matrix"

    # The frontend only ever sees images already resized by VisualPairSource,
    # so by the time it infers a "native size" from the tensor it was handed,
    # that size IS the working size - confirming the condition the bug needs.
    assert captured["frontend_native_size"] == (WORKING_HEIGHT, WORKING_WIDTH)

    native_matrix = np.array(
        [[200.0, 0.0, NATIVE_WIDTH / 2.0],
         [0.0, 200.0, NATIVE_HEIGHT / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    expected = resize_camera_matrix(
        native_matrix, (NATIVE_HEIGHT, NATIVE_WIDTH), (WORKING_HEIGHT, WORKING_WIDTH)
    )
    np.testing.assert_allclose(captured["camera_matrix"], expected, rtol=1e-5)
    # Pinned explicitly, not just left to the allclose above: the pre-fix
    # value is the untouched native matrix, off by exactly native/working
    # (4x on both axes here) - this is what silently corrupted the bearing
    # scale and the per-cell rotational field's coordinate grid.
    assert not np.allclose(captured["camera_matrix"], native_matrix)


def test_build_vo_dataset_forwards_distortion_to_the_image_source(flight):
    """Regression: ``calibration.distortion`` used to be parsed and then
    dropped - ``build_vo_dataset`` had no parameter to carry it to
    ``VisualPairSource``, so a calibration file with real lens distortion had
    it silently ignored by every caller."""

    pytest.importorskip("cv2")
    distortion = np.array([-0.05, 0.01, 0.0, 0.0, 0.0])
    dataset, _, _ = build_vo_dataset(
        flight,
        (0, TICKS),
        image_size=(WORKING_HEIGHT, WORKING_WIDTH),
        camera_matrix=np.array(
            [[200.0, 0.0, NATIVE_WIDTH / 2.0],
             [0.0, 200.0, NATIVE_HEIGHT / 2.0],
             [0.0, 0.0, 1.0]]
        ),
        calibration_image_size=(NATIVE_HEIGHT, NATIVE_WIDTH),
        distortion=distortion,
        window_length=40, stride=40, warmup=5, max_visual_events=4,
    )
    assert dataset.image_source._rectify_maps is not None


def test_evaluator_restores_the_trained_rotation_mode(flight, tmp_path, monkeypatch):
    """Regression: the evaluator used to always build a ``rotation_mode="field"``
    frontend regardless of what the checkpoint was trained with.

    Trains one epoch with the NON-default ``--rotation-mode constant`` - the
    default is "field", so training with the default would pass even with the
    bug, by coincidence - then evaluates that checkpoint. ``rotation.map`` is
    shaped ``(2, 3)`` for "constant" and ``(3, 3)`` for "field"
    (:class:`RotationalSearchField`), so restoring the wrong mode makes
    ``load_state_dict(strict=True)`` fail outright; this pins that it does not.
    """

    run_dir = tmp_path / "run"
    assert train_fixedwing_vo.main(
        _train_argv(flight, run_dir, rotation_mode="constant")
    ) == 0

    captured: dict = {}
    original_forward = VisionMambaFlowFrontend.forward

    def spying_forward(self, image0, image1, **kwargs):
        captured.setdefault("rotation_mode", self.rotation.mode)
        captured.setdefault("rotation_map_shape", tuple(self.rotation.map.weight.shape))
        return original_forward(self, image0, image1, **kwargs)

    monkeypatch.setattr(VisionMambaFlowFrontend, "forward", spying_forward)

    assert evaluate_velocity_horizons.main([
        str(run_dir / "best.pt"),
        "--dataset", str(flight),
        "--splits", "validation",
        "--no-plots", "--no-progress",
    ]) == 0
    assert captured["rotation_mode"] == "constant"
    assert captured["rotation_map_shape"] == (2, 3)


def _write_presplit_flight(root: Path, *, ticks: int, forward_speed: float) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _write_flight_csv(root / "flight.csv", ticks=ticks, forward_speed=forward_speed)
    _write_images(root / "images", ticks=ticks)
    return root


def test_evaluator_baseline_uses_the_training_folder_not_the_scored_one(
    tmp_path,
):
    """Regression: for a PRE-SPLIT run, ranges["train"] in the evaluator's
    ``_resolve_ranges`` maps to the WHOLE scored directory (every phase does -
    a pre-split folder IS its split, with no sub-range to reconstruct), so
    slicing the currently-loaded dataset by it reports the scored folder's own
    mean under the "train" label. Scoring ``data_split/validation`` would then
    compare the model against ITS OWN mean, flattering skill/baseline numbers
    exactly the way a validation-mean baseline would.

    Uses two flights with deliberately different forward speeds (25 vs 5 m/s)
    so the two possible baselines are unmistakable.
    """

    train_root = _write_presplit_flight(
        tmp_path / "train_flight", ticks=TICKS, forward_speed=25.0
    )
    validation_root = _write_presplit_flight(
        tmp_path / "validation_flight", ticks=300, forward_speed=5.0
    )
    _write_calibration(train_root / "calibration.json")

    run_dir = tmp_path / "run"
    assert train_fixedwing_vo.main(_train_argv(
        train_root, run_dir, validation_dataset=str(validation_root),
    )) == 0

    output = tmp_path / "eval.json"
    assert evaluate_velocity_horizons.main([
        str(run_dir / "best.pt"),
        "--dataset", str(validation_root),
        "--calibration", str(train_root / "calibration.json"),
        "--splits", "validation",
        "--no-plots", "--no-progress",
        "--output", str(output),
    ]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    baseline = report["train_mean_baseline_m_s"]
    # The TRAINING flight's forward speed (25), never the validation flight's
    # own (5) - which is what the pre-fix code would have reported here.
    assert baseline[0] == pytest.approx(25.0, abs=1.0)


def test_horizon_pass_reads_the_named_splits_own_file_in_a_presplit_run(
    tmp_path, monkeypatch,
):
    """Regression: ``run_horizon_pass`` used to always read ``datasets["train"]``
    and the training file's clock/body-rate, regardless of ``--horizon-split``.
    That is harmless when every phase shares one file (the fractional-split
    mode), but in a PRE-SPLIT run each phase is a different directory, so a
    "validation" horizon leg would silently score a slice of the TRAINING
    file's ticks under the validation label - training data leaking into a
    number reported as held out.

    Spies on ``stream_horizon_metrics`` to capture the target velocity it was
    actually asked to score, and checks it is the VALIDATION flight's own
    (5 m/s), never the training flight's (25 m/s).
    """

    train_root = _write_presplit_flight(
        tmp_path / "train_flight", ticks=TICKS, forward_speed=25.0
    )
    validation_root = _write_presplit_flight(
        tmp_path / "validation_flight", ticks=300, forward_speed=5.0
    )
    _write_calibration(train_root / "calibration.json")

    captured: dict = {}
    original = train_fixedwing_vo.stream_horizon_metrics

    def spying_stream_horizon_metrics(model, **kwargs):
        captured.setdefault(
            "target_velocity", np.asarray(kwargs["target_velocity"]).copy()
        )
        return original(model, **kwargs)

    monkeypatch.setattr(
        train_fixedwing_vo, "stream_horizon_metrics", spying_stream_horizon_metrics
    )

    run_dir = tmp_path / "run"
    assert train_fixedwing_vo.main(_train_argv(
        train_root, run_dir,
        validation_dataset=str(validation_root),
        horizon_minutes="0.02", horizon_split="validation",
    )) == 0

    assert "target_velocity" in captured, "the horizon pass never ran"
    assert float(np.mean(captured["target_velocity"][:, 0])) == pytest.approx(5.0, abs=1.0)


def test_eval_train_split_scores_training_data_with_dropout_off_on_schedule(
    flight, tmp_path,
):
    """``--eval-train-split`` settles PLAN.txt SS12: is a train/val gap real
    generalisation, or an artefact of train_* being measured mid-epoch with
    dropout on and averaged over a still-changing model? traineval_* answers
    that by scoring the training split through the same evaluate() path (eval
    mode, dropout off) validation uses.

    Also pins the --eval-train-split-every schedule: due on epoch 2 (2 % 2)
    and epoch 3 (the last epoch, forced), not on epoch 1.
    """

    run_dir = tmp_path / "run"
    assert train_fixedwing_vo.main(_train_argv(
        flight, run_dir,
        epochs="3", eval_train_split=True, eval_train_split_every="2",
    )) == 0

    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[0]["epoch"] == "1"
    assert rows[0]["traineval_vel_rmse"] == "", "epoch 1 is not due, must be blank"
    for row in rows[1:]:
        assert row["traineval_vel_rmse"] not in ("", None)
        assert np.isfinite(float(row["traineval_vel_rmse"]))
        assert np.isfinite(float(row["traineval_vel_rmse_y"]))


def test_eval_train_split_scores_every_window_even_when_not_divisible_by_batch_size(
    flight, tmp_path, monkeypatch,
):
    """Regression: --eval-train-split used to reuse loaders["train"], which is
    shuffle=True AND drop_last=True - dropping up to batch_size-1 windows
    EVERY call, a systematic gap whenever the split length is not a multiple
    of batch_size (a coincidence, not something a run controls). This flight's
    default fractional split gives the training phase 360 of its 600 ticks -
    9 windows at the default 40-tick window length; batch_size=4 does not
    divide 9 (9 % 4 == 1), the exact condition the bug needed.

    Spies on the dedicated loaders["train_eval"] pass (identified by its
    progress_description, distinct from the ordinary validation pass) and
    counts windows actually scored across every batch it yields.
    """

    captured: dict = {}
    original_evaluate = train_fixedwing_vo.evaluate

    def spying_evaluate(step, loader, *args, **kwargs):
        if str(kwargs.get("progress_description", "")).endswith("train(eval)"):
            captured["window_count"] = sum(
                int(batch["aiding"].shape[0]) for batch in loader
            )
        return original_evaluate(step, loader, *args, **kwargs)

    monkeypatch.setattr(train_fixedwing_vo, "evaluate", spying_evaluate)

    run_dir = tmp_path / "run"
    assert train_fixedwing_vo.main(_train_argv(
        flight, run_dir, epochs="1", eval_train_split=True, batch_size="4",
    )) == 0

    assert "window_count" in captured, "the eval-train-split pass never ran"
    assert captured["window_count"] == 9, (
        "drop_last=True on the reused training loader would have scored 8 "
        "(9 // 4 * 4), silently dropping the last window every call"
    )


def test_evaluator_json_carries_frontend_diagnostics_per_horizon(flight, tmp_path):
    """End-to-end: the frontend's per-pair diagnostics (boundary-hit rate,
    confidence, entropy, occupancy - see DIAGNOSTIC_NAMES) must actually reach
    tools/evaluate_velocity_horizons.py's JSON output, not just the library
    functions in isolation. A horizon short enough to fit the tiny synthetic
    flight is required, or every horizon is skipped and none of this runs."""

    run_dir = tmp_path / "run"
    assert train_fixedwing_vo.main(_train_argv(flight, run_dir)) == 0

    output = tmp_path / "eval.json"
    assert evaluate_velocity_horizons.main([
        str(run_dir / "best.pt"),
        "--dataset", str(flight),
        "--calibration", str(flight / "calibration.json"),
        "--splits", "validation",
        "--horizons", "0.01",
        "--no-plots", "--no-progress",
        "--output", str(output),
    ]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    horizon = report["splits"]["validation"]["horizons"]["h0.01m"]
    assert horizon["fits"], horizon.get("skipped")
    assert horizon["visual_events"] > 0
    for name in (
        "diag_boundary_hit_fraction", "diag_mean_usable_confidence",
        "diag_mean_entropy", "diag_occupied_fraction",
    ):
        assert name in horizon
        assert np.isfinite(horizon[name]), f"{name} was not finite: {horizon[name]}"


def test_aiding_vector_carries_body_rates_and_fusion_input_carries_visual_age(
    flight, tmp_path, monkeypatch
):
    """Increment 2: VO_AIDING_CHANNELS gained p/q/r body rates and fuse_stream
    gained a visual_age input alongside visual_present.

    Spies on VisionMambaVO.forward (the one method every VOStep call - train
    AND validation windows alike - actually reaches) to capture what really
    lands in the fusion input on a flight with a known constant yaw rate
    (0.02 rad/s, see _write_flight_csv) and the frontend's own deployment
    latency, which - at 0.35s against this test's 0.4s (40-tick) windows -
    already starves most of a window of a delivered token and so exercises
    visual_age well past the camera's own nominal cadence without needing a
    contrived dropout.
    """

    captured: dict = {"aiding": [], "visual_present": [], "visual_age": []}
    original_forward = VisionMambaVO.forward

    def spying_forward(self, aiding, visual_token, visual_present, visual_age, **kwargs):
        captured["aiding"].append(aiding.detach().clone())
        captured["visual_present"].append(visual_present.detach().clone())
        captured["visual_age"].append(visual_age.detach().clone())
        return original_forward(
            self, aiding, visual_token, visual_present, visual_age, **kwargs
        )

    monkeypatch.setattr(VisionMambaVO, "forward", spying_forward)

    run_dir = tmp_path / "run"
    assert train_fixedwing_vo.main(_train_argv(flight, run_dir)) == 0
    assert captured["aiding"], "VisionMambaVO.forward was never called"

    aiding = torch.cat([a.reshape(-1, a.shape[-1]) for a in captured["aiding"]])
    present = torch.cat([p.reshape(-1, 1) for p in captured["visual_present"]])
    age = torch.cat([a.reshape(-1, 1) for a in captured["visual_age"]])

    assert aiding.shape[-1] == 9
    # r_rad_s (index 7): the flight's yaw ramps at a constant 0.02 rad/s and
    # nothing else in the CSV moves, so the channel should sit close to that
    # value throughout - the pre-Increment-2 aiding vector could not have
    # carried this signal at all.
    assert float(aiding[:, 7].mean()) == pytest.approx(0.02, abs=0.002)
    # p_rad_s, q_rad_s (roll/pitch rate): both stay at zero the whole flight.
    assert float(aiding[:, 5].abs().max()) < 1e-6
    assert float(aiding[:, 6].abs().max()) < 1e-6

    assert age.shape == present.shape
    # The DEPLOYMENT LATENCY (0.35s, the trainer's own default), not zero,
    # exactly where a token just arrived - present fires at ready_tick, one
    # deployment latency after the second image was actually captured, so
    # the image is already that old the moment it becomes available. Growing
    # strictly between arrivals is the causal, resetting signal
    # visual_age_seconds is meant to produce, not a constant or a copy of
    # visual_present.
    on_arrival = age[present > 0]
    assert torch.allclose(on_arrival, torch.full_like(on_arrival, 0.35), atol=1e-4)
    assert float(age.max()) > 0.35


def test_tbptt_state_crosses_a_chunk_boundary_and_detach_does_not_change_values(flight):
    """Increment 3 scaffold (see PLAN_TBPTT.txt): ChronologicalWindowSampler +
    VOStep.forward_stream/forward_batch_stream carry recurrent state between
    chronologically-adjacent windows instead of resetting it every window,
    the way the default (shuffled, VOStep.forward) training path does.

    Runs the REAL frontend/dataset pipeline - not hand-built tensors - for
    two TBPTT steps and checks the two properties this scaffold exists for:
    carrying state into step 1 changes its output relative to a state=None
    run of the exact same batch (mirrors
    test_velocity_horizons.test_a_reset_actually_discards_the_past), and
    detaching the carried state between steps changes nothing about step 1's
    OWN forward values - only cuts the graph a backward pass would walk.
    """

    native_matrix = np.array(
        [[200.0, 0.0, NATIVE_WIDTH / 2.0],
         [0.0, 200.0, NATIVE_HEIGHT / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dataset, _, _ = build_vo_dataset(
        flight,
        (0, TICKS),
        image_size=(WORKING_HEIGHT, WORKING_WIDTH),
        camera_matrix=native_matrix,
        calibration_image_size=(NATIVE_HEIGHT, NATIVE_WIDTH),
        window_length=40, stride=40, warmup=5, max_visual_events=16,
    )
    sampler = ChronologicalWindowSampler(dataset, batch_size=2)
    assert len(sampler) >= 2, "the synthetic flight is too short for this test"

    camera_tensor = torch.from_numpy(
        resize_camera_matrix(
            native_matrix, (NATIVE_HEIGHT, NATIVE_WIDTH), (WORKING_HEIGHT, WORKING_WIDTH)
        )
    ).float()
    torch.manual_seed(0)
    frontend = VisionMambaFlowFrontend(
        visual_dim=8, d_model=8, depth=1, patch_size=8,
        image_size=(WORKING_HEIGHT, WORKING_WIDTH), context_grid=(4, 6),
        correlation_radius=2, token_grid=4,
    )
    model = VisionMambaVO(visual_dim=8, aiding_dim=8, fusion_dim=8, frontend=frontend)
    # The heads ship zero-initialised so an untrained model predicts straight
    # and level regardless of the fusion hidden state - which would make
    # "carrying state changes the output" vacuously true for the wrong
    # reason (nothing depends on state at all). Same fix as build_model() in
    # test_velocity_horizons.py.
    for head in (model.direction_head, model.log_rate_head, model.log_variance_head):
        torch.nn.init.normal_(head.weight, std=0.2)
    step = train_fixedwing_vo.VOStep(
        model, window_length=40, visual_dim=8, disable_visual=False,
        frontend_chunk=64, deployment_latency_s=0.35, camera_matrix=camera_tensor,
    )
    device = torch.device("cpu")

    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    batches = [batch for _, batch in zip(range(2), loader)]
    assert len(batches) == 2

    with torch.no_grad():
        _, _, _, state0, age_carry0 = train_fixedwing_vo.forward_batch_stream(
            step, batches[0], device, None
        )
        # Step 0's keep mask is all-False (nothing to carry into the very
        # first step of a stream), so state=None here is the exact
        # equivalent - both mean "every lane starts fresh".
        assert not sampler.continues_at(0).any()

        keep = sampler.continues_at(1)
        assert keep.all(), "single-range flight: step 1 should continue both lanes"
        carried_state = mask_stream_state(state0, keep)
        detached_state = detach_stream_state(carried_state)
        carried_age = mask_age_carry(age_carry0, keep)

        carried, _, _, _, _ = train_fixedwing_vo.forward_batch_stream(
            step, batches[1], device, detached_state, carried_age
        )
        fresh, _, _, _, _ = train_fixedwing_vo.forward_batch_stream(
            step, batches[1], device, None, None
        )
        # Detaching cuts the graph, not the numbers: re-running step 1 from
        # the UNDETACHED carried state must give bit-identical predictions.
        undetached, _, _, _, _ = train_fixedwing_vo.forward_batch_stream(
            step, batches[1], device, carried_state, carried_age
        )

    carried_velocity = carried["predicted_velocity"]
    fresh_velocity = fresh["predicted_velocity"]
    undetached_velocity = undetached["predicted_velocity"]

    assert not torch.allclose(carried_velocity, fresh_velocity), (
        "carrying state into step 1 must change its prediction relative to "
        "a fresh (state=None) run of the same batch"
    )
    torch.testing.assert_close(carried_velocity, undetached_velocity)


def _run_and_capture_fusion_input(flight, run_dir, monkeypatch, **overrides):
    """Trains one tiny epoch and returns everything VisionMambaVO.forward
    actually received, concatenated over every call - the aiding vector and
    visual_age, the two tensors --ablate-body-rate/--ablate-visual-age zero."""

    captured: dict = {"aiding": [], "visual_age": []}
    original_forward = VisionMambaVO.forward

    def spying_forward(self, aiding, visual_token, visual_present, visual_age, **kwargs):
        captured["aiding"].append(aiding.detach().clone())
        captured["visual_age"].append(visual_age.detach().clone())
        return original_forward(
            self, aiding, visual_token, visual_present, visual_age, **kwargs
        )

    monkeypatch.setattr(VisionMambaVO, "forward", spying_forward)
    assert train_fixedwing_vo.main(_train_argv(flight, run_dir, **overrides)) == 0
    aiding = torch.cat([a.reshape(-1, a.shape[-1]) for a in captured["aiding"]])
    age = torch.cat([a.reshape(-1, 1) for a in captured["visual_age"]])
    return aiding, age


def test_ablate_body_rate_zeros_only_its_own_channels_leaving_age_untouched(
    flight, tmp_path, monkeypatch
):
    """--ablate-body-rate must zero exactly VO_AIDING_CHANNELS' p/q/r columns,
    leave every other aiding channel and visual_age alone, and leave the
    aiding vector's WIDTH unchanged (9, not 6) - the tensor shape a checkpoint
    trained under this flag must still match one trained without it."""

    aiding, age = _run_and_capture_fusion_input(
        flight, tmp_path / "run", monkeypatch, ablate_body_rate=True,
    )
    assert aiding.shape[-1] == 9
    assert float(aiding[:, 5:8].abs().max()) == 0.0
    # cos_roll (index 1): the flight is level the whole time, so this should
    # sit at 1.0 - not zero, which is what it would read if ablation reached
    # channels it has no business touching.
    assert float(aiding[:, 1].mean()) == pytest.approx(1.0, abs=1e-4)
    # visual_age is a SEPARATE flag - ablating body rate must not silently
    # zero it too.
    assert float(age.max()) > 0.0


def test_ablate_visual_age_zeros_only_age_leaving_body_rate_untouched(
    flight, tmp_path, monkeypatch
):
    """--ablate-visual-age must zero the fusion input's age channel, leave the
    aiding vector (body rate included) alone, and leave every tensor's shape
    unchanged - age becomes a constant 0, not a dropped column."""

    aiding, age = _run_and_capture_fusion_input(
        flight, tmp_path / "run", monkeypatch, ablate_visual_age=True,
    )
    assert age.shape[-1] == 1
    assert float(age.abs().max()) == 0.0
    # r_rad_s (index 7): the flight's known 0.02 rad/s yaw rate, unaffected
    # by an ablation flag that is supposed to touch only visual_age.
    assert float(aiding[:, 7].mean()) == pytest.approx(0.02, abs=0.002)
