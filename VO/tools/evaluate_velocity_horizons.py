#!/usr/bin/env python
"""Body-velocity error as a function of how long the estimator has been running.

``metrics.csv`` reports one ``val_vel_rmse`` per epoch, measured on 600-tick
windows - six seconds, each starting from a zeroed state. This scores the same
checkpoint over legs of 1, 5, 10, 15, 20 and 30 minutes instead: the state is
reset at the start of every leg, the whole leg is streamed through the model
carrying state tick to tick, every tick after the warm-up is scored, and the
per-leg errors are averaged.

    python tools/evaluate_velocity_horizons.py runs/<run>/best.pt \\
        --dataset dataset --splits validation \\
        --output artifacts/velocity_horizons.json

Two tables come out: velocity error, and the dead-reckoned position that
velocity integrates to over each leg.

What to read. A VELOCITY curve flat in the horizon says the recurrent state is
not accumulating error and the six-second number generalises to a long leg; a
rising one says it is, and by how much. POSITION always grows with the leg - it
is an integral - so the number to watch there is drift_%: falling with the
horizon means the velocity error is largely unbiased and partly averages out,
flat or rising means it carries a bias that accumulates. The spread across legs
matters as much as the mean - a low mean with a wide spread is a model that
fails under some conditions, not one that works - and the max columns matter
independently of the RMS ones, because one bad second inside a ten-minute leg
barely moves an RMS and moves the maximum by all of it.

The position error rotates the prediction and the truth by the SAME reference
attitude, so it isolates the drift the velocity estimate causes. A deployed
system also carries attitude error, which this excludes by construction.

Horizons that do not fit the split are reported as skipped with the length they
needed, never silently dropped.

Three figures are written per split (``--plots``, on by default), the first two
from one extra continuous run over the WHOLE split with no resets: a trajectory
plot (the dead-reckoned path laid over the true one, worst velocity moment
circled), an error plot (velocity, direction and position error against flight
time, each maximum marked, a shared dashed line at the worst velocity moment),
and a summary plot of the tables' numbers against horizon length. See
vio/utils/horizon_plots.py for what each one is answering.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
# The package lives under src/; tools/ is imported as a package from the
# repository root. Both have to be importable when a script is run directly.
for _entry in (_ROOT / "src", _ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from vio.data.calibration import maybe_load_camera_calibration  # noqa: E402
from vio.data.fixedwing_vo import (  # noqa: E402
    VONormalizer,
    build_vo_dataset,
    reference_body_frame,
    reference_body_velocity,
)
from vio.data.image_pairs import resize_camera_matrix  # noqa: E402
from vio.models.frontend_factory import (  # noqa: E402
    build_frontend,
    frontend_class,
    resolve_velocity_mode,
)
from vio.models.velocity_horizons import (  # noqa: E402
    DEFAULT_HORIZON_MINUTES,
    encode_span_tokens,
    format_horizon_table,
    parse_horizon_minutes,
    run_span_horizons,
)
from vio.models.vision_mamba_vo import VisionMambaVO  # noqa: E402
from vio.utils.checkpoint_io import load_checkpoint  # noqa: E402
from vio.utils.stratify import (  # noqa: E402
    flight_conditions,
    format_stratified,
    stratified_errors,
)
from vio.utils.horizon_plots import save_split_plots  # noqa: E402

SPLIT_NAMES = ("train", "validation", "test", "full")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "checkpoint", type=Path, help="runs/<run>/best.pt from train_fixedwing_vo.py"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Flight directory. Defaults to the one recorded in the checkpoint, "
             "which is what makes a held-out number held out; point it "
             "elsewhere only to score a different capture.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Camera calibration for THIS dataset. Defaults to the one the "
             "checkpoint recorded, which describes the camera it was trained "
             "on - the loader rejects images whose resolution disagrees with "
             "it, so a capture from a different camera or render needs its "
             "own file named here.",
    )
    parser.add_argument(
        "--splits",
        default="validation",
        help="Comma-separated: train, validation, test, full. 'full' is the "
             "whole capture ignoring the split, which measures drift rather "
             "than held-out accuracy - label it that way if you quote it.",
    )
    parser.add_argument(
        "--horizons",
        default=",".join(f"{value:g}" for value in DEFAULT_HORIZON_MINUTES),
        help="Prefix lengths in minutes. Each one is scored over the FIRST H "
             "minutes of the split, taken out of a single unbroken run from "
             "the split's start to its end - not the split cut into repeated "
             "H-minute legs. A horizon longer than the split is reported "
             "skipped, never shortened.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Ticks excluded at the start of the run, where no image has "
             "arrived yet. Defaults to the trainer's --warmup.",
    )
    parser.add_argument(
        "--block-ticks",
        type=int,
        default=2000,
        help="Ticks per streamed block. Only affects memory and speed: the "
             "state is carried across blocks, so the result is identical.",
    )
    parser.add_argument(
        "--batch-pairs",
        type=int,
        default=8,
        help="Image pairs per frontend batch during the one token pass.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, default=None, help="Write JSON here.")
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save four figures per split: the dead-reckoned trajectory "
             "against the true one, body-frame velocity against the truth per "
             "axis, error growth over the run, and a summary against horizon "
             "length. The first three are drawn from the WHOLE split, not per "
             "horizon, because a horizon is a prefix of that same single run.",
    )
    parser.add_argument(
        "--stratify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Break the whole-split error down by turn rate, bank angle, "
             "altitude and ground speed (vio.utils.stratify), printed and "
             "written to the JSON under splits.<name>.stratified. A single RMSE "
             "averages cruise, where the camera sees a clean translation, with "
             "banked turns, where it does not; this says which one a model is "
             "losing on.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Figure directory (default: <output stem>_plots beside --output, "
             "or artifacts/velocity_horizons_plots without --output).",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable the progress bars."
    )
    parser.add_argument(
        "--no-position",
        action="store_true",
        help="Skip the dead-reckoning position metrics. They integrate the "
             "velocity over each leg through the REFERENCE attitude, applied "
             "to the prediction and the truth alike, so what they report is "
             "the drift the velocity estimate causes - a deployed system also "
             "carries attitude error, which this excludes by construction.",
    )
    parser.add_argument(
        "--disable-visual-input",
        action="store_true",
        help="Zero the visual token, as the trainer's flag of the same name "
             "does. The attitude-and-altitude-only floor at every horizon.",
    )
    return parser


def _resolve_ranges(
    saved: Dict[str, object], total: int, *, dataset_root: Optional[Path] = None
) -> Dict[str, Tuple[int, int]]:
    """The trainer's split, reconstructed from its own arguments.

    Deliberately not re-derived from a fresh guess: a horizon score is only
    held out if it uses the same ranges the run trained on.

    Two cases, distinguished by what the run recorded:

    * PRE-SPLIT (``validation_dataset`` recorded, produced by
      tools/split_dataset.py). Each phase was its own directory, used whole,
      so there are no fractions to reconstruct. Whichever directory is being
      scored is scored ENTIRELY, and every named split maps to the same full
      range: the file on disk already is the split. Passing the training
      directory here scores training data, and the label below says so.
    * FRACTIONS. The run cut one capture chronologically, so the ranges are
      rebuilt from the fractions it recorded. Nothing else is needed - no
      manifest, no file that has to still exist on the machine reading the
      results.
    """

    if saved.get("validation_dataset"):
        # The directory handed to this tool is one whole split. Naming every
        # range identically is honest: there is no sub-split to reconstruct,
        # and pretending otherwise would invent held-out ticks.
        print(
            "  pre-split run: this directory is scored whole "
            f"({total} ticks); the split is the folder, not a range"
        )
        return {"train": (0, total), "validation": (0, total),
                "test": (0, total), "full": (0, total)}

    gap = int(saved.get("window_length", 600))
    train_fraction = float(saved.get("train_fraction", 0.6))
    validation_fraction = float(saved.get("validation_fraction", 0.2))
    train_end = int(total * train_fraction)
    validation_end = train_end + int(total * validation_fraction)
    return {
        "train": (0, train_end),
        "validation": (train_end + gap, validation_end),
        "test": (validation_end + gap, total),
        "full": (0, total),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    progress = not args.no_progress

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    saved = dict(checkpoint.get("args", {}))
    if not saved:
        raise SystemExit(
            f"{args.checkpoint} carries no 'args' block, so the model it holds "
            "cannot be rebuilt. It is not a train_fixedwing_vo.py checkpoint."
        )

    # A PRE-SPLIT run's recorded "dataset" is its TRAINING folder, so it is
    # never the default to score: the held-out default is its validation
    # folder. And whatever folder is scored, if it is the training one the
    # report must not call the result held out.
    pre_split = bool(saved.get("validation_dataset"))
    if args.dataset is not None:
        dataset_root = Path(args.dataset)
    elif pre_split:
        dataset_root = Path(saved["validation_dataset"])
        print(f"no --dataset: scoring the run's validation folder {dataset_root}")
    else:
        dataset_root = Path(saved["dataset"])
    scoring_training_folder = (
        pre_split and dataset_root.resolve() == Path(saved["dataset"]).resolve()
    )
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset directory not found: {dataset_root}")
    horizons = parse_horizon_minutes(args.horizons)
    if not horizons:
        raise SystemExit("--horizons named no positive leg length")
    splits = [name.strip() for name in args.splits.split(",") if name.strip()]
    unknown = [name for name in splits if name not in SPLIT_NAMES]
    if unknown:
        raise SystemExit(f"Unknown split(s) {unknown}; choose from {list(SPLIT_NAMES)}")
    warmup = int(args.warmup if args.warmup is not None else saved.get("warmup", 20))

    camera_matrix = None
    native_size = None
    images_rectified = False
    distortion = None
    calibration = args.calibration if args.calibration is not None else saved.get("calibration")
    loaded = maybe_load_camera_calibration(calibration)
    if loaded is not None:
        camera_matrix = loaded.camera_matrix
        native_size = loaded.native_size
        images_rectified = loaded.images_rectified
        distortion = loaded.distortion if loaded.distortion.size else None

    image_size = tuple(int(value) for value in saved.get("image_size", (576, 1024)))
    normalizer = VONormalizer(**checkpoint["normalizer"])
    # What is wanted here is the dataset's per-tick arrays - aiding features,
    # log altitude, target velocity, image source - for the WHOLE flight. Legs
    # are cut from those directly, so the window list this builds is never
    # used; the smallest legal one (a single 2-tick window) is asked for so
    # that building it costs nothing. The normalizer comes from the checkpoint
    # rather than being refitted, because refitting on a different range would
    # shift the centred altitude channel the model was trained against.
    print(f"dataset:    {dataset_root}")
    dataset, _, attitude = build_vo_dataset(
        dataset_root,
        (0, 2),
        csv_name=saved.get("csv_name", "flight.csv"),
        time_column=saved.get("time_column", "Time"),
        time_scale=float(saved.get("time_scale", 1.0)),
        attitude_columns=saved.get("attitude_columns"),
        altitude_column=saved.get("altitude_column"),
        allow_reference_attitude=bool(saved.get("allow_reference_attitude", False)),
        image_folder=saved.get("image_folder", "images"),
        image_size=image_size,
        frame_gap=int(saved.get("frame_gap", 1)),
        pair_stride=int(saved.get("pair_stride", 1)),
        deployment_latency_s=float(saved.get("deployment_latency_s", 0.35)),
        # Reproduce the run's own alignment. Defaults match a checkpoint written
        # before these were recorded, so an older file scores exactly as before.
        max_frame_gap_s=saved.get("max_frame_gap_s"),
        image_time_offset_s=saved.get("image_time_offset_s", 0.0),
        # The lever arm defines the target, so scoring without the one the run
        # trained against would compare against a different quantity.
        lever_arm_m=saved.get("lever_arm_m"),
        camera_matrix=camera_matrix,
        calibration_image_size=native_size,
        images_rectified=images_rectified,
        distortion=distortion,
        normalizer=normalizer,
        window_length=2,
        stride=1,
        warmup=0,
        max_visual_events=1,
        # RGB frames for an RGB-trained stem; absent means grayscale, the only
        # thing a checkpoint written before --color existed can have used.
        grayscale=not bool(saved.get("color", False)),
    )
    total = int(attitude.times_s.size)
    ranges = _resolve_ranges(saved, total, dataset_root=dataset_root)
    duration_min = float(attitude.times_s[-1] - attitude.times_s[0]) / 60.0

    # Every horizon here is a COLD START: state is zeroed at the split's first
    # tick, and no visual event can exist before one deployment latency plus
    # roughly one camera frame interval have elapsed. --warmup shorter than
    # that scores ticks the model was blind on as if they were live, which
    # inflates the very early part of every horizon's error silently. This is
    # a diagnostic, not an auto-correction: the right warmup for a cold start
    # is still an open question (see PLAN.txt SS9), so it is surfaced rather
    # than guessed at.
    pair_dt = dataset.image_source.plan.pair_dt_s
    frame_interval = float(np.median(pair_dt)) if pair_dt.size else 0.0
    tick_interval = float(np.median(np.diff(attitude.times_s))) if total > 1 else 0.0
    deployment_latency_s = float(saved.get("deployment_latency_s", 0.35))
    # With non-overlapping pairs (--output-on-pairs) a cold start can wait up
    # to pair_stride - 1 more frames for the tiling's next first frame - the
    # same worst case the trainer's own --warmup default is built from.
    pair_stride = int(saved.get("pair_stride", 1) or 1)
    frame_gap = max(int(saved.get("frame_gap", 1) or 1), 1)
    frame_interval += (pair_stride - 1) * frame_interval / frame_gap
    if tick_interval > 0:
        min_cold_start_warmup = int(
            np.ceil((deployment_latency_s + frame_interval) / tick_interval - 1e-6)
        )
        if warmup < min_cold_start_warmup:
            print(
                f"  WARNING --warmup {warmup} ticks is shorter than the "
                f"worst-case cold-start blind period (~{min_cold_start_warmup} "
                f"ticks = deployment latency {deployment_latency_s:.3f}s + the "
                f"wait for a first pair {frame_interval:.3f}s, at a measured tick "
                f"interval of {tick_interval:.4f}s). The first "
                f"{min_cold_start_warmup - warmup} tick(s) of every horizon "
                "below have no visual event yet but are scored as if live."
            )
    print(f"attitude:   {', '.join(attitude.attitude_columns)} "
          f"| altitude: {attitude.altitude_column}")
    print(f"flight:     {total} ticks, {duration_min:.1f} min")

    # The frontend algorithm, checked BEFORE the weights are loaded. v2 and v3
    # have identical tensor shapes and different confidence semantics, so
    # strict=True below cannot tell them apart: it would load a v2 checkpoint
    # silently and score it under a measurement it was never trained with.
    saved_frontend = saved.get("frontend_id")
    # Which frontend CLASS the run used decides which id this build computes
    # for it; absent means the original frontend, the only one that existed.
    frontend_kind = str(saved.get("frontend", "mamba_correlation") or "mamba_correlation")
    current_frontend = str(frontend_class(frontend_kind).frontend_id)
    if saved_frontend is None:
        raise SystemExit(
            f"{args.checkpoint} records no frontend_id, so it predates "
            f"{current_frontend} and its weights were trained against a "
            "different flow measurement. Retrain, or score it with the "
            "revision that wrote it."
        )
    if str(saved_frontend) != current_frontend:
        raise SystemExit(
            f"{args.checkpoint} was trained with frontend {saved_frontend!r}, "
            f"this build computes {current_frontend!r}. The tensor shapes may "
            "still match, which is exactly why this is checked by name."
        )

    # The FUSION input contract - aiding-vector layout and what gets
    # concatenated onto the token - checked the same way and for the same
    # reason as frontend_id just above: AIDING_INPUT_DIM changing already
    # fails load_state_dict on a shape mismatch, but that only catches a
    # SHAPE change, not (say) an ablation flag that flips between training
    # and evaluation while keeping every tensor the same size.
    saved_temporal_input = saved.get("temporal_input_id")
    # Absent velocity_mode means the factored heads, the only mode a checkpoint
    # written before geometric_residual existed can have used.
    velocity_mode = resolve_velocity_mode(
        {"frontend": frontend_kind, "velocity_mode": saved.get("velocity_mode", "heads")}
    )
    current_temporal_input = str(VisionMambaVO.temporal_input_id_for(velocity_mode))
    if saved_temporal_input is None:
        raise SystemExit(
            f"{args.checkpoint} records no temporal_input_id, so it predates "
            f"{current_temporal_input} and its weights were trained against a "
            "different aiding-vector/fusion-input contract. Retrain, or "
            "score it with the revision that wrote it."
        )
    if str(saved_temporal_input) != current_temporal_input:
        raise SystemExit(
            f"{args.checkpoint} was trained with temporal input contract "
            f"{saved_temporal_input!r}, this build computes "
            f"{current_temporal_input!r}. The tensor shapes may still match, "
            "which is exactly why this is checked by name."
        )
    # Reproduced from the checkpoint, not exposed as a CLI override - same
    # convention rotation_mode above already uses, and for the same reason:
    # scoring under a DIFFERENT ablation than training used would silently
    # feed the model out-of-distribution inputs rather than measure it.
    ablate_body_rate = bool(saved.get("ablate_body_rate", False))
    ablate_visual_age = bool(saved.get("ablate_visual_age", False))
    if ablate_body_rate or ablate_visual_age:
        print(
            "ablation:   "
            + ", ".join(
                name for name, on in (
                    ("body-rate", ablate_body_rate),
                    ("visual-age", ablate_visual_age),
                ) if on
            )
            + " zeroed (recorded at training time)"
        )

    # Every setting that changes what the frontend MEASURES without changing a
    # tensor shape - rotation_mode, the reliability gate, the planar geometry -
    # is read back from the checkpoint by the same factory the trainer used,
    # never re-defaulted here: load_state_dict(strict=True) below cannot see
    # any of them, and frontend_id does not move with them. Absent keys mean a
    # checkpoint written before the setting existed, whose only possible value
    # is the historical one (rotation_mode "constant", gate off, grayscale).
    frontend = build_frontend(
        saved,
        camera_from_body=saved.get("camera_from_body"),
        prior_velocity=saved.get("prior_velocity_body"),
        dropout=0.0,
    ).to(device)
    frontend.load_state_dict(checkpoint["frontend"], strict=True)
    model = VisionMambaVO(
        visual_dim=int(saved.get("visual_dim", 64)),
        aiding_dim=int(saved.get("aiding_dim", 64)),
        fusion_dim=int(saved.get("fusion_dim", 96)),
        dropout=0.0,
        frontend=frontend,
        velocity_mode=velocity_mode,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    # A simple-loss run never put the uncertainty heads in its objective, so
    # they still hold their initialisation. Those values are shaped like
    # predictions and mean nothing, and the failure mode is someone quoting
    # them as a confidence. Absent key means a checkpoint predating the flag,
    # which could only have been trained under the NLL.
    if not bool(checkpoint.get("uncertainty_trained", True)):
        print(
            "warning: this checkpoint was trained with --velocity-loss "
            f"{checkpoint.get('velocity_loss', 'simple')}, which leaves the "
            "variance and direction-concentration heads OUT of the loss. They "
            "hold their initialisation (variance 1.0, kappa 1.0) and are NOT "
            "calibrated uncertainty - do not quote them."
        )
    selected = str(checkpoint.get("select_on", "vel_rmse_y"))
    selected_score = checkpoint.get(
        "val_score", checkpoint.get("val_vel_rmse_y", float("nan"))
    )
    print(f"checkpoint: {args.checkpoint} (epoch {checkpoint.get('epoch', '?')}, "
          f"chosen on val_{selected} {float(selected_score):.4f})")

    disable_visual = bool(
        args.disable_visual_input or saved.get("disable_visual_input", False)
    )
    if disable_visual:
        if bool(saved.get("output_on_pairs", False)):
            # One output per DELIVERED pair: with no pairs there is no tick to
            # score, and every horizon would come back NaN rather than a floor.
            raise SystemExit(
                "--disable-visual-input has no floor to report for an "
                "--output-on-pairs checkpoint: it scores only the ticks where a "
                "pair is delivered, and with visual input disabled none is."
            )
        print("visual input DISABLED: this is the attitude-and-altitude-only floor")

    # The same constant-velocity floor the trainer uses, from the same ticks.
    #
    # ``ranges["train"]`` is only the true training slice of THIS dataset
    # object in the FRACTIONAL-split mode, where the file being scored here is
    # the same file training cut its ranges from. In PRE-SPLIT mode every
    # phase name maps to the whole scored directory (see _resolve_ranges) -
    # "train" included - so slicing dataset.velocity_body by it would report
    # data_split/validation's (or .../test's) own mean under the "train"
    # label: a baseline that has seen the very data it is meant to be a
    # floor for, flattering every skill/baseline comparison exactly the way
    # scoring against a validation-mean baseline would.
    train_baseline = checkpoint.get("train_baseline_velocity_m_s")
    if train_baseline is not None:
        baseline = np.asarray(train_baseline, dtype=np.float64)
        print(
            "baseline:   train mean [%.2f %.2f %.2f] m/s (recorded at "
            "training time)" % tuple(baseline)
        )
    elif saved.get("validation_dataset"):
        train_root = Path(saved["dataset"])
        if not train_root.is_dir():
            raise SystemExit(
                f"{args.checkpoint} predates recording its own training-mean "
                f"baseline, and its training folder {train_root} is not "
                "reachable from here to recompute it (pre-split run: the "
                "folder being scored is not the training folder). Make the "
                "training folder reachable, or retrain to get a checkpoint "
                "that carries its own baseline."
            )
        _, train_velocity = reference_body_velocity(
            train_root / saved.get("csv_name", "flight.csv"),
            time_column=saved.get("time_column", "Time"),
            time_scale=float(saved.get("time_scale", 1.0)),
            lever_arm_m=saved.get("lever_arm_m"),
        )
        baseline = train_velocity.mean(axis=0)
        print(
            "baseline:   train mean [%.2f %.2f %.2f] m/s (recomputed from "
            "%s)" % (*baseline, train_root)
        )
    else:
        train_start, train_end = ranges["train"]
        baseline = dataset.velocity_body[train_start:train_end].mean(axis=0)
        print("baseline:   train mean [%.2f %.2f %.2f] m/s" % tuple(baseline))

    # The rotation that DEFINES the target, read only here and only to turn a
    # body-frame velocity back into a distance. It never reaches the model.
    _, _, reference_rotation = reference_body_frame(
        dataset_root / saved.get("csv_name", "flight.csv"),
        time_column=saved.get("time_column", "Time"),
        time_scale=float(saved.get("time_scale", 1.0)),
    )
    # --no-position drops the position columns from the TABLES, which is what
    # it is for. The trajectory figure needs the same rotation, and the run
    # that feeds it is the run the tables come from, so the integral is
    # computed whenever either wants it and the columns are dropped from the
    # table rows afterwards - asking for fewer columns is not a reason to hand
    # back a plot directory with a file missing from it.
    rotation = (
        reference_rotation if (args.plots or not args.no_position) else None
    )

    # camera_matrix is at NATIVE resolution; the frontend only ever sees images
    # already resized to image_size by VisualPairSource, so it must be handed
    # the WORKING-resolution matrix - the same one VisualPairSource computes
    # for itself internally - or its bearing scale and per-cell rotational
    # field are evaluated on the wrong coordinate grid. See the matching
    # comment in train_fixedwing_vo.py, whose training run this must score
    # under the same geometry as.
    camera_tensor = (
        None if camera_matrix is None
        else torch.from_numpy(
            resize_camera_matrix(camera_matrix, native_size, image_size)
        ).to(device)
    )
    plot_dir = args.plot_dir
    if args.plots and plot_dir is None:
        plot_dir = (
            args.output.parent / f"{args.output.stem}_plots"
            if args.output is not None
            else Path("artifacts/velocity_horizons_plots")
        )
    plotted: List[Path] = []
    report: Dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(dataset_root),
        "epoch": checkpoint.get("epoch"),
        "flight_ticks": total,
        "flight_minutes": duration_min,
        "horizons_minutes": list(horizons),
        "warmup_ticks": warmup,
        "disable_visual_input": disable_visual,
        "train_mean_baseline_m_s": [float(value) for value in baseline],
        "splits": {},
    }

    for name in splits:
        span = ranges[name]
        if span[1] - span[0] < 2:
            print(f"\n=== {name} ===  empty, skipped")
            continue
        minutes = float(attitude.times_s[span[1] - 1] - attitude.times_s[span[0]]) / 60.0
        print(f"\n=== {name} ===  ticks {span[0]}-{span[1]} ({minutes:.1f} min)")
        if name in ("train", "full") or scoring_training_folder:
            print("    NOT held out - a drift diagnostic, not an accuracy claim")
        tokens = encode_span_tokens(
            frontend,
            dataset.image_source,
            span=span,
            body_rate_rad_s=attitude.body_rate_rad_s,
            times_s=attitude.times_s,
            visual_dim=int(saved.get("visual_dim", 64)),
            device=device,
            camera_matrix=camera_tensor,
            batch_pairs=args.batch_pairs,
            num_workers=args.num_workers,
            disable_visual=disable_visual,
            progress=progress,
            attitude=attitude,
        )
        print(f"    {len(tokens)} visual events encoded")
        # ONE pass: the split streamed from its first tick to its last, state
        # zeroed once and never reset. Every horizon is a prefix of that run,
        # and the whole-split series the figures are drawn from falls out of
        # the same forward - there is no second pass over the model.
        results, span_entry = run_span_horizons(
            model,
            aiding=dataset.aiding,
            log_altitude=dataset.log_altitude,
            target_velocity=dataset.velocity_body,
            times_s=attitude.times_s,
            span=span,
            tokens=tokens,
            deployment_latency_s=deployment_latency_s,
            horizons_minutes=horizons,
            device=device,
            warmup_ticks=warmup,
            block_ticks=args.block_ticks,
            baseline=baseline,
            rotation_body_to_ned=rotation,
            collect_series=bool(args.plots or args.stratify),
            progress=progress,
            ablate_body_rate=ablate_body_rate,
            ablate_visual_age=ablate_visual_age,
            output_on_pairs=bool(saved.get("output_on_pairs", False)),
        )
        if args.no_position:
            for entry in results.values():
                for key in list(entry):
                    if key.startswith("pos_") or key == "path_length_m":
                        entry.pop(key)
        print(format_horizon_table(results))
        stratified = None
        if args.stratify and span_entry.get("fits") and "series" in span_entry:
            series = span_entry["series"]
            ticks = int(span[1] - span[0])
            predicted = np.asarray(series["vel_predicted_body"])[0, :ticks]
            target = np.asarray(series["vel_target_body"])[0, :ticks]
            conditions = flight_conditions(
                dataset.aiding[span[0]:span[1]],
                dataset.log_altitude[span[0]:span[1]],
                dataset.velocity_body[span[0]:span[1]],
            )
            stratified = stratified_errors(predicted, target, conditions)
            print("  whole-split error by flight condition:")
            print(format_stratified(stratified))
        if args.plots:
            # Two of the three figures are drawn from the run's series. If it
            # scored nothing they are skipped and only the summary figure -
            # which reads the tables instead - is written, which looks like a
            # plotting fault rather than a run that never happened. Say so.
            if not span_entry.get("fits"):
                print(
                    f"  ! the whole-split run scored nothing, so "
                    f"{name}_trajectory.png and {name}_errors.png are skipped: "
                    f"{span_entry.get('skipped', 'no reason recorded')}"
                )
            plotted.extend(save_split_plots(plot_dir, name, span_entry, results))
            span_entry.pop("series", None)
            report.setdefault("whole_span", {})[name] = span_entry
        span_entry.pop("series", None)
        report["splits"][name] = {
            "range": [int(span[0]), int(span[1])],
            "minutes": minutes,
            "held_out": name in ("validation", "test") and not scoring_training_folder,
            "visual_events": len(tokens),
            "horizons": results,
        }
        if stratified is not None:
            report["splits"][name]["stratified"] = stratified

    if plotted:
        print(f"\nwrote {len(plotted)} figure(s) to {plot_dir}")
        for path in plotted:
            print(f"  {path}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
