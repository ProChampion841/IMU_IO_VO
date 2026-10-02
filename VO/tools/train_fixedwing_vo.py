#!/usr/bin/env python3
"""Train the vision-Mamba VO estimator.

Inputs are images, the aircraft's own attitude, and its altitude. There is no
IMU and no flow cache: the scanned encoder is being trained, so its output
cannot be precomputed, and every image pair goes through the frontend inside
the training graph.

Three things this reports that a plain loss curve does not, and each exists
because of a specific way this problem has already misled us:

* **Per-axis RMSE.** The lateral axis fails independently of the other two.
  A single scalar RMSE is dominated by forward speed and stays flat while the
  axis that matters is not learning at all.

* **Skill against the training mean.** "Better than the previous run" is not
  the bar. Predicting the training split's mean velocity is free, and on the
  lateral axis it has previously beaten a trained model. Skill below zero
  means the model is worse than a constant.

* **Direction error, separately from speed error.** They have different
  observability - direction is recoverable from one frame pair and speed is
  not - so a model can be improving at the thing the camera can actually see
  while its RMSE is dominated by a scale it cannot.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import re
import sys
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm  # type: ignore[import-untyped]

_ROOT = Path(__file__).resolve().parents[1]
# The package lives under src/; tools/ is imported as a package from the
# repository root. Both have to be importable when a script is run directly.
for _entry in (_ROOT / "src", _ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from vio.data.calibration import maybe_load_camera_calibration
from vio.data.fixedwing_vo import (
    BODY_RATE_AIDING_SLICE,
    FixedWingVODataset,
    VO_AIDING_CHANNELS,
    VONormalizer,
    build_vo_dataset,
    hold_visual_velocity,
    normalize_index_ranges,
    reference_body_frame,
    scatter_visual_tokens,
    visual_age_seconds,
)
from vio.data.image_pairs import resize_camera_matrix
from vio.models.velocity_horizons import (
    encode_span_tokens,
    format_horizon_table,
    horizon_csv_row,
    horizon_metric_names,
    parse_horizon_minutes,
    stream_horizon_metrics,
)
from vio.models.frontend_factory import (
    build_frontend,
    frontend_class,
    resolve_velocity_mode,
)
from vio.models.vision_mamba_vo import (
    VOStreamState,
    VisionMambaFlowFrontend,
    VisionMambaVO,
)
from vio.utils.checkpoint_io import load_checkpoint
from vio.utils.metrics_csv import open_metrics_csv
from vio.utils.velocity_metrics import RunningVelocityStats, masked_velocity_stats


@dataclass
class Distributed:
    """Where this process sits in the job, and whether there is a job at all.

    Single-GPU runs get ``enabled=False`` and every collective below becomes a
    no-op, so exactly one code path is exercised either way. A trainer with two
    branches is a trainer where one of them is untested.
    """

    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def log(self, *parts, **kwargs) -> None:
        """Print from one rank. Twenty copies of the same line is not a log."""

        if self.is_main:
            print(*parts, **kwargs)

    def reduce_sum(self, values: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        return values

    def reduce_stats(self, stats: RunningVelocityStats) -> RunningVelocityStats:
        """Combine one epoch's velocity statistics across ranks.

        Sums reduce with SUM and maxima with MAX, which is the whole reason
        RunningVelocityStats keeps them apart: a true epoch RMS is the square
        root of the summed squares over the summed count, never the mean of
        per-rank RMS values, and a maximum is not an average of maxima either.
        """

        if not self.enabled:
            return stats
        sums = torch.as_tensor(stats.sums, dtype=torch.float64, device=self.device)
        maxima = torch.as_tensor(stats.maxima, dtype=torch.float64, device=self.device)
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        stats.sums = sums.cpu().numpy()
        stats.maxima = maxima.cpu().numpy()
        return stats

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def shutdown(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


#: Matches a --device that names several GPUs: "0,1,2" or "cuda:0,cuda:1".
_GPU_LIST = re.compile(r"^\s*(?:cuda:)?\d+\s*(?:,\s*(?:cuda:)?\d+\s*)+$")


def parse_gpu_list(device: str) -> Optional[List[int]]:
    """The GPU indices in a multi-device ``--device``, or None if it names one.

    ``"cuda"``, ``"cuda:0"``, ``"cpu"`` and ``"3"`` all name a single device and
    return None. ``"0,1,2"`` and ``"cuda:0,cuda:1"`` return the list.
    """

    if not isinstance(device, str) or not _GPU_LIST.match(device):
        return None
    indices = [int(piece.strip().removeprefix("cuda:")) for piece in device.split(",")]
    if len(set(indices)) != len(indices):
        raise ValueError(f"--device {device} repeats a GPU")
    return indices


def relaunch_under_torchrun(argv: Optional[Sequence[str]], device: str) -> int:
    """Re-exec this script under torchrun, one process per named GPU.

    A list of GPUs cannot simply be assigned: this trainer is DDP, one process
    per card, and the device comes from LOCAL_RANK. Handing the list to
    torchrun is what turns ``--device 0,1,2`` into the thing it obviously
    means, instead of ten processes all binding cuda:0 - see
    :func:`setup_distributed`.

    DataParallel would accept the list directly and is the wrong answer: it is
    deprecated, it replicates the model every step, and on ten cards it is
    dramatically slower than DDP.
    """

    if not torch.cuda.is_available():
        raise SystemExit(f"--device {device} names GPUs, but CUDA is not available")
    present = torch.cuda.device_count()
    missing = [index for index in parse_gpu_list(device) or [] if index >= present]
    if missing:
        raise SystemExit(
            f"--device {device} names GPU(s) {missing} but this machine has "
            f"{present} ({', '.join(str(i) for i in range(present))})"
        )

    indices = parse_gpu_list(device) or []
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Replace the list with a plain "cuda": under torchrun the card comes from
    # LOCAL_RANK, and CUDA_VISIBLE_DEVICES below has already narrowed the set.
    rewritten: List[str] = []
    skip = False
    for item in arguments:
        if skip:
            skip = False
            continue
        if item == "--device":
            rewritten += ["--device", "cuda"]
            skip = True
        elif item.startswith("--device="):
            rewritten.append("--device=cuda")
        else:
            rewritten.append(item)

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in indices)
    command = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc_per_node={len(indices)}",
        str(Path(__file__).resolve()), *rewritten,
    ]
    print(f"--device {device} -> {len(indices)} ranks via torchrun "
          f"(CUDA_VISIBLE_DEVICES={environment['CUDA_VISIBLE_DEVICES']})")
    return subprocess.call(command, env=environment)


def setup_distributed(args: argparse.Namespace) -> Distributed:
    """Join the torchrun job if there is one; otherwise run single-process.

    torchrun exports RANK/WORLD_SIZE/LOCAL_RANK, so their presence is the
    signal - no separate flag to forget. The device comes from LOCAL_RANK and
    not from ``--device``: ten processes all binding cuda:0 is the classic way
    to turn a ten-GPU node into a one-GPU node with ten times the memory
    pressure.
    """

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        device = torch.device(args.device)
        if device.type == "cuda":
            # "cuda" with no index is the common spelling and set_device
            # rejects it, so pin it to a concrete card here rather than
            # failing after the datasets have been built.
            device = torch.device("cuda", device.index or 0)
            torch.cuda.set_device(device)
        return Distributed(False, 0, 1, 0, device)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # The backend follows the device that was ASKED for, not whatever hardware
    # happens to be present. Deciding from torch.cuda.is_available() picks
    # NCCL for an explicit `--device cpu` run on a GPU box, and NCCL without a
    # CUDA tensor to communicate hangs at the first collective rather than
    # failing.
    want_cuda = torch.device(args.device).type == "cuda" and torch.cuda.is_available()
    if not dist.is_initialized():
        backend = "nccl" if want_cuda else "gloo"
        if getattr(args, "dist_file", None) is not None:
            posix = Path(args.dist_file).resolve().as_posix()
            dist.init_process_group(
                backend=backend,
                init_method="file:///" + posix.lstrip("/"),
                rank=rank, world_size=world_size,
            )
        else:
            dist.init_process_group(backend=backend)
    if want_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return Distributed(True, rank, world_size, local_rank, device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    data = parser.add_argument_group("data")
    data.add_argument("--dataset", type=Path, required=True)
    data.add_argument("--csv-name", default="flight.csv")
    data.add_argument("--image-folder", default="images")
    data.add_argument("--time-column", default="Time")
    data.add_argument("--time-scale", type=float, default=1.0)
    data.add_argument(
        "--validation-dataset", type=Path, default=None,
        help="A pre-split validation dataset, in the same layout as --dataset. "
             "When given, --dataset is used WHOLE as training and this whole "
             "directory as validation: no fractions are applied to either, and "
             "no leakage between them is possible because they are different "
             "files. Produced by tools/split_dataset.py.",
    )
    data.add_argument(
        "--test-dataset", type=Path, default=None,
        help="A pre-split test dataset. Requires --validation-dataset.",
    )
    data.add_argument(
        "--train-fraction", type=float, default=0.6,
        help="Chronological start-to-end split: this fraction of the flight, "
             "from the start, is training.",
    )
    data.add_argument(
        "--validation-fraction", type=float, default=0.2,
        help="The next fraction after training is validation; whatever "
             "remains after that (minus the one-window gaps) is test. "
             "Default 0.6/0.2/0.2.",
    )
    data.add_argument(
        "--attitude-columns", nargs=3, default=None,
        help="Defaults to NavEul*. GPSNavEul* is refused: it is the rotation "
             "that DEFINES the target, so feeding it in is target leakage.",
    )
    data.add_argument(
        "--altitude-column", default=None,
        help="Relative-altitude field. By default tries relativeAlt, "
             "RelatedAlt, then the legacy Barometer field.",
    )
    data.add_argument(
        "--calibration", type=Path, default=None,
        help="configs/vo/camera_fixedwing.json. Without it the measured "
             "flow is in fractions of the working image, which is "
             "comparable within one camera and meaningless across two; "
             "with it the token is normalised bearing per second.",
    )
    data.add_argument(
        "--allow-reference-attitude", action="store_true",
        help="Permit GPSNavEul* as an input. This is a leakage ablation and "
             "its result must be reported separately.",
    )
    data.add_argument(
        "--color", action="store_true",
        help="Load the frames as RGB (three channels) instead of grayscale. "
             "Worth it for a colour camera: vegetation, soil and water that are "
             "one grey level can be three distinct colours, which is texture the "
             "matcher can use. Triples image transport; no effect on JPEGs that "
             "are single-channel on disk.",
    )

    window = parser.add_argument_group("windows")
    window.add_argument("--window-length", type=int, default=600)
    window.add_argument("--stride", type=int, default=300)
    window.add_argument(
        "--warmup", type=int, default=None,
        help="Ticks at the start of every window left out of the loss (a "
             "window is a cold start). Default 20 for the original frontend; "
             "for --frontend planar, the ticks before a window's first image "
             "pair can arrive (pair interval + deployment latency) plus 5.",
    )
    window.add_argument(
        "--max-visual-events", type=int, default=120,
        help="Image pairs carried per window. A 600-tick window is 6 s, which "
             "at 20 Hz holds 120 pairs; anything lower subsamples them evenly "
             "and trains at a visual rate the deployed system will not have. "
             "Checkpointing makes this nearly free in memory - see "
             "--frontend-chunk.",
    )
    window.add_argument(
        "--frame-gap", type=int, default=None,
        help="Frames between the two images of a pair. Default 1 for the "
             "original frontend; for --frontend planar, however many frames "
             "span --planar-baseline-s (1.0 s) at this capture's median frame "
             "rate - 20 at 20 Hz. tools/check_motion_budget.py shows what each "
             "gap buys.",
    )
    window.add_argument(
        "--output-on-pairs", action="store_true",
        help="One velocity output per image pair, at the pair interval: pairs "
             "no longer overlap - (0, g), (g, 2g), ... for --frame-gap g - so a "
             "new measurement arrives every g frames (every 0.5 s for g = 10 at "
             "20 Hz), and the model's velocity is scored (and, in the "
             "evaluator, reported) only on the tick each pair is delivered, "
             "held in between. Default off: a pair ends on every frame and the "
             "output is scored on every telemetry tick.",
    )
    window.add_argument(
        "--random-pair-phase", action="store_true",
        help="Training augmentation for --output-on-pairs: each training window "
             "draws which frame its pair tiling starts on (0 .. g-1), so the same "
             "stretch of flight is seen as g different sets of image pairs "
             "across epochs instead of one. The pair interval and the output "
             "cadence are unchanged, and validation, test, --eval-train-split, "
             "the horizon pass and the evaluator all stay on the fixed tiling "
             "(phase 0), so their numbers remain comparable with runs without "
             "it. Default off.",
    )
    window.add_argument(
        "--planar-baseline-s", type=float, default=1.0,
        help="Time between the two images of a pair that the planar default "
             "--frame-gap aims for (default 1.0 s: about 13 cells of ground "
             "motion at 200 m and 20 m/s, 1 percent single-pair precision, "
             "80-90 percent image overlap).",
    )
    window.add_argument("--deployment-latency-s", type=float, default=0.35)
    window.add_argument(
        "--max-frame-gap-s", type=float, default=None,
        help="Reject an image pair whose two exposures are further apart than "
             "this. A dropout leaves two frames adjacent in the folder but far "
             "apart in time; across that gap the true displacement leaves the "
             "correlator's bounded search window, so what comes back is a "
             "confident wrong match rather than a large one. Unset means every "
             "pair is kept however long its interval for the original "
             "frontend, and 1.5x the nominal pair interval for --frontend "
             "planar.",
    )
    window.add_argument(
        "--image-time-offset", type=float, default=0.0,
        help="Constant seconds to add to every image timestamp, to put the "
             "camera clock on the telemetry clock. Positive means the image "
             "stamps read early.",
    )
    window.add_argument(
        "--image-time-offset-file", type=Path, default=None,
        help="JSON holding a time-varying camera-to-telemetry offset, as "
             '{\"times_s\": [...], \"offsets_s\": [...]} (the offset_table key '
             "of tools/estimate_time_offset.py output is also accepted). "
             "Interpolated per frame, so a drifting clock is corrected along "
             "the flight rather than by one average. Overrides "
             "--image-time-offset.",
    )
    window.add_argument(
        "--lever-arm", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="Offset from the GPS antenna to the camera, in BODY axes, metres "
             "(x forward, y right, z down). GPS measures velocity at the "
             "antenna, but the camera is what sees the motion the network has "
             "to explain, and a rigid body in a turn moves its points at "
             "different velocities: v_camera = v_gps + omega x r. On a wing "
             "mount the two are metres apart, so the label describes a point "
             "the camera is not at, by an amount proportional to turn rate -- "
             "correlated with the manoeuvres the estimator is judged on rather "
             "than averaging away. At 30 deg/s and 2 m that is about 1 m/s. "
             "Taken from the calibration's mounting.gps_to_camera_m when that "
             "is present; this flag overrides it. Unset means no correction.",
    )
    window.add_argument(
        "--image-size", type=int, nargs=2, default=(576, 1024),
        help="HEIGHT WIDTH the frontend works at. 16:9, matching the "
             "1920x1080 source: a non-matching ratio stretches the image "
             "anisotropically, which costs matching quality even though the "
             "flow magnitude stays correct. Bigger is better for "
             "correspondence; 1080 1920 is native and fits with "
             "--frontend-chunk 1.",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--visual-dim", type=int, default=64)
    model.add_argument("--stem-dim", type=int, default=64)
    model.add_argument("--stem-depth", type=int, default=2)
    model.add_argument("--patch-size", type=int, default=8)
    model.add_argument("--context-grid", type=int, nargs=2, default=(12, 16))
    model.add_argument("--token-grid", type=int, default=6)
    model.add_argument(
        "--correlation-radius", type=int, default=None,
        help="Correlation search radius in cells. Default 4 for the original "
             "frontend, 3 for --frontend planar (its search is already centred "
             "on the predicted motion).",
    )
    model.add_argument("--aiding-dim", type=int, default=64)
    model.add_argument("--fusion-dim", type=int, default=96)
    model.add_argument("--dropout", type=float, default=0.1)

    gate = parser.add_argument_group("visual reliability gate")
    gate.add_argument(
        "--min-pool-weight", type=float, default=1e-4,
        help="Pooled correlation weight below which a cell counts as "
             "unmeasured. The 1e-4 default is a numerical guard, not a quality "
             "filter - it lets through essentially every cell that correlated "
             "at all. Try 0.02/0.05/0.10/0.20 on a development split to make "
             "it a real one.",
    )
    gate.add_argument(
        "--max-cell-entropy", type=float, default=1.0,
        help="Reject a correlation cell whose normalized entropy reaches this. "
             "1.0 (the default) rejects nothing. Entropy is normalized by each "
             "cell's OWN candidate count, so 0.0 is a single certain match and "
             "1.0 is a flat, uninformative surface. This catches the failure "
             "the peak SCORE cannot: a blurry patch scores ~0.97 against every "
             "candidate, which looks excellent and means nothing.",
    )
    gate.add_argument(
        "--min-cell-confidence", type=float, default=0.0,
        help="Reject a cell whose usable confidence is at or below this. "
             "Usable confidence already has the uniform floor removed, so it "
             "reaches 0 for a no-information cell; a threshold here is a "
             "threshold on real evidence. 0.0 (the default) rejects nothing.",
    )
    gate.add_argument(
        "--min-score-margin", type=float, default=0.0,
        help="Reject a cell whose RAW top-1 minus top-2 correlation score is "
             "at or below this. The temperature-free alternative to "
             "--min-cell-confidence and --max-cell-entropy: those read the "
             "softmax, whose sharpness the temperature sets, while this reads "
             "the scores themselves and no temperature can move it. In "
             "correlation-score units, not on [0, 1], so sweep it separately. "
             "0.0 (the default) rejects nothing.",
    )
    gate.add_argument(
        "--reject-boundary-peaks", action="store_true",
        help="Reject a cell whose correlation peak sits on the edge of the "
             "search window. The soft-argmax cannot look outside the window, "
             "so such a cell reports a lower bound as though it were a "
             "measurement.",
    )
    gate.add_argument(
        "--min-reliable-cell-fraction", type=float, default=0.0,
        help="Refuse the WHOLE image pair when fewer than this fraction of "
             "cells survive the gates above. A refused pair is dropped, not "
             "zeroed: its tick keeps visual_present=0 and visual_age keeps "
             "growing, which is the state the fusion contract already has for "
             "'no image arrived'. No measurement beats a confident wrong one.",
    )
    model.add_argument(
        "--rotation-mode", default="field", choices=("field", "constant"),
        help="How the predicted rotation centres the correlation search. "
             "'field' evaluates the rotational optical-flow field at every "
             "cell; 'constant' is the old single displacement for the whole "
             "image, kept so the change can be ablated. A constant can only "
             "represent the order-zero terms of that field, and it provably "
             "removes NONE of the curl produced by rotation about the optical "
             "axis, because that field is odd about the principal point and so "
             "has zero mean - the best constant for it is exactly zero.",
    )
    model.add_argument(
        "--rotation-map", type=Path, default=None,
        help="artifacts/camera_imu_rotation.json. Seeds the search centre so "
             "rotation is removed from the first step instead of being learned "
             "through the correlator.",
    )
    model.add_argument(
        "--frontend", default="mamba_correlation",
        choices=("mamba_correlation", "planar"),
        help="'mamba_correlation' (default) is the original frontend: a learned "
             "small-angle rotation field centres the search and the network "
             "learns the scale. 'planar' removes rotation EXACTLY by warping "
             "with the attitude, matches coarse-to-fine, and solves the camera "
             "translation over a flat ground plane in closed form, with "
             "RelativeAlt pinning the vertical - a metric velocity per pair "
             "before any training. Built for a nadir camera well above the "
             "terrain relief (hundreds of metres) and a long --frame-gap; see "
             "tools/check_motion_budget.py for the gap and "
             "tools/estimate_camera_mounting.py for --camera-mounting.",
    )
    model.add_argument(
        "--velocity-mode", default="auto",
        choices=("auto", "heads", "geometric_residual"),
        help="How the output velocity is formed. 'heads': the original factored "
             "direction x altitude x bearing-rate heads. 'geometric_residual': "
             "the latest per-pair velocity from --frontend planar, held between "
             "pairs, plus a learned correction that starts at zero - so the "
             "untrained model already outputs the geometric estimate. 'auto' "
             "(default) picks geometric_residual for --frontend planar and heads "
             "otherwise.",
    )

    planar = parser.add_argument_group("planar frontend (--frontend planar)")
    planar.add_argument(
        "--camera-mounting", default=None, metavar="NAME|MATRIX|FILE|auto",
        help="Camera-to-body rotation. One of top_forward, right_forward, "
             "left_forward, bottom_forward (which image edge faces the nose, "
             "optical axis straight down); or a 3x3 camera_from_body matrix as "
             "JSON; or a JSON file with a camera_from_body key (the output of "
             "tools/estimate_camera_mounting.py); or 'auto' to measure it from "
             "the training images at start-up. Default: the calibration file's "
             "mounting.camera_from_body, and 'auto' when it has none. A wrong "
             "mounting swaps or negates the forward and lateral axes.",
    )
    planar.add_argument(
        "--prior-velocity", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
        help="Body velocity (m/s) the coarse search is centred on. Default: the "
             "training split's mean velocity. The coarse window reaches +/- "
             "(--coarse-radius x --coarse-factor) cells around it.",
    )
    planar.add_argument("--coarse-factor", type=int, default=None,
                        help="Feature pooling for the coarse stage. Default 4, or 2 "
                             "when the feature map is too small for 4.")
    planar.add_argument("--coarse-radius", type=int, default=None,
                        help="Coarse search radius in pooled cells. Default 6, "
                             "reduced to fit a small feature map.")
    planar.add_argument("--coarse-highpass", type=int, default=5,
                        help="Local-mean kernel removed from coarse features (odd; 1 = off).")
    planar.add_argument("--fine-highpass", type=int, default=9,
                        help="Local-mean kernel removed from fine features (odd; 1 = off).")
    planar.add_argument("--fine-iterations", type=int, default=2,
                        help="Fine correlate-and-fit passes (the last is differentiated).")
    planar.add_argument("--huber-cells", type=float, default=1.0,
                        help="Robust-fit residual, in cells, beyond which a cell stops counting fully.")
    planar.add_argument(
        "--altitude-constraint", type=float, default=300.0,
        help="Weight of the altimeter row n.t = h0 - h1 in the planar fit, "
             "relative to the image's own information. The image sees the "
             "vertical only as a percent-level scale change; the altimeter "
             "measures it. 0 = image only.",
    )
    planar.add_argument("--no-learn-mounting", action="store_true",
                        help="Freeze the mounting (no learned misalignment correction).")
    planar.add_argument("--mounting-lr-scale", type=float, default=0.1,
                        help="Learning-rate multiplier for the mounting correction.")
    planar.add_argument("--max-geometric-speed", type=float, default=80.0,
                        help="A per-pair velocity faster than this (m/s) is refused.")
    planar.add_argument("--min-fit-cells", type=float, default=8.0,
                        help="Minimum confidence-weighted cell count for a pair's fit.")

    train = parser.add_argument_group("training")
    train.add_argument(
        "--epochs", type=int, default=3000,
        help="There is no early stopping: the run goes to this number and "
             "stops. Pick it from OPTIMISER STEPS, not from wall clock - the "
             "step count is windows/(batch x ranks), so ten ranks at batch 4 "
             "over a 21-minute split is about 10 steps an epoch and 30 epochs "
             "is 300 steps, which is not a trained model. 3000 epochs is "
             "roughly 31k steps there.",
    )
    train.add_argument(
        "--select-on", default="vel_rmse",
        choices=("vel_rmse", "vel_rmse_x", "vel_rmse_y", "vel_rmse_z",
                 "vel_dir_rmse", "loss"),
        help="Validation metric that decides best.pt (smaller is better). "
             "With no early stopping and thousands of epochs, best.pt is the "
             "minimum of a noisy statistic over very many draws, so a "
             "single-axis metric picks the luckiest epoch on that axis while "
             "the other two may have got worse. The default is the 3-axis "
             "RMSE for that reason.",
    )
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--direction-weight", type=float, default=0.5)
    train.add_argument(
        "--velocity-loss", choices=("nll", "simple"), default=None,
        help="Default 'nll' for the original frontend and 'simple' for "
             "--frontend planar. "
             "'nll' is the Gaussian NLL plus von Mises-Fisher "
             "direction term, with learned variance and concentration. "
             "'simple' is deterministic Smooth L1 on velocity plus a FIXED "
             "cosine direction term, with no uncertainty heads in the loss at "
             "all. Use 'simple' when training loss is falling while "
             "val_vel_rmse is not: the uncertainty heads can buy loss by "
             "growing confident on memorised samples, which looks like "
             "progress and is not. Both paths score through the same function, "
             "so a run cannot optimise one and be selected on the other.",
    )
    train.add_argument(
        "--huber-delta", type=float, default=1.0, metavar="M_S",
        help="Smooth L1 transition point in m/s, for --velocity-loss simple. "
             "Below it the penalty is quadratic, above it linear, so a GPS "
             "glitch or a mis-timed frame costs a bounded amount instead of "
             "dominating its batch.",
    )
    train.add_argument(
        "--photometric-augment", type=float, default=0.0, metavar="STRENGTH",
        help="Training-only exposure jitter: random gain, offset and gamma per "
             "pair, slightly different between the two images of a pair (what an "
             "auto-exposure camera does). STRENGTH is the standard deviation of "
             "the shared log-gain; 0.1-0.2 is sensible. 0 (default) is off.",
    )
    train.add_argument(
        "--patience", type=int, default=0, metavar="EPOCHS",
        help="Stop when the --select-on metric has not improved by --min-delta "
             "for this many epochs. 0 (default) never stops early. best.pt is "
             "the selected checkpoint either way.",
    )
    train.add_argument("--min-delta", type=float, default=0.0,
                       help="Improvement smaller than this does not reset --patience.")
    train.add_argument(
        "--lr-warmup-epochs", type=int, default=0,
        help="Ramp the learning rate linearly from 1/N to full over the first N "
             "epochs before the cosine decay. Large effective batches "
             "(--lr-scaling linear over many ranks) are the usual reason a "
             "first epoch spikes; a short warm-up avoids it. 0 = none.",
    )
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument(
        "--lr-scaling", choices=("none", "linear", "sqrt"), default="none",
        help="Every rank processes its own windows, so N ranks make the "
             "effective batch N times larger and the optimizer takes N times "
             "FEWER steps per epoch. 'linear' multiplies the learning rate by "
             "N (the usual large-batch rule), 'sqrt' by sqrt(N), 'none' leaves "
             "it alone. Default matches the VIO trainer; change it and say so, "
             "because it makes runs at different rank counts incomparable.",
    )
    train.add_argument(
        "--dist-file", type=Path, default=None,
        help="Rendezvous through a shared FILE instead of a TCP store. Needed "
             "where a TCP store cannot be created (some Windows PyTorch builds "
             "ship without libuv) and handy for single-node runs. The path "
             "must be visible to every rank and must not exist beforehand.",
    )
    train.add_argument(
        "--no-progress", action="store_true",
        help="Disable the batch progress bars (useful when an external job "
             "scheduler captures stderr).",
    )
    train.add_argument(
        "--frontend-chunk", type=int, default=8,
        help="Recompute the frontend in the backward pass, this many "
             "pairs at a time. Activation memory becomes flat in the "
             "pair count instead of linear, which is what makes a large "
             "image with a full-rate window possible at all. Smaller = "
             "less memory, more recomputation. 0 disables it.",
    )
    train.add_argument("--run-dir", type=Path, default=Path("runs/vo"))
    train.add_argument(
        "--save-every", type=int, default=1, metavar="N",
        help="Keep a checkpoint of its own every N epochs, in "
             "<run-dir>/epochs/epoch_XXXX.pt, on top of best.pt and the "
             "overwritten last.pt. Default 1: every epoch. The payload is the "
             "same as last.pt, so any of them can be resumed from or scored - "
             "which also means each carries optimizer state as well as "
             "weights, about 8 MB at the default model size, so a 3000-epoch "
             "run is roughly 25 GB. The projected total is printed at the "
             "first save; --keep-last caps it and 0 disables the whole thing.",
    )
    train.add_argument(
        "--keep-last", type=int, default=0, metavar="K",
        help="Delete all but the K newest per-epoch checkpoints as they are "
             "written. 0 (the default) keeps every one. best.pt and last.pt "
             "are never touched.",
    )
    train.add_argument(
        "--resume", default=None, metavar="PATH",
        help="Continue an interrupted run from a checkpoint. 'auto' means "
             "$RUN_DIR/last.pt, which is written after every completed epoch. "
             "Re-run the ORIGINAL command with this added: the dataset, the "
             "input contract and the model shape are fingerprinted into the "
             "checkpoint and a mismatch is refused, because silently "
             "continuing on different data is worse than starting over.",
    )
    train.add_argument(
        "--disable-visual-input", action="store_true",
        help="Zero the visual token in BOTH training and validation. The "
             "attitude-and-altitude-only floor: anything that does not beat it "
             "is not using the camera.",
    )
    train.add_argument(
        "--ablate-body-rate", action="store_true",
        help="Zero the aiding vector's p/q/r channels (BODY_RATE_AIDING_SLICE) "
             "in both training and evaluation, WITHOUT changing AIDING_INPUT_DIM "
             "- an ablation for whether body rate is earning its place in the "
             "aiding vector, isolated from every other channel and from "
             "--ablate-visual-age. Recorded in the resume fingerprint and the "
             "checkpoint, so a checkpoint trained with one setting cannot be "
             "silently evaluated under the other.",
    )
    train.add_argument(
        "--ablate-visual-age", action="store_true",
        help="Zero the fusion input's visual_age channel in both training and "
             "evaluation, WITHOUT changing the fusion input width - an ablation "
             "for whether visual_age is earning its place, isolated from "
             "--ablate-body-rate. Recorded in the resume fingerprint and the "
             "checkpoint, same as --ablate-body-rate.",
    )
    train.add_argument(
        "--eval-train-split", action="store_true",
        help="Additionally score the TRAINING split through the same "
             "evaluate() path validation uses - eval mode, dropout off, one "
             "pass over the whole split - and log it as traineval_*, "
             "separate from train_* (which is measured mid-epoch, with "
             "dropout on, averaged over a model that is still changing). "
             "Settles whether a train/val gap is real generalisation or an "
             "artefact of those two differences: if traineval_vel_rmse tracks "
             "val_vel_rmse closely while train_vel_rmse does not, the gap was "
             "never about generalisation. Costs a full extra pass over the "
             "training data every --eval-train-split-every epochs.",
    )
    train.add_argument(
        "--eval-train-split-every", type=int, default=1, metavar="N",
        help="Run --eval-train-split every N epochs, and always on the last "
             "one. Only applies when --eval-train-split is set.",
    )

    horizon = parser.add_argument_group("horizons")
    horizon.add_argument(
        "--horizon-minutes", default="", metavar="LIST",
        help="OFF by default: training runs the windowed validation pass and "
             "nothing else. Pass a list of prefix lengths in MINUTES (e.g. "
             "0.5,1,5,10,15,20,30,40) to also score the held-out split during "
             "training. The split is streamed once from its first tick to its "
             "last, state zeroed once and never reset, and each horizon is "
             "the FIRST H minutes of that run - so it measures what a "
             "--window-length of 600 ticks (six seconds) cannot: whether the "
             "recurrent state accumulates error over a long run. Adds "
             "vel_rmse/vel_max_error/vel_dir_rmse/vel_dir_max_error and the "
             "dead-reckoned pos_error_final/_max/_rmse/pos_drift_percent "
             "columns per horizon to metrics.csv. It is not cheap - see "
             "--horizon-every. To get these numbers without paying for them "
             "every epoch, leave this empty and run "
             "tools/evaluate_velocity_horizons.py on a checkpoint instead.",
    )
    horizon.add_argument(
        "--horizon-split", default="validation",
        choices=("train", "validation", "test"),
        help="Which split the legs are cut from. Anything but validation or "
             "test is a drift diagnostic, not a held-out number.",
    )
    horizon.add_argument(
        "--horizon-every", type=int, default=1, metavar="N",
        help="Run the horizon pass every N epochs, and always on the last one. "
             "Only applies when --horizon-minutes is set, which it is not by "
             "default. Default 1: every epoch. Worth knowing what that costs - the "
             "frontend weights move each epoch, so its tokens cannot be cached "
             "across epochs, and the scan walks every tick of every leg: a "
             "full 1..30 min sweep is 486,000 scan ticks on top of one pass "
             "over the split's images, measured at roughly 20 minutes per pass "
             "on a 14-minute validation split. Raise this to sample the curve "
             "less often on a long capture.",
    )
    horizon.add_argument(
        "--horizon-block-ticks", type=int, default=2000,
        help="Ticks streamed per block. Memory and speed only - the state is "
             "carried across blocks, so the result does not depend on it.",
    )
    horizon.add_argument(
        "--horizon-batch-pairs", type=int, default=8,
        help="Image pairs per frontend batch in the horizon token pass.",
    )
    horizon.add_argument(
        "--no-horizon-position", action="store_true",
        help="Drop the dead-reckoning position columns. They integrate the "
             "velocity over each leg through the REFERENCE attitude, applied "
             "to the prediction and the truth alike, so they report the drift "
             "the VELOCITY estimate causes: a deployed system also carries "
             "attitude error, which this excludes by construction.",
    )
    return parser


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------


def resolve_ranges(args: argparse.Namespace, total: int) -> Dict[str, Tuple[int, int]]:
    """Telemetry index ranges per split: chronological, start to end, always.

    Condition-segment splitting (``--split-manifest``) is intentionally not
    consulted here - by request, every run uses the same start-to-end cut:
    the first ``--train-fraction`` of the flight is training, the next
    ``--validation-fraction`` is validation, and the remainder is test. A gap
    of one window is left between phases so no validation window overlaps a
    training window - overlapping windows share telemetry, and a model that
    memorised a window would score on it twice.
    """

    gap = args.window_length
    train_end = int(total * args.train_fraction)
    validation_end = train_end + int(total * args.validation_fraction)
    return {
        "train": (0, train_end),
        "validation": (train_end + gap, validation_end),
        "test": (validation_end + gap, total),
    }


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------


def resolve_image_time_offset(
    args: argparse.Namespace,
) -> "float | Dict[str, List[float]]":
    """The camera-to-telemetry clock correction named by the flags.

    A table is interpolated per frame, so a drifting offset is corrected along
    the flight rather than by one average; a float is one constant offset.
    Resolved through a function rather than a local so that the training loop
    and the resume fingerprint cannot disagree about what was applied.
    """

    if args.image_time_offset_file is not None:
        payload = json.loads(
            Path(args.image_time_offset_file).read_text(encoding="utf-8")
        )
        table = payload.get("offset_table", payload)
        if not (isinstance(table, dict) and "times_s" in table and "offsets_s" in table):
            raise SystemExit(
                f"{args.image_time_offset_file} has no times_s/offsets_s table. "
                "Expected {\"times_s\": [...], \"offsets_s\": [...]}, or that "
                "object under an \"offset_table\" key."
            )
        return {
            "times_s": [float(v) for v in table["times_s"]],
            "offsets_s": [float(v) for v in table["offsets_s"]],
        }
    return float(args.image_time_offset)


def resolve_lever_arm(args: argparse.Namespace) -> Optional[List[float]]:
    """GPS antenna -> camera offset in body axes, metres, or None.

    ``--lever-arm`` wins over the calibration's ``mounting.gps_to_camera_m`` so
    a mounting can be tried without editing the manifest. An all-zero arm is
    normalised to None: it is not a correction, and recording it as one would
    make two identical runs disagree on resume.
    """

    arm: Optional[List[float]] = None
    if args.lever_arm is not None:
        arm = [float(v) for v in args.lever_arm]
    elif args.calibration is not None:
        mounting = json.loads(
            Path(args.calibration).read_text(encoding="utf-8")
        ).get("mounting", {})
        recorded = mounting.get("gps_to_camera_m")
        if recorded is not None:
            arm = [float(v) for v in recorded]
    if arm is None:
        return None
    if len(arm) != 3 or not all(math.isfinite(v) for v in arm):
        raise SystemExit(
            "The lever arm must be three finite numbers: body x y z in metres."
        )
    return arm if any(v != 0.0 for v in arm) else None


def build_schedule(
    optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int = 0
) -> torch.optim.lr_scheduler.LRScheduler:
    """Cosine decay to zero over ``epochs``, stepped once per epoch.

    With ``warmup_epochs`` the first N epochs ramp linearly from 1/N of the
    rate to all of it, then the same cosine takes over. Without it this is
    exactly the CosineAnnealingLR every earlier run used, so resuming one of
    those restores the same schedule object it saved.
    """

    if warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    warm = int(warmup_epochs)
    total = max(int(epochs), 1)

    def factor(epoch: int) -> float:
        if epoch < warm:
            return (epoch + 1) / warm
        return 0.5 * (1.0 + math.cos(math.pi * min(epoch, total) / total))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def parse_camera_mounting(value: Any) -> List[List[float]]:
    """A camera_from_body rotation from a name, a matrix, JSON text or a file.

    Accepts what ``--camera-mounting`` and a calibration's
    ``mounting.camera_from_body`` may hold: one of
    :data:`~vio.models.planar_geometry.NADIR_MOUNTINGS`' names, a 3x3 nested
    list, that list as JSON text, or the path of a JSON file with a
    ``camera_from_body`` key (top level or under ``mounting``) - which is what
    ``tools/estimate_camera_mounting.py`` writes. Validated as a proper
    rotation either way.
    """

    from vio.models.planar_geometry import NADIR_MOUNTINGS, mounting_matrix

    candidate = value
    if isinstance(candidate, str):
        text = candidate.strip()
        if text in NADIR_MOUNTINGS:
            return mounting_matrix(text).tolist()
        if text.startswith("["):
            candidate = json.loads(text)
        else:
            path = Path(text).expanduser()
            if not path.is_file():
                raise SystemExit(
                    f"--camera-mounting {value!r} is not a mounting name "
                    f"({', '.join(sorted(NADIR_MOUNTINGS))}), a JSON matrix, or a file"
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
            candidate = payload.get("camera_from_body") or payload.get(
                "mounting", {}
            ).get("camera_from_body")
            if candidate is None:
                raise SystemExit(f"{path} has no camera_from_body")
            if isinstance(candidate, str):
                return mounting_matrix(candidate).tolist()
    try:
        return mounting_matrix(candidate).tolist()
    except ValueError as error:
        raise SystemExit(f"invalid camera mounting {value!r}: {error}") from error


def resolve_frontend_defaults(
    args: argparse.Namespace, *, frame_interval_s: float, tick_interval_s: float
) -> List[str]:
    """Fill the flags whose right value depends on the frontend.

    The original frontend keeps its historical defaults exactly (frame gap 1,
    warmup 20, radius 4, NLL loss, no pair-interval cap), so every older
    command line means what it always meant. The planar frontend gets the
    values its geometry needs, derived from THIS capture's clocks:

    * ``--frame-gap``: the frames spanning ``--planar-baseline-s`` at the
      median frame interval (20 at 20 Hz for 1 s);
    * ``--max-frame-gap-s``: 1.5x that pair interval, so a dropped frame's
      doubled interval is refused rather than matched outside the window;
    * ``--warmup``: the ticks before a window's first pair can arrive (pair
      interval + deployment latency) plus 5 - earlier ticks are blind;
    * ``--correlation-radius`` 3 and ``--velocity-loss simple``.

    Anything given on the command line is left alone. Mutates ``args`` - so
    the checkpoint, the fingerprint and the evaluator all see the resolved
    values - and returns one log line per value it chose.
    """

    planar = getattr(args, "frontend", "mamba_correlation") == "planar"
    chosen: List[str] = []
    usable_clock = frame_interval_s > 0 and math.isfinite(frame_interval_s)
    if args.frame_gap is None:
        if planar and usable_clock:
            args.frame_gap = max(1, int(round(args.planar_baseline_s / frame_interval_s)))
            chosen.append(
                f"--frame-gap {args.frame_gap} ({args.frame_gap * frame_interval_s:.2f} s "
                f"at the capture's {1.0 / frame_interval_s:.1f} Hz)"
            )
        else:
            args.frame_gap = 1
    if args.max_frame_gap_s is None and planar and usable_clock:
        args.max_frame_gap_s = round(1.5 * args.frame_gap * frame_interval_s, 3)
        chosen.append(f"--max-frame-gap-s {args.max_frame_gap_s:g}")
    args.pair_stride = int(args.frame_gap) if getattr(args, "output_on_pairs", False) else 1
    if args.pair_stride > 1:
        chosen.append(
            f"--output-on-pairs: one pair and one output every {args.pair_stride} frames"
            + (f" ({args.pair_stride * frame_interval_s:.2f} s)" if usable_clock else "")
        )
    if args.warmup is None:
        if planar and usable_clock and tick_interval_s > 0:
            # Non-overlapping pairs can leave a window waiting up to one more
            # pair interval for its first one. The epsilon keeps
            # 135.00000000000003 ticks from rounding up to 136.
            blind = math.ceil(
                (
                    args.deployment_latency_s
                    + (args.frame_gap + args.pair_stride - 1) * frame_interval_s
                ) / tick_interval_s
                - 1e-6
            )
            args.warmup = int(blind) + 5
            chosen.append(f"--warmup {args.warmup} (first pair arrives after ~{blind} ticks)")
        else:
            args.warmup = 20
    if planar and (args.coarse_factor is None or args.coarse_radius is None):
        # The coarse window must fit inside the pooled feature map; pick the
        # largest settings (up to 4x pooling, radius 6) that do, so a small
        # --image-size works instead of failing at start-up.
        short_side = min(int(args.image_size[0]), int(args.image_size[1])) // int(args.patch_size)
        wanted = args.coarse_radius if args.coarse_radius is not None else 6
        factor = args.coarse_factor
        if factor is None:
            factor = 4 if short_side // 4 >= 2 * wanted + 1 else 2
        radius = args.coarse_radius
        if radius is None:
            radius = max(1, min(6, (short_side // factor - 1) // 2))
        if args.coarse_factor is None or args.coarse_radius is None:
            chosen.append(f"--coarse-factor {factor} --coarse-radius {radius} (feature map short side {short_side} cells)")
        args.coarse_factor, args.coarse_radius = int(factor), int(radius)
    if args.correlation_radius is None:
        args.correlation_radius = 3 if planar else 4
        if planar:
            chosen.append("--correlation-radius 3")
    if args.velocity_loss is None:
        args.velocity_loss = "simple" if planar else "nll"
        if planar:
            chosen.append("--velocity-loss simple")
    return chosen


def auto_camera_mounting(
    args: argparse.Namespace,
    world: "Distributed",
    calibration,
    root: Path,
    image_time_offset,
    frame_interval_s: float,
) -> List[List[float]]:
    """Measure the camera mounting from the training images at start-up.

    The same measurement as ``tools/estimate_camera_mounting.py`` (its
    reference velocity is read only as a yardstick, never as a model input).
    Deterministic, so every rank - and a resume - arrives at the same matrix.
    """

    from tools.estimate_camera_mounting import estimate_mounting
    from vio.data.attitude import load_attitude_altitude

    csv_path = root / args.csv_name
    attitude = load_attitude_altitude(
        csv_path, time_column=args.time_column, time_scale=args.time_scale,
        attitude_columns=args.attitude_columns, altitude_column=args.altitude_column,
        allow_reference_attitude=args.allow_reference_attitude,
    )
    times, velocity_body = reference_body_frame(
        csv_path, time_column=args.time_column, time_scale=args.time_scale
    )[:2]
    # About 0.3 s between the two frames: enough motion to measure, little
    # enough rotation that most pairs qualify.
    gap = max(1, int(round(0.3 / frame_interval_s))) if frame_interval_s > 0 else 6
    world.log(f"camera mounting: measuring it from {root} (frame gap {gap}) ...")
    report = estimate_mounting(
        root,
        attitude=attitude,
        times=times,
        velocity_body=velocity_body,
        calibration=calibration,
        image_size=args.image_size,
        image_folder=args.image_folder,
        image_time_offset=image_time_offset,
        frame_gap=gap,
        pairs=200,
    )
    world.log(
        f"camera mounting: {report['best_mounting']}, misalignment "
        f"{report['yaw_misalignment_deg']:+.2f} deg, measured/predicted motion "
        f"{report['scale_ratio_median']:.3f} over {report['pairs_used']} pairs"
    )
    if abs(float(report["scale_ratio_median"]) - 1.0) > 0.05:
        world.log(
            "  WARNING measured and predicted ground motion disagree by more than "
            "5%: check that the altitude column is height above the ground, the "
            "focal length, and --image-time-offset before trusting this run"
        )
    return [[float(v) for v in row] for row in report["camera_from_body"]]


def resolve_camera_mounting(args: argparse.Namespace) -> Optional[List[List[float]]]:
    """``--camera-mounting``, else the calibration's mounting, else None.

    ``auto`` also returns None: the caller measures it (it needs the data).
    """

    if getattr(args, "camera_mounting", None) == "auto":
        return None
    if getattr(args, "camera_mounting", None):
        return parse_camera_mounting(args.camera_mounting)
    if args.calibration is not None:
        mounting = json.loads(
            Path(args.calibration).read_text(encoding="utf-8")
        ).get("mounting", {})
        recorded = mounting.get("camera_from_body")
        if recorded is not None:
            return parse_camera_mounting(recorded)
    return None


def resume_fingerprint(
    args: argparse.Namespace,
    source,
    normalizer: VONormalizer,
    ranges: Mapping[str, Union[Tuple[int, int], Sequence[Tuple[int, int]]]],
    *,
    frontend_id: Optional[str] = None,
    temporal_input_id: Optional[str] = None,
    planar: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Everything a resumed run must still agree with the original about.

    Not a hash: the parts are stored by name so a refusal can say WHICH one
    moved. The three groups are the data (a different flight, a different
    clock, a different split), the input contract (what the model is being
    shown), and the model's shape (which would fail to load anyway, but with a
    much worse message).

    Deliberately absent: --epochs, --learning-rate, --batch-size and the rest
    of the optimisation knobs. Changing those on a resume is a legitimate
    thing to do and is reported rather than refused.

    ``frontend_id`` / ``temporal_input_id`` are the BUILT modules' ids (the
    planar frontend and the geometric-residual fusion have their own);
    omitted, the original frontend's and fusion's are assumed. ``planar`` is
    the resolved flat-ground geometry (mounting, prior, search settings) or
    None for the original frontend.
    """

    camera: Dict[str, object] = {}
    if args.calibration is not None:
        camera = json.loads(
            Path(args.calibration).read_text(encoding="utf-8")
        ).get("camera", {})
    return {
        "data": {
            "flight_csv": str(Path(args.dataset).resolve() / args.csv_name),
            "ticks": int(source.times_s.size),
            "first_time_s": float(source.times_s[0]),
            "last_time_s": float(source.times_s[-1]),
            "ranges": {
                name: [list(pair) for pair in normalize_index_ranges(span)]
                for name, span in sorted(ranges.items())
            },
        },
        "contract": {
            "attitude_columns": list(source.attitude_columns),
            "altitude_column": source.altitude_column,
            "image_size": [int(value) for value in args.image_size],
            "frame_gap": int(args.frame_gap),
            # Which pairs exist and which ticks are scored.
            "pair_stride": int(getattr(args, "pair_stride", 1)),
            "output_on_pairs": bool(getattr(args, "output_on_pairs", False)),
            "deployment_latency_s": float(args.deployment_latency_s),
            # Alignment settings belong in the checkpoint: scoring a run under a
            # different clock offset or gap rule silently measures a different
            # thing than training did.
            "max_frame_gap_s": (
                None if args.max_frame_gap_s is None else float(args.max_frame_gap_s)
            ),
            "image_time_offset_s": resolve_image_time_offset(args),
            # The lever arm changes the supervision target itself, so a resume
            # that moves it is continuing one run's optimizer on another run's
            # problem.
            "lever_arm_m": resolve_lever_arm(args),
            # Lens distortion changes the images themselves - VisualPairSource
            # rectifies with it before the frontend ever sees a frame - so a
            # resume that moves it is training on visibly different pixels. A
            # top-level key rather than nested in "camera" so an older
            # checkpoint, which predates this being wired in at all, is judged
            # against FINGERPRINT_DEFAULTS' "no correction" default instead of
            # failing on the CAMERA dict gaining a key it never had an opinion
            # about.
            "distortion": [float(v) for v in camera.get("distortion", [])],
            "window_length": int(args.window_length),
            "stride": int(args.stride),
            "warmup": int(args.warmup),
            "max_visual_events": int(args.max_visual_events),
            "disable_visual_input": bool(args.disable_visual_input),
            # Same tensor shapes either way (zeroed, not omitted - see
            # VOStep._encode_visual/_maybe_ablate_body_rate) so a shape
            # mismatch at load_state_dict would NOT catch a resume that
            # silently flips one of these; the fingerprint has to.
            "ablate_body_rate": bool(args.ablate_body_rate),
            "ablate_visual_age": bool(args.ablate_visual_age),
            "normalizer": normalizer.as_dict(),
            # Grayscale and RGB frames are different pixels AND a different
            # stem shape; a resume across them would fail to load anyway, but
            # the refusal should say why.
            "color": bool(getattr(args, "color", False)),
            "camera": {
                key: camera.get(key)
                for key in ("fx", "fy", "cx", "cy", "width", "height",
                            "images_rectified")
            },
        },
        "model": {
            "visual_dim": int(args.visual_dim),
            "stem_dim": int(args.stem_dim),
            "stem_depth": int(args.stem_depth),
            "patch_size": int(args.patch_size),
            "context_grid": [int(value) for value in args.context_grid],
            "token_grid": int(args.token_grid),
            "correlation_radius": int(args.correlation_radius),
            "aiding_dim": int(args.aiding_dim),
            "fusion_dim": int(args.fusion_dim),
            # The reliability gate. Same tensor shapes whatever these are set
            # to, and frontend_id does not move with them, so nothing else
            # would notice a resume that changed them - yet they decide which
            # correlation cells count and whether a pair is delivered at all,
            # which is exactly the "agree on every dimension while measuring
            # motion differently" case the comment below describes. Present in
            # FINGERPRINT_DEFAULTS at their off-values, because a checkpoint
            # predating the gate provably ran ungated - unlike frontend_id,
            # whose absence proves nothing.
            "min_pool_weight": float(args.min_pool_weight),
            "max_cell_entropy": float(args.max_cell_entropy),
            "min_cell_confidence": float(args.min_cell_confidence),
            "min_score_margin": float(args.min_score_margin),
            "reject_boundary_peaks": bool(args.reject_boundary_peaks),
            "min_reliable_cell_fraction": float(args.min_reliable_cell_fraction),
            # The frontend ALGORITHM, not its shape. Every other key here is a
            # dimension, and two runs can agree on all of them while measuring
            # motion differently. Deliberately absent from FINGERPRINT_DEFAULTS:
            # a checkpoint predating this key cannot be shown to have used the
            # current algorithm, so it is refused rather than assumed.
            "frontend_id": str(frontend_id or VisionMambaFlowFrontend.frontend_id),
            # Changes the SHAPE of rotation.map (3x3 against 2x3) as well as
            # the geometry, so a mismatch is a hard load failure rather than a
            # silent one - but naming it here makes the refusal say why.
            "rotation_mode": str(args.rotation_mode),
            # The FUSION input contract, the same role frontend_id plays for
            # the frontend: AIDING_INPUT_DIM changing (6 -> 9) already fails
            # load_state_dict on its own, but that only catches a SHAPE
            # change. This catches an evaluator built against a different
            # aiding-vector/fusion-input semantics that happens to keep the
            # same shape - deliberately absent from FINGERPRINT_DEFAULTS, for
            # the same reason frontend_id is: a checkpoint predating this key
            # cannot be shown to have used the current contract.
            "temporal_input_id": str(temporal_input_id or VisionMambaVO.temporal_input_id),
            # Which frontend and how its output becomes a velocity. Both change
            # what the weights mean, and the planar geometry below changes
            # what the frontend measures without changing any tensor shape.
            "frontend": str(getattr(args, "frontend", "mamba_correlation")),
            "velocity_mode": resolve_velocity_mode(args),
            "planar": None if planar is None else dict(planar),
        },
    }


#: Fingerprint keys added after the format was already in use, with the value
#: their absence implies. A checkpoint written before a key existed carries no
#: entry for it, and refusing to resume over a key whose default is "apply no
#: correction" would be wrong: nothing about the problem moved. A run that sets
#: one of these to a NON-default value still differs from such a checkpoint, and
#: is still refused, which is the case that matters.
FINGERPRINT_DEFAULTS: Dict[str, object] = {
    "max_frame_gap_s": None,
    "image_time_offset_s": 0.0,
    "lever_arm_m": None,
    # Absence means the run predates distortion being wired through at all, in
    # which case none was ever applied - the same "no correction" meaning as
    # every other entry here.
    "distortion": [],
    # Unlike frontend_id, absence here IS informative: the field mode did not
    # exist when such a checkpoint was written, so it can only have been the
    # constant. In practice frontend_id refuses those checkpoints first; this
    # entry exists so the refusal names the real difference instead of
    # reporting a missing key.
    "rotation_mode": "constant",
    # Both ablations postdate this key existing at all, so absence means
    # neither was ever applied - the same "no correction" meaning as
    # max_frame_gap_s/lever_arm_m/distortion above.
    "ablate_body_rate": False,
    "ablate_visual_age": False,
    # The reliability gate postdates every checkpoint written before it, and
    # its off-values ARE what those runs did - 1e-4 was the hardcoded
    # min_pool_weight and no other filtering existed. So absence is
    # informative here, and a pre-gate checkpoint keeps resuming cleanly
    # instead of being refused for a key it could not have carried.
    "min_pool_weight": 1e-4,
    "max_cell_entropy": 1.0,
    "min_cell_confidence": 0.0,
    "min_score_margin": 0.0,
    "reject_boundary_peaks": False,
    "min_reliable_cell_fraction": 0.0,
    # The planar frontend, RGB input and the geometric output all postdate
    # every earlier checkpoint, which therefore ran the original frontend on
    # grayscale frames with the factored heads.
    "color": False,
    "pair_stride": 1,
    "output_on_pairs": False,
    "frontend": "mamba_correlation",
    "velocity_mode": "heads",
    "planar": None,
}


def fingerprint_differences(
    saved: Optional[Mapping[str, Any]], current: Mapping[str, Mapping[str, Any]]
) -> list:
    """Human-readable ``group.key`` differences, deepest key first."""

    if not saved:
        return ["the checkpoint carries no fingerprint (written by an older run)"]
    differences = []
    for group, values in current.items():
        before = (saved or {}).get(group, {})
        if not isinstance(before, dict):
            differences.append(f"{group}: missing from the checkpoint")
            continue
        for key, value in values.items():
            if key in before:
                was = before[key]
            elif key in FINGERPRINT_DEFAULTS:
                was = FINGERPRINT_DEFAULTS[key]
            else:
                was = "<absent>"
            if was != value:
                differences.append(f"{group}.{key}: {was!r} -> {value!r}")
    return differences


#: Per-epoch checkpoints, written by --save-every.
EPOCH_CHECKPOINT_GLOB = "epoch_*.pt"


def epoch_checkpoints(directory: Path) -> list:
    """The per-epoch checkpoints in ``directory``, oldest epoch first."""

    found = []
    for path in directory.glob(EPOCH_CHECKPOINT_GLOB):
        stem = path.stem.split("_", 1)[-1]
        if stem.isdigit():
            found.append((int(stem), path))
    return [path for _, path in sorted(found)]


def prune_epoch_checkpoints(directory: Path, keep_last: int) -> None:
    """Keep the ``keep_last`` newest per-epoch checkpoints, delete the rest.

    Deliberately narrow: only files this trainer wrote, matching
    ``epoch_<digits>.pt``, inside the run's own ``epochs/`` directory. best.pt
    and last.pt live one level up and are never candidates. ``keep_last`` of 0
    deletes nothing at all, which is the default.
    """

    if keep_last <= 0:
        return
    existing = epoch_checkpoints(directory)
    for path in existing[: max(len(existing) - keep_last, 0)]:
        path.unlink(missing_ok=True)


def announce_epoch_checkpoint_cost(
    written: Path, args: argparse.Namespace, world: "Distributed"
) -> None:
    """Say what --save-every will cost on disk, once, at the first save.

    A default run is 3000 epochs and a checkpoint carries optimizer state as
    well as weights. Discovering that at epoch 2000, out of disk, is worse than
    being told at epoch 1.
    """

    size = written.stat().st_size
    saves = max(args.epochs // max(args.save_every, 1), 1)
    if args.keep_last > 0:
        saves = min(saves, args.keep_last)
    total = size * saves / 1e9
    world.log(
        f"  per-epoch checkpoints: {size / 1e6:.1f} MB each, "
        f"{saves} retained -> {total:.1f} GB"
        + ("" if args.keep_last > 0 else "  (cap it with --keep-last)")
    )
    if total > 50.0 and args.keep_last <= 0:
        world.log(
            f"  WARNING that is {total:.0f} GB of checkpoints. --keep-last N "
            f"keeps only the N newest,\n"
            f"          or --save-every N writes one every N epochs. Neither "
            f"touches best.pt/last.pt."
        )


# ---------------------------------------------------------------------------
# loss and metrics
# ---------------------------------------------------------------------------


#: An axis whose mean-predictor RMSE is below this carries too little
#: variance for a skill score to mean anything.
SKILL_BASELINE_FLOOR_M_S = 0.25

#: The metrics.csv column stems evaluate() produces, shared by val_* and
#: (when --eval-train-split is set) traineval_*, so the two are directly
#: comparable column for column.
TRAIN_EVAL_METRIC_SUFFIXES: Tuple[str, ...] = (
    "loss", "cov_loss",
    "vel_rmse", "vel_max_error",
    "vel_rmse_x", "vel_rmse_y", "vel_rmse_z",
    "vel_dir_rmse", "vel_dir_max_error",
    "baseline_vel_rmse", "baseline_vel_rmse_y", "baseline_vel_dir_rmse",
    "skill_vs_mean_x", "skill_vs_mean_y", "skill_vs_mean_z",
    # How many image pairs the reliability gate delivered on THIS split. Worth
    # a column of its own per split rather than one global number: the same
    # thresholds meet different imagery in train and validation, and a
    # validation split that quietly receives half the visual measurements
    # would otherwise show up only as a worse velocity error.
    "visual_kept_fraction",
)


def simple_velocity_loss(
    prediction: Dict[str, torch.Tensor],
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    direction_weight: float,
    huber_delta: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Deterministic velocity regression: Smooth L1 plus a fixed cosine term.

    The learned-uncertainty version below can lower its loss two ways - by
    predicting velocity better, or by predicting its own confidence better -
    and those are not the same thing. On a run that is memorising, the second
    route is much the cheaper one: the heads reached kappa ~ 1100 (a claimed
    1.7 deg of heading spread) while real validation heading error was 7.5 deg,
    so training loss fell while validation RMSE did not move at all.

    This loss removes that route entirely. There is no variance and no
    concentration - only "how far is the predicted velocity from the measured
    one", which is the quantity ``vel_rmse`` reports. Training loss and the
    selection metric then move together, and a flat validation curve can no
    longer be hidden by a confident one.

    Smooth L1 rather than plain MSE because the labels have outliers - a GPS
    velocity glitch or a mis-timed frame produces a residual whose square
    would dominate the batch. Beyond ``huber_delta`` metres per second the
    penalty grows linearly, so a bad label costs a bounded amount.

    The direction term keeps a FIXED weight. It is the same speed-weighted
    ``1 - cos`` the NLL version reduces to at kappa = 1, so the two losses
    agree about what a good heading is and differ only in whether the model
    may argue about how sure it is.
    """

    weights = mask.unsqueeze(-1)
    total = weights.sum().clamp_min(1.0)
    residual = prediction["predicted_velocity"] - target
    per_component = F.smooth_l1_loss(
        residual, torch.zeros_like(residual), reduction="none", beta=huber_delta
    )
    velocity_term = (per_component * weights).sum() / (total * 3.0)

    speed = target.norm(dim=-1, keepdim=True).clamp_min(1e-3)
    unit_target = target / speed
    cosine = (prediction["predicted_direction"] * unit_target).sum(dim=-1)
    # Same speed weighting as the NLL version: the direction of a
    # near-stationary sample is noise, whatever the loss around it looks like.
    speed_weight = (speed.squeeze(-1) / speed.mean().clamp_min(1e-6)) * mask
    direction = ((1.0 - cosine) * speed_weight).sum() / speed_weight.sum().clamp_min(1.0)

    loss = velocity_term + direction_weight * direction
    return loss, {
        # Reported under the same key the NLL version uses so metrics.csv keeps
        # one column meaning "the velocity half of the loss" in both modes.
        "nll": float(velocity_term.detach()),
        "direction": float(direction.detach()),
        # No learned concentration in this mode. NaN rather than 0.0, because
        # zero is a real kappa and would read as "the model is maximally
        # unsure" instead of "this run had no such head in the loss".
        "direction_concentration": float("nan"),
    }


def velocity_loss(
    prediction: Dict[str, torch.Tensor],
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    direction_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Gaussian NLL on velocity, plus a von Mises-Fisher direction term.

    The direction term exists because the camera observes direction directly
    and speed only through altitude. Without it the gradient is dominated by
    the forward axis, where the scale error lives, and the lateral component -
    which is what the images actually carry - contributes almost nothing.

    Direction lives on a sphere, so its uncertainty is a von Mises-Fisher
    concentration rather than a variance. For S2 at usable concentration the
    negative log-likelihood reduces to

        kappa * (1 - cos) - log kappa

    which is the same shape as the Gaussian NLL above: a residual scaled by a
    learned precision, plus the log term that stops the precision collapsing
    to nothing to make the first term cheap. At kappa = 1, where the head is
    initialised, this is exactly the fixed (1 - cos) term it replaces, so
    enabling it does not move the starting point.

    ``direction_weight`` is kept as an outer multiplier. It no longer sets how
    much the model trusts its own heading - kappa does that, per tick - but it
    still sets how much the direction channel matters against speed, which is
    a modelling choice rather than something to be learned away.
    """

    weights = mask.unsqueeze(-1)
    total = weights.sum().clamp_min(1.0)
    velocity = prediction["predicted_velocity"]
    log_variance = prediction["velocity_log_variance"]
    residual = velocity - target
    nll = 0.5 * (residual.pow(2) * torch.exp(-log_variance) + log_variance)
    nll = (nll * weights).sum() / (total * 3.0)

    speed = target.norm(dim=-1, keepdim=True).clamp_min(1e-3)
    unit_target = target / speed
    cosine = (prediction["predicted_direction"] * unit_target).sum(dim=-1)
    # Weight by speed: the direction of a near-stationary sample is noise, and
    # letting it contribute equally would teach the head to fit that noise.
    speed_weight = (speed.squeeze(-1) / speed.mean().clamp_min(1e-6)) * mask

    log_concentration = prediction["direction_log_concentration"]
    concentration = torch.exp(log_concentration)
    # The speed weighting stays. It answers a different question from kappa:
    # kappa is how sure the model is, the weight is how much the LABEL is worth
    # believing, and the direction of a near-stationary sample is noise however
    # confident the head is about it.
    per_tick = concentration * (1.0 - cosine) - log_concentration
    direction = (per_tick * speed_weight).sum() / speed_weight.sum().clamp_min(1.0)

    loss = nll + direction_weight * direction
    return loss, {
        "nll": float(nll.detach()),
        "direction": float(direction.detach()),
        "direction_concentration": float(
            (concentration * speed_weight).sum().detach()
            / speed_weight.sum().clamp_min(1.0).detach()
        ),
    }


def compute_velocity_loss(
    prediction: Dict[str, torch.Tensor],
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    direction_weight: float,
    loss_mode: str,
    huber_delta: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Pick the loss by name, in one place.

    Training and validation both come through here, so a run cannot end up
    optimising one objective and being scored on another - the failure that
    makes a train/validation gap impossible to interpret.
    """

    if loss_mode == "simple":
        return simple_velocity_loss(
            prediction, target, mask,
            direction_weight=direction_weight, huber_delta=huber_delta,
        )
    if loss_mode == "nll":
        return velocity_loss(
            prediction, target, mask, direction_weight=direction_weight
        )
    raise ValueError(f"unknown loss mode {loss_mode!r}")


@torch.no_grad()
def evaluate(
    step: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    baseline: np.ndarray,
    direction_weight: float,
    loss_mode: str = "nll",
    huber_delta: float = 1.0,
    progress_description: Optional[str] = None,
    show_progress: bool = True,
    world: Optional[Distributed] = None,
) -> Dict[str, float]:
    """Held-out metrics, using the same accumulator as the VIO trainer.

    ``RunningVelocityStats`` is shared deliberately: ``vel_rmse`` there is the
    RMS of the error MAGNITUDE ``||v_pred - v_true||``, not the per-component
    RMS, and the two differ by a factor of sqrt(3). Computing it independently
    here would produce a column that looks comparable with a VIO run and is
    not.

    The mean-predictor baseline goes through a second accumulator rather than
    a separate formula, so the skill numbers compare like with like.
    """

    step.eval()
    stats = RunningVelocityStats()
    reference_stats = RunningVelocityStats()
    # loss sum, covariance-term sum, batch count, pairs offered, pairs delivered
    tallies = torch.zeros(5, dtype=torch.float64, device=device)
    batches = tqdm(
        loader,
        desc=progress_description or "validation",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
        disable=not show_progress,
    )
    for batch in batches:
        prediction, target, mask = forward_batch(step, batch, device)
        loss, parts = compute_velocity_loss(
            prediction, target, mask, direction_weight=direction_weight,
            loss_mode=loss_mode, huber_delta=huber_delta,
        )
        tallies[0] += float(loss)
        tallies[1] += parts["nll"]
        tallies[2] += 1.0
        # Same accounting as the training loop. Validation can receive far
        # fewer visual measurements than training - different imagery, same
        # thresholds - and without this the cause would be invisible: the
        # velocity error would simply be worse, with nothing to say the model
        # was flying half blind through the split.
        inner = step.module if hasattr(step, "module") else step
        tallies[3] += getattr(inner, "_pairs_offered", 0.0)
        tallies[4] += getattr(inner, "_pairs_delivered", 0.0)
        stats.update(
            masked_velocity_stats(prediction["predicted_velocity"], target, mask)
        )
        constant = torch.from_numpy(baseline).to(device, dtype=target.dtype)
        reference_stats.update(
            masked_velocity_stats(
                constant.reshape(1, 1, 3).expand_as(target), target, mask
            )
        )

    if world is not None:
        world.reduce_sum(tallies)
        world.reduce_stats(stats)
        world.reduce_stats(reference_stats)

    metrics = dict(stats.metrics())
    reference = reference_stats.metrics()
    count = max(float(tallies[2]), 1.0)
    metrics["loss"] = float(tallies[0]) / count
    metrics["cov_loss"] = float(tallies[1]) / count
    offered = float(tallies[3])
    metrics["visual_kept_fraction"] = (
        float(tallies[4]) / offered if offered > 0 else float("nan")
    )
    metrics["baseline_vel_rmse"] = reference["vel_rmse"]
    metrics["baseline_vel_rmse_y"] = reference["vel_rmse_y"]
    metrics["baseline_vel_dir_rmse"] = reference["vel_dir_rmse"]
    for axis in ("x", "y", "z"):
        model_rmse = metrics[f"vel_rmse_{axis}"]
        floor = reference[f"vel_rmse_{axis}"]
        # Skill needs an axis with something to beat. On a still-air flight the
        # lateral baseline RMSE is a few centimetres per second, and dividing
        # by it turns an excellent model into a skill of -37000 - a number that
        # reads as a bug and buries the axes where skill is meaningful. Below
        # the floor the axis is reported as undefined rather than as noise.
        metrics[f"skill_vs_mean_{axis}"] = (
            float(1.0 - model_rmse / floor)
            if floor >= SKILL_BASELINE_FLOOR_M_S
            else float("nan")
        )
    return metrics


#: metrics.csv prefix per split, matching the existing ``val_`` convention.
HORIZON_COLUMN_PREFIX = {"train": "train_", "validation": "val_", "test": "test_"}


@torch.no_grad()
def run_horizon_pass(
    model: VisionMambaVO,
    frontend: VisionMambaFlowFrontend,
    args: argparse.Namespace,
    *,
    span: Tuple[int, int],
    horizons: Sequence[float],
    dataset: FixedWingVODataset,
    times_s: np.ndarray,
    body_rate_rad_s: np.ndarray,
    device: torch.device,
    camera_matrix: Optional[torch.Tensor],
    baseline: np.ndarray,
    rotation_body_to_ned: Optional[np.ndarray],
    prefix: str,
) -> Dict[str, float]:
    """Stream the split once, start to end, and score each horizon prefix.

    The windowed validation above measures six seconds from a cold start. This
    measures what happens after ten minutes of running, which is a different
    question and can have a very different answer: a state that drifts is
    invisible in a 600-tick window by construction.

    The state is zeroed once at the split's first tick and never reset, and a
    horizon is the FIRST H minutes of that run. The frontend runs once over
    the span and so does the scan, so the cost is one pass however many
    horizons are asked for.
    """

    tokens = encode_span_tokens(
        frontend,
        dataset.image_source,
        span=span,
        body_rate_rad_s=body_rate_rad_s,
        times_s=times_s,
        visual_dim=args.visual_dim,
        device=device,
        camera_matrix=camera_matrix,
        batch_pairs=args.horizon_batch_pairs,
        num_workers=args.num_workers,
        disable_visual=args.disable_visual_input,
        progress=not args.no_progress,
        attitude=dataset.attitude,
    )
    results = stream_horizon_metrics(
        model,
        aiding=dataset.aiding,
        log_altitude=dataset.log_altitude,
        target_velocity=dataset.velocity_body,
        times_s=times_s,
        span=span,
        tokens=tokens,
        deployment_latency_s=args.deployment_latency_s,
        horizons_minutes=horizons,
        device=device,
        warmup_ticks=args.warmup,
        block_ticks=args.horizon_block_ticks,
        baseline=baseline,
        rotation_body_to_ned=rotation_body_to_ned,
        progress=not args.no_progress,
        ablate_body_rate=args.ablate_body_rate,
        ablate_visual_age=args.ablate_visual_age,
        output_on_pairs=args.output_on_pairs,
    )
    print(f"horizons on {args.horizon_split} ({len(tokens)} visual events)")
    print(format_horizon_table(results))
    # The column set is fixed by whether a rotation was supplied, not by
    # whether any horizon happened to fit - otherwise a split too short for
    # every horizon would drop the position columns out of a row whose header
    # keeps them.
    return horizon_csv_row(
        results, prefix=prefix, position=rotation_body_to_ned is not None
    )


def _correlation_candidate_gib(
    pairs: int, channels: int, candidates: int, height: int, width: int
) -> float:
    """Estimated peak size, in GiB, of ONE frontend call's largest internal
    tensor: LocalCorrelation's (pairs, channels, candidates, height, width)
    all-candidates-at-once float32 buffer (see
    ``LocalCorrelation._all_grid_samples``/``_all_integer_shifts``). This is
    what ``chunk`` in :func:`encode_pairs` bounds ``pairs`` to - at the
    trainer's real defaults (64 channels, 81 candidates, 72x128) an unchunked
    480-pair validation batch (4 windows x 120 events) materialises this at
    just over 85 GiB, on top of everything else the frontend and fusion
    stack hold; a chunk of 8 is 1.4 GiB.
    """

    return (pairs * channels * candidates * height * width * 4) / (1024**3)


def encode_pairs(
    frontend: VisionMambaFlowFrontend,
    image0: torch.Tensor,
    image1: torch.Tensor,
    pair_dt_s: torch.Tensor,
    body_rate_rad_s: torch.Tensor,
    *,
    camera_matrix: Optional[torch.Tensor] = None,
    chunk: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode every pair in a batch, optionally recomputing in the backward.

    Returns ``(tokens, qualities, reliable)``. ``reliable`` is the frontend's
    per-pair verdict on whether enough of the correlation grid survived the
    reliability gate to be worth delivering at all - see
    :meth:`VisionMambaFlowFrontend._reliable_cells`. It is carried alongside
    rather than folded into the token because a zeroed token and an absent one
    mean opposite things to the fusion model.

    Without checkpointing the activation memory is LINEAR in the number of
    pairs, and that is what caps resolution: measured at 576x1024 with an 8-pixel
    patch, 2.24 GiB per pair, so eight pairs is 17.9 GiB and a full 20 Hz window
    is out of reach on any card.

    Checkpointing makes it flat instead - only the chunk being recomputed holds
    activations - at the cost of one extra forward pass. Measured on the same
    configuration: 4.42 / 8.95 / 17.87 GiB for 2 / 4 / 8 pairs plain, against a
    constant 4.45 GiB checkpointed. That turns the pair count into a free
    parameter and the image size into the only thing worth spending memory on,
    which is the right way round: correspondence is limited by resolution, not
    by how many pairs are held in the graph at once.

    ``chunk`` trades the two off. Larger chunks recompute less often and use
    more memory; 8 is a reasonable default on an 80 GiB card, 1-2 on a small
    one. Zero disables chunking entirely - the whole batch goes through the
    frontend, and its correlator's all-candidates-at-once tensor, in one call.

    Chunking is honoured WHETHER OR NOT gradients are enabled. Only the
    ``checkpoint()`` recompute-in-backward wrapper is training-only: under
    ``torch.no_grad()`` (``evaluate()``, ``run_horizon_pass()``) there is no
    backward to recompute for, so each chunk calls the frontend directly. The
    memory ``chunk`` bounds - the correlator's (pairs, channels, candidates,
    height, width) tensor, see :func:`_correlation_candidate_gib` - is exactly
    as real during a no_grad validation pass as during training; a version of
    this function that only chunked under an active autograd graph would let a
    validation batch's full pair count reach the correlator at once, which at
    the trainer's real defaults is tens of gigabytes for a single tensor.

    Every image is converted from its stored dtype (uint8) to float and
    normalised to [0, 1] INSIDE this function, one chunk at a time - never for
    the whole batch at once - so that conversion cannot itself be the largest
    live tensor when chunking was supposed to bound memory.
    """

    encoded = encode_pair_batch(
        frontend, image0, image1, pair_dt_s, body_rate_rad_s,
        camera_matrix=camera_matrix, chunk=chunk,
    )
    return encoded["visual_token"], encoded["visual_quality"], encoded["pair_reliable"]


#: Per-pair frontend outputs :func:`encode_pair_batch` collects, beyond the
#: token/quality/verdict every frontend returns. Only the flat-ground frontend
#: produces them.
GEOMETRIC_OUTPUTS = ("geometric_velocity", "geometric_log_variance", "geometric_valid")

#: Per-pair inputs the flat-ground frontend reads, in the order they are
#: chunked. Keyword names of PlanarFlowFrontend.forward.
PAIR_GEOMETRY_INPUTS = ("relative_rotation", "down_body", "altitude_m")


def photometric_parameters(
    pairs: int, channels: int, strength: float, *, device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Random per-pair exposure changes, ``(pairs, 2, channels + 2)``.

    For each of the two images: a per-channel gain (brightness and, on RGB, a
    white-balance shift), an additive offset, and a gamma. The two images
    share most of it - a scene lit differently along the flight - and differ
    by a smaller amount, which is what an auto-exposure camera does between
    two frames a second apart. ``strength`` is the standard deviation of the
    shared log-gain; everything else is scaled from it. Drawn OUTSIDE the
    checkpointed frontend call and handed in as a tensor, so the recompute in
    the backward pass applies exactly the same change as the forward did.
    """

    def normal(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, device=device)

    shared_gain = normal(pairs, 1, 1) * strength
    colour = normal(pairs, 1, channels) * (0.25 * strength if channels > 1 else 0.0)
    per_image = normal(pairs, 2, 1) * (0.5 * strength)
    gain = torch.exp(shared_gain + colour + per_image)
    offset = normal(pairs, 2, 1) * (0.1 * strength)
    gamma = torch.exp(normal(pairs, 1, 1) * (0.5 * strength) + normal(pairs, 2, 1) * (0.2 * strength))
    return torch.cat((gain, offset, gamma), dim=-1)


def apply_photometric(image: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    """``image`` ``(N, C, H, W)`` in [0, 1], ``parameters`` ``(N, C + 2)``."""

    channels = image.shape[1]
    gain = parameters[:, :channels].view(-1, channels, 1, 1)
    offset = parameters[:, channels:channels + 1].view(-1, 1, 1, 1)
    gamma = parameters[:, channels + 1:channels + 2].view(-1, 1, 1, 1)
    return (image.clamp_min(1e-6).pow(gamma) * gain + offset).clamp(0.0, 1.0)


def encode_pair_batch(
    frontend: torch.nn.Module,
    image0: torch.Tensor,
    image1: torch.Tensor,
    pair_dt_s: torch.Tensor,
    body_rate_rad_s: torch.Tensor,
    *,
    camera_matrix: Optional[torch.Tensor] = None,
    chunk: int = 0,
    pair_geometry: Optional[Mapping[str, torch.Tensor]] = None,
    photometric: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """:func:`encode_pairs`, generalised: any per-pair inputs, a dict out.

    ``pair_geometry`` holds :data:`PAIR_GEOMETRY_INPUTS` for the flat-ground
    frontend, one row per pair, and is chunked with the images.
    ``photometric`` is :func:`photometric_parameters`' output, applied to the
    float images inside the chunk. The returned dict always has
    ``visual_token``, ``visual_quality`` and ``pair_reliable``, plus
    :data:`GEOMETRIC_OUTPUTS` when the frontend produces them. Chunking and
    checkpointing behave exactly as :func:`encode_pairs` documents.
    """

    # A frontend that takes pair geometry also returns a metric velocity; the
    # attribute, not the outputs of one call, decides it, so every chunk of
    # every batch returns the same tuple layout.
    geometric = bool(getattr(frontend, "requires_pair_geometry", False))
    geometry_names = tuple(pair_geometry) if pair_geometry is not None else ()
    geometry_values = tuple(pair_geometry[name] for name in geometry_names) if pair_geometry else ()
    with_photometric = photometric is not None

    def encode(first, second, interval, rate, *extra):
        first = first.float().div(255.0)
        second = second.float().div(255.0)
        if with_photometric:
            parameters = extra[-1]
            extra = extra[:-1]
            first = apply_photometric(first, parameters[:, 0])
            second = apply_photometric(second, parameters[:, 1])
        keywords = dict(zip(geometry_names, extra))
        out = frontend(
            first, second, pair_dt_s=interval,
            body_rate_rad_s=rate, camera_matrix=camera_matrix, **keywords,
        )
        results = [out["visual_token"], out["visual_quality"], out["pair_reliable"]]
        if geometric:
            results += [out[name] for name in GEOMETRIC_OUTPUTS]
        return tuple(results)

    names = ["visual_token", "visual_quality", "pair_reliable"]
    inputs = (image0, image1, pair_dt_s, body_rate_rad_s, *geometry_values)
    if with_photometric:
        inputs = inputs + (photometric,)

    if chunk <= 0:
        pieces = [encode(*inputs)]
    else:
        checkpointing = torch.is_grad_enabled()
        pieces = []
        for start in range(0, image0.shape[0], chunk):
            stop = min(start + chunk, image0.shape[0])
            sliced = tuple(value[start:stop] for value in inputs)
            if checkpointing:
                # use_reentrant=False keeps this working with a frontend whose
                # inputs do not all require grad, which is the case here: the
                # images never do. checkpoint() carries no return annotation,
                # so a checker infers it from torch's own branches and lands
                # on "... | None"; it returns exactly what `encode` returned.
                pieces.append(
                    cast(tuple, checkpoint(encode, *sliced, use_reentrant=False))
                )
            else:
                # Nothing will ever call backward() through this, so skip
                # checkpoint()'s save-input/recompute machinery. Chunking still
                # bounds the correlator's peak tensor to this chunk.
                pieces.append(encode(*sliced))
    if geometric:
        names += list(GEOMETRIC_OUTPUTS)
    return {
        name: torch.cat([piece[index] for piece in pieces])
        for index, name in enumerate(names)
    }


class VOStep(torch.nn.Module):
    """The whole step - encode pairs, scatter, fuse - as one module.

    This exists for DDP. The reducer starts an iteration when the wrapped
    module's ``forward`` is entered and expects every participating parameter
    to be reached inside it. Calling the frontend separately and only wrapping
    the fusion model leaves the frontend's gradients outside that bookkeeping,
    which shows up as a hang or a "marked ready twice" error partway through
    the first epoch rather than as anything legible.

    Single-GPU runs go through exactly this path too, so there is no untested
    second branch.

    With the flat-ground frontend (``requires_pair_geometry``) each pair also
    carries its attitude geometry in, and a metric velocity out; that velocity
    is held between pairs (:func:`hold_visual_velocity`) and handed to a
    ``geometric_residual`` model as the base of its prediction.
    ``photometric_augment`` (training only) jitters each pair's exposure before
    the frontend sees it - see :func:`photometric_parameters`.
    """

    #: Declared so a type checker knows what register_buffer put here.
    #: nn.Module.__getattr__ is typed to return "Tensor | Module", so without
    #: this every use of self.camera_matrix reads as possibly a Module. The
    #: annotation carries no assignment, so it does not shadow the buffer -
    #: this is the same pattern torch uses for _BatchNorm.running_mean.
    camera_matrix: Optional[torch.Tensor]

    def __init__(
        self,
        model: VisionMambaVO,
        *,
        window_length: int,
        visual_dim: int,
        disable_visual: bool,
        frontend_chunk: int,
        deployment_latency_s: float,
        camera_matrix: Optional[torch.Tensor] = None,
        ablate_body_rate: bool = False,
        ablate_visual_age: bool = False,
        photometric_augment: float = 0.0,
        output_on_pairs: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        # With --output-on-pairs the velocity is an output only on the tick a
        # pair is delivered; the prediction dict then carries that mask as
        # "output_mask" and forward_batch folds it into the loss mask.
        self.output_on_pairs = bool(output_on_pairs)
        self.window_length = int(window_length)
        self.visual_dim = int(visual_dim)
        self.disable_visual = bool(disable_visual)
        self.frontend_chunk = int(frontend_chunk)
        self.deployment_latency_s = float(deployment_latency_s)
        self.ablate_body_rate = bool(ablate_body_rate)
        self.ablate_visual_age = bool(ablate_visual_age)
        self.photometric_augment = float(photometric_augment)
        frontend = model.frontend
        self.requires_geometry = bool(
            frontend is not None and getattr(frontend, "requires_pair_geometry", False)
        )
        self.geometric_residual = model.velocity_mode == "geometric_residual"
        if self.geometric_residual and not self.requires_geometry and not disable_visual:
            raise ValueError(
                "velocity_mode 'geometric_residual' needs a frontend that produces "
                "a metric velocity (--frontend planar)"
            )
        # Last batch's pair accounting, for logging only. Plain floats rather
        # than buffers: they are a report on the batch that just went through,
        # not state the model depends on, and a buffer would be checkpointed
        # and DDP-synchronised for no reason. The caller reads them right
        # after forward() and accumulates its own epoch totals.
        self._pairs_offered = 0.0
        self._pairs_delivered = 0.0
        # The held-velocity carry from the last forward_stream call, for a
        # TBPTT caller to mask and hand back (see forward_batch_stream).
        self.last_velocity_carry: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        # A buffer, so .to(device) and DDP's device checks handle it and it is
        # never silently left on the CPU while the images are not.
        self.register_buffer(
            "camera_matrix",
            None if camera_matrix is None else camera_matrix.clone(),
            persistent=False,
        )

    def _encode_visual(
        self,
        aiding: torch.Tensor,
        image0: torch.Tensor,
        image1: torch.Tensor,
        pair_dt_s: torch.Tensor,
        body_rate: torch.Tensor,
        offsets: torch.Tensor,
        valid: torch.Tensor,
        times_s: torch.Tensor,
        age_carry: Optional[torch.Tensor] = None,
        geometry: Optional[Mapping[str, torch.Tensor]] = None,
        velocity_carry: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """The frontend + scatter + age (+ held velocity) pipeline shared by
        :meth:`forward` and :meth:`forward_stream` - the two differ only in
        which of ``VisionMambaVO.forward``/``forward_stream`` they hand this to.

        Returns a dict: ``token``, ``quality``, ``present``, ``age``,
        ``age_carry`` and - for a ``geometric_residual`` model -
        ``visual_velocity``, ``visual_velocity_valid`` and ``velocity_carry``.
        ``forward`` discards the carries (a reset every window, by design);
        only :meth:`forward_stream` threads them onward.
        """

        batch_size, events = image0.shape[:2]
        held = held_valid = new_velocity_carry = None
        if self.disable_visual:
            shape = (batch_size, self.window_length, self.visual_dim)
            token = aiding.new_zeros(shape)
            quality = aiding.new_zeros((batch_size, self.window_length, 1))
            present = aiding.new_zeros((batch_size, self.window_length, 1))
            if self.geometric_residual:
                held = aiding.new_zeros((batch_size, self.window_length, 3))
                held_valid = aiding.new_zeros((batch_size, self.window_length, 1))
        else:
            # disable_visual and "the model has a frontend" are separate facts,
            # so this can be None here. Saying so is both what narrows the type
            # and what turns an AttributeError from inside encode_pairs into a
            # sentence naming the actual mistake.
            frontend = self.model.frontend
            if frontend is None:
                raise RuntimeError(
                    "visual input is enabled but the model has no frontend; "
                    "pass one to VisionMambaVO or set disable_visual"
                )
            pair_geometry = None
            if self.requires_geometry:
                if geometry is None:
                    raise RuntimeError(
                        "the planar frontend needs each pair's geometry "
                        "(visual_event_rotation/down/altitude in the batch)"
                    )
                pair_geometry = {
                    "relative_rotation": geometry["rotation"].flatten(0, 1),
                    "down_body": geometry["down"].flatten(0, 1),
                    "altitude_m": geometry["altitude"].flatten(0, 1),
                }
            photometric = None
            if self.training and self.photometric_augment > 0.0:
                photometric = photometric_parameters(
                    batch_size * events, int(image0.shape[2]), self.photometric_augment,
                    device=image0.device,
                )
            encoded = encode_pair_batch(
                frontend,
                # Raw uint8 CHW pairs, deliberately not converted to float
                # here: encode_pair_batch does that per chunk, so the whole
                # batch's images are never live as float at once.
                image0.flatten(0, 1),
                image1.flatten(0, 1),
                pair_dt_s.flatten(0, 1),
                body_rate.flatten(0, 1),
                camera_matrix=self.camera_matrix,
                chunk=self.frontend_chunk,
                pair_geometry=pair_geometry,
                photometric=photometric,
            )
            # A pair the frontend judged unreliable becomes an ABSENT pair,
            # not a zeroed token delivered as though an image had arrived:
            # ``scatter_visual_tokens`` leaves its presence bit at zero, so
            # visual_age keeps growing from the last pair that WAS trusted -
            # exactly "no image arrived", the state the fusion contract
            # already knows how to represent. Re-injecting the previous token
            # instead would deliver one measurement twice and let a stale
            # reading masquerade as a fresh one.
            #
            # Passed as ``delivered`` rather than folded into ``valid`` so the
            # refused pair is still multiplied through the scatter: dropping it
            # from the index set would take the frontend out of the autograd
            # graph entirely on a batch where every pair was refused, which
            # static_graph DDP does not survive. See scatter_visual_tokens.
            reliable = encoded["pair_reliable"].reshape(batch_size, events)
            delivered = valid.to(reliable.dtype) * reliable
            self._pairs_offered = float(valid.sum().detach())
            self._pairs_delivered = float(delivered.sum().detach())
            token, quality, present = scatter_visual_tokens(
                encoded["visual_token"].reshape(batch_size, events, self.visual_dim),
                encoded["visual_quality"].reshape(batch_size, events, 1),
                offsets, valid,
                window_length=self.window_length,
                visual_dim=self.visual_dim,
                delivered=reliable,
            )
            if self.geometric_residual:
                held, held_valid, new_velocity_carry = hold_visual_velocity(
                    encoded["geometric_velocity"].reshape(batch_size, events, 3),
                    offsets, valid,
                    window_length=self.window_length,
                    delivered=reliable,
                    carry=velocity_carry,
                )
        age, new_age_carry = visual_age_seconds(
            present, times_s, self.deployment_latency_s, carry=age_carry
        )
        if self.ablate_visual_age:
            # Zeroed, not omitted: the fusion input's shape (and therefore
            # every downstream weight's shape) must not depend on which
            # ablation flags a run happens to set, or a checkpoint from one
            # combination could never even load into another.
            age = torch.zeros_like(age)
        return {
            "token": token,
            "quality": quality,
            "present": present,
            "age": age,
            "age_carry": new_age_carry,
            "visual_velocity": held,
            "visual_velocity_valid": held_valid,
            "velocity_carry": new_velocity_carry,
        }

    def _maybe_ablate_body_rate(self, aiding: torch.Tensor) -> torch.Tensor:
        if not self.ablate_body_rate:
            return aiding
        aiding = aiding.clone()
        aiding[..., BODY_RATE_AIDING_SLICE] = 0.0
        return aiding

    @staticmethod
    def _geometry(
        rotation: Optional[torch.Tensor],
        down: Optional[torch.Tensor],
        altitude: Optional[torch.Tensor],
    ) -> Optional[Dict[str, torch.Tensor]]:
        if rotation is None or down is None or altitude is None:
            return None
        return {"rotation": rotation, "down": down, "altitude": altitude}

    def forward(
        self,
        aiding: torch.Tensor,
        log_altitude: torch.Tensor,
        image0: torch.Tensor,
        image1: torch.Tensor,
        pair_dt_s: torch.Tensor,
        body_rate: torch.Tensor,
        offsets: torch.Tensor,
        valid: torch.Tensor,
        times_s: torch.Tensor,
        rotation: Optional[torch.Tensor] = None,
        down: Optional[torch.Tensor] = None,
        altitude: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        aiding = self._maybe_ablate_body_rate(aiding)
        visual = self._encode_visual(
            aiding, image0, image1, pair_dt_s, body_rate, offsets, valid, times_s,
            geometry=self._geometry(rotation, down, altitude),
        )
        prediction = self.model(
            aiding, visual["token"], visual["present"], visual["age"],
            visual_quality=visual["quality"], log_altitude=log_altitude,
            visual_velocity=visual["visual_velocity"],
            visual_velocity_valid=visual["visual_velocity_valid"],
        )
        if self.output_on_pairs:
            prediction["output_mask"] = visual["present"].squeeze(-1).detach()
        return prediction

    def forward_stream(
        self,
        aiding: torch.Tensor,
        log_altitude: torch.Tensor,
        image0: torch.Tensor,
        image1: torch.Tensor,
        pair_dt_s: torch.Tensor,
        body_rate: torch.Tensor,
        offsets: torch.Tensor,
        valid: torch.Tensor,
        times_s: torch.Tensor,
        state: Optional[VOStreamState],
        age_carry: Optional[torch.Tensor] = None,
        rotation: Optional[torch.Tensor] = None,
        down: Optional[torch.Tensor] = None,
        altitude: Optional[torch.Tensor] = None,
        velocity_carry: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], VOStreamState, torch.Tensor]:
        """:meth:`forward` for TBPTT: threads recurrent state across
        chronologically-adjacent windows instead of resetting it every window.

        ``age_carry`` is the age-side counterpart of ``state`` - see
        :func:`~vio.data.fixedwing_vo.visual_age_seconds` and
        :func:`~vio.data.fixedwing_vo.mask_age_carry` - so a continuing lane's
        staleness keeps growing from where the previous chunk left it instead
        of resetting to 0 at every window boundary. ``velocity_carry`` is the
        same for a ``geometric_residual`` model's held velocity; the new one is
        left in :attr:`last_velocity_carry` (mask it with
        :func:`~vio.data.fixedwing_vo.mask_velocity_carry`).

        Local-only scaffold (see ``PLAN_TBPTT.txt``): correct only when
        ``self`` is the bare module, not wrapped in
        ``DistributedDataParallel`` - DDP's gradient-sync hooks fire on the
        WRAPPED ``forward()``, not on a method called directly like this one,
        so a distributed TBPTT step needs its own call path. That is
        explicitly out of scope here.
        """

        aiding = self._maybe_ablate_body_rate(aiding)
        visual = self._encode_visual(
            aiding, image0, image1, pair_dt_s, body_rate, offsets, valid, times_s,
            age_carry=age_carry,
            geometry=self._geometry(rotation, down, altitude),
            velocity_carry=velocity_carry,
        )
        self.last_velocity_carry = visual["velocity_carry"]
        prediction, state = self.model.forward_stream(
            aiding, visual["token"], visual["present"], visual["age"],
            visual_quality=visual["quality"], log_altitude=log_altitude, state=state,
            visual_velocity=visual["visual_velocity"],
            visual_velocity_valid=visual["visual_velocity_valid"],
        )
        if self.output_on_pairs:
            prediction["output_mask"] = visual["present"].squeeze(-1).detach()
        return prediction, state, visual["age_carry"]


#: Batch keys carrying each pair's attitude geometry, in VOStep.forward's
#: trailing positional order. Always produced by FixedWingVODataset; only the
#: flat-ground frontend reads them.
PAIR_GEOMETRY_KEYS = ("visual_event_rotation", "visual_event_down", "visual_event_altitude")


def _geometry_arguments(batch: Mapping[str, torch.Tensor], to) -> List[Optional[torch.Tensor]]:
    return [to(name) if name in batch else None for name in PAIR_GEOMETRY_KEYS]


def forward_batch(
    step: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Move one batch to the device and run it through the step module.

    ``step`` is the DDP wrapper when distributed and the bare VOStep otherwise;
    the call is identical either way, which is the point of putting the whole
    computation inside one module.
    """

    to = lambda name: batch[name].to(device, non_blocking=True)
    prediction = step(
        to("aiding"),
        to("log_altitude"),
        to("visual_image0"),
        to("visual_image1"),
        to("visual_event_dt_s"),
        to("visual_event_body_rate"),
        to("visual_event_offset"),
        to("visual_event_valid"),
        to("telemetry_time_s"),
        *_geometry_arguments(batch, to),
    )
    mask = to("loss_mask")
    if "output_mask" in prediction:
        # --output-on-pairs: only the ticks where a pair was delivered are
        # outputs, so only they are scored - in training and validation alike.
        mask = mask * prediction["output_mask"].to(mask.dtype)
    return prediction, to("target_velocity_body"), mask


def forward_batch_stream(
    step: "VOStep",
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    state: Optional[VOStreamState],
    age_carry: Optional[torch.Tensor] = None,
    velocity_carry: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, VOStreamState, torch.Tensor]:
    """TBPTT counterpart of :func:`forward_batch`: threads recurrent state
    across chronologically-adjacent windows (see ChronologicalWindowSampler in
    ``vio.data.fixedwing_vo``) instead of resetting it every window.

    Local-only scaffold: ``step`` must be the bare :class:`VOStep`, never a
    ``DistributedDataParallel`` wrapper - see :meth:`VOStep.forward_stream`.
    A ``geometric_residual`` model's new held-velocity carry is left in
    ``step.last_velocity_carry``.
    """

    to = lambda name: batch[name].to(device, non_blocking=True)
    rotation, down, altitude = _geometry_arguments(batch, to)
    prediction, state, age_carry = step.forward_stream(
        to("aiding"),
        to("log_altitude"),
        to("visual_image0"),
        to("visual_image1"),
        to("visual_event_dt_s"),
        to("visual_event_body_rate"),
        to("visual_event_offset"),
        to("visual_event_valid"),
        to("telemetry_time_s"),
        state,
        age_carry,
        rotation=rotation,
        down=down,
        altitude=altitude,
        velocity_carry=velocity_carry,
    )
    mask = to("loss_mask")
    if "output_mask" in prediction:
        mask = mask * prediction["output_mask"].to(mask.dtype)
    return prediction, to("target_velocity_body"), mask, state, age_carry


# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    # A multi-GPU --device becomes a torchrun job, unless we ARE one already -
    # the child re-enters here with --device cuda and WORLD_SIZE set, and
    # without this guard it would fork itself forever.
    if parse_gpu_list(args.device) is not None and "WORLD_SIZE" not in os.environ:
        return relaunch_under_torchrun(argv, args.device)
    world = setup_distributed(args)
    device = world.device
    # Same seed on every rank: the model must start identical, and DDP only
    # synchronises gradients, never the initial weights. DistributedSampler
    # supplies the per-rank shuffle difference from its own epoch seed.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if world.is_main:
        args.run_dir.mkdir(parents=True, exist_ok=True)
    world.barrier()
    if world.enabled:
        world.log(f"distributed: {world.world_size} ranks, backend "
                  f"{dist.get_backend()}, this rank on {device}")
    try:
        return _train(args, world, device)
    finally:
        world.shutdown()


def _train(args: argparse.Namespace, world: Distributed, device: torch.device) -> int:

    camera_matrix = None
    native_size = None
    images_rectified = False
    distortion = None
    calibration = maybe_load_camera_calibration(args.calibration)
    if calibration is not None:
        camera_matrix = calibration.camera_matrix
        native_size = calibration.native_size
        images_rectified = calibration.images_rectified
        distortion = calibration.distortion if calibration.distortion.size else None
        focal_x, focal_y = calibration.focal_lengths
        world.log(f"camera: {native_size[1]}x{native_size[0]} "
                  f"fx={focal_x:.1f} fy={focal_y:.1f}"
                  + ("" if distortion is None else f" distortion={list(distortion)}"))


    # Build once, then re-window per split so the attitude file is read once.
    source = None
    datasets: Dict[str, FixedWingVODataset] = {}
    normalizer: Optional[VONormalizer] = None
    image_time_offset = resolve_image_time_offset(args)
    if isinstance(image_time_offset, dict):
        world.log(
            f"image clock: {len(image_time_offset['times_s'])}-point offset table "
            f"from {args.image_time_offset_file}"
        )
    elif image_time_offset:
        world.log(f"image clock: constant offset {image_time_offset:+.6f} s")

    lever_arm = resolve_lever_arm(args)
    if lever_arm is not None:
        world.log(
            "lever arm: GPS->camera "
            f"[{lever_arm[0]:+.3f}, {lever_arm[1]:+.3f}, {lever_arm[2]:+.3f}] m; "
            "the target is the velocity of the CAMERA, not of the antenna"
        )

    # Where each phase's data comes from. With pre-split directories every
    # phase is a different folder used WHOLE, so no fraction is applied and
    # leakage between phases is not merely guarded against but impossible.
    phase_roots: Dict[str, Path] = {}
    if args.validation_dataset is not None:
        phase_roots["train"] = Path(args.dataset)
        phase_roots["validation"] = Path(args.validation_dataset)
        if args.test_dataset is not None:
            phase_roots["test"] = Path(args.test_dataset)
        world.log(
            "pre-split datasets: each directory is used whole; "
            "--train-fraction / --validation-fraction are not applied"
        )
    else:
        for name in ("train", "validation", "test"):
            phase_roots[name] = Path(args.dataset)

    from vio.data.attitude import AttitudeAltitude, load_attitude_altitude

    sources: Dict[str, AttitudeAltitude] = {}
    for phase in ("train", "validation", "test"):
        root = phase_roots.get(phase)
        if root is None:
            continue
        loaded = load_attitude_altitude(
            root / args.csv_name,
            time_column=args.time_column, time_scale=args.time_scale,
            attitude_columns=args.attitude_columns,
            altitude_column=args.altitude_column,
            allow_reference_attitude=args.allow_reference_attitude,
        )
        sources[phase] = loaded
        if source is None:
            source = loaded
            world.log("attitude:", ", ".join(loaded.attitude_columns),
                      "| altitude:", loaded.altitude_column)
        for note in loaded.notes:
            world.log(f"  {phase}: {note}" if args.validation_dataset else f"  {note}")
        if not args.validation_dataset:
            break

    if source is None:
        raise SystemExit(
            "no telemetry was loaded; --dataset must contain "
            f"{args.csv_name}"
        )
    ranges: Dict[str, Tuple[int, int]]
    if args.validation_dataset is not None:
        ranges = {
            phase: (0, int(sources[phase].times_s.size))
            for phase in sources
        }
    else:
        ranges = resolve_ranges(args, source.times_s.size)
    world.log("ranges:", json.dumps(ranges))

    # The flags whose default depends on the frontend (see
    # resolve_frontend_defaults) are settled here, from the training images'
    # own clock, before anything reads them.
    from vio.data.images import numeric_image_manifest

    from vio.data.image_pairs import DEFAULT_IMAGE_PATTERN, DEFAULT_IMAGE_TIME_SCALE

    _, capture_times = numeric_image_manifest(
        phase_roots["train"] / args.image_folder, DEFAULT_IMAGE_PATTERN, DEFAULT_IMAGE_TIME_SCALE
    )
    frame_interval = float(np.median(np.diff(capture_times))) if capture_times.size > 1 else 0.0
    tick_interval = float(np.median(np.diff(source.times_s))) if source.times_s.size > 1 else 0.0
    for line in resolve_frontend_defaults(
        args, frame_interval_s=frame_interval, tick_interval_s=tick_interval
    ):
        world.log(f"  {args.frontend} default: {line}")
    if args.warmup >= args.window_length:
        raise SystemExit(
            f"--warmup {args.warmup} ticks does not fit in --window-length "
            f"{args.window_length}: with --frame-gap {args.frame_gap} the first image "
            "pair of a window arrives that late. Lengthen the window."
        )
    if args.random_pair_phase and args.pair_stride < 2:
        raise SystemExit(
            "--random-pair-phase varies where the pair tiling starts, which only "
            "exists with --output-on-pairs and a --frame-gap above 1 (pair stride "
            f"is {args.pair_stride} here: every frame already starts a pair)."
        )

    for phase in ("train", "validation", "test"):
        if phase not in phase_roots:
            continue
        span = ranges.get(phase)
        if span is None:
            continue
        span_ranges = normalize_index_ranges(span)
        if not any(end - start >= args.window_length for start, end in span_ranges):
            world.log(f"  {phase}: too short for one window, skipped")
            continue
        dataset, built, _ = build_vo_dataset(
            phase_roots[phase], span,
            csv_name=args.csv_name, time_column=args.time_column,
            time_scale=args.time_scale, attitude_columns=args.attitude_columns,
            altitude_column=args.altitude_column,
            allow_reference_attitude=args.allow_reference_attitude,
            image_folder=args.image_folder, image_size=tuple(args.image_size),
            frame_gap=args.frame_gap, deployment_latency_s=args.deployment_latency_s,
            max_frame_gap_s=args.max_frame_gap_s,
            pair_stride=args.pair_stride,
            image_time_offset_s=image_time_offset,
            lever_arm_m=lever_arm,
            camera_matrix=camera_matrix,
            calibration_image_size=native_size,
            images_rectified=images_rectified,
            distortion=distortion,
            normalizer=normalizer, window_length=args.window_length,
            stride=args.stride, warmup=args.warmup,
            max_visual_events=args.max_visual_events,
            grayscale=not args.color,
            # Training only: every other split, and every pass that scores the
            # training split, stays on the fixed tiling (phase 0).
            random_pair_phase=bool(args.random_pair_phase and phase == "train"),
        )
        if normalizer is None:
            normalizer = built
        datasets[phase] = dataset
        world.log(f"  {phase}: {len(dataset)} windows over ticks {span}")
        if dataset.random_pair_phase:
            pair_source = dataset.image_source
            distinct = sum(
                int(pair_source.plan_for(p).ready_tick.size)
                for p in range(pair_source.phase_count)
            )
            world.log(
                f"  {phase}: --random-pair-phase, {pair_source.phase_count} tiling "
                f"phases, {distinct} distinct image pairs (phase 0 alone: "
                f"{int(pair_source.plan.ready_tick.size)})"
            )

    # Never drop data silently: a gap cap that quietly removes a tenth of the
    # frames reads as "the capture is small" rather than "frames are missing".
    # Reported once, and named as a whole-flight figure, because each phase
    # builds its own VisualPairSource over the SAME image folder -- the count is
    # identical in all three and printing it per phase would read as three
    # separate sets of rejections.
    if datasets:
        any_source = next(iter(datasets.values())).image_source
        rejected = getattr(any_source, "rejected_gap_pairs", 0)
        if rejected:
            kept = int(any_source.plan.ready_tick.size)
            world.log(
                f"image gaps: rejected {rejected} pair(s) across the flight spanning "
                f"more than --max-frame-gap-s {args.max_frame_gap_s:g} s; {kept} kept"
            )

    if "train" not in datasets:
        raise SystemExit("The training split is shorter than one window.")

    # Every window is a cold start: nothing is delivered before one pair
    # interval plus the deployment latency has passed since the window began.
    # Ticks before that are scored as if the model had vision - with a long
    # --frame-gap (one second at 20 Hz for gap 20) that is a fifth of a 6 s
    # window of pure attitude-only guessing mixed into every metric.
    train_pairs = datasets["train"].image_source.plan.pair_dt_s
    if train_pairs.size and not args.disable_visual_input:
        tick_interval = float(np.median(np.diff(source.times_s)))
        if tick_interval > 0:
            blind_ticks = int(np.ceil(
                (args.deployment_latency_s + float(np.median(train_pairs))) / tick_interval
            ))
            if args.warmup < blind_ticks:
                world.log(
                    f"  WARNING --warmup {args.warmup} ticks is shorter than the "
                    f"~{blind_ticks} ticks every window spends before its first "
                    f"image pair can arrive (pair interval "
                    f"{float(np.median(train_pairs)):.3f} s + latency "
                    f"{args.deployment_latency_s:.3f} s); those ticks are scored "
                    f"blind. Consider --warmup {blind_ticks + 5}."
                )
    if "validation" not in datasets:
        # Training without validation is not a degraded run, it is an unscored
        # one: nothing selects a checkpoint, nothing detects overfitting, and
        # the saved "best" model is simply the last epoch under another name.
        # Far better to stop here than to discover it in the metrics file.
        span = ranges.get("validation")
        have = (
            0
            if span is None
            else sum(end - start for start, end in normalize_index_ranges(span))
        )
        needed = args.window_length
        total = source.times_s.size
        raise SystemExit(
            f"No validation split: {have} ticks available, {needed} needed for "
            f"one window.\n"
            f"A chronological split leaves one window ({needed} ticks) between "
            f"phases so no\nvalidation window overlaps a training one. On "
            f"{total} ticks at "
            f"{args.train_fraction:.0%}/{args.validation_fraction:.0%} that "
            f"leaves too little.\n"
            f"Either shorten the window (--window-length) or raise "
            f"--validation-fraction."
        )

    # Save the actual input contract next to the metrics. This is intentionally
    # independent of the checkpoint: a shared training log should be enough to
    # establish whether a run consumed raw IMU, which attitude was selected,
    # and which altitude supplied metric scale.
    input_contract = {
        "product": "visual_odometry_velocity",
        "raw_imu_used": False,
        "image_input": {
            "kind": "rgb_frame_pair" if args.color else "grayscale_frame_pair",
            "height": int(args.image_size[0]),
            "width": int(args.image_size[1]),
            "frame_gap": int(args.frame_gap),
        },
        "telemetry_input": {
            "source_attitude_columns": list(source.attitude_columns),
            "source_altitude_column": source.altitude_column,
            "temporal_aiding_channels": list(VO_AIDING_CHANNELS),
            "yaw_usage": (
                "NavEulZ is used in the relative quaternion rotation between "
                "image exposures; absolute yaw is not concatenated"
            ),
        },
        "output": [
            "body_velocity_m_s_x",
            "body_velocity_m_s_y",
            "body_velocity_m_s_z",
            "velocity_log_variance_x",
            "velocity_log_variance_y",
            "velocity_log_variance_z",
        ],
    }
    input_contract["frontend"] = str(args.frontend)
    input_contract["output_rate"] = (
        f"one velocity per image pair (every {int(args.pair_stride)} frames)"
        if args.output_on_pairs
        else "every telemetry tick"
    )
    input_contract["velocity_mode"] = resolve_velocity_mode(args)
    if args.frontend == "planar":
        input_contract["telemetry_input"]["planar_geometry"] = (
            "per pair: relative rotation between the exposures (NavEul*, exact), "
            "ground normal from roll/pitch, and altitude at both exposures; "
            "the image is de-rotated by warping and the translation solved over "
            "a flat ground plane"
        )
    (args.run_dir / "input_contract.json").write_text(
        json.dumps(input_contract, indent=2) + "\n", encoding="utf-8"
    )
    world.log("input contract: images + " + ", ".join(source.attitude_columns)
          + f" + {source.altitude_column}; raw IMU: NO")

    # The honest baseline: the TRAINING split's mean velocity, scored on
    # validation. Using the validation mean would be a baseline that saw the
    # answer, and would flatter every model compared against it. Concatenated
    # over every range assigned to train - a condition-segment split can hand
    # train several disjoint spans, and slicing only the first/last would mean
    # something different than "the training split's mean".
    train_ranges = normalize_index_ranges(ranges["train"])
    train_velocity = np.concatenate(
        [datasets["train"].velocity_body[start:end] for start, end in train_ranges]
    )
    baseline = train_velocity.mean(axis=0)
    world.log("train mean velocity baseline: [%.2f %.2f %.2f] m/s" % tuple(baseline))

    velocity_mode = resolve_velocity_mode(args)
    camera_from_body = resolve_camera_mounting(args)
    if args.frontend == "planar" and camera_from_body is None and calibration is not None:
        camera_from_body = auto_camera_mounting(
            args, world, calibration, phase_roots["train"], image_time_offset, frame_interval
        )
    prior_velocity = (
        [float(v) for v in args.prior_velocity]
        if args.prior_velocity is not None
        else [float(v) for v in baseline]
    )
    planar_settings: Optional[Dict[str, Any]] = None
    if args.frontend == "planar":
        if camera_matrix is None:
            raise SystemExit(
                "--frontend planar needs --calibration: a metric velocity is "
                "focal length times angle"
            )
        if camera_from_body is None:
            raise SystemExit(
                "--frontend planar needs the camera mounting: pass --camera-mounting "
                "(top_forward / right_forward / left_forward / bottom_forward, or the "
                "matrix tools/estimate_camera_mounting.py prints), or put "
                "mounting.camera_from_body in the calibration file"
            )
        if args.rotation_map is not None:
            raise SystemExit(
                "--rotation-map seeds the learned small-angle rotation field, which "
                "--frontend planar does not have: it removes rotation exactly from "
                "the attitude"
            )
        planar_settings = {
            "camera_from_body": camera_from_body,
            "prior_velocity_body": prior_velocity,
            "coarse_factor": int(args.coarse_factor),
            "coarse_radius": int(args.coarse_radius),
            "coarse_highpass": int(args.coarse_highpass),
            "fine_highpass": int(args.fine_highpass),
            "fine_iterations": int(args.fine_iterations),
            "huber_cells": float(args.huber_cells),
            "altitude_constraint": float(args.altitude_constraint),
            "learn_mounting": not args.no_learn_mounting,
            "max_geometric_speed": float(args.max_geometric_speed),
            "min_fit_cells": float(args.min_fit_cells),
        }
    elif velocity_mode == "geometric_residual":
        raise SystemExit(
            "--velocity-mode geometric_residual needs --frontend planar, the only "
            "frontend that measures a metric velocity"
        )
    frontend = build_frontend(
        args, camera_from_body=camera_from_body, prior_velocity=prior_velocity,
        dropout=args.dropout,
    ).to(device)
    if args.frontend == "planar":
        mounting_text = np.asarray(camera_from_body).round(3).tolist()
        world.log(
            f"frontend: planar (camera_from_body {mounting_text}, prior velocity "
            f"[{prior_velocity[0]:.1f} {prior_velocity[1]:.1f} {prior_velocity[2]:.1f}] m/s, "
            f"velocity mode {velocity_mode})"
        )
    if args.rotation_map is not None:
        payload = json.loads(Path(args.rotation_map).read_text(encoding="utf-8"))
        try:
            jacobian = payload["fit"]["jacobian_px_per_rad"]
        except (KeyError, TypeError) as error:
            raise SystemExit(
                f"{args.rotation_map} has no fit.jacobian_px_per_rad. This "
                "repository has no tool that produces that artifact - "
                "supply a JSON of the shape "
                '{"fit": {"jacobian_px_per_rad": [[dx_wx, dx_wy, dx_wz], '
                '[dy_wx, dy_wy, dy_wz]]}}, or drop --rotation-map and let '
                "the map be learned instead of seeded."
            ) from error
        if native_size is None:
            raise SystemExit(
                "--rotation-map needs --calibration too: the Jacobian is in "
                "NATIVE pixels per radian and the correlator works in feature "
                "cells of a resized image, so the native size is what converts "
                "between them. Seeding without it would be wrong by the resize "
                "factor - a rotation correction several times too large or too "
                "small, applied to every pair."
            )
        # A rotation map is a property of ONE camera on ONE airframe. Seeding
        # from another capture's map is worse than not seeding: the search
        # centre is then confidently wrong, in proportion to turn rate, and the
        # correlator spends its bounded window looking in the wrong place.
        recorded = str(payload.get("dataset", ""))
        if recorded and Path(recorded).name != Path(args.dataset).resolve().name:
            world.log(
                f"  WARNING {args.rotation_map} was calibrated on "
                f"'{Path(recorded).name}', not '{Path(args.dataset).resolve().name}'.\n"
                "          Seeding from another capture's mounting points the "
                "search window\n          the wrong way on every pair. "
                "Re-calibrate a Jacobian for THIS dataset (this repository "
                "has no tool that\n          does so; see "
                "RotationalSearchField.seed_from_jacobian for the expected "
                "JSON shape),\n          or drop --rotation-map and let it "
                "be learned."
            )
        frontend.rotation.seed_from_jacobian(
            torch.tensor(jacobian, dtype=torch.float32),
            patch_size=args.patch_size,
            resize=(
                args.image_size[0] / native_size[0],
                args.image_size[1] / native_size[1],
            ),
        )
        world.log(f"seeded the search centre from {args.rotation_map}")

    model = VisionMambaVO(
        visual_dim=args.visual_dim, aiding_dim=args.aiding_dim,
        fusion_dim=args.fusion_dim, dropout=args.dropout, frontend=frontend,
        velocity_mode=velocity_mode,
    ).to(device)
    # The frontend is a submodule of the model, so model.parameters()
    # already covers it. Listing both hands the optimizer a duplicate
    # group, which double-counts weight decay on every frontend weight.
    parameters = list(model.parameters())
    frontend_size = sum(p.numel() for p in frontend.parameters())
    world.log(f"parameters: {sum(p.numel() for p in parameters):,} "
          f"(frontend {frontend_size:,})")

    # camera_matrix is at NATIVE (calibration) resolution, but the frontend
    # receives images already resized to args.image_size by VisualPairSource -
    # it never sees a native-resolution frame. Its own native/working rescale
    # (_bearing_scale, _normalised_grid) infers "native size" from the image
    # tensor it is actually handed, which by then IS args.image_size, so that
    # rescale is always a no-op. Handing it the native matrix directly - as a
    # previous version of this trainer did - silently used the native fx/fy/cx/cy
    # as if they already belonged to the working image: at a typical 1920x1080
    # native / 576x1024 working size that overstates focal length by ~1.875x and
    # misplaces the principal point by hundreds of working pixels, corrupting
    # both the bearing scale and the per-cell rotational field's coordinate
    # grid. Resizing here, once, with the same helper VisualPairSource uses
    # internally, is what makes the two agree.
    camera_tensor = (
        None if camera_matrix is None
        else torch.from_numpy(
            resize_camera_matrix(camera_matrix, native_size, tuple(args.image_size))
        ).to(device)
    )

    step = VOStep(
        model,
        window_length=args.window_length,
        visual_dim=args.visual_dim,
        disable_visual=args.disable_visual_input,
        frontend_chunk=args.frontend_chunk,
        deployment_latency_s=args.deployment_latency_s,
        camera_matrix=camera_tensor,
        ablate_body_rate=args.ablate_body_rate,
        ablate_visual_age=args.ablate_visual_age,
        photometric_augment=args.photometric_augment,
        output_on_pairs=args.output_on_pairs,
    ).to(device)
    if world.enabled:
        # static_graph lets DDP cope with gradient checkpointing, which
        # otherwise re-enters the autograd graph and trips "marked ready
        # twice". With --disable-visual-input the frontend is never reached, so
        # its parameters are genuinely unused and DDP has to be told rather
        # than left to hang waiting for gradients that never arrive.
        #
        # --velocity-loss simple leaves the variance and concentration heads
        # out of the loss, so their parameters are unused for the same reason
        # and need the same treatment. Without this a multi-GPU run hangs
        # partway through the first epoch instead of reporting anything.
        has_unused_parameters = (
            args.disable_visual_input or args.velocity_loss == "simple"
        )
        step = DistributedDataParallel(  # type: ignore[assignment]
            step,
            device_ids=[world.local_rank] if device.type == "cuda" else None,
            output_device=world.local_rank if device.type == "cuda" else None,
            find_unused_parameters=has_unused_parameters,
            static_graph=not args.disable_visual_input,
            # gradient_as_bucket_view is deliberately off: the depthwise
            # convolutions produce gradients whose strides do not match DDP's
            # bucket layout, so it copies anyway - losing the memory saving and
            # printing a stride-mismatch warning on every backward.
        )

    scale = {"none": 1.0, "linear": float(world.world_size),
             "sqrt": math.sqrt(world.world_size)}[args.lr_scaling]
    learning_rate = args.learning_rate * scale
    # The mounting correction is a physical angle, not a feature weight: it gets
    # a smaller step (its gradient is large - every pixel of every pair moves
    # with it) and no weight decay (zero is not a prior worth pulling towards
    # harder than the data says).
    mounting_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.endswith("mounting_correction")
    ]
    mounting_ids = {id(parameter) for parameter in mounting_parameters}
    groups: List[Dict[str, Any]] = [
        {"params": [p for p in parameters if id(p) not in mounting_ids]}
    ]
    if mounting_parameters:
        groups.append({
            "params": mounting_parameters,
            "lr": learning_rate * float(args.mounting_lr_scale),
            "lr_scale": float(args.mounting_lr_scale),
            "weight_decay": 0.0,
        })
    optimizer = torch.optim.AdamW(
        groups, lr=learning_rate, weight_decay=args.weight_decay
    )
    schedule = build_schedule(optimizer, args.epochs, args.lr_warmup_epochs)

    samplers: Dict[str, Optional[DistributedSampler]] = {}
    loaders = {}
    for phase, dataset in datasets.items():
        training = phase == "train"
        sampler = None
        if world.enabled:
            # drop_last on the training sampler keeps every rank doing the same
            # number of steps; an uneven last batch deadlocks the all-reduce.
            # Validation keeps every window and DistributedSampler pads the
            # short rank by repeating a few, which is counted consistently in
            # both the numerator and the denominator of every metric below.
            sampler = DistributedSampler(
                dataset, num_replicas=world.world_size, rank=world.rank,
                shuffle=training, drop_last=training,
            )
        samplers[phase] = sampler
        loaders[phase] = DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=(training and sampler is None),
            sampler=sampler, num_workers=args.num_workers,
            drop_last=training, pin_memory=(device.type == "cuda"),
        )
    if args.eval_train_split:
        # A SEPARATE loader over datasets["train"], deterministic and
        # complete - shuffle=False, drop_last=False - never loaders["train"]
        # itself, which is shuffled and drop_last=True for the optimiser.
        # Reusing that one would silently under-report: drop_last discards up
        # to batch_size-1 windows EVERY epoch, a systematic gap whenever the
        # split length is not a multiple of batch_size, not an occasional one.
        # Phase 0 only, so a --random-pair-phase run scores its training split
        # on the same fixed tiling as validation, every epoch.
        train_eval_dataset = datasets["train"].fixed_phase_view()
        train_eval_sampler = None
        if world.enabled:
            train_eval_sampler = DistributedSampler(
                train_eval_dataset, num_replicas=world.world_size, rank=world.rank,
                shuffle=False, drop_last=False,
            )
        samplers["train_eval"] = train_eval_sampler
        loaders["train_eval"] = DataLoader(
            train_eval_dataset, batch_size=args.batch_size,
            shuffle=False, sampler=train_eval_sampler, num_workers=args.num_workers,
            drop_last=False, pin_memory=(device.type == "cuda"),
        )
    train_pairs_per_batch = args.batch_size * args.max_visual_events
    world.log(
        f"training work: {len(loaders['train'])} steps/epoch per rank, up to "
        f"{train_pairs_per_batch} image pairs/step "
        f"({args.batch_size} windows x {args.max_visual_events} events)",
        flush=True,
    )
    if not args.disable_visual_input:
        # The correlator's largest intermediate is (pairs, channels,
        # candidates, height, width) float32 - see
        # _correlation_candidate_gib - and encode_pairs bounds "pairs" to
        # --frontend-chunk for BOTH training and validation. Printed here,
        # not buried in a docstring, because this is the number that decides
        # whether a run OOMs the moment validation (or a large --batch-size)
        # is reached.
        candidates = (2 * args.correlation_radius + 1) ** 2
        grid_height = args.image_size[0] // args.patch_size
        grid_width = args.image_size[1] // args.patch_size
        if args.frontend_chunk > 0:
            chunk_gib = _correlation_candidate_gib(
                min(args.frontend_chunk, train_pairs_per_batch),
                args.stem_dim, candidates, grid_height, grid_width,
            )
            world.log(
                f"  frontend chunk: {args.frontend_chunk} pairs per frontend/"
                f"correlator call (train AND validation) -> est. peak "
                f"candidate tensor {chunk_gib:.2f} GiB",
                flush=True,
            )
        else:
            whole_gib = _correlation_candidate_gib(
                train_pairs_per_batch, args.stem_dim, candidates,
                grid_height, grid_width,
            )
            world.log(
                f"  frontend chunk: DISABLED (--frontend-chunk 0) -> every "
                f"step's {train_pairs_per_batch} pairs reach the correlator "
                f"in one call, est. peak candidate tensor {whole_gib:.2f} "
                "GiB - this is a per-GPU figure, not shared across ranks",
                flush=True,
            )
    if world.enabled:
        # Worth printing loudly: the same --epochs on more ranks is fewer
        # optimizer steps, so a run that converged on one GPU may look
        # undertrained on twenty for reasons that have nothing to do with the
        # model.
        world.log(
            f"  effective batch {args.batch_size * world.world_size} windows "
            f"({world.world_size} ranks x {args.batch_size}), "
            f"{len(loaders['train'])} steps/epoch "
            f"(a single rank would take {world.world_size}x more)"
        )
        world.log(f"  learning rate {learning_rate:g} "
                  f"(--lr-scaling {args.lr_scaling})")

    metrics_path = args.run_dir / "metrics.csv"
    # Same names as the VIO trainer's metrics.csv, so a VO run and a VIO run
    # can be compared column by column. vel_rmse is the RMS of the error
    # MAGNITUDE, not the per-component RMS - the two differ by sqrt(3), and
    # sharing RunningVelocityStats is what keeps them the same quantity.
    columns = [
        "epoch",
        "train_loss", "val_loss",
        "train_cov_loss", "val_cov_loss",
        "train_vel_rmse", "train_vel_max_error",
        "train_vel_rmse_x", "train_vel_rmse_y", "train_vel_rmse_z",
        "train_vel_dir_rmse", "train_vel_dir_max_error",
        "val_vel_rmse", "val_vel_max_error",
        "val_vel_rmse_x", "val_vel_rmse_y", "val_vel_rmse_z",
        "val_vel_dir_rmse", "val_vel_dir_max_error",
        "val_baseline_vel_rmse", "val_baseline_vel_rmse_y",
        "val_baseline_vel_dir_rmse",
        "val_skill_vs_mean_x", "val_skill_vs_mean_y", "val_skill_vs_mean_z",
        "train_direction_loss", "learning_rate",
        # Fraction of offered image pairs the reliability gate delivered. With
        # the gate off this is 1.0 every epoch; open_metrics_csv retires an
        # older file whose header lacks it rather than silently appending
        # mislabelled rows.
        "train_visual_kept_fraction",
        # The same fraction measured on the validation split, so the two can be
        # compared directly. A gap between them means the thresholds meet
        # different imagery on either side of the split.
        "val_visual_kept_fraction",
        # The learned correlation temperature. The gate deliberately does not
        # read it (see LocalCorrelation.gate_temperature); this column is how a
        # reader sees whether it has drifted from the fixed reference the
        # thresholds were chosen against.
        "correlation_temperature",
    ]
    # Horizon columns are appended, never interleaved, so an existing
    # metrics.csv stays readable column-for-column against a run without them.
    horizons = parse_horizon_minutes(args.horizon_minutes)
    horizon_span = ranges.get(args.horizon_split) if horizons else None
    # A pre-split run's phases live in DIFFERENT directories - datasets["train"]
    # covers only the training file's ticks - so a horizon leg for another
    # split additionally needs that split's own dataset to exist. It is built
    # from the same span this checks, so its absence here would mean it was
    # skipped as "too short for one window" (see the dataset-building loop
    # above): failing the same way as no span at all, for the same reason.
    if horizons and args.horizon_split not in datasets:
        horizon_span = None
    if horizons and horizon_span is None:
        world.log(
            f"  --horizon-minutes ignored: there is no '{args.horizon_split}' "
            "split in this dataset"
        )
        horizons = ()
    elif horizons and horizon_span is not None:
        # A horizon leg is one unbroken run cut from a SINGLE contiguous span,
        # which every chronological phase always is. normalize_index_ranges
        # is called anyway, cheaply, so this stays correct if ``ranges`` is
        # ever produced some other way in the future.
        horizon_span = max(
            normalize_index_ranges(horizon_span), key=lambda pair: pair[1] - pair[0]
        )
    horizon_prefix = HORIZON_COLUMN_PREFIX[args.horizon_split]
    horizon_rotation = None
    if horizons and not args.no_horizon_position:
        # The rotation that DEFINES the target, read here and nowhere near the
        # model, purely so a body-frame velocity can be integrated into a
        # distance. Loading it once up front rather than per epoch: it is a
        # whole extra pass over the CSV.
        #
        # Read from the HORIZON SPLIT's own directory, not args.dataset - in a
        # pre-split run those are different files (phase_roots["validation"]
        # is not phase_roots["train"]), and reading the training file's
        # rotation to score a validation leg would silently mismatch the
        # attitude used here against the one the leg's ticks actually came
        # from. phase_roots[phase] equals args.dataset for every phase in the
        # fractional-split mode, so this is a no-op there.
        _, _, horizon_rotation = reference_body_frame(
            phase_roots[args.horizon_split] / args.csv_name,
            time_column=args.time_column,
            time_scale=args.time_scale,
        )
    if horizons:
        columns.extend(
            f"{horizon_prefix}{name}"
            for name in horizon_metric_names(
                horizons, position=horizon_rotation is not None
            )
        )
        world.log(
            "horizons: "
            + ", ".join(f"{value:g} min" for value in horizons)
            + f" over the {args.horizon_split} split, every "
            + (f"{args.horizon_every} epochs" if args.horizon_every > 1 else "epoch")
        )
        world.log(
            "  each leg starts from a zeroed state and is streamed unbroken to "
            "its end"
        )
        # Every horizon leg is a cold start too; the general --warmup check
        # after the datasets are built already warns when the warm-up is
        # shorter than the time to the first image pair.

    # Same as the val_* set, appended rather than interleaved for the same
    # reason the horizon columns are: an existing metrics.csv without
    # --eval-train-split stays readable column-for-column.
    if args.eval_train_split:
        columns.extend(
            f"traineval_{suffix}" for suffix in TRAIN_EVAL_METRIC_SUFFIXES
        )
        world.log(
            "eval-train-split: scoring the training split through evaluate() "
            f"(dropout off) every "
            + (
                f"{args.eval_train_split_every} epochs"
                if args.eval_train_split_every > 1
                else "epoch"
            )
        )

    # The header is written only when the file is new, so a resumed run appends
    # to the same file instead of truncating the history that justified
    # resuming in the first place. A run resumed after the column set changed -
    # --horizon-minutes added, say - retires the old file rather than appending
    # rows in a new order under the old header, which would mislabel every
    # column with nothing in the file to say so.
    metrics = open_metrics_csv(metrics_path, columns) if world.is_main else None

    start_epoch = 1
    best = float("inf")
    # Epochs since the selection metric last improved by --min-delta, for
    # --patience. Carried in every checkpoint so a resumed run keeps counting.
    stale_epochs = 0
    # The score patience measures against. It only moves when an epoch beats
    # it by more than --min-delta, so a run of small gains that add up to
    # more than min_delta still resets patience when the sum crosses it.
    patience_reference = float("inf")
    if normalizer is None:
        raise SystemExit(
            "no split produced a dataset, so there is nothing to train on; "
            "check --window-length against the split lengths above"
        )
    fingerprint = resume_fingerprint(
        args, source, normalizer, ranges,
        frontend_id=str(frontend.frontend_id),
        temporal_input_id=str(model.temporal_input_id),
        planar=planar_settings,
    )
    if args.resume is not None:
        resume_path = (
            args.run_dir / "last.pt" if args.resume == "auto" else Path(args.resume)
        )
        # Only rank 0 writes runs/, so on a multi-node job only its machine is
        # guaranteed to hold the file: the leader reads it once and broadcasts.
        # Every rank still checks the fingerprint against its own locally built
        # data, so a stale copy on another node fails loudly here.
        saved: Dict[str, Any]
        if world.enabled:
            # A bare [None] is a list of None, which makes every later use of
            # `saved` read as None - the broadcast is what fills it, and that
            # is invisible to a type checker.
            resume_payload: List[Optional[Dict[str, Any]]] = [None]
            if world.is_main:
                if not resume_path.is_file():
                    raise SystemExit(f"Resume checkpoint not found: {resume_path}")
                resume_payload[0] = load_checkpoint(resume_path, map_location="cpu")
            dist.broadcast_object_list(resume_payload, src=0)
            received = resume_payload[0]
            if received is None:
                # Rank 0 raises above rather than broadcasting nothing, so this
                # means the broadcast itself did not deliver - worth saying,
                # because the alternative is an AttributeError on the next line.
                raise SystemExit(
                    f"rank {world.rank} received no checkpoint from the resume "
                    "broadcast"
                )
            saved = received
        else:
            if not resume_path.is_file():
                raise SystemExit(f"Resume checkpoint not found: {resume_path}")
            saved = load_checkpoint(resume_path, map_location="cpu")

        differences = fingerprint_differences(saved.get("fingerprint"), fingerprint)
        if differences:
            raise SystemExit(
                f"{resume_path} was written under different conditions:\n  "
                + "\n  ".join(differences)
                + "\n\nResuming would continue one run's optimizer state on "
                "another run's problem.\nRe-run the ORIGINAL command with "
                "--resume, or start a new --run-dir."
            )
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = int(saved["epoch"]) + 1
        saved_epochs = int(saved.get("epochs", args.epochs))
        saved_warmup = int(saved.get("args", {}).get("lr_warmup_epochs", 0) or 0)
        # A schedule saved under a different shape (another --epochs, or a
        # warm-up added/removed - which also changes the scheduler CLASS) is
        # not loaded but rebuilt and fast-forwarded below.
        reshaped = saved_epochs != args.epochs or saved_warmup != args.lr_warmup_epochs
        if not reshaped:
            schedule.load_state_dict(saved["scheduler"])
        # The metric best.pt was chosen on is recorded, so a resume that
        # changes --select-on starts its search again rather than comparing
        # this run's numbers against a different quantity from the last one.
        saved_metric = saved.get("select_on", "vel_rmse_y")
        if saved_metric == args.select_on:
            best = float(
                saved.get("best_val_score", saved.get(f"best_val_{saved_metric}", float("inf")))
            )
            stale_epochs = int(saved.get("stale_epochs", 0))
            patience_reference = float(saved.get("patience_reference", best))
        else:
            world.log(
                f"selection metric changed ({saved_metric} -> {args.select_on}), "
                "so best.pt is contested from scratch"
            )
            best = float("inf")
        if reshaped:
            # A restored scheduler carries the T_max it was built with, so a
            # changed --epochs would otherwise be ignored - and silently, in
            # the worst way: a run cut short at its own T_max has already
            # annealed to eta_min, so extending it would train every further
            # epoch at a learning rate of zero. Rebuilding and fast-forwarding
            # puts the optimizer exactly where the NEW cosine says this epoch
            # should be, which is what extending a run has to mean.
            for group in optimizer.param_groups:
                # Each group keeps its own multiple of the base rate (the
                # mounting correction runs at --mounting-lr-scale of it).
                rate = learning_rate * float(group.get("lr_scale", 1.0))
                group["lr"] = rate
                group["initial_lr"] = rate
            schedule = build_schedule(optimizer, args.epochs, args.lr_warmup_epochs)
            # Walking the schedule forward without stepping the optimizer is
            # exactly what is wanted here - the optimizer's own state was
            # restored above - so torch's "step() before optimizer.step()"
            # warning is describing the intent, not a mistake.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                for _ in range(start_epoch - 1):
                    schedule.step()
            world.log(
                f"  schedule changed (--epochs {saved_epochs} -> {args.epochs}, "
                f"--lr-warmup-epochs {saved_warmup} -> {args.lr_warmup_epochs}): "
                f"rebuilt and fast-forwarded, resuming at lr "
                f"{optimizer.param_groups[0]['lr']:g}"
            )
        world.log(
            f"resumed {resume_path} at epoch {start_epoch} of {args.epochs} "
            f"(best val_{args.select_on} so far {best:.4f}, "
            f"lr {optimizer.param_groups[0]['lr']:g})"
        )
        if start_epoch > args.epochs:
            world.log("  nothing to do: the checkpoint is already the last epoch")

    best_path = args.run_dir / "best.pt"
    last_path = args.run_dir / "last.pt"
    epochs_dir = args.run_dir / "epochs"
    announced_disk = False
    if world.is_main and args.save_every > 0:
        epochs_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint_payload(epoch: int, score: float) -> Dict[str, object]:
        """`model` is the bare VisionMambaVO - DDP wraps `step`, not it - so
        the saved keys carry no "module." prefix and load straight into an
        unwrapped model at inference."""

        # `frontend` is a submodule of `model`, so its tensors appear twice
        # here - and cost nothing twice: torch.save writes shared storages
        # once, measured identical to the byte with the entry removed. It is
        # kept because checkpoints in the wild are loaded through it.
        return {
            "model": model.state_dict(),
            "frontend": frontend.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": schedule.state_dict(),
            "normalizer": normalizer.as_dict(),
            "fingerprint": fingerprint,
            "args": vars(args) | {
                "dataset": str(args.dataset),
                "run_dir": str(args.run_dir),
                "calibration": None if args.calibration is None else str(args.calibration),
                "rotation_map": None if args.rotation_map is None else str(args.rotation_map),
                "dist_file": None if args.dist_file is None else str(args.dist_file),
                "validation_dataset": (
                    None
                    if args.validation_dataset is None
                    else str(args.validation_dataset)
                ),
                "test_dataset": (
                    None if args.test_dataset is None else str(args.test_dataset)
                ),
                "image_time_offset_file": (
                    None
                    if args.image_time_offset_file is None
                    else str(args.image_time_offset_file)
                ),
                # The frontend algorithm, so the evaluator can refuse a
                # checkpoint whose weights fit but whose semantics do not.
                "frontend_id": str(frontend.frontend_id),
                # Same idea, for the fusion side: the aiding-vector layout and
                # what gets concatenated onto the token. ablate_body_rate/
                # ablate_visual_age need no separate entry here - they are
                # plain argparse flags, already carried by vars(args) above.
                "temporal_input_id": str(model.temporal_input_id),
                # RESOLVED planar geometry, under the names the evaluator reads:
                # the flags alone do not capture a mounting read from the
                # calibration file or a prior defaulted to the training mean.
                "velocity_mode": velocity_mode,
                "camera_from_body": camera_from_body,
                "prior_velocity_body": prior_velocity,
                # RESOLVED alignment, under the names the evaluator reads.
                # vars(args) carries the raw flags -- image_time_offset and
                # lever_arm -- and an evaluator asking for image_time_offset_s
                # or lever_arm_m would get the default and silently score
                # against a different clock and a different target than the run
                # trained with. The offset may also be a table loaded from a
                # file, which the flag alone does not capture.
                "image_time_offset_s": resolve_image_time_offset(args),
                "lever_arm_m": resolve_lever_arm(args),
            },
            "epoch": epoch,
            "epochs": int(args.epochs),
            # WHICH metric decided this file, and its value. The old
            # val_vel_rmse_y / best_val_vel_rmse_y keys are kept because the
            # evaluator prints them, but they are only the y axis when that is
            # what was selected on - a key that names one metric and holds
            # another is worse than no key.
            "select_on": str(args.select_on),
            # Which objective trained this checkpoint. Under "simple" the
            # variance and concentration heads receive NO gradient at all, so
            # they hold their initialisation - a variance of exp(0)=1 m^2/s^2
            # on every axis and kappa=1 on every heading, which look like real
            # predictions and are not. Recorded so a reader, and
            # evaluate_velocity_horizons.py, can say so instead of quoting an
            # untrained number as an uncertainty.
            "velocity_loss": str(args.velocity_loss),
            "uncertainty_trained": args.velocity_loss != "simple",
            "val_score": score,
            "best_val_score": best,
            "stale_epochs": stale_epochs,
            "patience_reference": patience_reference,
            "val_vel_rmse_y": score if args.select_on == "vel_rmse_y" else float("nan"),
            "best_val_vel_rmse_y": best if args.select_on == "vel_rmse_y" else float("nan"),
            "world_size": world.world_size,
            # The TRAINING split's own mean velocity, carried in the
            # checkpoint rather than left for a later reader to recompute.
            # For a pre-split run --dataset is a different directory than
            # whatever gets scored later, so re-deriving "the training mean"
            # from the folder an evaluator happens to be pointed at would
            # silently substitute that folder's own mean - flattering every
            # skill/baseline comparison exactly the way scoring against a
            # validation-mean baseline would. Storing the number computed
            # here, once, from the real training ticks, is what makes it
            # reproducible without depending on the training folder still
            # being reachable at evaluation time.
            "train_baseline_velocity_m_s": [float(value) for value in baseline],
        }

    for epoch in range(start_epoch, args.epochs + 1):
        step.train()
        # Without this every epoch draws the same per-rank shuffle, so the
        # model sees the same windows in the same order for the whole run.
        train_sampler = samplers.get("train")
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        running = {
            "loss": 0.0, "nll": 0.0, "direction": 0.0, "n": 0,
            "pairs_offered": 0.0, "pairs_delivered": 0.0, "skipped": 0,
        }
        train_stats = RunningVelocityStats()
        train_batches = tqdm(
            loaders["train"],
            desc=f"epoch {epoch}/{args.epochs} train",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
            disable=args.no_progress or not world.is_main,
        )
        for batch in train_batches:
            optimizer.zero_grad(set_to_none=True)
            prediction, target, mask = forward_batch(step, batch, device)
            loss, parts = compute_velocity_loss(
                prediction, target, mask,
                direction_weight=args.direction_weight,
                loss_mode=args.velocity_loss, huber_delta=args.huber_delta,
            )
            loss.backward()
            # One non-finite batch (a corrupt frame, a telemetry glitch) must
            # not poison every weight: NaN gradients survive clipping and one
            # optimizer step writes NaN into all of them, after which every
            # later loss is NaN too. The norm is taken AFTER DDP's all-reduce,
            # so every rank sees the same value and skips the same step.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, args.grad_clip if args.grad_clip > 0 else float("inf")
            )
            if not (torch.isfinite(grad_norm) and torch.isfinite(loss.detach())):
                optimizer.zero_grad(set_to_none=True)
                running["skipped"] += 1
                if running["skipped"] <= 3:
                    world.log(
                        f"  WARNING non-finite loss/gradient at epoch {epoch} "
                        f"batch {running['n'] + running['skipped']}: step skipped"
                    )
                continue
            optimizer.step()
            running["loss"] += float(loss.detach())
            running["nll"] += parts["nll"]
            running["direction"] += parts["direction"]
            running["n"] += 1
            # How many image pairs the gate let through this batch. Read off
            # the bare VOStep (DDP wraps it) right after the forward that set
            # them. Logged as a fraction per epoch, because the raw counts
            # depend on batch size and rank count and the fraction does not.
            inner = step.module if world.enabled else step
            running["pairs_offered"] += inner._pairs_offered
            running["pairs_delivered"] += inner._pairs_delivered
            # Measured DURING the epoch, so it reflects a model that is still
            # changing - which is what makes the train/val gap readable rather
            # than an artefact of evaluating one and not the other.
            train_stats.update(
                masked_velocity_stats(
                    prediction["predicted_velocity"], target, mask
                )
            )
            train_batches.set_postfix(
                loss=f"{running['loss'] / running['n']:.4f}",
                nll=f"{running['nll'] / running['n']:.4f}",
                refresh=False,
            )
        # Read BEFORE stepping: the column names the rate this epoch actually
        # trained at, not the one the next epoch will use.
        if running["skipped"]:
            world.log(
                f"  epoch {epoch}: skipped {running['skipped']} step(s) with a "
                "non-finite loss or gradient - if this is more than a rare "
                "batch, check the data (altitude, timestamps, blank frames)"
            )
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        schedule.step()
        # Each rank saw a different shard, so the reported training loss is the
        # sum over ranks divided by the total batch count - not rank 0's own
        # average, which would be a fifth of the data reported as if it were
        # all of it.
        tallies = torch.tensor(
            [running["loss"], running["nll"], running["direction"], running["n"],
             running["pairs_offered"], running["pairs_delivered"]],
            dtype=torch.float64, device=device,
        )
        world.reduce_sum(tallies)
        world.reduce_stats(train_stats)
        batches = max(float(tallies[3]), 1.0)
        # Summed across ranks first, so the fraction describes the whole epoch
        # rather than whichever shard rank 0 happened to draw.
        pairs_offered = float(tallies[4])
        visual_kept_fraction = (
            float(tallies[5]) / pairs_offered if pairs_offered > 0 else float("nan")
        )

        row = {
            "epoch": epoch,
            "train_loss": float(tallies[0]) / batches,
            "train_cov_loss": float(tallies[1]) / batches,
            "train_direction_loss": float(tallies[2]) / batches,
            "learning_rate": epoch_learning_rate,
            # 1.0 means the gate rejected nothing. Watch this against
            # val_vel_rmse when tuning the thresholds: a gate that drops most
            # pairs and does not move the error is removing signal, not noise.
            "train_visual_kept_fraction": visual_kept_fraction,
            # Read off the frontend directly rather than from a diagnostic,
            # because it is a single parameter of the run, not a per-pair
            # measurement to be averaged.
            "correlation_temperature": (
                float(torch.exp(frontend.correlation.log_temperature).detach())
                if frontend.correlation.learnable_temperature
                else float(frontend.correlation.temperature)
            ),
        }
        row.update({f"train_{k}": v for k, v in train_stats.metrics().items()})
        if "validation" in loaders:
            validation_sampler = samplers.get("validation")
            if validation_sampler is not None:
                validation_sampler.set_epoch(epoch)
            validation = evaluate(
                step, loaders["validation"], device, baseline=baseline,
                direction_weight=args.direction_weight, world=world,
                loss_mode=args.velocity_loss, huber_delta=args.huber_delta,
                progress_description=f"epoch {epoch}/{args.epochs} validation",
                show_progress=not args.no_progress and world.is_main,
            )
            row.update({f"val_{k}": v for k, v in validation.items()})
            # Every rank has the same reduced metrics, so every rank agrees on
            # whether this epoch is the best - but only one writes the file.
            score = validation[args.select_on]
            # --min-delta only decides whether patience resets; best.pt still
            # follows every improvement, however small.
            if score < patience_reference - float(args.min_delta):
                patience_reference = score
                stale_epochs = 0
            else:
                stale_epochs += 1
            if score < best:
                best = score
                if world.is_main:
                    torch.save(checkpoint_payload(epoch, score), best_path)
            skill = validation["skill_vs_mean_y"]
            skill_text = "n/a" if math.isnan(skill) else f"{skill:+.3f}"
            world.log(
                f"epoch {epoch:3d}  loss {row['train_loss']:7.4f}/"
                f"{validation['loss']:.4f}  "
                f"cov {row['train_cov_loss']:7.4f}/"
                f"{validation['cov_loss']:.4f}  "
                f"val rmse {validation['vel_rmse']:6.3f} "
                f"(max {validation['vel_max_error']:6.2f})  "
                f"y {validation['vel_rmse_y']:6.3f} (skill {skill_text})  "
                f"dir {validation['vel_dir_rmse']:5.2f} deg "
                f"(max {validation['vel_dir_max_error']:5.1f})"
            )
        else:
            world.log(
                f"epoch {epoch:3d}  loss {row['train_loss']:7.4f}  "
                f"cov {row['train_cov_loss']:7.4f}  (no validation split)"
            )

        eval_train_due = (
            epoch % max(args.eval_train_split_every, 1) == 0 or epoch == args.epochs
        )
        if args.eval_train_split and eval_train_due:
            # loaders["train_eval"]: a dedicated deterministic (shuffle=False,
            # drop_last=False) pass built above - NOT loaders["train"], which
            # is shuffled and drops up to batch_size-1 windows every call and
            # so cannot promise it scored the whole split. Its sampler is
            # order-only (shuffle=False), so no set_epoch is needed.
            traineval = evaluate(
                step, loaders["train_eval"], device, baseline=baseline,
                direction_weight=args.direction_weight, world=world,
                loss_mode=args.velocity_loss, huber_delta=args.huber_delta,
                progress_description=f"epoch {epoch}/{args.epochs} train(eval)",
                show_progress=not args.no_progress and world.is_main,
            )
            row.update({f"traineval_{k}": v for k, v in traineval.items()})
            # The number PLAN.txt SS12 asks for: train_vel_rmse is measured
            # WITH dropout and averaged over a model that changed all epoch;
            # traineval_vel_rmse is the same split, same dropout-off eval mode
            # validation uses. If it tracks val_vel_rmse while train_vel_rmse
            # does not, the train/val gap was never about generalisation.
            world.log(
                f"           traineval rmse {traineval['vel_rmse']:6.3f}  "
                f"y {traineval['vel_rmse_y']:6.3f}  "
                f"dir {traineval['vel_dir_rmse']:5.2f} deg"
            )

        due = epoch % max(args.horizon_every, 1) == 0 or epoch == args.epochs
        if horizons and due and horizon_span is not None:
            # One rank only. The legs are a single contiguous stream, so there
            # is nothing to shard, and `model`/`frontend` here are the bare
            # modules - DDP wraps `step`, not them - so a forward under no_grad
            # runs no collective and the other ranks simply wait at the barrier
            # below.
            if world.is_main:
                row.update(
                    run_horizon_pass(
                        model,
                        frontend,
                        args,
                        span=horizon_span,
                        horizons=horizons,
                        # datasets[args.horizon_split], not datasets["train"]:
                        # in the FRACTIONAL-split mode every phase's dataset
                        # wraps the same file and this would be the same
                        # object either way, but in PRE-SPLIT mode each
                        # phase's dataset is built from a DIFFERENT directory
                        # (phase_roots[phase]), and reading "train"'s arrays
                        # while indexing them with a span sized for another
                        # split's file would score the wrong ticks of the
                        # wrong flight under that split's name - silent
                        # training-data leakage into a number reported as
                        # held out. Each dataset carries its own attitude
                        # (FixedWingVODataset.attitude), so this needs no
                        # separate ``sources`` lookup - which does not even
                        # have an entry for a non-train phase in fractional
                        # mode.
                        dataset=datasets[args.horizon_split],
                        times_s=datasets[args.horizon_split].attitude.times_s,
                        body_rate_rad_s=(
                            datasets[args.horizon_split].attitude.body_rate_rad_s
                        ),
                        device=device,
                        camera_matrix=camera_tensor,
                        baseline=baseline,
                        rotation_body_to_ned=horizon_rotation,
                        prefix=horizon_prefix,
                    )
                )
            world.barrier()

        if world.is_main:
            payload = checkpoint_payload(
                epoch, row.get(f"val_{args.select_on}", float("nan"))
            )
            torch.save(payload, last_path)
            if args.save_every > 0 and (
                epoch % args.save_every == 0 or epoch == args.epochs
            ):
                written = epochs_dir / f"epoch_{epoch:04d}.pt"
                torch.save(payload, written)
                if not announced_disk:
                    announce_epoch_checkpoint_cost(written, args, world)
                    announced_disk = True
                prune_epoch_checkpoints(epochs_dir, args.keep_last)
            if metrics is not None:
                metrics.append(row)

        if args.patience > 0 and "validation" in loaders and stale_epochs >= args.patience:
            # Every rank reduced the same validation metrics, so every rank
            # reaches this decision on the same epoch - no collective is left
            # waiting on a rank that stopped.
            world.log(
                f"early stop: val_{args.select_on} has not improved by "
                f"{args.min_delta:g} for {stale_epochs} epochs (--patience "
                f"{args.patience}); best {best:.4f} is in {best_path}"
            )
            break

    world.log(f"\nbest val_{args.select_on} {best:.4f}  ->  {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
