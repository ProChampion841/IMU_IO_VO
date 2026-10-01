# Fixed-Wing Visual Odometry

A causal, fully learned monocular visual-odometry estimator for fixed-wing
aircraft. There is no IMU, no ESKF, no factor graph, no RANSAC pose backend:
the aircraft's own attitude solution (`NavEul*`) supplies the rotation that
de-rotates the image motion, its altitude supplies the scale that turns
angular motion into metres per second, and a causal Mamba fusion stack turns
that plus a stream of image pairs into body velocity, emitted every
telemetry tick.

```text
attitude (roll, pitch), body rates p/q/r, altitude, dt  ──┐
                                                            ├─► causal fusion Mamba ──► body velocity xyz
image pair ──► VisionMamba stem ──► rotation-centred       │
               local correlation ──► visual token ─────────┘
               (arrives late, at its measured deployment time)
```

Three geometry choices, not architecture, are what keep the lateral axis from
being lost (see the module docstring of
[`src/vio/models/vision_mamba_vo.py`](src/vio/models/vision_mamba_vo.py) for
the full reasoning):

- The predicted rotational displacement **centres the correlation search
  window** before matching, instead of being learned as a bias afterwards -
  a yaw rate that would otherwise look like several m/s of lateral drift.
- Altitude is **added in log space** to a predicted log bearing rate
  (`v = h * u`), making the scale relation exact rather than something a
  linear layer has to discover from a concatenated channel.
- Direction and speed are **predicted as separate heads**, because they have
  different observability (direction is recoverable from one frame pair,
  speed is not) and a scale error must not corrupt the crab angle.

This is a **research estimator**. Do not use it as an aircraft's only
navigation source.

## Layout

```text
src/vio/
  data/     flight loading, attitude/pose reference, image timing, calibration
  models/   the causal Mamba stem/fusion, local correlation, velocity horizons
tools/      runnable scripts (see below)
tests/      run with pytest
configs/vo/ camera calibration manifest
```

The package sits under `src/` and is importable once installed:

```bash
python -m pip install -e ".[dev]"
```

Every script under `tools/` also puts `src/` on `sys.path` itself, so a bare
checkout runs the tools without the install step; the install is only needed
to `import vio` from elsewhere (a notebook, a different tool).

`data/`, `data_split/`, `artifacts/`, and `runs/` are generated or supplied
and are not tracked by git.

## Dataset format

A dataset is one folder holding one telemetry file and one image folder:

```text
<dataset>/flight.csv        telemetry, one row per tick
<dataset>/images/<ms>.jpg   frames named by capture time, in milliseconds
```

`flight.csv` needs a time column (`Time` by default), the attitude columns
the estimator consumes (`NavEulX/Y/Z`, roll/pitch/yaw), an altitude column
(`relativeAlt`, `RelatedAlt`, or the legacy `Barometer` field - autodetected),
and the **target** the model is trained against: `GPSNavVnX/Y/Z` (velocity)
rotated by `GPSNavEulX/Y/Z` (attitude). The two `GPSNav*` families are refused
as *inputs* - they define the label the model is scored against, so feeding
them in would be leaking the answer.

The split is chronological only (see [Splitting](#splitting) below) - there
is no condition-segment or multi-sequence manifest support in this tree.

## Calibration

`configs/vo/camera_fixedwing.json` is the calibration for the real camera.
Only its `camera` block is read (intrinsics, distortion, whether the images
on disk are already rectified) - a file may carry other blocks and they are
ignored, so the same file can also describe IMU fields another pipeline
wants without this one caring.

Before trusting a real-data run, confirm three things in it:

1. **`images_rectified`** - it currently reads `false` against unrectified
   1920x1080 source images, so the loader is doing the lens-distortion
   correction itself on every load. If the images you are pointing at have
   *already* been rectified upstream, this must flip to `true` or the
   correction is applied twice.
2. **`mounting.gps_to_camera_m`** - currently `[0, 0, 0]` (body axes: x
   forward, y right, z down). GPS measures velocity at the antenna, the
   camera is what the model actually sees the motion from, and a rigid body
   in a turn moves those two points at different velocities
   (`v_camera = v_gps + omega x r`) - at 30 deg/s and a 2 m lever arm that is
   about 1 m/s of avoidable error. **Measure the real offset on the airframe
   and fill it in** (or pass `--lever-arm X Y Z` to override it) rather than
   leaving it at zero.
3. **`--deployment-latency-s`** (trainer/evaluator flag, default `0.35`) -
   confirm this matches how long it actually takes a captured image to
   become available downstream on the real system. It sets how stale a
   delivered image is treated as (`visual_age` starts at this value, not at
   zero, the instant an image arrives) and how a delivered image's
   attitude-at-exposure is read.

## Before training on real data

Run these once per capture, in order, before splitting or training. Each
one writes a JSON report under `artifacts/` worth reading, not just running.

```bash
# 1. camera clock offset - the camera and telemetry clocks are not the same
#    clock, and this is not visible in the timestamps alone
python tools/estimate_time_offset.py \
    --dataset data --output artifacts/time_offset.json

# 2. can these images even support a visual result, at the offset just found?
python tools/check_dataset_sync.py \
    --dataset data --image-time-offset <MEASURED_OFFSET_SECONDS> \
    --frame-gap 1 --output artifacts/sync.json

# 3. is the velocity target actually independent of what the model reads,
#    or does it secretly come from a navigation solution that already saw it?
python tools/audit_velocity_provenance.py \
    --dataset data --output artifacts/provenance.json
```

`check_dataset_sync.py` separates three things that are easy to confuse:
whether the clocks overlap, whether image motion follows the attitude
(consistent focal length on both axes), and whether fine detail survives
between frames far enough apart to correlate - images can be perfectly
synchronised and still carry nothing a matcher can use.

## Splitting

```bash
python tools/split_dataset.py \
    --dataset data --output-root data_split \
    --image-time-offset <MEASURED_OFFSET_SECONDS> \
    --frame-gap 1 --deployment-latency-s 0.35 \
    --link
```

Chronological only: `--train-fraction`/`--validation-fraction` (default
0.6/0.2, remainder is test) cut the flight from the start, with a
`--gap-ticks` guard band between splits so a training window and a
validation window never end and begin a fraction of a second apart. `--link`
hard-links the images instead of copying them (same bytes on disk once, not
twice - the source must be on the same filesystem). Pass the *same*
`--image-time-offset`, `--frame-gap` and `--deployment-latency-s` here that
training will use, since they change how many ticks near each split boundary
are actually usable.

A chronological split means train/validation/test share terrain, weather,
airframe trim and sensor bias - useful for verifying the pipeline and for
A/B comparisons where both arms see the identical split, but **not** a
generalisation number.

## Training

```bash
python tools/train_fixedwing_vo.py \
    --dataset data_split/train --validation-dataset data_split/validation \
    --test-dataset data_split/test --calibration configs/vo/camera_fixedwing.json \
    --rotation-mode field --run-dir runs/vo_field_s0 \
    --device 0,1,2,3,4,5,6,7,8,9 --num-workers 2 --lr-scaling sqrt \
    --epochs 300 --seed 0 --select-on vel_rmse
```

The runnable version of this, staged as smoke test -> real train -> evaluate,
is [`commands.txt`](commands.txt).

**Multi-GPU is one node, self-launching.** A comma-separated `--device`
(`0,1,2,...`) re-execs the script under `torchrun --standalone` itself, one
rank per named GPU (`RANK`/`WORLD_SIZE`/`LOCAL_RANK` are how it knows it is
already inside that job on the second entry) - there is no separate
`torchrun` invocation to remember, and no multi-node support in this tree.
`--lr-scaling sqrt` (or `linear`) follows the optimizer's effective batch as
it grows with the number of ranks; the start banner prints `world=…` and
`effective_batch=…` to confirm it did what you intended.

**Early stopping is opt-in.** `--epochs` is the budget; pick it from
*optimiser steps* (windows / (batch x ranks) per epoch), not wall clock -
300 epochs at ten ranks on a short split can be a few hundred steps, which is
a convergence pilot, not a trained model. `--patience N` stops a run whose
`--select-on` metric has not improved by `--min-delta` for N epochs (the count
survives a resume); `best.pt` is the selected checkpoint either way. Watch
`runs/<name>/metrics.csv`. `--lr-warmup-epochs N` ramps the learning rate in
over the first N epochs - worth it with `--lr-scaling linear` over many ranks,
where the first epochs otherwise take very large steps.

**Resuming** re-runs the *original* command with `--resume <path>` added
(`last.pt` is written after every completed epoch). The dataset, the input
contract and the model shape are fingerprinted into the checkpoint, and a
resume that disagrees with any of them is refused, naming the field that
moved - continuing one run's optimizer state on another run's problem is
worse than starting over. `--test-dataset` must stay identical across a
resume for the same reason: dropping it changes the fingerprinted split
ranges.

### Ablations

Three flags zero part of the model's input without changing its shape, so a
checkpoint trained under one setting cannot silently be evaluated under
another - each is recorded in the resume fingerprint and the checkpoint
itself, and the evaluator reproduces it automatically from a checkpoint
rather than needing to be told:

- `--disable-visual-input` - zero the visual token entirely. The
  attitude-and-altitude-only floor; anything that does not beat it is not
  using the camera.
- `--ablate-body-rate` - zero the aiding vector's p/q/r channels only.
- `--ablate-visual-age` - zero the fusion input's visual-age channel only.

### What a run writes

| file | when |
|---|---|
| `best.pt` | whenever the `--select-on` validation metric improves |
| `last.pt` | after every completed epoch, overwritten |
| `epochs/epoch_0042.pt` | every `--save-every` epochs, kept up to `--keep-last` |

Every one is a complete checkpoint (weights + optimizer state), so any of
them can be resumed from or scored directly.

### TBPTT: scaffolded, not wired in

`ChronologicalWindowSampler`, `VOStreamState`/`mask_stream_state`, and
`VOStep.forward_stream` exist and are tested (see
[`PLAN_TBPTT.txt`](PLAN_TBPTT.txt)), but there is **no `--tbptt` flag**. The
production epoch loop above always resets state at every window. Truncated
backprop through time is explicitly **not** DDP-safe yet - it calls a method
that bypasses `DistributedDataParallel`'s gradient-sync hooks - so do not
attempt to enable it in a multi-GPU run; it is a later, separate piece of
work.

## High altitude: the planar frontend

At a few hundred metres one frame of separation moves the ground less than
one correlation cell (0.66 cells at 200 m, 20 m/s, 20 Hz, 1024 px wide with
f = 1052 px), so the per-pair speed is a sub-cell measurement and is at the
mercy of matching noise; a model trained on it memorises the flight instead
of measuring it. The remedy is a longer baseline - but a one-second baseline
brings ten-plus degrees of rotation between the two frames, which the
learned small-angle rotation field cannot remove. `--frontend planar` is built
for this regime:

1. the second frame is **warped by the exact interframe rotation** from the
   attitude log (not a linearised field), so what remains is translation;
2. a **coarse search, voted on by the whole image**, finds the ground motion
   around a nominal velocity (`--prior-velocity`, default the training mean),
   on features with their local mean removed (`--coarse-highpass`,
   `--fine-highpass`) so a large field or forest cannot pose as a match;
3. a **fine per-cell search** around the motion that estimate predicts,
   then a **robust least-squares fit of the camera translation over a flat
   ground plane**, with the ground tilt from roll/pitch and the altimeter
   pinning the vertical component (`--altitude-constraint`);
4. the output is a **metric velocity per pair**, and the default
   `--velocity-mode geometric_residual` makes the model's prediction that
   velocity (held between pairs) plus a learned correction that starts at
   zero - an untrained model already outputs the closed-form estimate.

On a flight rendered by `tools/make_synthetic_flight.py` at 200 m, 20 m/s,
20 Hz and 1024x576 (f = 1052 px) with S-turns to 25 deg of bank, the
**untrained** planar frontend measures velocity to 0.20 m/s RMS per pair at a
one-second baseline and 0.33 m/s at half a second; the model it replaces
reports 2.6-3.7 m/s per axis on the real flight. The real flight will be
harder (texture, lens residual, clock offset, rolling shutter), but those
are the numbers to beat.

Before the first planar run, once per capture:

```bash
# how far the ground moves per --frame-gap, and which gap to use
python tools/check_motion_budget.py --dataset data \
    --calibration configs/vo/camera_fixedwing.json --output artifacts/motion_budget.json

# which way the camera points on the airframe (and a scale check that
# RelativeAlt, the focal length and the clock agree)
python tools/estimate_camera_mounting.py --dataset data \
    --calibration configs/vo/camera_fixedwing.json --output artifacts/camera_mounting.json
```

`estimate_camera_mounting.py` prints a `mounting.camera_from_body` block to
paste into the calibration file (or pass `--camera-mounting
artifacts/camera_mounting.json`). A wrong mounting is not a small error - it
swaps or negates the forward and lateral axes - so do not guess it. Its
"measured / predicted motion" ratio should be within a few percent of 1.0;
if it is not, fix the altitude column, the focal length or the image time
offset before training.

`--frontend planar` needs no other flag: it sets `--frame-gap` to the frames
in `--planar-baseline-s` (1.0 s: 20 at 20 Hz), caps the pair interval at 1.5x
that, sets `--warmup` past the first pair's arrival, uses radius 3 and the
`simple` loss, and - when neither `--camera-mounting` nor the calibration
gives a mounting - measures the mounting from the training images at
start-up. Every value it chose is printed and recorded in the checkpoint;
any flag given explicitly wins. The commands are the "200 m" section of
[`commands.txt`](commands.txt). `--color` loads RGB frames for a colour
camera, and `--photometric-augment 0.15` jitters exposure between the two
frames of a pair during training, as an auto-exposure camera does.

## Evaluating

```bash
python tools/evaluate_velocity_horizons.py runs/vo_field_s0/best.pt \
    --dataset data_split/validation --calibration configs/vo/camera_fixedwing.json \
    --device cuda:0 --output artifacts/eval_field.json --plot-dir artifacts/plots_field
```

Scores velocity and dead-reckoned position error over `--horizons` (default
1/5/10/15/20/30 minutes): the split is streamed once from its first tick to
its last with state zeroed only at the start, and each horizon is the first
H minutes of that one run - so it measures what a short training window
cannot, whether the recurrent state accumulates error over a long flight.
`--splits full` scores the whole capture and is a drift diagnostic, not a
held-out number. `--plots` (on by default) writes a dead-reckoned trajectory
figure, a per-axis velocity-vs-truth figure, an error-growth figure, and a
summary against horizon length. `--stratify` (on by default) also breaks the
whole-split error down by turn rate, bank angle, altitude and ground speed,
so a model that is good in cruise and poor in turns says so.

Run it three ways per checkpoint before trusting a result: on `validation`
while iterating, on `validation --disable-visual-input` as the visual-blind
floor, and on `--splits test` exactly once, at the end.

## Exporting to ONNX

```bash
pip install onnx onnxruntime
python tools/export_onnx.py runs/vo_planar_500ms/best.pt --output-dir export/onnx
python tools/onnx_inference.py export/onnx --dataset data_split/test \
    --checkpoint runs/vo_planar_500ms/best.pt --output artifacts/onnx_velocity_test.csv
```

Two graphs, because the model runs at two rates:

* `frontend.onnx` - once per image pair: both frames at the working size
  (float 0-1, `(1, C, H, W)`), `pair_dt_s`, and the pair geometry
  (`relative_rotation`, `down_body`, `altitude_m`). The working-resolution
  intrinsics are baked in. Returns the visual token, quality, whether the pair
  is delivered (`pair_reliable`) and the per-pair metric velocity.
* `temporal_step.onnx` - once per telemetry tick: the aiding vector, the
  current visual inputs and the carried recurrent state in; the velocity
  (m/s, body frame) and the next state out. Batch is dynamic.

`vo_onnx.json` records everything a runtime must reproduce: image
preprocessing, intrinsics, the aiding channels and normaliser, the per-tick
rules (presence, visual age, held velocity, state feedback), timing, and the
measured ONNX-vs-PyTorch difference. The export fails if any output differs
from PyTorch by more than `--tolerance` (1e-3).

`tools/onnx_inference.py` is the deployable runtime: numpy, onnxruntime and
Pillow only (OpenCV only for lens distortion), no torch, no project code.
Copy it with the export folder. `VOOnnxRuntime.add_frame(image, t)` takes
every camera frame, `add_telemetry(t, roll, pitch, yaw, rel_alt)` every
telemetry row and returns that tick's velocity; `emitted` is True on the
ticks that carry a new output (one per pair with `--output-on-pairs`). It
reproduces the training rules exactly - pairing, delivery one latency after
the second exposure, attitude interpolated at both exposures, refused pairs
not delivered, held velocity, visual age. Given `--dataset` it replays a
flight folder through the same class; `--checkpoint` also runs the PyTorch
model (via `tools/onnx_reference.py`) and fails unless both deliver pairs on
the same ticks and agree within `--tolerance` (0.05 m/s).

ONNX has no matrix inverse and cannot trace the adaptive pooling here, so
during export only the 3x3 solves use the closed-form adjugate and adaptive
pooling uses exact averaging matrices; both are covered by
`tests/test_onnx_export.py`.

### Inference time against the real-time budget

```bash
python tools/benchmark_inference.py --checkpoint runs/vo_planar_s0/best.pt --threads 1 4
python tools/benchmark_inference.py --frontend both          # untrained, trainer defaults
python tools/benchmark_inference.py --frontend planar --frame-gap 10 --output-on-pairs
```

Prints both graphs' inputs and outputs, parameters per component, and the
median/p90 latency of the frontend (per pair) and the temporal step (per
tick), in PyTorch and in onnxruntime. It also times the runtime's per-frame
JPEG decode, undistort and resize. It then sets these against the trained
rates: frontend duty (pairs/s x (2 x preprocess + frontend), because the
runtime preprocesses both frames of every pair), tick duty (telemetry Hz x
step), and the latency margin against `--deployment-latency-s`, which a pair
must beat or its token arrives later than the model was trained to expect.
Timing does not depend on the weights, so the untrained mode is enough to
size hardware. Run it on the target (`--providers CUDAExecutionProvider
CPUExecutionProvider` for a GPU): numbers from another machine do not
transfer.

## GPU gate, before spending server time on a full run

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.device_count())"
python -m pytest -q
python -m pytest tests/test_vision_mamba_vo.py::test_mask_stream_state_moves_the_mask_to_the_state_devices_own_device -q
python tools/benchmark_correlation.py --device cuda:0 --warmup 10 --iterations 30
```

The CUDA-specific test above is skip-marked in every CI run in this
repository so far (no GPU there) - on the server it must actually **pass**,
not skip. The benchmark should fit comfortably in GPU memory and gives a
real forward/forward+backward timing at the trainer's actual resolution,
unlike any number quoted from a CPU run.

## Smoke test

Two GPUs, two epochs, a few minutes - staged as step 0 in
[`commands.txt`](commands.txt). Proceed to the real pilot in step 1 only if
**all** of the following hold:

- Both ranks initialised on different GPUs (check the start banner).
- Losses and gradients stayed finite the whole run.
- `runs/smoke_field_ddp2/last.pt` and `best.pt` were both produced.
- Resume works (rerun with `--resume` added; it continues, it doesn't
  restart from epoch 0).
- GPU memory has comfortable headroom, not a near-OOM margin.
- The sync/provenance reports from [Before training](#before-training-on-real-data)
  are credible, not just "ran without crashing."

For the first full pilot, keep `--num-workers 2` (not higher - ten GPUs at 8
workers each is 80 loader processes) and run only `--rotation-mode field`,
seed 0. Treat 300 epochs as a convergence pilot, not a finished budget. Do
not launch the full field/constant x 3-seed comparison matrix, or the
`--ablate-body-rate`/`--ablate-visual-age` arms, until that pilot and its
visual-blind evaluation look right - each of those is a `--rotation-mode`,
`--seed`, or ablation flag away from the same command in `commands.txt`,
run one at a time once the single pilot is trusted.

## Tests

```bash
python -m pytest tests -q
```

## Attribution

This repository began as a fork of
[MzeroMiko/VMamba](https://github.com/MzeroMiko/VMamba) (MIT). Its image
classification, detection and segmentation pipelines and the VMamba backbone
have been removed; the causal Mamba stem and fusion stack here are this
project's own. See the git history to recover any of the removed VMamba
code.
