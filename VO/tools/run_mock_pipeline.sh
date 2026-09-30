#!/usr/bin/env bash
# End-to-end rehearsal on mock data shaped like the real capture:
# 200 m, ~20 m/s, 20 Hz RGB frames (timestamps jittered, ~1% dropped),
# 100 Hz telemetry with the real CSV columns, a nadir camera.
#
# Runs every step of the "200 m" section of commands.txt in order - the
# checks, the split, the untrained floor, a short training run with one
# output per 500 ms, and the horizon evaluation - and stops at the first
# failure. Use it on the training machine before the real data.
#
#   bash tools/run_mock_pipeline.sh [OUT_DIR] [DEVICE] [EPOCHS]
#   bash tools/run_mock_pipeline.sh runs/mock cuda 3
#
# Defaults are CPU-sized (960x540 frames, 288x512 working size, 4 minutes);
# set IMAGE_SIZE="576 1024" and a bigger WIDTH/HEIGHT/FOCAL for full size.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${1:-runs/mock}
DEVICE=${2:-cpu}
EPOCHS=${3:-2}
WIDTH=${WIDTH:-960}; HEIGHT=${HEIGHT:-540}; FOCAL=${FOCAL:-986}
IMAGE_SIZE=${IMAGE_SIZE:-"288 512"}
DURATION=${DURATION:-240}

echo "== 1/8 mock flight -> $OUT/data"
python tools/make_synthetic_flight.py --output "$OUT/data" --duration-s "$DURATION" \
    --altitude-m 200 --speed-m-s 20 --image-width "$WIDTH" --image-height "$HEIGHT" \
    --focal-px "$FOCAL" --ground-metres-per-texel 0.2 --texture-size 4096 --turn-period-s 120

echo "== 2/8 colour frames, jittered timestamps, ~1% dropped"
python - "$OUT/data" <<'EOF'
import sys, numpy as np
from pathlib import Path
from PIL import Image
root = Path(sys.argv[1]) / "images"
rng = np.random.default_rng(1)
for i, f in enumerate(sorted(root.glob("*.jpg"), key=lambda p: int(p.stem))):
    if i > 0 and rng.random() < 0.01:
        f.unlink(); continue
    g = np.asarray(Image.open(f).convert("L"), dtype=np.float32)
    rgb = np.stack((g * 0.95 + 8, g * 1.02, g * 0.85 + 15), -1).clip(0, 255).astype(np.uint8)
    t = int(f.stem) + int(rng.integers(-4, 5))
    f.unlink()
    Image.fromarray(rgb).save(root / f"{t}.jpg", quality=92)
EOF
CAL="$OUT/data/calibration.json"

echo "== 3/8 checks: motion budget and camera mounting"
python tools/check_motion_budget.py --dataset "$OUT/data" --calibration "$CAL" \
    --image-size $IMAGE_SIZE --output "$OUT/motion_budget.json" | tail -3
python tools/estimate_camera_mounting.py --dataset "$OUT/data" --calibration "$CAL" \
    --image-size $IMAGE_SIZE --output "$OUT/camera_mounting.json" | head -4

echo "== 4/8 split"
python tools/split_dataset.py --dataset "$OUT/data" --output-root "$OUT/data_split" \
    --frame-gap 20 --deployment-latency-s 0.35 --link | tail -4

COMMON=(--dataset "$OUT/data_split/train" --validation-dataset "$OUT/data_split/validation"
        --test-dataset "$OUT/data_split/test" --calibration "$CAL" --image-size $IMAGE_SIZE
        --frontend planar --color --device "$DEVICE" --num-workers 2 --no-progress)

echo "== 5/8 untrained floor (learning rate 0)"
python tools/train_fixedwing_vo.py "${COMMON[@]}" --learning-rate 0 --epochs 1 \
    --run-dir "$OUT/runs/untrained" | grep -E "default|mounting:|epoch|best"

echo "== 6/8 train: one output per 500 ms"
python tools/train_fixedwing_vo.py "${COMMON[@]}" --frame-gap 10 --output-on-pairs \
    --photometric-augment 0.15 --dropout 0.2 --lr-warmup-epochs 1 --epochs "$EPOCHS" \
    --patience 5 --run-dir "$OUT/runs/pairs500" | grep -E "default|epoch|best|WARN|skipped"

echo "== 7/8 evaluate"
python tools/evaluate_velocity_horizons.py "$OUT/runs/pairs500/best.pt" \
    --dataset "$OUT/data_split/validation" --horizons 0.25,0.5 --device "$DEVICE" \
    --no-progress --output "$OUT/eval_pairs500.json" --plot-dir "$OUT/plots_pairs500"

echo "== 8/8 no NaN anywhere"
python - "$OUT" <<'EOF'
import csv, json, math, sys
from pathlib import Path
out = Path(sys.argv[1])
for run in ("untrained", "pairs500"):
    rows = list(csv.DictReader((out / "runs" / run / "metrics.csv").open()))
    for row in rows:
        for key in ("train_loss", "val_loss", "val_vel_rmse"):
            assert math.isfinite(float(row[key])), (run, row["epoch"], key, row[key])
    print(f"{run}: {len(rows)} epoch(s), last val_vel_rmse {float(rows[-1]['val_vel_rmse']):.3f} m/s")
report = json.loads((out / "eval_pairs500.json").read_text())
for label, entry in report["splits"]["validation"]["horizons"].items():
    if entry.get("fits"):
        assert math.isfinite(entry["vel_rmse"]), (label, entry["vel_rmse"])
        print(f"{label}: vel_rmse {entry['vel_rmse']:.3f}  dir_rmse {entry['vel_dir_rmse']:.2f} deg  "
              f"pos_error_final {entry.get('pos_error_final', float('nan')):.2f} m")
print("MOCK PIPELINE OK")
EOF
