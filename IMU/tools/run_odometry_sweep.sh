#!/bin/sh
# Odometry-relevant evaluation sweep.
#
# Everything reported on this project so far used --usegtrot, which feeds GROUND-TRUTH
# rotation into the integrator for gravity compensation (utils/integrate.py:35).  That is
# the AirIMU paper's convention and is reasonable for sensor fusion, where attitude comes
# from elsewhere.  It is NOT reasonable for dead reckoning: it hides attitude drift, which
# is the dominant long-horizon error source, and it means an accelerometer-only network
# (codeposenet zeroes the gyro correction) is never penalised for leaving rotation alone.
#
# This sweep therefore crosses two axes:
#   * horizon  -- 2 s / 10 s / 20 s relative windows, plus whole-trajectory ATE/AOE
#   * gravity  -- gt rotation (fusion-like) vs free-running attitude (true dead reckoning)
#
# Results land in result/<model>_h<seqlen>_<gt|free>/loss_result.json.
set -u
PY=C:/Users/Admin/anaconda3/envs/mvvio/python.exe
DC=configs/datasets/UAV/uav_1000.conf
MAXJOBS=${MAXJOBS:-4}

run_one() {
    model=$1; seqlen=$2; mode=$3
    out="result/${model}_h${seqlen}_${mode}"
    if [ -f "$out/loss_result.json" ]; then
        echo "SKIP  $out (exists)"; return 0
    fi
    if [ "$mode" = "gt" ]; then gt="--usegtrot"; else gt=""; fi
    mkdir -p "$out"
    # shellcheck disable=SC2086
    $PY evaluation/evaluate_state.py --dataconf "$DC" \
        --exp "experiments/UAV/$model" --seqlen "$seqlen" $gt \
        --savedir "$out" > "$out/eval.log" 2>&1
    if [ -f "$out/loss_result.json" ]; then
        echo "OK    $out"
    else
        echo "FAIL  $out (see $out/eval.log)"; tail -5 "$out/eval.log"
    fi
}

MODELS=${MODELS:-"const3 const6 codeposenet_v3 codeposenet_v5"}
SEQLENS=${SEQLENS:-"200 1000 2000"}

for model in $MODELS; do
    for seqlen in $SEQLENS; do
        for mode in gt free; do
            run_one "$model" "$seqlen" "$mode" &
            while [ "$(jobs -r | wc -l)" -ge "$MAXJOBS" ]; do wait -n 2>/dev/null || sleep 2; done
        done
    done
done
wait
echo "=== sweep complete ==="
