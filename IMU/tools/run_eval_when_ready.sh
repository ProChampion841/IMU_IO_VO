#!/bin/sh
# Wait for N completed epochs, then run the full evaluation: odometry
# (inference -> evaluate_state, AirIMU vs raw on held-out test flights) and
# covariance calibration, so accuracy and uncertainty are read together.
N=${1:-6}
PY=C:/Users/Admin/anaconda3/envs/mvvio/python.exe
while [ "$(tr '\r' '\n' < logs/train_full.log | grep -c 'train loss')" -lt "$N" ]; do
  sleep 20
done
echo "=== $N epochs complete ==="
$PY tools/summarize_training.py
echo
echo "=== inference on held-out test flights ==="
$PY inference.py --config configs/exp/UAV/codenet.conf --device cuda:0 > logs/inference.log 2>&1
echo "inference exit=$?"
tail -3 logs/inference.log
echo
echo "=== evaluate_state: AirIMU vs raw integration ==="
$PY evaluation/evaluate_state.py --dataconf configs/datasets/UAV/uav_1000.conf \
    --exp experiments/UAV/codenet --seqlen 200 --usegtrot --savedir result/uav \
    > logs/evaluate.log 2>&1
echo "evaluate exit=$?"
tail -3 logs/evaluate.log
$PY - <<'PYEOF'
import json, numpy as np, os
p="result/uav/loss_result.json"
if not os.path.isfile(p): raise SystemExit("no loss_result.json produced")
r=json.load(open(p))
if not r: raise SystemExit("loss_result.json is empty")
print("\n%-14s %12s %12s %10s"%("metric","raw","AirIMU","change"))
print("-"*52)
for k,u in [("ROE","deg"),("RTE","m"),("RVE","m/s"),("RO_RMSE","deg"),("RP_RMSE","m"),("AOE","deg"),("ATE","m")]:
    try:
        a=float(np.mean([x[k+"(raw)"] for x in r])); b=float(np.mean([x[k+"(AirIMU)"] for x in r]))
    except KeyError: continue
    print("%-14s %12.4f %12.4f %9.1f%%"%(k+" ["+u+"]",a,b,100*(b-a)/a if a else float('nan')))
print("\nmean over %d held-out test flights; R* are 2 s windows (the trustworthy ones),"%len(r))
print("A* are whole-trajectory and inherit the drift of the integrated GT position.")
PYEOF
echo
echo "=== covariance calibration ==="
$PY tools/eval_covariance.py --max_batches 8 --batch_size 16 2>&1 | grep -viE "userwarning|warnings.warn|^ *from|^loaded: data"
