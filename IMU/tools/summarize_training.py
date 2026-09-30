"""Summarise per-epoch progress from a train.py log.

The composite `loss` is a poor progress signal because it includes the Gaussian
NLL covariance term, which is unbounded below -- it can fall while the state
errors get worse.  The per-channel position/rotation/velocity errors are the
numbers to read.

Note: tqdm rewrites its bar every iteration, so each epoch contributes hundreds
of "training epoch:"/"testing losses:" lines carrying a *running* mean.  Only the
last line before each "train loss:" summary is the epoch's final value; taking
matches in order instead silently reports the first few iterations of epoch 0.
"""
import argparse, io, re

ap = argparse.ArgumentParser()
ap.add_argument("--log", default="logs/train_full.log")
a = ap.parse_args()

txt = io.open(a.log, encoding="utf-8", errors="replace").read().replace("\r", "\n")
# split into per-epoch blocks at the epoch summary line
parts = re.split(r"train loss: ([-\d.]+) test loss: ([-\d.]+)", txt)
TR = re.compile(r"training epoch: (\d+), losses: [-\d.]+, position, ([\d.]+) rotation ([\d.]+)")
TE = re.compile(r"testing losses: [-\d.]+, position, ([\d.]+) rotation ([\d.]+), vel ([\d.]+)")
EV = re.compile(r"eval pos: ([\d.]+) eval rot: ([\d.]+)")

print("ep |  train_pos  train_rot |   test_pos   test_rot   test_vel |   eval_pos  eval_rot")
print("-" * 88)
best = None
for i in range((len(parts) - 1) // 3):
    block = parts[i * 3]                      # text preceding this epoch's summary
    tl, sl = parts[i * 3 + 1], parts[i * 3 + 2]
    trm = TR.findall(block)
    tem = TE.findall(block)
    evm = EV.findall(parts[i * 3 + 3] if i * 3 + 3 < len(parts) else "")
    tp, tr_ = (float(trm[-1][1]), float(trm[-1][2])) if trm else (float("nan"),) * 2
    sp, sr, sv = (map(float, tem[-1]) if tem else (float("nan"),) * 3)
    ep_, er = (map(float, evm[-1]) if evm else (float("nan"),) * 2)
    star = ""
    if best is None or float(sl) < best:
        best, star = float(sl), " *"
    print("%2d | %10.3f %10.4f | %10.3f %10.4f %10.4f | %10.3f %9.4f%s"
          % (i, tp, tr_, sp, sr, sv, ep_, er, star))
print("\n* = new best test loss (drives checkpointing).")
print("raw-integration baseline, final frame of a 10 s window: pos ~11 m, rot ~0.037 rad (2.1 deg)")
