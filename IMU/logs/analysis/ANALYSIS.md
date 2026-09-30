# IMU `logs/` analysis — `tilt_aware` run

Source files: `logs/parameters.yaml`, `logs/metric.csv` (1122 epochs, 0–1121),
`logs/tensorboard/events.out.tfevents.*` (39 scalars, 2026-09-27 04:05 → 2026-09-30 16:25,
about 4.5 min per epoch). Plot: `tilt_aware_training.png`.

## 1. What the run is

- **Config:** `configs/exp/UAV/tilt_aware.conf`: `hybridnet` (GRU 3 s + Mamba ~20 s), additive
  accel correction, `gtrot: True` (NavEul attitude), **new feature: `att_input: gravity`**
  (body-frame gravity direction, in_dim 6 → 9), `const_correction: False`, `cov_model: bias`, seed 0.
- **Splits:** 55 train flights plus `train.csv`. `test` and `eval` use **the same 11 flights**
  plus `valid1.csv`; they are the checkpoint-selection flights, **not held-out data**.
  `inference` has the 11 held-out flights, and it **is not scored anywhere in this log.**
- **Horizons:** **training windows are 40 s** (4000 frames, step 1000 s per `parameters.yaml`). Primary
  validation is **60 s**, because `metric_horizons[0]` = 6000 overrides the test window; the extra val
  horizons are 30 s and 120 s. The `eval/*` columns use 40 s windows.
  *Correction:* an earlier version of this note said training ran at 60 s. The CSV column
  `train_pos_error_60s` is **mislabelled**: train.py names it after `metric_horizons[0]`, but the
  training windows were 40 s.
  The checkpoint is selected on a 3-epoch trailing mean of `val_pos_error_60s`.

## 2. Headline: this arm learns, and the earlier arms did not

The model/raw ratio uses the same windows on both sides, so it is valid (see §4 for why the
absolute values are not):

| epochs | LR | val 30 s | val 60 s | val 120 s | train (40 s windows) |
|---|---|---|---|---|---|
| 0–9 | 1e-3 | 0.978 ± 0.047 | 0.965 ± 0.044 | 0.949 ± 0.053 | 0.949 |
| 10–49 | 1e-3 | 0.961 ± 0.020 | 0.941 ± 0.018 | 0.906 ± 0.023 | 0.904 |
| 50–125 | 1e-3 | 0.973 ± 0.050 | 0.959 ± 0.053 | 0.918 ± 0.066 | 0.850 |
| 126–308 | 3e-4 → 1e-5 | 0.957 ± 0.008 | 0.930 ± 0.009 | 0.886 ± 0.010 | 0.771 |
| 309–599 | 1e-5 (floor) | 0.958 ± 0.004 | 0.928 ± 0.005 | 0.881 ± 0.005 | 0.753 |
| 900–1121 | 1e-5 | 0.959 ± 0.005 | 0.929 ± 0.005 | **0.871 ± 0.006** | 0.746 |

- **Best single epochs:** 30 s 0.925 (ep 120), 60 s 0.906 (ep 127), 120 s 0.856 (ep 932).
  Endpoint velocity at 60 s reaches 0.892 (ep 209).
- **Selected `best_model.ckpt` (trailing mean-3 on 60 s):** epoch 129, with 30 s 0.950,
  60 s 0.922 and 120 s 0.882.
- The model beats the raw integration in 1106 of 1122 epochs at all three horizons. The 16
  exceptions are all spikes at epochs 9–125, while the LR was 1e-3.
- **Pre-registered criterion (c) holds on val:** the best epochs are 120, 127 and 932. The four
  earlier arms all peaked at epochs 1–4 and then got worse than 1.0. This is the first arm on
  this project whose validation score improves with training.
- The `eval/pos_error` column (every 10 epochs, eval split) agrees: 73.5 m → 64.7 m (−12 %).

**Caveat:** the 30 s and 60 s numbers (0.958, 0.929) look well below the held-out
constant-bias bar (0.9994 / 0.9987). They come from a different split, so they cannot be compared.
Only `--splits inference --per_flight` can decide pass or fail (§6).

## 3. Training dynamics

- **The generalisation gap is large and still growing.** Train (40 s windows) goes 0.95 → 0.75; val at 60 s
  goes 0.96 → 0.93. The horizons differ (40 s vs 60 s), so the gap is indicative, not exact. From epoch 309 (LR floor) onward, val 30 s and 60 s are flat (slope ≈ +0.0001 per 100 epochs), but
  **val 120 s is still improving by −0.0018 per 100 epochs** and train by −0.0013. The run
  still fits the train set. Only the 120 s horizon, 3× the 40 s training window, keeps
  benefiting.
- **LR schedule:** four cuts at epochs 126, 187, 248 and 309. The floor has been reached since epoch 309, and
  about 800 epochs have run there. The cut at epoch 126 is what stabilised val: the ±0.05 spikes stop and
  the sd drops to about 0.005. The optimiser was too noisy at 1e-3, so a lower starting LR or warm-down
  would likely have reached the same place in far fewer epochs.
- **The budget does not match the config comments.** The header says "200 → 40 epochs, patience
  12 → 6", but the file sets `max_epoches: 4000, patience: 60`. The run has no early stop, and at
  about 4.5 min per epoch it needs about 11 more days to finish.
- **The covariance head peaked early:** `val_cov_loss` is lowest at epoch 42 (0.2135) and has sat at about 0.2187 since.
  The velocity NLL behaves the same way (7.17 at epoch 30, about 7.9 now). The correction keeps improving while
  the uncertainty calibration slowly degrades. The checkpoint picked for position is therefore not the
  best one for covariance.
- **Per-step (relative) velocity got slightly worse:** `val/vel_rel_error_mps` rose from 0.955 to 0.993
  (raw is 0.955). Endpoint drift improved (−9 % at 60 s), but the 0.5 s increments are about 4 % noisier.
  If a VIO front-end consumes short-horizon increments, it will see this.

## 4. Bugs found in how the logged numbers are computed (`train.py`)

Neither bug changes the model/raw **ratios**, because both arms are scaled by the same factor. Both
make every **absolute** value in `metric.csv` and TensorBoard wrong.

1. **`train()` never increments `n_win`** (`train.py:589`, divided by at `:645–655`).
   Everything is divided by `max(1, 0) = 1`, so every `train_*` column is a **sum over
   batches**, not a mean. That is why `train_pos_error_60s` (actually 40 s windows) is about 33 000 m while val is about 700 m, and
   why `train_loss` is about 37–47 while `val_loss` is about 0.68. It also affects `train_cov_*` and `pred_cov_*`.
2. **`test()` weights by batch size but divides by batch count.** It accumulates `x * bs` and
   then divides by `(i+1)` (`train.py:727–735`) instead of `n_win`. That multiplies every val
   column by the mean batch size. `test_loader` uses `batch_size = per_rank_bs = 6`, not
   `eval_batch_size`, and the 120 s loader uses 6 // 2 = 3. So:
   - `val_*_30s` and `val_*_60s` are **about 6× too large**, and `val_*_120s` is **about 3× too large**.
   - Corrected raw baselines: 217.1 / 6 ≈ **36.2 m** (30 s), 738.1 / 6 ≈ **123.0 m** (60 s),
     1074.8 / 3 ≈ **358.3 m** (120 s). These match the numbers in `UPDATES_2026-09-07/08.txt`
     (about 36.5 / 123.9 / 357–386 m), which likely also explains the open "imu_metric.csv baseline" question.
   - Side effect: in the CSV, 120 s appears only 1.46× worse than 60 s. In reality it is about 2.9× worse.
   - The window-weighting fix from 2026-09-08 was only half applied: it added `* bs` but kept `/(i+1)`.

   The fix is to return `v / n_win` in both functions and add `n_win += bs` in `train()`. Old CSVs
   stay comparable by ratio only.

## 5. Smaller observations

- `val_raw_*` is constant across all epochs, so the val set is deterministic and ratios are cleanly paired.
  `train_raw_*` varies (sd ≈ 55 on about 33 300), which is expected from `aug_*_bias_std` augmentation and shuffling.
- `ReduceLROnPlateau` steps on `val_loss`, while checkpoint selection uses `pos_error_60s`.
  Both hit their minimum early (loss at epoch 64, pos at 60 s at epoch 127). That is fine here, but they are two different signals.
- `log_raw_baseline` reruns the raw integrator on every train batch every epoch. That cost is paid about 1100 times for a
  number that never changes on val and is only noise on train.

## 6. Recommended next steps

1. **Run the pre-registered test now.** Nothing more will be learned from val 30 s and 60 s.
   ```
   python -m tools.eval_vel_horizons --config configs/exp/UAV/tilt_aware.conf \
     --ckpt experiments/UAV/tilt_aware/ckpt/best_model.ckpt \
     --splits inference --nested --horizons 3000 6000 12000 --per_flight
   python -m tools.const_bias_control --config configs/exp/UAV/tilt_aware.conf \
     --ckpt experiments/UAV/tilt_aware/ckpt/best_model.ckpt \
     --fit_split train --score_split inference --horizons 3000 6000 12000
   ```
   Pass condition: pooled 120 s < 0.9585, the LOFO range is entirely < 1.0, and flight `143_23` is checked separately.
   Also score a **late checkpoint (for example ep ~930 or the latest)**. The 120 s result keeps improving after epoch 129,
   and selecting on 60 s may have left 120 s performance behind.
2. **Stop or shorten the run.** LR has been at the floor for about 800 epochs, 30 s and 60 s are flat, and the train/val gap
   keeps widening. Set `max_epoches` and `patience` to what the header intends.
3. **Fix the two normalisation bugs (§4)** so absolute metres in logs are real.
4. **Rerun with seed 1 or 2** to confirm the effect is larger than seed spread. The gap from the earlier arms (0.99 → 0.93 at 60 s on val)
   is much larger than the ≈0.005 epoch-to-epoch sd, but only one seed exists.
5. **Ablate the second change.** The arm changed both `att_input` and `cov_model: bias`. Run `cov_model: propagated`
   to attribute the gain to the gravity input.
