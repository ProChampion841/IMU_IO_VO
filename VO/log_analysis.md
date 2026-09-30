---
title: Log Analysis of Validation Stagnation
---

## Potential Causes Identified

| # | Hypothesis | Why it is logically plausible |
|---|------------|--------------------------------|
| 1 | **Incorrect `log_altitude` input** | `VisionMambaVO.forward` reconstructs velocity as `v = h * u` where `h` must be the *raw* log‑altitude.  If the validation pipeline feeds the *centered* encoder output instead, `h` becomes near‑zero, drastically under‑estimating speed and inflating `val_vel_rmse` and the cosine‑direction error. |
| 2 | **Mismatched `rotation_mode`** | The model is trained with `rotation_mode='field'` (cell‑wise rotation map).  Validation defaults to `rotation_mode='constant'`, which applies a single rotation to the whole image.  This mis‑alignment corrupts the cost‑volume, leading to poor velocity and direction predictions. |
| 3 | **TBPTT stream‑state leakage** | `mask_stream_state` and `detach_stream_state` must clear hidden states at sequence boundaries.  If validation does not reset the state, hidden information from previous batches either leaks (data‑leakage) or remains stale, breaking causality and causing a validation‑only performance gap. |
| 4 | **Training‑validation data distribution mismatch** | The validation set contains log‑altitude / speed ranges that are under‑represented in training (e.g., very high altitude or speed).  The model has never seen those regimes, so RMSE remains high on validation while training loss continues to drop. |
| 5 | **Fixed learning‑rate plateau** | A constant learning‑rate (e.g., 1e‑3) can lead to a loss plateau after the initial descent.  Without a schedule that reduces the LR, the optimizer cannot make the fine‑grained updates needed to improve validation metrics. |

## How to Verify Each Hypothesis

### 1. Incorrect `log_altitude` Input
1. **Add logging** in `run_validation.py` (or the validation loop) to print the values passed to `VisionMambaVO.forward`:
   ```python
   logger.info(f"log_altitude_raw={log_altitude_raw.shape}, log_altitude_centered={log_altitude_centered.shape}")
   ```
2. **Run a single validation batch** twice:
   * a) Pass the raw `log_altitude`.
   * b) Pass the centered version instead.
3. Compare the resulting velocity tensor `v`.  The raw version should yield speeds in the expected range (10‑30 m/s), while the centered version will produce near‑zero speeds.
4. Compute `val_vel_rmse` for the two runs.  A large drop when using the raw altitude confirms the hypothesis.

### 2. Mismatched `rotation_mode`
1. **Print the argument** used when constructing `VisionMambaVO` inside the validation script.
2. **Execute the same validation batch** with `rotation_mode='field'` and then with the default `'constant'`.
3. Record `val_vel_rmse` and `val_vel_cosine` for both runs.
4. Optionally visualise the rotation map returned by `RotationalSearchField` (e.g., using `matplotlib.imshow`).  The `'field'` mode should show spatially varying patterns, while `'constant'` will be uniform.  Improvement in metrics when using `'field'` validates the cause.

### 3. TBPTT State Leakage
1. Insert a log line before each batch:
   ```python
   logger.info(f"State norm before batch: {state.norm().item() if state is not None else 'None'}")
   ```
2. Ensure `detach_stream_state(state)` (or `state = None`) is called at the start of **every** validation batch.
3. Run validation twice:
   * a) With the current code (no explicit reset).
   * b) With an explicit reset at the beginning of each batch.
4. Compare the metrics.  A noticeable reduction in RMSE and a more stable cosine score after resetting confirms the issue.

### 4. Data Distribution Mismatch
1. Plot histograms of `log_altitude` and `velocity` for both training and validation sets (e.g., using `plt.hist`).
2. Identify any regions present only in validation.
3. Create a filtered validation subset that only contains samples falling inside the training range and recompute `val_vel_rmse` on this subset.
4. If the subset RMSE is dramatically lower, the distribution gap is a primary cause.
5. As a further check, swap the train/validation split (train on the former validation set) and observe whether the original validation metrics improve.

### 5. Fixed Learning‑Rate Plateau
1. Review the training log to see if the loss plateaus after a certain epoch while the learning‑rate stays constant.
2. Retrain the model for a few more epochs using a learning‑rate scheduler such as:
   ```python
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
   ```
3. Track validation metrics after the scheduler reduces the LR.  A continued drop in `val_vel_rmse` would confirm that the fixed LR was limiting performance.
4. Alternatively, repeat training with a globally smaller LR (e.g., 5e‑4) for the same number of epochs and compare the validation curve.

## Summary of Verification Strategy
- **Log‑based inspection** (printing arguments, state norms) gives immediate, low‑overhead evidence.
- **Controlled single‑batch experiments** isolate the effect of a single variable (altitude input, rotation mode, state reset).
- **Statistical visualisation** (histograms, rotation‑map heat‑maps) reveals distribution or architectural mismatches.
- **Re‑training with a scheduler** tests whether optimisation dynamics are the bottleneck.

By executing the above checks, you can definitively confirm which of the five hypotheses is the dominant factor behind the stagnant validation error and then apply the corresponding fix.
