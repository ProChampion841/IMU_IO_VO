import os
import csv
import copy
import torch
import numpy as np

import torch.utils.data as Data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

import tqdm
from utils import move_to
from utils.distributed import (parse_devices, pick_backend, ddp_setup, ddp_cleanup,
                               is_dist, is_main, get_rank, get_world_size,
                               reduce_metrics, unwrap)
from model import net_dict
from pyhocon import ConfigFactory
from pyhocon import HOCONConverter as conf_convert

from datasets import SeqeuncesDataset, collate_fcs
from model.losses import get_loss, loss_
from eval import evaluate

# TensorBoard is optional at import time only so that a broken/absent install
# degrades to "CSV only" instead of killing a training run that was going to
# work.  tensorboard 2.21.0 is installed in the `mvvio` env and verified.
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_IMPORT_ERROR = None
except Exception as _e:                                   # pragma: no cover
    SummaryWriter = None
    _TB_IMPORT_ERROR = repr(_e)

# NOTE: upstream shipped `torch.autograd.set_detect_anomaly(True)` unconditionally
# here.  It is a debugging aid that instruments every autograd node, and it was
# measured to cost 11-12x on this model (batch 256: 15.7 -> 190.6 windows/s).
# It is now opt-in via --detect_anomaly.


# ---------------------------------------------------------------------------
# WHAT THE DISPLAYED NUMBERS ARE, read off model/losses.py rather than guessed
# ---------------------------------------------------------------------------
# loss       the training objective actually minimised: Huber terms scaled by
#            pos_weight / vel_weight (/ rot_weight) plus the covariance NLL.
#            Loss units, not physical units -- it is not comparable across
#            configs with different weights.
# pos_error  model/losses.py:64, state_losses['pos'] = pos_dist[:,-1,:].norm().mean()
#            where pos_dist = inte_state['pos'] - data['gt_pos'] sampled at
#            `sampling`.  `[:, -1, :]` is the LAST sample of the window, so this
#            is the END-OF-WINDOW position drift, averaged over the batch.
#            gt_pos is datasets/UAVdataset.py's gt_translation, which is
#            cumtrapz of GPSNavVn (m/s) -> METRES.
# vel_error  same construction against gt_vel -> METRES PER SECOND.
# Both are RAW errors, not the Huber/weighted loss terms, so they are the honest
# physical read on the model and are comparable across configs.
#
# NO ROTATION IS DISPLAYED OR LOGGED.  Attitude is an INPUT to these networks
# (att_input / att_source) and `use_rot_loss: False` means the integrated
# rotation is neither trained on nor part of the objective, so a rotation column
# is noise between the two numbers that matter.  Note rotation is still
# COMPUTED inside model/losses.py -- it has to be: the covariance NLL consumes
# rot_dist (losses.py:79) even when the rotation state term is off -- and
# rot_loss is still returned for configs that DO train on rotation, so nothing
# that used to be reported for them disappears.
# ---------------------------------------------------------------------------

# Exactly the columns the user asked for, plus the eval block.  No rotation.
# Two error notions x two arms, all in METRES or METRES PER SECOND -- no ratios.
#
#                     model                 no-network arm (see raw_baseline_errors)
#   absolute (40 s)   *_pos_error           *_raw_pos_error
#   relative (0.5 s)  *_pos_rel_error       *_raw_pos_rel_error
#
# ABSOLUTE is the drift at the last supervision point, i.e. over the whole window.
# RELATIVE is the drift accumulated over ONE `sampling`-length sub-window (0.5 s at
# sampling 50), which is the horizon a VIO front-end actually integrates across.
# The raw arm is computed by raw_baseline_errors() on the SAME batch and windows, so
# `pos_error` against `raw_pos_error` answers the only question that matters: is the
# network beating doing nothing?
def horizon_tag(frames):
    """6000 -> '60s', 30000 -> '300s'.  Frames are 100 Hz on this corpus."""
    return "%gs" % (frames / 100.0)


def csv_fields(horizons, log_raw=True):
    """The metric.csv schema, built from the horizons this run reports.

    `metric_horizons` in a train config is a list of WINDOW LENGTHS IN FRAMES.
    The FIRST one is the primary horizon: it is the window the train pass already
    uses, so `train_*` is reported at that length and nowhere else, and its
    validation loss is the `val_loss` that checkpoint selection reads.  Every
    horizon gets its own validation columns.

    Absent, `metric_horizons` falls back to the legacy fixed schema, so every
    config written before 2026-09-04 produces a byte-identical header.

    The raw-integration columns are kept alongside each horizon on purpose: on
    this corpus a position error is only interpretable against what doing nothing
    would have cost -- the same number is a win on one flight and a loss on
    another.  Set `log_raw_baseline: False` to drop them.
    """
    if not horizons:
        return list(_LEGACY_CSV_FIELDS)
    h0 = horizon_tag(horizons[0])
    f = ["epoch", "lr", "train_loss",
         "train_pos_error_" + h0, "train_vel_error_" + h0]
    if log_raw:
        f += ["train_raw_pos_error_" + h0, "train_raw_vel_error_" + h0]
    # Covariance tracking.  `cov_loss` is what the covariance NLL adds to the
    # objective; the three cov_nll_* are the UNWEIGHTED per-block NLLs, so a move in
    # cov_loss can be attributed to the model rather than to a weight change.  Empty
    # cells when propcov is off.
    f += ["train_cov_loss", "train_cov_nll_rot", "train_cov_nll_vel", "train_cov_nll_pos"]
    f += ["val_loss", "val_cov_loss", "val_cov_nll_rot", "val_cov_nll_vel", "val_cov_nll_pos"]
    for h in horizons:
        t = horizon_tag(h)
        f += ["val_pos_error_" + t, "val_vel_error_" + t]
        if log_raw:
            f += ["val_raw_pos_error_" + t, "val_raw_vel_error_" + t]
    return f


_LEGACY_CSV_FIELDS = ["epoch", "lr",
              "train_loss", "train_pos_error", "train_vel_error",
              "train_pos_rel_error", "train_vel_rel_error",
              "train_raw_pos_error", "train_raw_vel_error",
              "train_raw_pos_rel_error", "train_raw_vel_rel_error",
              "val_loss", "val_pos_error", "val_vel_error",
              "val_pos_rel_error", "val_vel_rel_error",
              "val_raw_pos_error", "val_raw_vel_error",
              "val_raw_pos_rel_error", "val_raw_vel_rel_error",
              "eval_pos_error", "eval_vel_error", "eval_pos_rmse", "eval_vel_rmse"]
CSV_FIELDS = _LEGACY_CSV_FIELDS


class EpochLogger:
    """metric.csv + TensorBoard, one row / one step per epoch.

    Always on.  It is a local-file logger with no login, no network and no
    per-step cost (a dozen scalars per epoch).  This is the ONLY logger in the
    repo -- wandb was removed 2026-09-07 at the user's request.  `tensorboard:
    False` in a train config turns the TensorBoard half off; metric.csv is
    unconditional because a killed run has to leave its history behind.
    """

    def __init__(self, exp_dir, start_epoch=0, use_tb=True, fields=None):
        self.fields = list(fields) if fields else list(_LEGACY_CSV_FIELDS)
        os.makedirs(exp_dir, exist_ok=True)
        self.csv_path = os.path.join(exp_dir, "metric.csv")
        self._f = None
        self._csv = None
        self.tb = None
        self._open_csv(start_epoch)
        print('[metric.csv]  "%s"' % os.path.abspath(self.csv_path))

        if use_tb and SummaryWriter is not None:
            tb_dir = os.path.join(exp_dir, "tensorboard")
            # purge_step drops any events already written for epochs that are
            # about to be recomputed after a --load_ckpt resume, so the curve
            # does not fork.
            kwargs = {"purge_step": start_epoch} if start_epoch > 0 else {}
            self.tb = SummaryWriter(log_dir=tb_dir, **kwargs)
            print('[tensorboard] tensorboard --logdir "%s"' % os.path.abspath(tb_dir))
        elif use_tb:
            print("[tensorboard] disabled: torch.utils.tensorboard failed to "
                  "import (%s).  metric.csv is unaffected." % _TB_IMPORT_ERROR)
        else:
            print("[tensorboard] disabled by config (tensorboard: False).")

    def _open_csv(self, start_epoch):
        """Append to an existing metric.csv instead of truncating it.

        On a --load_ckpt resume the loop restarts AT checkpoint['epoch'], i.e.
        it recomputes that epoch, so rows with epoch >= start_epoch are dropped
        and rewritten rather than duplicated.  A file whose header does not
        match the current schema is rotated to metric.csv.bak* and a fresh one
        started -- never silently mixed.
        """
        rows, n_read, header_ok = [], 0, False
        if os.path.isfile(self.csv_path):
            try:
                with open(self.csv_path, "r", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    header_ok = (header == self.fields)
                    if header_ok:
                        for row in reader:
                            if not row:
                                continue
                            n_read += 1
                            try:
                                epoch_of_row = int(float(row[0]))
                            except (ValueError, IndexError):
                                continue
                            if epoch_of_row < start_epoch:
                                rows.append(row)
            except OSError as err:
                print("[metric.csv] could not read existing file (%s); starting a new one" % err)
                header_ok = False

            if header_ok and n_read == len(rows):
                # Nothing to drop: plain append, and NO second header row.
                self._f = open(self.csv_path, "a", newline="")
                self._csv = csv.writer(self._f)
                print("[metric.csv] appending after %d existing row(s)" % n_read)
                return

            if not header_ok:
                backup = self.csv_path + ".bak"
                i = 1
                while os.path.exists(backup):
                    backup = self.csv_path + ".bak%d" % i
                    i += 1
                os.replace(self.csv_path, backup)
                print("[metric.csv] header mismatch; previous file kept as %s" % os.path.basename(backup))
            else:
                print("[metric.csv] resuming at epoch %d: keeping %d of %d row(s)"
                      % (start_epoch, len(rows), n_read))

        self._f = open(self.csv_path, "w", newline="")
        self._csv = csv.writer(self._f)
        self._csv.writerow(self.fields)
        for row in rows:
            self._csv.writerow(row)
        self._f.flush()

    @staticmethod
    def _fmt(v):
        return "" if v is None else "%.8g" % float(v)

    def log_epoch(self, epoch, lr, train_loss, test_loss, eval_metrics=None,
                  horizons=None, val_by_horizon=None, log_raw=True):
        """One CSV row.

        `horizons` + `val_by_horizon` select the new schema: `val_by_horizon` maps a
        window length in FRAMES to that horizon's validation dict.  `test_loss` stays
        the primary horizon's dict -- it is what `val_loss` and checkpoint selection
        read -- so passing neither argument reproduces the legacy row exactly.
        """
        eval_metrics = eval_metrics or {}
        if horizons:
            vb = val_by_horizon or {}
            row = [epoch, self._fmt(lr), self._fmt(train_loss["loss"]),
                   self._fmt(train_loss["pos_loss"]), self._fmt(train_loss["vel_loss"])]
            if log_raw:
                row += [self._fmt(train_loss.get("raw_pos")), self._fmt(train_loss.get("raw_vel"))]
            row += [self._fmt(train_loss.get("cov_loss")),
                    self._fmt(train_loss.get("cov_nll_rot")),
                    self._fmt(train_loss.get("cov_nll_vel")),
                    self._fmt(train_loss.get("cov_nll_pos"))]
            row += [self._fmt(test_loss["loss"]),
                    self._fmt(test_loss.get("cov_loss")),
                    self._fmt(test_loss.get("cov_nll_rot")),
                    self._fmt(test_loss.get("cov_nll_vel")),
                    self._fmt(test_loss.get("cov_nll_pos"))]
            for h in horizons:
                d = vb.get(h) or {}
                row += [self._fmt(d.get("pos_loss")), self._fmt(d.get("vel_loss"))]
                if log_raw:
                    row += [self._fmt(d.get("raw_pos")), self._fmt(d.get("raw_vel"))]
        else:
            def arm(d):
                return [self._fmt(d["loss"]), self._fmt(d["pos_loss"]), self._fmt(d["vel_loss"]),
                        self._fmt(d.get("pos_rel")), self._fmt(d.get("vel_rel")),
                        self._fmt(d.get("raw_pos")), self._fmt(d.get("raw_vel")),
                        self._fmt(d.get("raw_pos_rel")), self._fmt(d.get("raw_vel_rel"))]
            row = ([epoch, self._fmt(lr)] + arm(train_loss) + arm(test_loss)
                   + [self._fmt(eval_metrics.get("pos_error")),
                      self._fmt(eval_metrics.get("vel_error")),
                      self._fmt(eval_metrics.get("pos_rmse")),
                      self._fmt(eval_metrics.get("vel_rmse"))])
        assert len(row) == len(self.fields), ("row/header mismatch: %d vs %d"
                                              % (len(row), len(self.fields)))
        self._csv.writerow(row)
        # Flush every epoch: a run killed at epoch 17 must still have 17 rows.
        self._f.flush()

        if horizons and self.tb is not None:
            for h, d in (val_by_horizon or {}).items():
                t = horizon_tag(h)
                if d.get("pos_loss") is not None:
                    self.tb.add_scalar("val_%s/pos_error_m" % t, d["pos_loss"], epoch)
                    self.tb.add_scalar("val_%s/vel_error_mps" % t, d["vel_loss"], epoch)
                if d.get("raw_pos") is not None:
                    self.tb.add_scalar("val_%s/raw_pos_error_m" % t, d["raw_pos"], epoch)
                # raw_vel too: with wandb gone TensorBoard is the only live view, and
                # the model/raw ratio is the thing being watched.  It was already in
                # metric.csv (val_raw_vel_error_<h>), just not plotted.
                if d.get("raw_vel") is not None:
                    self.tb.add_scalar("val_%s/raw_vel_error_mps" % t, d["raw_vel"], epoch)

        if self.tb is None:
            return
        self.tb.add_scalar("lr", lr, epoch)
        for tag, d in (("train", train_loss), ("val", test_loss)):
            self.tb.add_scalar("%s/loss" % tag, d["loss"], epoch)
            self.tb.add_scalar("%s/pos_error_m" % tag, d["pos_loss"], epoch)
            self.tb.add_scalar("%s/vel_error_mps" % tag, d["vel_loss"], epoch)
            for _k, _t, _u in (("pos_rel", "pos_rel_error_m", "m"),
                               ("vel_rel", "vel_rel_error_mps", "m/s"),
                               ("raw_pos", "raw_pos_error_m", "m"),
                               ("raw_vel", "raw_vel_error_mps", "m/s"),
                               ("raw_pos_rel", "raw_pos_rel_error_m", "m"),
                               ("raw_vel_rel", "raw_vel_rel_error_mps", "m/s")):
                if d.get(_k) is not None:
                    self.tb.add_scalar("%s/%s" % (tag, _t), d[_k], epoch)
            # Covariance diagnostics, position/velocity only.
            for k, name in (("pred_cov_pos", "pred_cov_pos"), ("pred_cov_vel", "pred_cov_vel")):
                if k in d:
                    self.tb.add_scalar("%s/%s" % (tag, name), d[k], epoch)
        for k, v in eval_metrics.items():
            self.tb.add_scalar("eval/%s" % k, v, epoch)
        # Flush so a killed run keeps its events too.
        self.tb.flush()

    def close(self):
        if self.tb is not None:
            try:
                self.tb.close()
            except Exception as err:                       # pragma: no cover
                print("[tensorboard] close failed: %r" % err)
            self.tb = None
        if self._f is not None:
            try:
                self._f.flush()
                self._f.close()
            except Exception as err:                       # pragma: no cover
                print("[metric.csv] close failed: %r" % err)
            self._f = None


@torch.no_grad()
def raw_baseline_errors(network, data, init_state, label, confs):
    """Absolute and relative error of the NO-NETWORK arm.

    NOT "raw, uncorrected IMU" -- that label was wrong and is corrected here.  What
    this arm removes is the NETWORK's correction, nothing else.  Under every shipped
    UAV config the `freeze_hist_s: 15.0` key is set on each dataset section, so
    datasets/dataset.py subtracts a per-window constant bias from acc/gyro inside
    __getitem__, BEFORE the batch reaches either arm.  Both arms therefore integrate
    an already-bias-corrected signal, and `*_raw_*` is "the 15 s pre-window freeze
    alone", not "unaided integration".  The comparison is still exactly paired and
    valid; only the NAME was misleading.  To measure against genuinely unaided
    integration, set freeze_hist_s: 0.0 and re-run -- the two are not the same number
    (measured on `test`: the freeze is worth 2% at 30 s and HURTS by 3% at 60 s).

    PAIRED with the model arm: same batch, same windows, same init_state, same
    `sampling`, same gtrot setting -- only the signal differs.  That pairing is the
    point of the feature.  Window difficulty varies enormously across this corpus,
    so a baseline measured on some other set of windows would be mostly noise and
    the comparison would not mean much.

    Costs one extra integrator pass per batch under no_grad.  Set
    `log_raw_baseline: False` in a train config to turn it off.
    """
    net = unwrap(network)
    iv = getattr(net, "interval", 0)
    d = dict(data)
    d["corrected_acc"] = data["acc"][:, iv:, :]
    d["corrected_gyro"] = data["gyro"][:, iv:, :]
    out = net.integrate(init_state=init_state, data=d,
                        cov_state={"acc_cov": None, "gyro_cov": None})
    _, pos_dist = loss_(None, out["pos"], label["gt_pos"], sampling=confs.sampling)
    _, vel_dist = loss_(None, out["vel"], label["gt_vel"], sampling=confs.sampling)
    res = {"raw_pos": pos_dist[:, -1, :].norm(dim=-1).mean().item(),
           "raw_vel": vel_dist[:, -1, :].norm(dim=-1).mean().item()}
    if pos_dist.shape[1] > 1:
        res["raw_pos_rel"] = (pos_dist[:, 1:, :] - pos_dist[:, :-1, :]).norm(dim=-1).mean().item()
        res["raw_vel_rel"] = (vel_dist[:, 1:, :] - vel_dist[:, :-1, :]).norm(dim=-1).mean().item()
    else:
        res["raw_pos_rel"], res["raw_vel_rel"] = res["raw_pos"], res["raw_vel"]
    return res


def _aug_std(v):
    """Normalise an aug_*_bias_std config value to a (1, 1, 3) tensor, or None if off.

    Accepts a scalar (isotropic) or a 3-list (per axis).  Per-axis is the faithful
    setting: the MEASURED per-window bias spread is anisotropic --
    accel [0.0798, 0.0986, 0.0164] m/s^2 and gyro [0.0492, 0.0320, 0.0611] deg/s --
    so a single scalar at the x/y spread over-augments accel z by about 5x, making
    that axis a harder problem than the real one.
    """
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        t = torch.tensor([float(x) for x in v], dtype=torch.float64)
        if t.numel() != 3:
            raise ValueError("aug_*_bias_std must be a scalar or 3 values, got %d" % t.numel())
    else:
        t = torch.full((3,), float(v), dtype=torch.float64)
    return None if float(t.abs().max()) <= 0.0 else t.reshape(1, 1, 3)


def bias_augment(data, confs, correct_gyro):
    """Inject a random CONSTANT bias per window, redrawn every time it is sampled.

    WHY.  MEASURED on this corpus: the per-window fitted bias points in a different
    direction on every flight -- cosine of each flight's fit against the pooled mean
    is -0.068 for accel, +0.336 for gyro.  There is no shared bias to learn, so the
    only way to cut TRAINING error is to key on flight-specific cues, and 929k
    parameters over 56 flights make that easy.  The result, same checkpoint, same
    4000-frame windows, 40 s horizon:

        train flights   139.76 m vs raw 242.66 m   ratio 0.576
        val flights     126.10 m vs raw 126.67 m   ratio 0.995

    -42% on flights it has seen, -0.5% on flights it has not.

    Redrawing a bias per window breaks that shortcut: "this flight -> that bias" is
    worthless when the bias changes on every draw, so the only way left to reduce the
    loss is to ESTIMATE the bias from the signal -- which is the skill that transfers.

    WHAT IT DOES NOT FIX.  If a constant bias is simply not identifiable from a
    window of this data, augmentation cannot conjure the information: train error
    will rise and val will stay flat.  That outcome is informative, not a bug -- it
    would say the per-window correction is unlearnable here, and the honest answer is
    a constant-noise-density model rather than a network.

    The bias is added to the WHOLE tensor, padding frames included.  padding_collate
    prepends 9 synthetic frames holding a clean rotated gravity vector; leaving those
    unbiased would put a step at the boundary that the network could read the
    injected bias straight off, which is a shortcut with no runtime counterpart.

    Gyro augmentation is skipped when correct_gyro is False, because the network then
    has no gyro head and could only absorb the injected bias as unremovable error.

    Both stds default to 0.0, which is an exact no-op, so configs that do not set
    them train bit-for-bit as before.  Applied in train() only -- validation and
    evaluation always see the clean signal, so val columns stay comparable across
    runs.  The raw baseline is computed from this same augmented `data`, which keeps
    raw_baseline_errors' paired comparison honest; expect train_raw_* to RISE when
    augmentation is on.
    """
    sa = _aug_std(confs.get("aug_acc_bias_std", 0.0))
    sg = _aug_std(confs.get("aug_gyro_bias_std", 0.0))
    if not correct_gyro:
        sg = None
    if sa is None and sg is None:
        return data
    out = dict(data)
    for key, std in (("acc", sa), ("gyro", sg)):
        if std is None:
            continue
        x = out[key]
        # one draw per window (B, 1, 3), broadcast over time: a BIAS, not noise.
        b = torch.randn(x.shape[0], 1, x.shape[2], dtype=x.dtype, device=x.device)
        out[key] = x + b * std.to(dtype=x.dtype, device=x.device)
    return out


# ---------------------------------------------------------------------------
# EMA WEIGHTS and SMOOTHED BEST-CHECKPOINT SELECTION -- both opt-in, conf.train
# ---------------------------------------------------------------------------
# WHY.  MEASURED on hybrid_v9_acconly (26 epochs, 13 val flights at 40 s):
# val_pos_error saw-tooths epoch to epoch with sd 2.55 m and lag-1 autocorrelation
# -0.49, against an effect size of ~1 m (126.10 m model vs 126.59 m raw).  So
# best_model.ckpt (epoch 2 of 25, 121.66 m) is the epoch that drew the lucky half
# of the oscillation, not the best model.  Two independent, separately switchable
# fixes, both exact no-ops at their defaults:
#
#   ema_decay: d      exponential moving average of the PARAMETERS, updated after
#                     every optimizer.step():  ema = d*ema + (1-d)*live.  Averages
#                     the oscillation out of the weights.  Time constant 1/(1-d)
#                     steps.  This corpus is 32,610 train windows / batch 128 =
#                     255 steps per epoch, so 0.998 averages over ~500 steps
#                     (~2.0 epochs) and 0.9985 over ~670 (~2.6 epochs); 0.999 is
#                     ~3.9 epochs, already long against an 80-epoch run with a
#                     plateau scheduler.  0.0 (default) = off, nothing is built.
#   select_smooth: K  best_model.ckpt is chosen on the MEAN of the last K epochs'
#                     val loss instead of the single-epoch value.  The window is
#                     never shorter than K, so nothing is selected before epoch
#                     K-1 (see smoothed_criterion for why).  K=1 (default) is the
#                     old rule bit for bit: sum([x])/1 == x exactly.
#
# WHAT THE VAL COLUMNS MEAN WHEN ema_decay > 0.  test() and evaluate() run on the
# EMA copy, because that is the model that gets saved and selected; the training
# pass still runs the live model.  So in metric.csv / TensorBoard the val_* and
# eval_* columns describe the EMA weights and train_* the live ones.
#
# DDP.  After every all-reduced step each rank holds identical parameters and the
# update is deterministic, so every rank maintains its own EMA copy in lock-step:
# nothing is broadcast, rank 0 saves.  BUFFERS (BatchNorm running stats, in_scale,
# acc_std/gyro_std) are COPIED from the live model, not averaged -- they are
# statistics, not optimised weights, and averaging running_var across epochs would
# de-pair it from running_mean.
#
# CHECKPOINT LAYOUT.  'model_state_dict' is always the weights every downstream
# tool loads: the EMA copy when it is on, the live model when it is off.  With EMA
# on the live weights are ALSO stored under 'live_state_dict' so --load_ckpt can
# resume both; a checkpoint written without EMA has no such key and resume then
# seeds the EMA from the live weights.  'best_loss' is the SMOOTHED criterion
# (mean over K epochs) -- with K=1 that is the plain val loss as before.
# 'val_loss_history' (per-epoch val loss, index == epoch) is what the smoothing
# reads, stored always so a resume continues the same window.
# ---------------------------------------------------------------------------

def set_seed(seed, rank=0):
    """Seed every RNG that can change a reported number.

    WHY THIS EXISTS.  Until 2026-09-08 train.py seeded NOTHING: weight init
    (model/cnn.py:21 and friends), DataLoader shuffling, and dropout were all drawn
    from an unseeded global RNG.  Every config A/B on this project therefore
    confounded the config with the seed -- and the effects being compared are ~1-2%
    while the epoch-to-epoch sd of val_pos_error_60s is ~0.5-1.8 m on a ~124 m
    baseline.  A/B differences smaller than the seed spread were not measurable.

    RANK.  Weights must be IDENTICAL across DDP ranks, and DistributedSampler already
    partitions the data, so every rank seeds the same.  `rank` is folded only into the
    DataLoader worker seeds via worker_init_fn below, never into init.

    NOT made bitwise deterministic on purpose: torch.use_deterministic_algorithms
    would force a slow cuDNN path and the pypose integrator loop dominates anyway.
    This makes runs REPRODUCIBLE-ISH -- same init, same batch order -- which is what
    an A/B needs.  Residual GPU nondeterminism (atomics) is far below the effects here.
    """
    import random as _random
    seed = int(seed)
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def seed_worker(worker_id):
    """DataLoader worker seeding -- torch gives each worker a base_seed, derive from it."""
    import random as _random
    ws = torch.initial_seed() % 2 ** 32
    np.random.seed(ws)
    _random.seed(ws)


def build_ema(network, decay):
    """A detached copy of the bare module on the same device, or None when off."""
    if decay is None or float(decay) <= 0.0:
        return None
    ema = copy.deepcopy(unwrap(network))
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema.eval()


@torch.no_grad()
def ema_update(ema, network, decay):
    """ema_p = d*ema_p + (1-d)*p over parameters; buffers copied.  After optimizer.step()."""
    live = unwrap(network)
    d = float(decay)
    for e, p in zip(ema.parameters(), live.parameters()):
        e.mul_(d).add_(p.detach(), alpha=1.0 - d)
    for e, b in zip(ema.buffers(), live.buffers()):
        e.copy_(b)


def smoothed_criterion(history, k):
    """Mean of the last k entries; +inf until k entries exist.  k=1 returns the value itself.

    The window is NEVER shorter than k.  A shorter warm-up window is a noisier
    estimate, and a running minimum over noisier draws is biased low, so it hands
    the early epochs exactly the advantage this smoothing exists to remove.
    MEASURED on hybrid_v9_acconly's 26-epoch history: with a warm-up window, K=5
    picked epoch 2 on a 3-sample mean (0.595068); the argmin over genuine 5-epoch
    windows is epoch 21 (0.600293).  So no best_model.ckpt is written before epoch
    K-1 (newest.ckpt still is); K=1 is unaffected.
    """
    k = max(1, int(k))
    if len(history) < k:
        return float("inf")
    tail = list(history)[-k:]
    return sum(tail) / k


def train(network, loader, confs, epoch, optimizer, ema=None, ema_decay=0.0):
    """
    Train network for one epoch using a specified data loader
    Outputs all targets, predicts, predicted covariance params, and losses
    """
    network.train()
    losses, pos_losses, rot_losses, vel_losses = 0, 0, 0, 0
    acc = {k: 0.0 for k in ('pos_rel','vel_rel','raw_pos','raw_vel','raw_pos_rel','raw_vel_rel')}
    log_raw = confs.get("log_raw_baseline", True)
    pred_cov_rot, pred_cov_vel, pred_cov_pos = 0, 0, 0
    cov_loss_sum, cov_nll = 0.0, {'rot': 0.0, 'vel': 0.0, 'pos': 0.0}
    acc_covs, gyro_covs = 0, 0
    # When rotation is not scored it is not accumulated either.  losses.py still
    # returns the key (the covariance NLL needs rot_dist), we simply do not read
    # it, which also drops one .item() GPU sync per batch.
    use_rot_loss = confs.get("use_rot_loss", True)
    n_win = 0
    t_range = tqdm.tqdm(loader, desc="train ep %03d" % epoch, disable=not is_main())
    for i, (data, init_state, label) in enumerate(t_range):
        data, init_state, label = move_to([data, init_state, label], confs.device)
        # Augment BEFORE both arms so the model and the raw baseline see the same
        # signal and stay a paired comparison.  No-op unless aug_*_bias_std is set.
        data = bias_augment(data, confs, confs.get("correct_gyro", True))
        inte_state = network(data, init_state)
        loss_state = get_loss(inte_state, label, confs)
        # Per-BATCH means are accumulated below, so this counts batches.  It used to be
        # declared and never incremented, so every train_* column was divided by
        # max(1, 0) = 1 -- a SUM over ~all batches, not a mean (train_pos_error_60s read
        # ~33,000 m against ~120 m per window).  Only the final batch can be short, so
        # the batch mean is the window mean to within that one batch.
        n_win += 1

        # statistics
        losses += loss_state['loss'].item()
        pos_losses += loss_state['pos'].item()
        vel_losses += loss_state['vel'].item()
        acc['pos_rel'] += loss_state['pos_rel'].item()
        acc['vel_rel'] += loss_state['vel_rel'].item()
        if log_raw:
            for _k, _v in raw_baseline_errors(network, data, init_state, label, confs).items():
                acc[_k] += _v
        if use_rot_loss:
            rot_losses += loss_state['rot'].item()

        if confs.propcov:
            # acc_cov/gyro_cov are HybridNet's IMU-CORRECTION uncertainties.  A model
            # that does not correct the IMU (velnet regresses velocity directly) has
            # no such quantity; its uncertainty is reported through pred_cov_vel and
            # the cov_loss/cov_nll_* columns instead, so nothing is lost by skipping.
            if "acc_cov" in inte_state:
                acc_covs += inte_state["acc_cov"].mean().item()
                gyro_covs += inte_state["gyro_cov"].mean().item()
            pred_cov_pos += loss_state['pred_cov_pos'].mean().item()
            pred_cov_rot += loss_state['pred_cov_rot'].mean().item()
            pred_cov_vel += loss_state['pred_cov_vel'].mean().item()
            # What the covariance NLL contributes to `loss`, plus the unweighted
            # per-block NLL so a move in cov_loss is attributable to the model and
            # not to a weight edit.  Absent when propcov is off.
            if 'cov_loss' in loss_state:
                cov_loss_sum += loss_state['cov_loss'].item()
                for _k in cov_nll:
                    cov_nll[_k] += loss_state['cov_nll_' + _k].item()

        # Running means, in fixed-width columns, position AND velocity, no
        # rotation.  set_postfix keeps the metrics in a stable column layout
        # instead of reflowing the description on every refresh.
        t_range.set_postfix(loss_avg="%9.6f" % (losses / (i + 1)),
                            pos_err_avg_m="%8.4f" % (pos_losses / (i + 1)),
                            vel_err_avg_mps="%8.4f" % (vel_losses / (i + 1)),
                            refresh=False)
        t_range.refresh()

        optimizer.zero_grad()
        loss_state['loss'].backward()
        optimizer.step()
        if ema is not None:
            ema_update(ema, network, ema_decay)

    out = {"loss": (losses/max(1, n_win)), "pos_loss": (pos_losses/max(1, n_win)), "vel_loss":((vel_losses)/max(1, n_win)),
           **{k: v/max(1, n_win) for k, v in acc.items()},
           "pred_cov_rot": (pred_cov_rot/max(1, n_win)), "pred_cov_vel": (pred_cov_vel/max(1, n_win)), "pred_cov_pos": (pred_cov_pos/max(1, n_win)),
           "cov_loss": (cov_loss_sum/max(1, n_win)) if confs.get("propcov", False) else None,
           "cov_nll_rot": (cov_nll['rot']/max(1, n_win)) if confs.get("propcov", False) else None,
           "cov_nll_vel": (cov_nll['vel']/max(1, n_win)) if confs.get("propcov", False) else None,
           "cov_nll_pos": (cov_nll['pos']/max(1, n_win)) if confs.get("propcov", False) else None}
    if use_rot_loss:
        # Kept for the configs that DO train rotation.  Never displayed, never
        # written to metric.csv/TensorBoard.
        out["rot_loss"] = (rot_losses/max(1, n_win))
    return out


def test(network, loader, confs, epoch=None):
    network.eval()
    use_rot_loss = confs.get("use_rot_loss", True)
    log_raw = confs.get("log_raw_baseline", True)
    with torch.no_grad():
        losses, pos_losses, rot_losses, vel_losses = 0, 0, 0, 0
        acc = {k: 0.0 for k in ('pos_rel','vel_rel','raw_pos','raw_vel','raw_pos_rel','raw_vel_rel')}
        pred_cov_rot, pred_cov_vel, pred_cov_pos = 0, 0, 0
        cov_loss_sum, cov_nll = 0.0, {'rot': 0.0, 'vel': 0.0, 'pos': 0.0}
        acc_covs, gyro_covs = [], []

        # WINDOW-WEIGHTED, not batch-weighted.  Every term below is a per-batch MEAN
        # over that batch's windows, so averaging the batch means with equal weight
        # over-counts the short final batch.  Measured on the shipped configs before
        # this fix: 101 val windows at batch 64 -> batches of [64, 37], so the last 37
        # windows carried 50% of val_pos_error_60s instead of 37%; at 120 s the split
        # is [32, 10] and the final 10 windows carried half.  Weighting by `bs` makes
        # every reported column the true mean over windows.
        #
        # THIS CHANGES REPORTED NUMBERS.  Runs before 2026-09-08 are not directly
        # comparable: the old rule read 123.75784 m where the true mean is 121.47393 m
        # (60 s, epoch 2 of accel_const).  The bias ran AGAINST the model, so old
        # model/raw ratios were slightly pessimistic, not flattering.
        n_win = 0
        desc = "val   ep %03d" % epoch if epoch is not None else "val"
        t_range = tqdm.tqdm(loader, desc=desc, disable=not is_main())
        for i, (data, init_state, label) in enumerate(t_range):

            data, init_state, label = move_to([data, init_state, label], confs.device)
            inte_state = network(data, init_state)
            bs = int(inte_state['vel'].shape[0])
            n_win += bs

            loss_state = get_loss(inte_state, label, confs)
            # statistics
            losses += loss_state['loss'].item() * bs
            pos_losses += loss_state["pos"].item() * bs
            vel_losses += loss_state['vel'].item() * bs
            acc['pos_rel'] += loss_state['pos_rel'].item() * bs
            acc['vel_rel'] += loss_state['vel_rel'].item() * bs
            if log_raw:
                for _k, _v in raw_baseline_errors(network, data, init_state, label, confs).items():
                    acc[_k] += _v * bs
            if use_rot_loss:
                rot_losses += loss_state["rot"].item() * bs

            if confs.propcov:
                if "acc_cov" in inte_state:      # see the note in train()
                    acc_covs.append(inte_state["acc_cov"].reshape(-1))
                    gyro_covs.append(inte_state["gyro_cov"].reshape(-1))
                pred_cov_pos += loss_state['pred_cov_pos'].mean().item() * bs
                pred_cov_rot += loss_state['pred_cov_rot'].mean().item() * bs
                pred_cov_vel += loss_state['pred_cov_vel'].mean().item() * bs
                if 'cov_loss' in loss_state:
                    cov_loss_sum += loss_state['cov_loss'].item() * bs
                    for _k in cov_nll:
                        cov_nll[_k] += loss_state['cov_nll_' + _k].item() * bs

            t_range.set_postfix(loss_avg="%9.6f" % (losses / n_win),
                                pos_err_avg_m="%8.4f" % (pos_losses / n_win),
                                vel_err_avg_mps="%8.4f" % (vel_losses / n_win),
                                refresh=False)
            t_range.refresh()

        if acc_covs:
            acc_covs = torch.cat(acc_covs)
        if gyro_covs:
            gyro_covs = torch.cat(gyro_covs)

    # Every accumulator above is SUM(batch_mean * bs), so the window mean is / n_win.
    # It used to be / (i+1) -- the batch COUNT -- which multiplied every val column by
    # the mean batch size: x6 at 30/60 s and x3 at 120 s with batch_size 6 (e.g. the
    # 60 s raw baseline logged 738.1 m for a true 123.0 m).  Ratios were unaffected.
    n_win = max(1, n_win)
    out = {"loss": (losses/n_win), "pos_loss":(pos_losses/n_win), "vel_loss":(vel_losses/n_win),
           **{k: v/n_win for k, v in acc.items()},
           "pred_cov_rot": (pred_cov_rot/n_win), "pred_cov_vel": (pred_cov_vel/n_win), "pred_cov_pos": (pred_cov_pos/n_win),
           "acc_covs": acc_covs, "gyro_covs": gyro_covs,
           "cov_loss": (cov_loss_sum/n_win) if confs.get("propcov", False) else None,
           "cov_nll_rot": (cov_nll['rot']/n_win) if confs.get("propcov", False) else None,
           "cov_nll_vel": (cov_nll['vel']/n_win) if confs.get("propcov", False) else None,
           "cov_nll_pos": (cov_nll['pos']/n_win) if confs.get("propcov", False) else None}
    if use_rot_loss:
        out["rot_loss"] = (rot_losses/n_win)
    return out


def save_ckpt(network, optimizer, scheduler, epoch_i, test_loss, conf, save_best = False,
              ema=None, val_history=None):
    # One payload, written to up to three files.  'model_state_dict' is what every
    # downstream tool loads: the EMA copy when on, else the live model.  `test_loss`
    # is the running best of the SMOOTHED criterion (see smoothed_criterion); with
    # select_smooth 1 it is the plain val loss, exactly as before.
    payload = {
        'epoch': epoch_i,
        'model_state_dict': (ema if ema is not None else network).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': test_loss,
    }
    if ema is not None:
        payload['live_state_dict'] = network.state_dict()
    if val_history is not None:
        payload['val_loss_history'] = list(val_history)

    if epoch_i%conf.train.save_freq==conf.train.save_freq-1:
        torch.save(payload, os.path.join(conf.general.exp_dir, "ckpt/%04d.ckpt"%epoch_i))

    if save_best:
        print("saving the best model", test_loss)
        torch.save(payload, os.path.join(conf.general.exp_dir, "ckpt/best_model.ckpt"))

    torch.save(payload, os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt"))


def main_worker(local_rank, device_ids, args):
    """One training process.

    `local_rank` is 0..world_size-1 and `device_ids[local_rank]` is the CUDA
    ordinal it owns.  Called directly for a single device, or once per GPU by
    torch.multiprocessing.spawn / torchrun.
    """
    world_size = len(device_ids) if device_ids else 1
    distributed = world_size > 1

    if distributed:
        ddp_setup(local_rank, world_size, device_ids[local_rank])
        device = "cuda:%d" % device_ids[local_rank]
    elif device_ids:
        device = "cuda:%d" % device_ids[0]
        torch.cuda.set_device(device_ids[0])
    else:
        device = "cpu"

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    conf = ConfigFactory.parse_file(args.config)
    conf.train.device = device
    # BEFORE any dataset or model is built: both draw from the global RNG.
    _seed = set_seed(conf.train.get("seed", args.seed), rank=local_rank)
    conf.train["seed"] = _seed          # recorded into parameters.yaml
    if is_main():
        print("[seed] %d (override with --seed, or `seed:` in the train config)" % _seed)
    exp_folder = os.path.split(conf.general.exp_dir)[-1]
    conf_name = os.path.split(args.config)[-1].split(".")[0]
    conf['general']['exp_dir'] = os.path.join(conf.general.exp_dir, conf_name)

    if is_main():
        print("[device] %s | world_size %d | backend %s"
              % (device, world_size, pick_backend() if distributed else "-"))
        if distributed:
            print("[device] batch_size %d is the GLOBAL batch: DistributedSampler gives "
                  "each of the %d ranks ~%.1f samples/step, so the effective batch -- and "
                  "therefore the optimisation at a given lr -- is UNCHANGED from a "
                  "single-GPU run."
                  % (conf.train.batch_size, world_size, conf.train.batch_size / world_size))

    collate_fn = collate_fcs[conf.dataset.collate] if 'collate' in conf.dataset.keys() else collate_fcs['base']

    # `metric_horizons` (FRAMES) is read before the datasets because its FIRST entry
    # defines the primary validation window -- the one whose loss becomes `val_loss`
    # and drives checkpoint selection.  conf.dataset.test carries its own
    # window_size, and leaving it in place would silently report the primary column
    # at a different horizon than the one named in the header.
    metric_horizons = [int(h) for h in conf.train.get("metric_horizons", [])]

    train_dataset = SeqeuncesDataset(data_set_config=conf.dataset.train)
    # The train_* CSV columns are NAMED after metric_horizons[0], but the TRAIN window
    # is not overridden -- it stays conf.dataset.train.window_size.  Say so when the
    # two differ: the tilt_aware run logged `train_pos_error_60s` from 40 s windows.
    if metric_horizons and is_main():
        _tw = sorted({int(e["window_size"]) for e in conf.dataset.train.data_list})
        if _tw != [metric_horizons[0]]:
            print("[metric] NOTE: train windows are %s frames but the train_* columns are "
                  "labelled %s (metric_horizons[0]); the label is the VAL horizon, not "
                  "the train window." % (_tw, horizon_tag(metric_horizons[0])))
    _test_conf = conf.dataset.test
    if metric_horizons:
        _test_conf = copy.deepcopy(conf.dataset.test)
        for entry in _test_conf.data_list:
            entry["window_size"] = metric_horizons[0]
            entry["step_size"] = metric_horizons[0]
    test_dataset = SeqeuncesDataset(data_set_config=_test_conf)
    eval_dataset = SeqeuncesDataset(data_set_config=conf.dataset.eval)
    num_workers = conf.train.get('num_workers', 0)

    train_sampler = test_sampler = None
    if distributed:
        train_sampler = Data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True, drop_last=True)
        test_sampler = Data.distributed.DistributedSampler(
            test_dataset, num_replicas=world_size, rank=local_rank, shuffle=False, drop_last=False)

    per_rank_bs = max(1, conf.train.batch_size // world_size)
    _gen = torch.Generator(); _gen.manual_seed(_seed)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=per_rank_bs,
                                   shuffle=(train_sampler is None), sampler=train_sampler,
                                   collate_fn=collate_fn, num_workers=num_workers,
                                   generator=_gen, worker_init_fn=seed_worker)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=per_rank_bs,
                                  shuffle=False, sampler=test_sampler, collate_fn=collate_fn)

    # ---- extra validation horizons ------------------------------------------
    # `metric_horizons` (FRAMES) asks for the validation error at more than one
    # window length.  Each extra horizon gets its own dataset built from the SAME
    # flights as conf.dataset.test with window_size/step_size overridden, so the
    # only thing that differs between the columns is how far the integration runs.
    #
    # The FIRST horizon reuses test_loader unchanged -- it is the primary one, its
    # loss is `val_loss`, and it is what checkpoint selection reads.  Set it equal to
    # the TRAIN window so `train_*` and the first `val_*` pair are the same horizon.
    #
    # Cost: one extra validation pass per horizon per epoch, and the pass at horizon
    # H is ~H/H0 times the primary one, because the pypose integrator loop is linear
    # in window length.  Flights shorter than a horizon simply yield no window there,
    # so a long horizon silently uses FEWER flights -- the startup line below prints
    # how many, and it is the number to check before trusting a long column.
    horizon_loaders = {}
    if metric_horizons:
        for h in metric_horizons[1:]:
            hc = copy.deepcopy(conf.dataset.test)
            for entry in hc.data_list:
                entry["window_size"] = h
                entry["step_size"] = h
            hds = SeqeuncesDataset(data_set_config=hc)
            hsampler = (Data.distributed.DistributedSampler(
                hds, num_replicas=world_size, rank=local_rank, shuffle=False,
                drop_last=False) if distributed else None)
            horizon_loaders[h] = Data.DataLoader(
                dataset=hds, batch_size=max(1, per_rank_bs // max(1, h // metric_horizons[0])),
                shuffle=False, sampler=hsampler, collate_fn=collate_fn)
            if is_main():
                print("[metric] extra val horizon %s: %d windows"
                      % (horizon_tag(h), len(hds)))
        if is_main():
            print("[metric] horizons (frames): %s | primary %s drives val_loss and "
                  "checkpoint selection" % (metric_horizons, horizon_tag(metric_horizons[0])))
    # Upstream hardcoded batch_size=1 here; the integrator step is latency-bound, so
    # batch 64 is two orders faster.  drop_last is required above batch 1 because
    # eval.py:58 concatenates per-batch states along dim=-2.
    eval_loader = Data.DataLoader(dataset=eval_dataset, batch_size=conf.train.get('eval_batch_size', 1),
                                  shuffle=False, collate_fn=collate_fn, drop_last=True)

    if is_main():
        os.makedirs(os.path.join(conf.general.exp_dir, "ckpt"), exist_ok=True)
        with open(os.path.join(conf.general.exp_dir, "parameters.yaml"), "w") as f:
            f.write(conf_convert.to_yaml(conf))
    if distributed:
        torch.distributed.barrier()

    network = net_dict[conf.train.network](conf.train).to(device=device, dtype=train_dataset.get_dtype())

    if is_main():
        _sa = _aug_std(conf.train.get("aug_acc_bias_std", 0.0))
        _sg = _aug_std(conf.train.get("aug_gyro_bias_std", 0.0))
        if not conf.train.get("correct_gyro", True):
            _sg = None
        if _sa is not None or _sg is not None:
            _f = lambda t, k: "off" if t is None else np.array2string(
                t.reshape(3).numpy() * k, precision=4)
            print("[bias_augment] ACTIVE on the training split only, redrawn per window: "
                  "acc %s m/s^2, gyro %s deg/s.  train_raw_* will RISE -- the raw "
                  "baseline sees the same augmented signal; val/eval never do."
                  % (_f(_sa, 1.0), _f(_sg, 180.0 / np.pi)))
        else:
            print("[bias_augment] off (set aug_acc_bias_std / aug_gyro_bias_std to enable)")

    if distributed:
        if conf.train.get("sync_bn", False):
            # Otherwise each rank computes BatchNorm statistics on its own shard,
            # which at batch 128 over 10 ranks is ~13 samples per statistic.
            network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(network)
        # correct_gyro False leaves the two gyro heads without gradient (8 tensors).
        # DDP raises "Expected to have finished reduction in the prior iteration" on
        # unused parameters unless it is told to expect them.
        find_unused = conf.train.get("ddp_find_unused_parameters",
                                     not conf.train.get("correct_gyro", True))
        network = torch.nn.parallel.DistributedDataParallel(
            network, device_ids=[device_ids[local_rank]],
            output_device=device_ids[local_rank], find_unused_parameters=find_unused)
        if is_main():
            print("[ddp] find_unused_parameters=%s sync_bn=%s"
                  % (find_unused, conf.train.get("sync_bn", False)))

    # FILTER BY requires_grad.  A frozen Stage A gyro module has requires_grad=False
    # on every parameter; passing network.parameters() unfiltered would still hand them
    # to Adam, and Adam's weight decay / state would mutate them even with zero grads.
    # eval() alone is NOT freezing -- tests/test_gyro_stages.py asserts byte-identity
    # across a real optimizer step.
    _trainable = [q for q in network.parameters() if q.requires_grad]
    _frozen = sum(1 for q in network.parameters() if not q.requires_grad)
    if _frozen and is_main():
        print("[freeze] %d frozen params excluded from the optimizer, %d trainable"
              % (_frozen, sum(q.numel() for q in _trainable)))
    optimizer = torch.optim.Adam(_trainable, lr=conf.train.lr,
                                 weight_decay=conf.train.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=conf.train.factor,
                                  patience=conf.train.patience, min_lr=conf.train.min_lr)
    best_loss = np.inf
    epoch = 0

    # See the EMA / SELECTION block above train() for why and what the columns mean.
    ema_decay = float(conf.train.get("ema_decay", 0.0))
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1), got %r" % ema_decay)
    select_k = max(1, int(conf.train.get("select_smooth", 1)))
    ema = build_ema(network, ema_decay)         # None when ema_decay is 0
    val_history = []                            # per-epoch val loss, index == epoch

    if args.load_ckpt:
        ckpt_path = os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt")
        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            # 'live_state_dict' exists only in checkpoints written with EMA on, and
            # there 'model_state_dict' is the EMA copy (see save_ckpt).
            live_sd = checkpoint.get("live_state_dict", checkpoint["model_state_dict"])
            unwrap(network).load_state_dict(live_sd)
            if ema is not None:
                if "live_state_dict" in checkpoint:
                    ema.load_state_dict(checkpoint["model_state_dict"])
                else:
                    ema.load_state_dict(live_sd)
                    if is_main():
                        print("[ema] checkpoint has no EMA weights: EMA seeded from the live weights")
            elif "live_state_dict" in checkpoint and is_main():
                print("[ema] checkpoint was written with EMA on but ema_decay is 0 now: "
                      "resuming from its LIVE weights")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            # The loop restarts AT `epoch` and recomputes it (EpochLogger._open_csv
            # drops that row too), so keep only the history strictly before it.
            val_history = list(checkpoint.get("val_loss_history", []))[:epoch]
            if is_main():
                print("loaded state dict %s best_loss %f" % (ckpt_path, best_loss))
                if select_k > 1 and len(val_history) < epoch:
                    print("[select] checkpoint carries %d of %d past val losses; no "
                          "best_model.ckpt is written until %d epochs of history exist"
                          % (len(val_history), epoch, select_k))
        elif is_main():
            print("Can't find the checkpoint")

    if is_main():
        n_steps = len(train_loader)
        if ema is not None:
            horizon = 1.0 / (1.0 - ema_decay)
            print("[ema] ACTIVE decay %g: time constant %.0f steps = %.1f epochs at %d steps/epoch.  "
                  "val_*/eval_* columns and model_state_dict in every .ckpt are the EMA weights; "
                  "train_* is the live model, kept under live_state_dict."
                  % (ema_decay, horizon, horizon / n_steps, n_steps))
        else:
            print("[ema] off (ema_decay 0; 0.998 averages ~%.1f epochs at %d steps/epoch)"
                  % (500.0 / n_steps, n_steps))
        # Name the ACTUAL criterion.  This line used to say "val loss" unconditionally,
        # so a run selecting on select_metric: pos_error_120s still announced the loss
        # -- and the banner is what someone reads months later when deciding what a
        # checkpoint means.
        _sm = str(conf.train.get("select_metric", "loss"))
        _what = ("the val objective (loss)" if _sm == "loss"
                 else "val_%s -- NOT the objective" % _sm)
        if select_k > 1:
            print("[select] best_model.ckpt chosen on the MEAN of %s over the last %d "
                  "epochs (never fewer: none is written before epoch %d); best_loss in "
                  "the checkpoint is that mean" % (_what, select_k, select_k - 1))
        else:
            print("[select] best_model.ckpt chosen on the single-epoch %s "
                  "(select_smooth 1)" % _what)

    logger = None
    if is_main():
        logger = EpochLogger(conf.general.exp_dir, start_epoch=epoch,
                             fields=csv_fields(metric_horizons,
                                               conf.train.get('log_raw_baseline', True)),
                             use_tb=conf.train.get("tensorboard", True))

    try:
        for epoch_i in range(epoch, conf.train.max_epoches):
            if train_sampler is not None:
                # Without set_epoch the shuffle is identical every epoch: a real
                # training bug that is invisible in the loss curve.
                train_sampler.set_epoch(epoch_i)

            train_loss = train(network, train_loader, conf.train, epoch_i, optimizer,
                               ema=ema, ema_decay=ema_decay)
            # Validate (and below, evaluate) the model that gets SAVED and SELECTED:
            # the EMA copy when it is on.  The live model is validated only when
            # EMA is off, so val_* columns always describe model_state_dict.
            eval_net = network if ema is None else ema
            test_loss = test(eval_net, test_loader, conf.train, epoch_i)
            # Extra validation horizons.  Same flights, longer integration.
            # Log the whole batch, not this rank's shard.
            train_loss = reduce_metrics(train_loss, device)
            test_loss = reduce_metrics(test_loss, device)

            # AFTER the all-reduce, not before.  This used to store `test_loss` into
            # val_by_horizon and only then rebind the name to the reduced dict, so
            # under DDP the PRIMARY horizon's val_pos_error / val_vel_error columns
            # were rank 0's shard while every other horizon was the true all-reduced
            # value -- two different quantities in adjacent columns of one row, and
            # the primary is the one checkpoint selection reads.  Single-GPU runs were
            # unaffected (reduce_metrics is a no-op there), which is why it survived.
            val_by_horizon = {}
            if metric_horizons:
                val_by_horizon[metric_horizons[0]] = test_loss
                for _h, _ld in horizon_loaders.items():
                    val_by_horizon[_h] = reduce_metrics(
                        test(eval_net, _ld, conf.train, epoch_i), device)
            lr = scheduler.optimizer.param_groups[0]['lr']

            if is_main():
                print("epoch %03d | lr %.3e | train loss %.6f pos %.4f m vel %.4f m/s"
                      " | val loss %.6f pos %.4f m vel %.4f m/s"
                      % (epoch_i, lr,
                         train_loss["loss"], train_loss["pos_loss"], train_loss["vel_loss"],
                         test_loss["loss"], test_loss["pos_loss"], test_loss["vel_loss"]))

            eval_metrics = None
            if epoch_i % conf.train.eval_freq == conf.train.eval_freq - 1:
                # evaluate() concatenates per-batch states and is not sharded, so run
                # it on rank 0 while the others wait rather than duplicating it.
                if is_main():
                    eval_state = evaluate(network=unwrap(eval_net), loader=eval_loader, confs=conf.train)
                    eval_metrics = {
                        "pos_error": eval_state['loss']['pos_dist'].mean().item(),
                        "vel_error": eval_state['loss']['vel_dist'].mean().item(),
                        "pos_rmse":  eval_state['loss']['pos'].mean().item(),
                        "vel_rmse":  eval_state['loss']['vel'].mean().item(),
                    }
                    print("eval pos: %f m (rmse %f) eval vel: %f m/s (rmse %f)"
                          % (eval_metrics["pos_error"], eval_metrics["pos_rmse"],
                             eval_metrics["vel_error"], eval_metrics["vel_rmse"]))
                if distributed:
                    torch.distributed.barrier()

            if logger is not None:
                logger.log_epoch(epoch_i, lr, train_loss, test_loss, eval_metrics,
                                 horizons=metric_horizons or None,
                                 val_by_horizon=val_by_horizon,
                                 log_raw=conf.train.get("log_raw_baseline", True))

            # Every rank steps the scheduler on the SAME reduced value, so the LR
            # cannot diverge between ranks.
            scheduler.step(test_loss['loss'])
            # ---- WHAT best_model.ckpt IS CHOSEN ON --------------------------------
            # `select_metric` names the validation quantity to minimise.  Default
            # "loss" is the old rule, bit for bit.
            #
            # WHY IT IS WORTH CHANGING.  The objective and the reported metric are not
            # the same quantity and they do not bottom out together.  MEASURED on
            # velnet_v1 (40 epochs): val_loss -- Huber on body-frame velocity plus the
            # covariance NLL -- was best at epoch 17, while val_pos_error_120s was best
            # at epoch 5 and val_pos_error_60s at epoch 8.  Selecting on the loss
            # therefore shipped a checkpoint 5.6% worse at 120 s than one the run had
            # already passed through, and with save_freq 100 those weights were gone.
            #
            # Accepted values: "loss" (the objective) or any horizon column name that
            # the metric schema already produces, e.g. "pos_error_120s",
            # "vel_error_60s".  Unknown names raise at the FIRST selection rather than
            # silently falling back, because a silent fallback here means an entire
            # multi-hour run was selected on something other than what was asked for.
            # The CSV column is `val_pos_error_120s`; the dict key behind it is
            # `pos_loss`.  Accept the column name, because that is what the user
            # reads and therefore what they will write in the config.
            _COLUMN_TO_KEY = {"pos_error": "pos_loss", "vel_error": "vel_loss",
                              "pos_loss": "pos_loss", "vel_loss": "vel_loss"}
            _sel = str(conf.train.get("select_metric", "loss"))
            if _sel == "loss":
                _crit_value = test_loss['loss']
            else:
                _h_frames, _base = None, None
                for _h in (metric_horizons or []):
                    _suffix = "_" + horizon_tag(_h)
                    if _sel.endswith(_suffix):
                        _h_frames, _base = _h, _sel[:-len(_suffix)]
                        break
                _src = (val_by_horizon or {}).get(_h_frames) if _h_frames else None
                _key = _COLUMN_TO_KEY.get(_base)
                if _src is None or _key is None or _src.get(_key) is None:
                    raise KeyError(
                        "select_metric=%r is not available. Use 'loss', or "
                        "<channel>_<horizon> with channel in %s and horizon in %s -- "
                        "e.g. 'pos_error_120s'.  Horizons come from metric_horizons."
                        % (_sel, sorted(set(_COLUMN_TO_KEY)),
                           [horizon_tag(h) for h in (metric_horizons or [])]))
                _crit_value = float(_src[_key])
            val_history.append(_crit_value)
            criterion = smoothed_criterion(val_history, select_k)
            save_best = criterion < best_loss
            if save_best:
                best_loss = criterion
            if is_main():
                # unwrap: the DDP wrapper prefixes every state_dict key with
                # "module.", which breaks every single-GPU tool in this repo.
                save_ckpt(unwrap(network), optimizer, scheduler, epoch_i, best_loss,
                          conf, save_best=save_best, ema=ema, val_history=val_history)
            if distributed:
                torch.distributed.barrier()
    finally:
        if logger is not None:
            logger.close()
        if distributed:
            ddp_cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/exp/UAV/hybrid_best.conf', help='config file path')
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="'cpu', 'cuda:0', '0', or a comma-separated GPU list "
                             "'0,1,2,...' for multi-GPU DDP (one process per GPU)")
    parser.add_argument('--load_ckpt', default=False, action="store_true",
                        help="If True, try to load the newest.ckpt in the exp_dir specified in the config.")
    parser.add_argument('--seed', type=int, default=0,
                        help="RNG seed for weight init, shuffling and dropout. A `seed:` "
                             "key in the train config overrides this. Runs before 2026-09-08 "
                             "were UNSEEDED, so they are not reproducible and their "
                             "config-to-config differences confound seed with config.")
    parser.add_argument('--detect_anomaly', default=False, action="store_true",
                        help="enable torch autograd anomaly detection (11-12x slower; debugging only)")
    args = parser.parse_args()
    print(args)

    kind, device_ids = parse_devices(args.device)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Launched by torchrun, which already created the processes: do not spawn.
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", rank))
        ids = device_ids if len(device_ids) == world else list(range(world))
        ddp_setup(rank, world, ids[local])
        main_worker(local, ids, args)
    elif len(device_ids) > 1:
        import torch.multiprocessing as mp
        print("[device] spawning %d DDP processes on GPUs %s (backend %s)"
              % (len(device_ids), device_ids, pick_backend()))
        mp.spawn(main_worker, args=(device_ids, args), nprocs=len(device_ids), join=True)
    else:
        main_worker(0, device_ids, args)
