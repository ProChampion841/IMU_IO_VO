"""HybridNet: a two-branch (GRU + Mamba) IMU correction network.

What this is
------------
A drop-in replacement for ``CodeNet`` / ``CodePoseNet`` that answers four
requests at once:

1. **It emits corrected signals, not corrections.**  ``inference()`` returns
   ``corrected_acc`` / ``corrected_gyro`` as the primary output.  It *also*
   returns ``correction_acc`` / ``correction_gyro`` (= corrected - raw), because
   the whole downstream stack -- ``inference.py``, ``SeqInfDataset``,
   ``evaluation/evaluate_state.py``, ``tools/eval_horizons.py``,
   ``tools/eval_covariance.py`` -- consumes the *difference*.  Emitting both is
   not redundancy, it is the interface change the user asked for without a
   flag-day rewrite of five other files.

2. **Two branches with genuinely different time constants.**  The CNN token
   stream (one token per ``interval`` = 9 frames = 90 ms) is fed to

     * a **GRU branch** at full token rate -- the short-term, low-latency path.
       ``gru_window`` (in FRAMES; 0 or absent = the whole window, the original
       behaviour) runs it over non-overlapping chunks with ``h0 = 0`` at each
       chunk start, so its memory is a HARD cutoff rather than a soft decay;
     * a **Mamba branch** at ``mamba_stride`` token rate -- the long-horizon
       path.  Its state updates once every ``9 * mamba_stride`` frames, so at
       stride 8 it advances once per 0.72 s and at stride 16 once per 1.44 s.

   They are then fused.  See ``tools/hybrid_receptive_field.py`` for the
   *measured* influence-decay of each branch; the answer is not "whatever the
   architecture diagram says", and it depends on ``window_size``.

   What actually sets each horizon, measured on this box at initialisation
   (``python -m tools.hybrid_receptive_field``, float64, window 4000):

     * GRU, unbounded: influence is at 1/e by 0.45 s and at 1% by 1.26 s, and it
       hits the float64 noise floor past ~5 s.  ``gru_window: 300`` (= 33 tokens
       = 297 frames = 2.97 s) does not shorten that soft decay -- it makes it
       EXACT: no output can depend on a token more than 32 tokens back, and an
       output sitting on a chunk boundary depends on nothing before itself.
     * Mamba: influence falls to 1% after ~7 *SSM steps* regardless of what a
       step is worth, so the horizon in SECONDS is ``steps x stride x 0.09 s``
       and ``mamba_stride`` is the lever -- linearly.  Measured: stride 8 ->
       1% at 5.04 s, stride 16 -> 1% at 10.08 s, stride 24 -> 1% at 15.12 s.
       ``dt_min``/``dt_max`` are NOT a lever and are deliberately not exposed:
       shrinking dt slows the state decay but shrinks the input gain
       ``delta*B*u`` by the same factor, and the measured decay over the first
       four steps is identical (1.000 / 0.0397 / 0.039 / 0.00126) across a 100x
       range of dt.
     * ``window_size`` is not an influence lever at initialisation either -- it
       is the CEILING on what training can later learn, because ``h0 = 0`` at
       every window start for both branches, and it sets ``K = ceil(T/stride)``,
       the number of SSM steps.  Raising it is the expensive direction: the
       selective_scan backward is O(K^2) here, and 60-88% of a training step is
       the pypose integrator's Python chunk loop, which is linear in
       ``window_size`` and does not care about any of this.

3. **Attitude as an input.**  Not Euler angles -- see ``model/attitude.py`` for
   why -- but ``g_body = R.Inv() @ [0,0,1]``, the yaw-invariant part of the
   attitude, optionally plus sin/cos of roll and pitch.

4. **The rotation loss is optional.**  That lives in ``model/losses.py``
   (``use_rot_loss``), not here; this file only has to make the gyro correction
   *meaningful* when it is off, which is what ``gtrot: False`` does.

The initialisation trap, and how the "corrected output" request is honoured
--------------------------------------------------------------------------
A head that regresses the corrected specific force *directly* emits ~0 at
initialisation, i.e. "the aircraft is in free fall with no thrust".  The
preintegrator then diverges on step one, and the quantity we actually want to
learn (a ~0.03-0.06 m/s^2 bias) is 0.15-0.3% of an output whose dynamic range is
+-20 m/s^2 -- a signal-to-representation ratio that no optimiser will find.

So the *interface* returns corrected values while the *parameterisation* keeps a
skip connection:

    corrected = raw * (1 + s) + b           # correction_mode: "affine"  (default)
    corrected = raw + b                     # correction_mode: "additive"
    corrected = direct_scale * head(feat)   # correction_mode: "direct"  (see below)
    corrected = Exp(dtheta) @ raw + b       # correction_mode: "rotate"  (acc only;
                                            #   see _rotate_small for why)

``affine`` is the default because of a measured property of this corpus: fitting
one constant accel bias per split gives a bias *direction* that is essentially a
corpus constant (cosine 0.9652 train-vs-val, 0.9951 train-vs-test) while its
*magnitude* nearly doubles across splits (0.03054 / 0.04647 / 0.06275 m/s^2).
What varies flight to flight is a scale along a fixed axis, not a free 3-vector.
A diagonal scale term expresses that directly: in near-level flight
``raw ~ +9.81 * g_body`` (measured: whole-flight mean(raw - 9.81*g_body) is
[0.20, -0.12, 0.48] m/s^2 against [1.24, 0.18, 19.41] for the minus sign), so
``raw * s`` is a bias along the gravity direction
whose magnitude is ``9.81 * s`` -- exactly the observed structure, and with
``scale_std = 0.05`` its range (+-0.49 m/s^2) is ~10x the observed spread.

``direct`` is implemented for completeness and is expected to be bad.  It is the
literal "regress the corrected value with no skip" reading of the request; the
module prints a warning when it is selected.

The two coherent gyro configurations
------------------------------------
``gtrot: True``  -- the integrator takes its rotation from the external attitude
    solution for gravity removal.  Note (measured, not assumed) that this does
    **not** make the gyro gradient zero in pypose 0.9.5: ``incre_r``, the
    cumulative ``Exp(gyro*dt)`` product, still rotates the accel into the
    integration frame, so d(pos+vel loss)/d(corrected_gyro) has norm 4.06e-2
    versus 2.85e-1 at ``gtrot: False``.  It is attenuated ~7x and chunk-local
    (``sampling: 50``), not absent.  If you want a genuinely inert gyro path you
    must say so: set ``correct_gyro: False``, which is what
    ``hybrid_v6_gtrot.conf`` does and what ``CodePoseNet`` did by hard-coding it.

``gtrot: False`` -- rotation is propagated inside each chunk from the corrected
    gyro, initialised from ``init_state['rot']``.  The pos/vel loss then
    supervises the gyro correction through gravity leakage.  This is the
    configuration in which "output a corrected gyro" actually means something,
    and it is the default (``hybrid_v6.conf``).

Attitude source
---------------
``att_source: "gt"`` uses ``data['rot']`` (the GPS-aided nav filter).  That is
partially circular: the pos/vel labels come from the same filter.
``att_source: "mti"`` uses ``data['mti_rot']``, the independent magnetometer/MTI
solution, which is noisier (median g_body error 2.0 deg, p99 8.4 deg) but is
what the aircraft will actually have at runtime.  Because the feature is
yaw-invariant, the MTI's unusable heading (within-flight drift of tens of
degrees) does not enter.  The *integrator* is still initialised from
``init_state['rot']``; initialising it from the MTI would rotate the whole
integrated trajectory in the horizontal plane by the heading error and the
pos/vel loss would be dominated by that rather than by IMU error.
"""

import numpy as np
import torch
import torch.nn as nn

from model.attitude import (attitude_feature, attitude_feature_dim, input_dim,
                            pad_rotation, select_attitude)
from model.cnn import CNNEncoder
from model.mamba_block import MambaBranch, has_mamba
from model.net import ModelBase


def _head(d_in, d_hidden, d_out=3):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))


def _zero_last(seq):
    """Zero the final Linear of a head so it emits exactly 0 at initialisation.

    The upstream heads rely on ``default_init * acc_std`` being "small".  It is
    not especially small: measured, the untrained CodeNet emits a correction of
    ~0.02-0.03 m/s^2, i.e. the same size as the bias it is supposed to learn, so
    an untrained network is not the identity in any useful sense.  Zeroing the
    last layer makes the identity exact at step 0 without harming trainability:
    the gradient w.r.t. that layer's weight is ``delta * penultimate_activation``
    and the penultimate layer is still randomly initialised, so there is no
    symmetry to break.
    """
    last = seq[-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


class HybridNet(ModelBase):
    """Parallel GRU (short-term) + Mamba (long-term) correction network."""

    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf

        gyro_std = np.pi / 180
        if "gyro_std" in conf:
            gyro_std = conf.gyro_std
        self.register_buffer("gyro_std", torch.tensor(gyro_std))

        acc_std = 0.1
        if "acc_std" in conf:
            acc_std = conf.acc_std
        self.register_buffer("acc_std", torch.tensor(acc_std))

        # Output affine for correction_mode "direct".  A direct head has no skip from
        # the raw signal, so at initialisation it emits O(1) values -- in physical units
        # that is "zero specific force", and the preintegrator diverges before any
        # learning happens.  Mapping the head through the corpus statistics
        #     corrected = head(feature) * std + mean
        # puts it in the right physical range from step 0: with the last layer
        # zero-initialised the network starts by emitting the MEAN specific force
        # (~9.96 m/s^2 on z, i.e. gravity), which integrates smoothly, instead of
        # free-fall.  Measured on the 57 training flights, 3,379,618 airborne frames.
        _dstat = lambda k, d: torch.tensor(
            [float(v) for v in conf.get(k, d)], dtype=torch.get_default_dtype())
        self.register_buffer("direct_acc_mean", _dstat(
            "direct_acc_mean", [0.558294, -0.023284, 9.960953]))
        self.register_buffer("direct_acc_std", _dstat(
            "direct_acc_std", [1.998676, 1.474464, 4.209704]))
        self.register_buffer("direct_gyro_mean", _dstat(
            "direct_gyro_mean", [0.002250, -0.019711, -0.001787]))
        self.register_buffer("direct_gyro_std", _dstat(
            "direct_gyro_std", [0.224214, 0.215214, 0.133847]))

        # The correction is piecewise-constant over `interval` frames because the
        # CNN downsamples by stride 3 x stride 3.  interval is a CONSEQUENCE of
        # the encoder stride, not an independent knob -- keep them in sync.
        self.interval = 9
        # `causal_cnn: True` makes the whole correction path causal.  Two things have
        # to move together: the CNN pads on the left only (model/cnn.py), and a frame
        # is owned by the token that ENDS at or before it (inter_head 0) instead of
        # the token centred on it.  MEASURED before this switch: the correction at
        # window frame f depended on frames up to f+12 (120 ms of future IMU).
        # Default False keeps every existing checkpoint bit-identical.
        self.causal_cnn = bool(conf.get("causal_cnn", False))
        self.inter_head = 0 if self.causal_cnn else int(np.floor(self.interval / 2.0))
        self.inter_tail = self.interval - self.inter_head

        self.att_input = str(conf.get("att_input", "gravity"))
        self.att_source = str(conf.get("att_source", "gt"))
        self.correction_mode = str(conf.get("correction_mode", "affine"))
        if self.correction_mode not in ("affine", "additive", "direct", "rotate"):
            raise ValueError("correction_mode must be affine|additive|direct|rotate, got %r"
                             % self.correction_mode)
        self.scale_std = float(conf.get("scale_std", 0.05))
        # correction_mode "rotate": corrected_acc = Exp(dtheta) @ raw_acc + b.
        # dtheta is a small body-frame ATTITUDE correction, bounded to rotate_max_deg
        # by a tanh.  See _rotate_small() for why this is the right functional form for
        # the error measured on this corpus.  2 deg is ~5x the 0.387 deg tilt that the
        # fitted "bias" corresponds to.
        self.rotate_max = float(conf.get("rotate_max_deg", 2.0)) * np.pi / 180.0
        # `cov_stop_grad: True` feeds the covariance heads a DETACHED feature, so the
        # covariance NLL trains only its own two heads and cannot pull the shared trunk.
        # MEASURED on the tilt_aware run: cov_loss is ~0.219 of a ~0.678 val_loss (32%),
        # it bottomed at epoch 42 while the correction kept improving to epoch 120-930,
        # i.e. the two objectives want different trunks.  Default False == before.
        self.cov_stop_grad = bool(conf.get("cov_stop_grad", False))
        # `cov_model` / `cov_bias_std_init` / `cov_calib_lr_mult` appear in
        # tilt_aware.conf but NO code in this repository reads them, so that run used
        # the plain propagated covariance.  Say so instead of ignoring them silently.
        _cm = conf.get("cov_model", None)
        if _cm not in (None, "propagated"):
            print("[hybrid] WARNING: cov_model=%r is not implemented in this code base "
                  "and is IGNORED (cov_bias_std_init / cov_calib_lr_mult likewise). "
                  "The covariance is the plain propagated one." % (_cm,))
        self.direct_scale = float(conf.get("direct_scale", 10.0))
        self.correct_gyro = bool(conf.get("correct_gyro", True))

        # `gyro_source` ("native" / "identity" / "pretrained") and the frozen Stage A
        # GyroCorrector it loaded were removed on 2026-09-07 along with the gyro-first
        # tooling. The path was unreachable: no config set it, no Stage A checkpoint
        # existed to load, and its trainer (tools/train_gyro_stage1.py) is gone. The
        # line itself is measured closed -- the Stage A module learned no per-flight
        # structure (per-flight cosine 0.171 against corpus-mean 0.993) and a
        # module-free corpus constant matched or beat it.
        # `correct_gyro: False` already does everything "identity" did.
        # Recover with: git show <commit>^:model/gyro_corrector.py

        gru_hidden = int(conf.get("gru_hidden", 256))
        # `gru_window` is in FRAMES -- the unit every other window key in this
        # repo uses, and the unit the request was made in -- but the GRU runs on
        # TOKENS, one per `interval` = 9 frames.  Round to the nearest whole
        # token: 300 / 9 = 33.33 -> 33 tokens = 297 frames = 2.97 s at 100 Hz.
        # (Nearest rounds down here, so the realised window is <= the requested
        # one; the difference is 3 frames = 30 ms.)  0 or absent means one chunk
        # = the whole window, i.e. exactly the previous behaviour.
        gru_window_frames = int(conf.get("gru_window", 0))
        self.gru_window = (max(1, int(round(gru_window_frames / float(self.interval))))
                           if gru_window_frames > 0 else 0)
        self.gru_window_frames = self.gru_window * self.interval
        mamba_dim = int(conf.get("mamba_dim", 128))
        mamba_layers = int(conf.get("mamba_layers", 2))
        self.mamba_stride = int(conf.get("mamba_stride", 8))
        # See _correct() for what this does and why it is a CUMULATIVE (causal) mean.
        self.const_correction = bool(conf.get("const_correction", False))
        self.fuse_mode = str(conf.get("fuse", "concat"))
        if self.fuse_mode not in ("concat", "gate"):
            raise ValueError("fuse must be concat|gate, got %r" % self.fuse_mode)
        feat_dim = int(conf.get("fuse_dim", gru_hidden))
        head_hidden = int(conf.get("head_hidden", 128))
        cnn_dim = int(conf.get("cnn_dim", 64))

        # Input width is 6 / 9 / 13 depending on att_input, plus 1 for airspeed --
        # built from the real width, never hard-coded to 6.
        self.in_dim = input_dim(self.att_input) + (
            1 if bool(conf.get("use_airspeed", False)) else 0)

        # ------------------------------------------------------------------
        # INPUT CHANNEL SCALING -- `normalize_input: True`
        # ------------------------------------------------------------------
        # The raw channels are concatenated and handed straight to a Conv1d.  They
        # are in wildly different physical units, and nothing normalises them: the
        # BatchNorm in CNNEncoder (model/cnn.py:16-21) sits on the conv OUTPUT, which
        # rescales the sum but cannot restore a channel whose contribution to that sum
        # has already been swamped.
        #
        # MEASURED from the corpus std buffers registered just above:
        #     acc  var 23.890  ->  98.93%
        #     gyro var  0.1145 ->   0.474%      acc:gyro variance ratio 208.6 : 1
        #     att  var  0.1428 ->   0.591%
        # At Kaiming init every input channel gets the same weight variance, so those
        # shares ARE each group's share of the first-layer pre-activation variance.
        # The gyro channel enters the network at half a percent of the signal and has
        # to grow its weights ~15x to compete -- against a weight_decay that penalises
        # exactly that.  An independent measurement put the gyro at 0.44% of the first
        # conv at init and still only 0.73% after five epochs.
        #
        # This is the SAME bug this file already fixes one layer later: `short_norm`
        # (see the fuse section) exists because the GRU branch entered the fusion at
        # 1.2% against the Mamba branch's 98.8%.  The argument was never made at the
        # input.
        #
        # Dividing by a fixed per-channel scale is the whole fix.  Mean-centring is
        # unnecessary -- the following BatchNorm removes the DC offset, including
        # acc_z's +9.96 -- but the SCALE is not something BatchNorm can undo.
        #
        # WHY att uses one shared RMS rather than per-channel std: with
        # att_input "gravity_sincos", model/attitude.py returns
        # [gx, gy, gz, sin_r, cos_r, sin_p, cos_p] where sin_p = -gx EXACTLY
        # (attitude.py:129) and cos_p/cos_r are near-constant.  Per-channel
        # standardisation would amplify near-constant channels to full scale and
        # duplicate gx at equal weight.  A single group scale leaves their relative
        # magnitudes alone.  The 0.1661 default is the MEASURED group RMS std of g_body
        # over 774,912 airborne training frames (per-channel std
        # [0.1424, 0.2139, 0.1294]); acc is 2.8220 and gyro 0.1954 by the same
        # measure, so this puts all three groups at ~1/3 of the input variance.
        # `att_input: gravity` avoids the duplicate-channel issue entirely and loses
        # nothing: all four extra channels are deterministic functions of (gx,gy,gz).
        # ------------------------------------------------------------------
        # AIRSPEED -- `use_airspeed: True`, one extra input channel
        # ------------------------------------------------------------------
        # The ONLY channel on this corpus that observes something an IMU cannot.
        # MEASURED on the eval split: body-frame ground velocity carries 5.36 m/s
        # of speed variation and an implied 14.1 deg rms sideslip (p95 24.8 deg) --
        # a fixed wing does not sideslip 25 deg, that is wind-driven crab.  Neither
        # is visible to an accelerometer: level flight at 15 and at 30 m/s produce
        # the same specific force, and no inertial sensor feels wind.  A pitot does.
        #
        # It is a genuine onboard sensor, independent of the GPS solution that
        # supplies the labels, so it is NOT leakage -- unlike `att_source: gt`.
        # It is biased and uncalibrated (a fitted scale of 0.93-1.38 flight to
        # flight, see UAVdataset.py), which the network has to absorb.
        #
        # CENTRED, not just scaled.  Airspeed is mean 22.47 with std 2.99 (MEASURED
        # on the 57 TRAIN flights, 3,380,162 airborne frames -- train only, so no
        # val/test statistic enters).  Dividing by std alone would hand the first
        # Conv1d a channel sitting at 7.5 with unit variation, i.e. almost pure DC.
        # The BatchNorm downstream removes that DC from the conv OUTPUT, but by then
        # the useful 1-sigma signal has already been summed in at 1/7.5 of the
        # channel's weight. Subtracting the mean first is what makes the variation
        # the signal.
        self.use_airspeed = bool(conf.get("use_airspeed", False))
        self.airspeed_mean = float(conf.get("airspeed_mean", 22.4659))
        self.airspeed_std = float(conf.get("airspeed_std", 2.9918))

        self.normalize_input = bool(conf.get("normalize_input", False))
        _att_dim = attitude_feature_dim(self.att_input)
        _scale = [self.direct_acc_std, self.direct_gyro_std]
        _offset = [torch.zeros(3), torch.zeros(3)]
        if _att_dim:
            _scale.append(torch.full((_att_dim,), float(conf.get("att_input_scale", 0.1661))))
            _offset.append(torch.zeros(_att_dim))
        if self.use_airspeed:
            _scale.append(torch.tensor([self.airspeed_std]))
            _offset.append(torch.tensor([self.airspeed_mean]))
        self.register_buffer("in_scale", torch.cat(_scale).clamp_min(1e-3))
        # Zero everywhere except airspeed, so `(x - in_offset) / in_scale` is
        # bit-identical to the old `x / in_scale` for every pre-existing config.
        self.register_buffer("in_offset", torch.cat(_offset))
        self.cnn = CNNEncoder(c_list=[self.in_dim, 32, cnn_dim], k_list=[7, 7], s_list=[3, 3],
                              causal=self.causal_cnn)

        # `branches` selects which of the two paths is BUILT.  "mamba" and "gru" skip
        # constructing the other branch entirely rather than building it and not calling
        # it: an unused parameter makes DDP raise unless find_unused_parameters is on,
        # and that flag costs a full graph traversal every step.  It also keeps the
        # checkpoint free of dead tensors, so a mamba-only run cannot silently load
        # GRU weights.
        self.branches = str(conf.get("branches", "both"))
        if self.branches not in ("both", "mamba", "gru"):
            raise ValueError("branches must be both|mamba|gru, got %r" % (self.branches,))
        self.use_gru = self.branches in ("both", "gru")
        self.use_mamba = self.branches in ("both", "mamba")
        if self.branches != "both" and self.fuse_mode == "gate":
            # A gate between one branch and nothing is just a learned scalar on that
            # branch; say so rather than silently training a meaningless sigmoid.
            raise ValueError("fuse: gate needs two branches; use fuse: concat with "
                             "branches: %s" % self.branches)

        # --- short-term branch: full token rate --------------------------------
        if self.use_gru:
            self.gru1 = nn.GRU(input_size=cnn_dim, hidden_size=128, num_layers=1, batch_first=True)
            self.gru2 = nn.GRU(input_size=128, hidden_size=gru_hidden, num_layers=1,
                               batch_first=True)
        # A bare GRU emits std ~0.077 while MambaBranch ends in a LayerNorm and emits
        # std ~1.0.  Concatenated into one Linear, the fused variance splits purely by
        # scale x fan-in -- 0.077^2*256 : 1.0^2*128 = 1.2% : 98.8% -- so the short branch
        # would carry ~2% of the signal for 40% of the parameters, before any training.
        # Normalising here makes the split reflect usefulness rather than output scale.
            self.short_norm = nn.LayerNorm(gru_hidden)

        # --- long-term branch: 1/mamba_stride token rate -----------------------
        if self.use_mamba:
            self.mamba = MambaBranch(cnn_dim, mamba_dim, n_layer=mamba_layers,
                                     stride=self.mamba_stride)

        # --- fusion ------------------------------------------------------------
        fuse_in = (gru_hidden if self.use_gru else 0) + (mamba_dim if self.use_mamba else 0)
        if self.fuse_mode == "concat":
            self.fuse_lin = nn.Linear(fuse_in, feat_dim)
        else:
            self.gru_proj = nn.Linear(gru_hidden, feat_dim)
            self.mamba_proj = nn.Linear(mamba_dim, feat_dim)
            self.gate_lin = nn.Linear(gru_hidden + mamba_dim, feat_dim)
        self.fuse_act = nn.GELU()
        self.fuse_norm = nn.LayerNorm(feat_dim)
        # Opt-in dropout on the fused feature -- the one tensor every head reads,
        # so it regularises all six heads at once.  0.0 (default) is an EXACT
        # no-op: nn.Identity draws no random numbers, so train() and eval()
        # outputs are bit-identical to the pre-existing behaviour.
        self.fuse_dropout = float(conf.get("fuse_dropout", 0.0))
        if not 0.0 <= self.fuse_dropout < 1.0:
            # nn.Dropout rejects >= 1 itself; a NEGATIVE value would otherwise fall
            # through to Identity and silently train with no dropout at all.
            raise ValueError("fuse_dropout must be in [0, 1), got %r" % self.fuse_dropout)
        self.fuse_drop = (nn.Dropout(self.fuse_dropout) if self.fuse_dropout > 0.0
                          else nn.Identity())

        # --- heads --------------------------------------------------------------
        self.accdecoder = _head(feat_dim, head_hidden)
        self.gyrodecoder = _head(feat_dim, head_hidden)
        self.accscale_decoder = _head(feat_dim, head_hidden)
        self.gyroscale_decoder = _head(feat_dim, head_hidden)
        self.acccov_decoder = _head(feat_dim, head_hidden)
        self.gyrocov_decoder = _head(feat_dim, head_hidden)

        if bool(conf.get("zero_init_correction", True)):
            for h in (self.accdecoder, self.gyrodecoder,
                      self.accscale_decoder, self.gyroscale_decoder):
                _zero_last(h)

        if self.correction_mode == "direct":
            print("[hybrid] correction_mode='direct': the heads regress the corrected "
                  "signal itself, with NO skip from the raw input. Output is mapped "
                  "through the corpus mean/std and the last layer is zero-initialised, "
                  "so step 0 emits the mean specific force rather than free fall. Note "
                  "the network must reconstruct the full dynamics while the error to be "
                  "removed is ~0.15%% of the output; hybrid_v6.conf (affine) is the "
                  "control arm.")
        gw = ("whole window" if self.gru_window == 0
              else "%d tok = %d frames = %.2f s" % (self.gru_window, self.gru_window_frames,
                                                    self.gru_window_frames / 100.0))
        print("[hybrid] branches=%s in_dim=%d (att_input=%s, att_source=%s) mode=%s%s "
              "causal_cnn=%s cov_stop_grad=%s "
              "fuse=%s fuse_dropout=%.2f%s%s correct_gyro=%s params=%d"
              % (self.branches, self.in_dim, self.att_input, self.att_source,
                 self.correction_mode,
                 ("(max %.2f deg)" % np.degrees(self.rotate_max))
                 if self.correction_mode == "rotate" else "",
                 self.causal_cnn, self.cov_stop_grad,
                 self.fuse_mode, self.fuse_dropout,
                 (" gru(hidden=%d, window=%s)" % (gru_hidden, gw)) if self.use_gru else "",
                 (" mamba(dim=%d, layers=%d, stride=%d -> %.2f s/step, kernel=%s)"
                  % (mamba_dim, mamba_layers, self.mamba_stride,
                     self.mamba_stride * self.interval / 100.0,
                     "mamba-ssm" if has_mamba() else "pure-pytorch")) if self.use_mamba else "",
                 self.correct_gyro, sum(p.numel() for p in self.parameters())))

    # ------------------------------------------------------------------
    # feature extraction
    # ------------------------------------------------------------------
    def _attitude_channels(self, data):
        """(B, F, C) attitude feature aligned with the (possibly padded) acc/gyro.

        ``padding9`` grows acc/gyro to ``window_size + 9`` but leaves ``rot`` at
        ``window_size``.  Concatenating without padding the rotation either
        crashes or -- worse -- silently misaligns the attitude by 9 samples.  The
        pad is ``init_rot`` repeated, matching what ``padding_collate`` invents
        for acc.  ``rot[:, :1]`` *is* ``init_state['rot']`` for every dataset in
        this repo (both ``SeqeuncesDataset`` and ``SeqDataset`` slice the same
        array from the same frame_id), which is what lets ``inference()`` work
        from ``data`` alone -- ``inference.py`` never passes ``init_state``.
        """
        if self.att_input == "none":
            return None
        rot, _src = select_attitude(data, source=self.att_source)
        pad_len = data["acc"].shape[1] - rot.lshape[1]
        if pad_len < 0:
            raise RuntimeError(
                "rotation is longer than acc (%d vs %d); the collate is not one of "
                "the padding_collate family" % (rot.lshape[1], data["acc"].shape[1]))
        if pad_len > 0:
            rot = pad_rotation(rot, rot[:, :1], pad_len)
        return attitude_feature(rot, self.att_input).to(data["acc"].dtype)

    def _airspeed_channel(self, data):
        """(B, F, 1) pitot airspeed aligned with the (possibly padded) acc/gyro.

        `padding_collate` grows airspeed by the same `pad_len` it grows acc/gyro by,
        so under every collate in this repo it already matches.  This raises rather
        than broadcasting if it does not: a length mismatch here would silently
        misalign speed against the IMU by `pad_len` samples, which is exactly the
        class of bug `pad_rotation` exists to prevent for the attitude channel.
        """
        va = data.get("airspeed", None)
        if va is None:
            raise RuntimeError(
                "use_airspeed is True but the batch carries no 'airspeed'. The UAV "
                "loader publishes it and datasets/dataset.py forwards it; a dataset "
                "without a pitot cannot run this config.")
        if va.shape[1] != data["acc"].shape[1]:
            raise RuntimeError(
                "airspeed is %d frames against acc's %d -- the collate did not pad "
                "them together, so the channels are misaligned"
                % (va.shape[1], data["acc"].shape[1]))
        return va.to(data["acc"].dtype)

    def _net_input(self, data):
        """Assemble and normalise the encoder input: (B, F, in_dim).

        ONE definition, shared by HybridNet and VelocityNet.  They used to keep a
        copy each, which is how an input channel gets added to one model and not the
        other and the difference shows up as an unexplained accuracy gap.
        """
        channels = [data["acc"], data["gyro"]]
        att = self._attitude_channels(data)
        if att is not None:
            channels.append(att)
        if self.use_airspeed:
            channels.append(self._airspeed_channel(data))
        net_in = torch.cat(channels, dim=-1)
        if self.normalize_input:
            # See the in_scale / in_offset buffers in __init__.  in_offset is zero
            # for every channel but airspeed, so this is bit-identical to the old
            # `net_in / in_scale` whenever use_airspeed is False.
            net_in = (net_in - self.in_offset.to(net_in.dtype)) / self.in_scale.to(net_in.dtype)
        return net_in

    def _gru_stack(self, tokens):
        h, _ = self.gru1(tokens)                                      # h0 = 0
        h, _ = self.gru2(h)
        return h                                                      # (B, T, gru_hidden)

    def short_branch(self, tokens):
        """(B, T, cnn_dim) -> (B, T, gru_hidden).  Short-term branch, pre-LayerNorm.

        With ``gru_window`` unset this is the original two-layer GRU over the
        whole window.  With it set, the GRU runs over NON-OVERLAPPING chunks of
        ``self.gru_window`` tokens with ``h0 = 0`` at the start of each chunk.
        That is a hard cutoff, not a faster decay: output ``u`` is a function of
        tokens ``[floor(u/W)*W .. u]`` and of nothing else, so the maximum
        lookback is ``W - 1`` tokens and an output sitting on a chunk boundary
        looks back zero.  (A sliding window would give every output the same
        ``W`` tokens of history, but costs one full GRU pass per offset;
        chunking is what was asked for and is free.)

        The chunk axis is folded into the BATCH axis, so this is one GRU call
        however many chunks there are -- a Python loop over the 14 chunks of a
        4000-frame window would serialise 14 cuDNN launches per layer per step.
        A token count that is not a multiple of ``W`` is zero-padded at the END;
        because the GRU is unidirectional the outputs at the real positions are
        bit-identical to running the short final chunk on its own (checked in
        ``tools/hybrid_receptive_field.py --checks``), and the padding is sliced
        off, so no token is dropped or duplicated.
        """
        B, T, C = tokens.shape
        W = self.gru_window
        if W <= 0 or W >= T:
            return self._gru_stack(tokens)
        n = (T + W - 1) // W
        pad = n * W - T
        if pad:
            tokens = torch.cat([tokens, tokens.new_zeros(B, pad, C)], dim=1)
        h = self._gru_stack(tokens.reshape(B * n, W, C))              # (B*n, W, H)
        return h.reshape(B, n * W, -1)[:, :T, :]

    def encoder(self, x):
        """(B, F, in_dim) -> (B, T, feat_dim), T ~ F/interval."""
        tokens = self.cnn(x.transpose(-1, -2)).transpose(-1, -2)      # (B, T, cnn_dim)

        parts = []
        if self.use_gru:
            short = self.short_norm(self.short_branch(tokens))        # (B, T, gru_hidden)
            parts.append(short)
        if self.use_mamba:
            long = self.mamba(tokens)                                 # (B, T, mamba_dim)
            parts.append(long)

        if self.fuse_mode == "concat":
            fused = self.fuse_lin(parts[0] if len(parts) == 1
                                  else torch.cat(parts, dim=-1))
        else:
            gate = torch.sigmoid(self.gate_lin(torch.cat([short, long], dim=-1)))
            fused = gate * self.gru_proj(short) + (1.0 - gate) * self.mamba_proj(long)
        return self.fuse_drop(self.fuse_norm(self.fuse_act(fused)))

    def cov_decoder(self, x):
        if self.cov_stop_grad:
            x = x.detach()
        acc = torch.exp(self.acccov_decoder(x) - 5.0)
        gyro = torch.exp(self.gyrocov_decoder(x) - 5.0)
        return torch.cat([acc, gyro], dim=-1)

    def _update(self, to_update, feat, frame_len):
        """Broadcast one per-token value across its `interval` frames.

        Token ``i`` owns frames ``[i*interval - inter_head, i*interval + inter_tail)``
        clipped to ``[0, frame_len)``, so frame ``f`` is owned by exactly one token,
        ``floor((f + inter_head) / interval)``, clamped to the last real token.  That
        was VERIFIED before this rewrite: the coverage count is exactly 1 at every
        frame for window 4000, 6000 and 12000 -- no gaps, no double-adds.

        WHY IT IS NOT THE ORIGINAL PYTHON LOOP ANY MORE.  The loop ran once per
        token (up to 1334 of them) and was called six times per forward, one per
        head.  MEASURED on an RTX 4070 at batch 8, window 6000: 25 ms per call,
        150 ms per forward -- against 6.0 ms for the ENTIRE network (CNN + GRU +
        Mamba + fusion).  The broadcast cost 25x the model it was broadcasting.

        The ``index_select`` form is bit-identical (``torch.equal``, checked at all
        three window lengths) and 228x faster: 25.0 ms -> 0.109 ms.

        ``to_update`` is still ACCUMULATED into rather than overwritten, exactly as
        the loop did.  Every caller in this repo passes zeros, but the += semantics
        are what the callers were written against and changing them silently would
        be a different function wearing the same name.
        """
        idx = torch.div(
            torch.arange(frame_len, device=feat.device) + self.inter_head,
            self.interval, rounding_mode="floor").clamp_(max=feat.shape[1] - 1)
        return to_update + feat.index_select(1, idx)

    # ------------------------------------------------------------------
    # correction
    # ------------------------------------------------------------------
    @staticmethod
    def _rotate_small(dtheta, v):
        """Exp(dtheta) @ v by Rodrigues, exact, smooth at dtheta = 0.

        WHY A ROTATION.  The integrator computes world specific force as R_used @ acc.
        If the attitude it is handed is off by a small body-frame rotation dtheta,
        i.e. R_true = R_used Exp(dtheta), then R_used @ (Exp(dtheta) @ acc) is exactly
        R_true @ acc -- so the error is removed by rotating the specific force, not by
        adding a vector to it.  MEASURED (UPDATES_2026-09-08, tilt_aware.conf header):
        the best constant "bias" grows 0.0123 -> 0.0229 m/s^2 from the 30 s to the
        120 s fit, which is the signature of attitude leakage g*sin(theta), not of an
        accelerometer bias.  An additive head can only approximate g*(dtheta x g_body)
        and must learn the dependence on g_body from ~440 distinct windows; this form
        has it built in, with 3 bounded numbers per token.

        EXACT, not first order: at 2 deg the second-order term is ~6e-3 m/s^2, the
        same size as the corrections being learned.  sin(t)/t and (1-cos t)/t^2 use
        their Taylor series below 1e-4 rad so the gradient at dtheta = 0 is finite.
        """
        t2 = (dtheta * dtheta).sum(-1, keepdim=True)
        t = torch.sqrt(t2.clamp_min(1e-12))
        small = t2 < 1e-8
        a = torch.where(small, 1.0 - t2 / 6.0, torch.sin(t) / t)
        b = torch.where(small, 0.5 - t2 / 24.0, (1.0 - torch.cos(t)) / t2.clamp_min(1e-12))
        c1 = torch.cross(dtheta, v, dim=-1)
        c2 = torch.cross(dtheta, c1, dim=-1)
        return v + a * c1 + b * c2

    def _correct(self, raw, feature, frame_len, bias_head, scale_head, std,
                 channel="acc"):
        """raw (B, F', 3) -> corrected (B, F', 3) under the configured mode."""
        zero = torch.zeros_like(raw)
        if self.correction_mode == "direct":
            # Regress the corrected signal itself -- no skip from `raw`.  The head's
            # O(1) output is mapped through the corpus mean/std so it is in physical
            # units from step 0; see the buffers in __init__ for why that is needed.
            m = self.direct_acc_mean if channel == "acc" else self.direct_gyro_mean
            sd = self.direct_acc_std if channel == "acc" else self.direct_gyro_std
            value = bias_head(feature) * sd.to(feature.dtype) + m.to(feature.dtype)
            return self._update(zero.clone(), value, frame_len)

        # ---- CONSTANT-CORRECTION MODE (const_correction: True) -----------------
        # MEASURED motivation: with NavEul attitude and the 15 s freeze applied, the
        # best possible CONSTANT accel bias per window removes 46-62% of the residual
        # velocity error (2.94->1.21, 4.28->1.62, 5.57->3.01 m/s at 30/60/120 s), and
        # the implied |b| is a stable 0.094-0.104 m/s^2 across horizons.  So the thing
        # worth predicting is ONE 3-vector per window, not a per-frame signal -- and
        # the per-frame model has now twice peaked inside 5 epochs and then memorised.
        #
        # CAUSAL CUMULATIVE MEAN, not a global mean over the window.  A global mean
        # would make the correction at t=0 depend on data from t=120 s, which breaks
        # the property tools/eval_vel_horizons.py --nested relies on: that reading the
        # 30 s prefix of a 120 s run equals running a 30 s window (VERIFIED identical
        # to 0.000e+00).  A cumulative mean keeps that, and is also what a real system
        # does -- refine the bias estimate as more data arrives, converging to a
        # constant.
        f = feature
        if self.const_correction:
            n = torch.arange(1, f.shape[1] + 1, device=f.device, dtype=f.dtype)
            f = torch.cumsum(f, dim=1) / n.view(1, -1, 1)
        bias = self._update(zero.clone(), bias_head(f) * std, frame_len)
        if self.correction_mode == "additive" or (
                self.correction_mode == "rotate" and channel != "acc"):
            # rotate is an accelerometer-only form; a gyro correction stays additive.
            return raw + bias
        if self.correction_mode == "rotate":
            # scale_head is reused as the rotation head: it is zero-initialised, so
            # dtheta = 0 exactly at step 0 and the model starts as the identity.
            # NORM-bounded: |dtheta| < rotate_max whatever the direction (a per-axis
            # tanh would allow sqrt(3) x rotate_max on the diagonal).  tanh(n)/n -> 1
            # as n -> 0, so this is smooth and exactly 0 at a zero head output.
            h = scale_head(f)
            n = h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            dtheta = self._update(zero.clone(),
                                  self.rotate_max * torch.tanh(n) / n * h,
                                  frame_len)
            return self._rotate_small(dtheta, raw) + bias
        # affine: bounded diagonal scale, exactly 0 at init (tanh(0) = 0)
        scale = self._update(zero.clone(),
                             self.scale_std * torch.tanh(scale_head(feature)),
                             frame_len)
        return raw * (1.0 + scale) + bias

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # in_scale is a pure function of the config, not a learned tensor, but it is
        # a PERSISTENT buffer so it travels with the checkpoint.  Checkpoints written
        # before normalize_input existed (hybrid_v6 .. hybrid_v8) do not carry it and
        # would fail the strict=True load that eval.py and inference.py use.  Fill it
        # from the freshly built buffer instead: the value is the same either way.
        # load_state_dict() hands us a shallow copy, so this never touches the
        # caller's dict.
        # `in_offset` is new (it arrived with use_airspeed) and is likewise a pure
        # function of the config, so every checkpoint written before it existed --
        # hybrid_v6 .. v13, velnet_v1 -- must keep loading under strict=True.
        for name in ("in_scale", "in_offset"):
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = getattr(self, name).detach().clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def inference(self, data):
        """Pure network output.  Consumed by inference.py, which passes `data` only."""
        frame_len = data["acc"].shape[1] - self.interval

        feature = self.encoder(self._net_input(data))[:, 1:, :]

        raw_acc = data["acc"][:, self.interval:, :]
        raw_gyro = data["gyro"][:, self.interval:, :]

        corrected_acc = self._correct(raw_acc, feature, frame_len,
                                      self.accdecoder, self.accscale_decoder, self.acc_std,
                                      channel="acc")
        if self.correct_gyro:
            corrected_gyro = self._correct(raw_gyro, feature, frame_len,
                                           self.gyrodecoder, self.gyroscale_decoder, self.gyro_std,
                                           channel="gyro")
        else:
            # Genuinely inert, not "attenuated": no head output enters the graph.
            # gtrot=True alone does NOT achieve this (measured: |g| 4.06e-2 still
            # reaches the gyro through pypose's incre_r), so it has to be explicit.
            corrected_gyro = raw_gyro

        # The evaluation stack adds these to the raw signal, so they must be the
        # exact difference -- including the multiplicative part.
        correction_acc = corrected_acc - raw_acc
        correction_gyro = corrected_gyro - raw_gyro

        cov_state = {"acc_cov": None, "gyro_cov": None}
        if self.conf.propcov:
            cov = self.cov_decoder(feature)
            cov_state["acc_cov"] = self._update(torch.zeros_like(raw_acc), cov[..., :3], frame_len)
            cov_state["gyro_cov"] = self._update(torch.zeros_like(raw_gyro), cov[..., 3:], frame_len)

        return {"cov_state": cov_state,
                "corrected_acc": corrected_acc, "corrected_gyro": corrected_gyro,
                "correction_acc": correction_acc, "correction_gyro": correction_gyro}

    def forward(self, data, init_state):
        inference_state = self.inference(data)

        data["corrected_acc"] = inference_state["corrected_acc"]
        data["corrected_gyro"] = inference_state["corrected_gyro"]

        out_state = self.integrate(init_state=init_state, data=data,
                                   cov_state=inference_state["cov_state"])

        return {**out_state,
                "correction_acc": inference_state["correction_acc"],
                "correction_gyro": inference_state["correction_gyro"],
                "corrected_acc": inference_state["corrected_acc"],
                "corrected_gyro": inference_state["corrected_gyro"]}
