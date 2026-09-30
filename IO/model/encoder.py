"""Encoder: the two-branch (GRU + Mamba) IMU feature extractor.

What this is
------------
The shared trunk that turns a window of raw IMU into a per-token feature vector.
It carries no task head of its own -- ``model/velocity_net.py`` subclasses it and
supplies the heads that make it a model.

It was extracted from ``HybridNet``, the IMU-CORRECTION arm this project ran
before ``velnet_v1``.  That arm and its six correction heads were removed on
2026-09-07; what is left is the encoder, which both arms always shared verbatim
(``_net_input`` was already a single definition for exactly that reason).  The
surviving parameter NAMES are unchanged, so an existing velnet checkpoint still
loads -- see ``_load_from_state_dict``, which also drops the correction-head
tensors such a checkpoint still carries.
Recover the deleted arm with ``git show <commit>^:model/hybrid.py``.

Two branches with genuinely different time constants
----------------------------------------------------
The CNN token stream (one token per ``interval`` = 9 frames = 90 ms) is fed to

  * a **GRU branch** at full token rate -- the short-term, low-latency path.
    ``gru_window`` (in FRAMES; 0 or absent = the whole window) runs it over
    non-overlapping chunks with ``h0 = 0`` at each chunk start, so its memory is
    a HARD cutoff rather than a soft decay;
  * a **Mamba branch** at ``mamba_stride`` token rate -- the long-horizon path.
    Its state updates once every ``9 * mamba_stride`` frames, so at stride 8 it
    advances once per 0.72 s and at stride 96 (velnet_v1) once per 8.64 s.

They are then fused.

What actually sets each horizon, measured at initialisation (float64, window 4000):

  * GRU, unbounded: influence is at 1/e by 0.45 s and at 1% by 1.26 s, and it
    hits the float64 noise floor past ~5 s.  ``gru_window: 300`` (= 33 tokens
    = 297 frames = 2.97 s) does not shorten that soft decay -- it makes it
    EXACT: no output can depend on a token more than 32 tokens back, and an
    output sitting on a chunk boundary depends on nothing before itself.
  * Mamba: influence falls to 1% after ~7 *SSM steps* regardless of what a step
    is worth, so the horizon in SECONDS is ``steps x stride x 0.09 s`` and
    ``mamba_stride`` is the lever -- linearly.  Measured: stride 8 -> 1% at
    5.04 s, stride 16 -> 10.08 s, stride 24 -> 15.12 s.  ``dt_min``/``dt_max``
    are NOT a lever and are deliberately not exposed: shrinking dt slows the
    state decay but shrinks the input gain ``delta*B*u`` by the same factor, and
    the measured decay over the first four steps is identical
    (1.000 / 0.0397 / 0.039 / 0.00126) across a 100x range of dt.
  * ``window_size`` is not an influence lever at initialisation either -- it is
    the CEILING on what training can later learn, because ``h0 = 0`` at every
    window start for both branches, and it sets ``K = ceil(T/stride)``, the
    number of SSM steps.

  The tool that produced these numbers, ``tools/hybrid_receptive_field.py``, was
  removed with the correction arm because it constructed ``HybridNet`` directly.
  Recover it with ``git show <commit>^:tools/hybrid_receptive_field.py``.

Attitude as an input
--------------------
Not Euler angles -- see ``model/attitude.py`` for why -- but
``g_body = R.Inv() @ [0,0,1]``, the yaw-invariant part of the attitude,
optionally plus sin/cos of roll and pitch.  ``velnet_v1.conf`` sets
``att_input: none``, so on the shipped config this path is inert and the encoder
sees acc + gyro only.

Attitude source
---------------
``att_source: "gt"`` uses ``data['rot']`` (the GPS-aided nav filter).  That is
partially circular: the pos/vel labels come from the same filter.
``att_source: "mti"`` uses ``data['mti_rot']``, the independent magnetometer/MTI
solution, which is noisier (median g_body error 2.0 deg, p99 8.4 deg) but is
what the aircraft will actually have at runtime.  Because the feature is
yaw-invariant, the MTI's unusable heading (within-flight drift of tens of
degrees) does not enter.
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

    CURRENTLY UNCALLED, and kept for that reason.  Its only call site was
    ``HybridNet.__init__``, which applied it to the four correction heads before
    those were deleted.  ``VelocityNet.__init__`` (model/velocity_net.py:73-74)
    builds ``vel_decoder`` / ``velcov_decoder`` with plain ``_head()`` and never
    calls this -- so the comment above those two lines, which states they are
    zero-initialised, does NOT describe the code.  Measured consequence at step 0:
    ``vel_body`` is a random field of mean norm ~3.2 m/s instead of 0, and
    ``vel_cov`` spans ~0.41-2.67 (m/s)^2 instead of sitting at ``velnet_cov_init``.
    Calling this on both heads is the two-line fix; it CHANGES training behaviour,
    so it is deliberately not applied here.

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


class Encoder(ModelBase):
    """Parallel GRU (short-term) + Mamba (long-term) IMU feature encoder."""

    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf

        # PER-CHANNEL CORPUS STD, measured on the 57 training flights (3,379,618
        # airborne frames).  These feed `in_scale` below, which is what
        # `normalize_input: True` divides the encoder input by.
        #
        # The `direct_` prefix is historical: they arrived with the deleted correction
        # arm's "direct" mode, which also needed the matching MEANS.  The means are
        # gone with that arm; the stds keep their original names because they are
        # PERSISTENT buffers and renaming them would break the strict=True load of
        # every checkpoint this repo has ever written.
        _dstat = lambda k, d: torch.tensor(
            [float(v) for v in conf.get(k, d)], dtype=torch.get_default_dtype())
        self.register_buffer("direct_acc_std", _dstat(
            "direct_acc_std", [1.998676, 1.474464, 4.209704]))
        self.register_buffer("direct_gyro_std", _dstat(
            "direct_gyro_std", [0.224214, 0.215214, 0.133847]))

        # The correction is piecewise-constant over `interval` frames because the
        # CNN downsamples by stride 3 x stride 3.  interval is a CONSEQUENCE of
        # the encoder stride, not an independent knob -- keep them in sync.
        self.interval = 9
        self.inter_head = np.floor(self.interval / 2.0).astype(int)
        self.inter_tail = self.interval - self.inter_head

        self.att_input = str(conf.get("att_input", "gravity"))
        self.att_source = str(conf.get("att_source", "gt"))

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
        self.cnn = CNNEncoder(c_list=[self.in_dim, 32, cnn_dim], k_list=[7, 7], s_list=[3, 3])

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

        gw = ("whole window" if self.gru_window == 0
              else "%d tok = %d frames = %.2f s" % (self.gru_window, self.gru_window_frames,
                                                    self.gru_window_frames / 100.0))
        print("[encoder] branches=%s in_dim=%d (att_input=%s, att_source=%s) "
              "fuse=%s fuse_dropout=%.2f%s%s params=%d"
              % (self.branches, self.in_dim, self.att_input, self.att_source,
                 self.fuse_mode, self.fuse_dropout,
                 (" gru(hidden=%d, window=%s)" % (gru_hidden, gw)) if self.use_gru else "",
                 (" mamba(dim=%d, layers=%d, stride=%d -> %.2f s/step, kernel=%s)"
                  % (mamba_dim, mamba_layers, self.mamba_stride,
                     self.mamba_stride * self.interval / 100.0,
                     "mamba-ssm" if has_mamba() else "pure-pytorch")) if self.use_mamba else "",
                 sum(p.numel() for p in self.parameters())))

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

        ONE definition, shared by the Encoder and VelocityNet.  They used to keep a
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

        # The correction arm was deleted on 2026-09-07 and its six heads and four
        # scalar buffers went with it.  VelocityNet INHERITED those tensors, so every
        # velnet checkpoint written before that date still carries them and would fail
        # the strict=True load in eval.py / inference.py / tools.eval_vel_horizons with
        # "Unexpected key(s)".  Drop them here: they were never read on the velnet path
        # (VelocityNet overrides inference() and forward() and touches no correction
        # head), so discarding them loses nothing that affected a velnet prediction.
        _dead = ("accdecoder.", "gyrodecoder.", "accscale_decoder.", "gyroscale_decoder.",
                 "acccov_decoder.", "gyrocov_decoder.",
                 "acc_std", "gyro_std", "direct_acc_mean", "direct_gyro_mean")
        for key in [k for k in state_dict
                    if k.startswith(prefix) and k[len(prefix):].startswith(_dead)]:
            del state_dict[key]

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

