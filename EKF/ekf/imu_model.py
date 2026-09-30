"""The learned IMU correction (IMU/tools/export_onnx.py model), run CAUSALLY for the stream.

The ONNX model takes a fixed window: N frames plus 9 history samples, and returns
the corrected acc/gyro for the N frames.  Live, that becomes a SLIDING window:

  * every `every` IMU samples the model runs on the LATEST N + 9 real samples
    (only the past: nothing after "now" is used);
  * the model's CNN looks ~12 samples ahead, so the newest outputs of a run are
    computed against zero padding instead of data.  A sample is therefore only
    released once a run has seen at least `delay` samples AFTER it -- the EKF is
    fed the corrected IMU `delay` samples (+ up to `every`) late.  Measurements
    (VO, attitude, GPS) simply wait in the EKF's queue until the IMU catches up,
    because every message keeps its own timestamp;
  * before N + 9 samples exist (warm-up, 40 s for a 40 s model) the raw sample is
    released as it is.  Export a shorter model (--frames 1000 = 10 s) for a
    shorter warm-up.

INPUTS THE MODEL WAS TRAINED ON.  Training windows had the 15 s pre-window bias
freeze removed (IMU/tools/prewindow_align.freeze_biases, fitted against GPS
velocity).  `set_freeze(b_acc, b_gyro)` subtracts that constant before the model;
it is fitted at the GPS outage in run_stream.py.  With no GPS at all (--no_gt) it
stays zero, which is NOT what the model saw in training -- a known mismatch.
`active=False` passes the raw sample through (used before the outage).
"""
from collections import deque

import numpy as np

INTERVAL = 9
G_PAD = 9.81007


class StreamImuCorrector:
    def __init__(self, model, every=10, delay=16):
        self.m = model                      # tools.onnx_inference.OnnxModel
        self.n_in = model.n_in              # N + 9
        self.every = max(1, int(every))
        self.delay = max(0, int(delay))
        if "airspeed" in model.inputs:
            raise ValueError("this IMU model needs airspeed; not supported in the stream")
        self.buf = deque(maxlen=self.n_in)  # (t, acc_raw, gyro_raw, g_body)
        self.count = 0                      # samples pushed
        self.released = 0                   # samples released
        self.corr = {}                      # sample index -> (acc_c, gyro_c)
        self.freeze = (np.zeros(3), np.zeros(3))
        self.active = True
        self.runs = 0

    def set_freeze(self, b_acc, b_gyro):
        self.freeze = (np.asarray(b_acc, float), np.asarray(b_gyro, float))

    def push(self, t, acc, gyro, R_att):
        """One raw IMU sample (+ the nav attitude at it).  Returns the samples that
        are now final, oldest first: [(t, acc, gyro), ...]."""
        acc, gyro = np.asarray(acc, float), np.asarray(gyro, float)
        self.buf.append((float(t), acc, gyro, np.asarray(R_att, float)[2, :].copy(),
                         self.active))
        self.count += 1
        if self.count >= self.n_in and (self.count - self.n_in) % self.every == 0:
            self._run()
        return self._release(final=False)

    def flush(self):
        """End of stream: release everything left (with the latest outputs)."""
        return self._release(final=True)

    # ------------------------------------------------------------------
    def _run(self):
        b_acc, b_gyro = self.freeze
        acc = np.array([s[1] for s in self.buf]) - b_acc
        gyro = np.array([s[2] for s in self.buf]) - b_gyro
        g = np.array([s[3] for s in self.buf])
        feeds = {"acc": acc[None], "gyro": gyro[None]}
        if "g_body" in self.m.inputs:
            feeds["g_body"] = g[None]
        out = self.m(**feeds)
        self.runs += 1
        first = self.count - (self.n_in - INTERVAL)           # index of output frame 0
        last_ok = self.count - 1 - self.delay                  # newest sample with lookahead
        ca, cg = out["corrected_acc"][0], out["corrected_gyro"][0]
        for j in range(len(ca)):
            i = first + j
            if i > last_ok:
                break
            if i >= self.released:
                self.corr[i] = (ca[j].astype(float), cg[j].astype(float))

    def _release(self, final):
        out = []
        newest = self.count - 1
        while self.released <= newest:
            i = self.released
            pos = len(self.buf) - (self.count - i)             # position in the buffer
            if pos < 0:                                        # fell out: release raw
                self.released += 1
                continue
            t, acc, gyro, _, active = self.buf[pos]
            if not active:
                out.append((t, acc, gyro))
            elif i in self.corr:
                a, g = self.corr.pop(i)
                out.append((t, a, g))
            elif self.count < self.n_in:                       # warm-up: raw
                out.append((t, acc, gyro))
            elif final:
                out.append((t, acc - self.freeze[0], gyro - self.freeze[1]))
            else:
                break                                          # wait for its run
            self.released += 1
        return out


def freeze_at(fl, t_out, hist_s=15.0, k_sub=10, gravity=9.81007):
    """The 15 s pre-outage bias freeze, exactly as the IMU model was trained with
    (IMU/tools/prewindow_align.freeze_biases on the last hist_s of aided data)."""
    import pypose as pp
    import torch
    from tools.prewindow_align import freeze_biases
    from utils import pypose_compat
    pypose_compat.apply()
    t = fl["t"]
    j = int(np.searchsorted(t, t_out))
    H = int(round(hist_s / float(np.median(np.diff(t)))))
    if j < H + 1:
        return np.zeros(3), np.zeros(3)
    sl = slice(j - H, j)
    d = lambda x: torch.tensor(x[sl], dtype=torch.float64)[None]
    dt = torch.tensor(np.diff(t)[sl], dtype=torch.float64)[None, :, None]
    rot = pp.mat2SO3(torch.tensor(fl["R_gt"][sl], dtype=torch.float64))[None]
    with torch.no_grad():
        ba, bg = freeze_biases(d(fl["acc"]), d(fl["gyro"]), dt, rot, d(fl["v_gt"]), gravity,
                               k_sub=k_sub)
    return ba[0].numpy(), bg[0].numpy()


def stream_imu_fn(fl, corrector, att_source="gt", activate_at=None, freeze=None):
    """imu_fn for events.build_events: push the kept raw samples through the causal
    corrector in time order.  Each released sample ARRIVES at the time of the sample
    whose push released it (that is the latency) and keeps its own timestamp.
    activate_at: the model is used from this time on (raw before), and `freeze` is
    set then -- e.g. the GPS outage.  None: from the start, freeze as given."""
    R_att = fl["R_gt"] if att_source == "gt" else fl["R_mti"]

    def fn(kept):
        c = corrector
        c.buf.clear(); c.corr.clear(); c.count = c.released = c.runs = 0
        c.set_freeze(*(freeze if freeze is not None and activate_at is None
                       else (np.zeros(3), np.zeros(3))))
        c.active = activate_at is None
        out = []
        t = fl["t"]
        for k in kept:
            if not c.active and activate_at is not None and t[k] >= activate_at:
                c.active = True
                if freeze is not None:
                    c.set_freeze(*freeze)
            for ts, a, g in c.push(t[k], fl["acc"][k], fl["gyro"][k], R_att[k]):
                out.append((t[k], ts, a, g))
        for ts, a, g in c.flush():
            out.append((t[kept[-1]], ts, a, g))
        return out
    return fn
