"""Position and velocity error -- RMSE and WORST CASE -- per horizon, per split.

Answers "how far off is the trajectory after N seconds of unaided integration, and
how bad does it get in the worst window?" for a trained checkpoint, on whichever
splits you ask for.  It is offline and read-only: it costs the training run
nothing and can be pointed at any checkpoint after the fact.

TWO STATISTICS PER CHANNEL, both on the error MAGNITUDE at the END of a window of
the given length -- ||p_pred - p_gt|| in metres, ||v_pred - v_gt|| in m/s:

    rmse      sqrt(mean over windows of the squared magnitude).  Accumulated as a
              sum of squares and square-rooted ONCE -- averaging per-batch RMSEs
              is not an RMSE and would understate the spread.
    max       the single worst window in the split.  A mean hides exactly the case
              a dead-reckoning user cares about, so this is reported next to it.

POSITION IS NOT AN INDEPENDENT MEASUREMENT ON THIS CORPUS, AND MUST BE QUOTED AS
SUCH.  These logs carry no measured position: `datasets/UAVdataset.py` builds
`gt_translation` as the cumulative trapezoidal integral of the GPS velocity, and
every window here is initialised from the ground-truth state.  So

    pos_err(T) = integral from 0 to T of vel_err(t) dt

exactly -- the position row is a time-weighted restatement of the velocity row,
not a second opinion on it.  It is still the number worth reporting (a report
about dead reckoning is about metres of drift, and the integral is what turns a
sustained bias into the dominant error mode), but a good pos number cannot
corroborate a good vel number, and the two ratios will move together.  The
altitude cross-check against the logged barometric/GPS height put the label's own
error at a median 4.5 m, which bounds how much of a small pos_rmse is real.

TWO MODES, AND THEY ANSWER DIFFERENT QUESTIONS.

DEFAULT -- one pass per horizon, each at its own window length.  Every column gets
all the data it can have: a 600 s window survives on 6 of 13 validation flights
while a 60 s window survives on 12, so the short rows are computed on more
flights.  The price is that the rows are SEPARATE EXPERIMENTS on different start
frames and different flights, and must not be differenced -- a change between two
rows mixes "the error grew" with "the windows changed".

`--nested` -- one pass at the LONGEST horizon, every shorter horizon read off a
PREFIX of the same integration.  Identical flights, identical start frames,
identical initial state, one trajectory per window, so the rows ARE a growth
curve and the equal window count across them is the proof.  The price is the
mirror image: every horizon now uses only the flights long enough for the longest
one.  Use this when you want error-vs-duration; use the default when you want the
best estimate of each horizon on its own.

Either way the flight and window counts are printed per row, so a thin column is
visible rather than implied.

The RAW arm is the same windows, same initial states, same integrator, with the
network's correction removed -- the only thing that differs is the signal.  Read
`ratio` (model / raw), never the absolute number: window difficulty varies
enormously across this corpus, so the same 3 m/s is a good result on one flight
and a bad one on another.

`--splits train` uses conf.dataset.train, i.e. the flights the model FIT ON.  It
is there to show the train/val gap, which on this corpus is the whole story --
do not read it as performance.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import copy
import csv

import torch
import torch.utils.data as Data
from pyhocon import ConfigFactory

from datasets import SeqeuncesDataset, collate_fcs
from model import net_dict
from model.losses import loss_
from utils import move_to


def horizon_tag(frames):
    return "%gs" % (frames / 100.0)


def flight_lengths(section_conf):
    """{flight: n_frames} by counting CSV lines -- no loader, no decode."""
    out = {}
    for entry in section_conf.data_list:
        root = entry.get("data_root", "data")
        for f in entry["data_drive"]:
            path = os.path.join(root, f)
            if not os.path.exists(path):
                out[f] = 0
                continue
            with open(path, "rb") as fh:
                out[f] = max(0, sum(1 for _ in fh) - 1)
    return out


def check_lengths(conf, sections, horizons):
    """Which flights are long enough for each horizon, and which get dropped.

    A flight shorter than the horizon supplies NO window at that horizon -- it is
    silently absent rather than an error -- so a long-horizon row can end up
    computed on a fraction of the split.  On this corpus that is not a corner case:
    the 10 min horizon keeps 6 of 13 validation flights and 8 of 11 test flights.
    Run this before reading a long column as signal.
    """
    for sec in sections:
        if sec not in conf.dataset:
            print("  [%s] not in this config" % sec)
            continue
        lens = flight_lengths(conf.dataset[sec])
        print("\n=== split '%s' -- %d flights ===" % (sec, len(lens)))
        print("%-38s %9s %8s   %s"
              % ("flight", "frames", "seconds", " ".join("%8s" % horizon_tag(h)
                                                         for h in horizons)))
        print("-" * (58 + 9 * len(horizons)))
        for f in sorted(lens):
            n = lens[f]
            cells = " ".join("%8s" % (n // h if n // h else "-") for h in horizons)
            print("%-38s %9d %8.0f   %s" % (f[:38], n, n / 100.0, cells))
        print("-" * (58 + 9 * len(horizons)))
        print("%-38s %9s %8s   %s"
              % ("flights usable", "", "",
                 " ".join("%8s" % ("%d/%d" % (sum(1 for v in lens.values() if v > h + 1),
                                              len(lens))) for h in horizons)))
        print("%-38s %9s %8s   %s"
              % ("windows", "", "",
                 " ".join("%8d" % sum(v // h for v in lens.values()) for h in horizons)))
        print("%-38s %9s %8s   %s"
              % ("dates", "", "",
                 " ".join("%8d" % len({f[:10] for f, v in lens.items() if v > h + 1})
                          for h in horizons)))
        for h in horizons:
            drop = sorted(f for f, v in lens.items() if v <= h + 1)
            if drop:
                print("  DROPPED at %-6s (%d): %s"
                      % (horizon_tag(h), len(drop), ", ".join(d[:24] for d in drop)))


# Channels whose error is the NORM of a residual.  `vel_dir` is not one of them --
# it is an angle between two vectors, so it gets its own function below and is
# appended only for reporting.
CHANNELS = (("pos", "gt_pos", "m"), ("vel", "gt_vel", "m/s"))
REPORT = CHANNELS + (("vel_dir", "gt_vel", "deg"),)


def endpoint_dir_err(pred_vel, gt_vel, sampling):
    """Angle between the predicted and true VELOCITY VECTORS, in degrees.

    WHY THIS IS WORTH SEPARATING FROM vel_rmse.  ||v_pred - v_gt|| mixes two failures
    that a dead-reckoning user cares about differently: flying the right heading at
    the wrong speed, and flying the right speed in the wrong direction.  The second
    is what curves the trajectory away and dominates position error at long horizons,
    while the first mostly scales the along-track distance.  A model can improve the
    magnitude while degrading the direction, and vel_rmse alone will not show it.

    This is the direction of the VELOCITY VECTOR -- a property of the trajectory, not
    the attitude solution.  It is not the rotation-state error, and nothing here reads
    the integrated rotation.

    FULL 3-D ANGLE, not horizontal course.  Vertical is a small share of the velocity
    on this corpus so the two nearly coincide, but they are not the same number; if
    you want ground-track course error instead, take the angle of the xy components
    only.  Degenerate cases (a near-zero velocity, where direction is undefined) are
    guarded by clamping the norms; on airborne UAV data at ~22 m/s this never fires.
    """
    _, _ = None, None
    if sampling:
        p = pred_vel[:, sampling - 1::sampling, :]
        g = gt_vel[:, sampling - 1::sampling, :]
    else:
        p, g = pred_vel[:, -1:, :], gt_vel[:, -1:, :]
    p, g = p[:, -1, :], g[:, -1, :]
    cos = ((p * g).sum(dim=-1)
           / (p.norm(dim=-1).clamp(min=1e-9) * g.norm(dim=-1).clamp(min=1e-9)))
    return torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0)))


def endpoint_err(pred, gt, sampling):
    """||pred - gt|| at the last supervised checkpoint, one value per window.

    Routed through `loss_(None, ...)` rather than differenced here so that the
    sampling stride is applied exactly as the training objective applies it; with
    `fc=None` it returns the residual and builds no loss term.  `dim=` is passed by
    keyword -- `norm(-1)` is a p=-1 NORM over every element, which collapses to a
    plausible-looking scalar instead of erroring.
    """
    _, dist = loss_(None, pred, gt, sampling=sampling)
    return dist[:, -1, :].norm(dim=-1)


# ---------------------------------------------------------------------------
# NESTED HORIZONS -- one window, read at several durations
# ---------------------------------------------------------------------------
# The default mode above gives every horizon its own window set, so the 60 s and
# 120 s rows are computed from different start frames and (on this corpus) a
# different number of flights.  Those columns cannot be differenced: a change
# between them mixes "the error grew" with "the windows changed".
#
# `--nested` builds ONE dataset at the LONGEST horizon and reads every shorter
# horizon off a PREFIX of the same integration.  Identical flights, identical
# start frames, identical initial state, one trajectory per window -- so the
# rows are a growth curve rather than three separate experiments.
#
# WHAT IT COSTS, stated plainly: every horizon now uses only the flights long
# enough for the LONGEST one.  At 40 min that is a small subset of this corpus,
# and the short rows get worse statistics than they could have had.  That is the
# trade the mode exists to make; `--check` prints which flights survive.
#
# ON MONOTONICITY.  Because the prefix is the same trajectory, the error at
# 2 min is the 1 min error plus what accumulated after it, so it USUALLY grows
# with duration -- but it is not guaranteed to: a trajectory that loops back can
# reduce a position error, and `vel_dir` is an angle that can close again.  The
# summary prints the measured fraction of windows that grow, so the assumption
# is checked rather than assumed.

def endpoint_index(h, sampling):
    """Output index whose state is the END of an `h`-frame prefix.

    `pos[k]` is the state after integrating `dt[0..k]`, i.e. the state at frame
    `start + k + 1`, and `label['gt_pos'][k]` is `gt_translation[start+k+1]`.  So
    an `h`-frame window ends at index `h - 1` -- NOT `h`, and not `h - 2`.

    When `sampling` is set, the per-horizon mode scores the last checkpoint at or
    before the end (`loss_` slices `[sampling-1::sampling]` then takes `[-1]`).
    This reproduces that rule exactly, so a nested row and a per-horizon row of
    the same length are the same quantity and may be compared directly.
    """
    h, s = int(h), int(sampling) if sampling else 0
    idx = (h // s) * s - 1 if s else h - 1
    if idx < 0:
        raise ValueError(
            "horizon %d frames is shorter than one sampling interval (%d), so no "
            "supervised checkpoint falls inside it" % (h, s))
    return idx


def err_at(pred, gt, idx):
    """||pred - gt|| at one output index, one value per window."""
    return (pred[:, idx, :] - gt[:, idx, :]).norm(dim=-1)


# ---------------------------------------------------------------------------
# PEAK ERROR -- the worst moment INSIDE the window, not the value at the end
# ---------------------------------------------------------------------------
# `*_rmse` is the error AT the horizon: "how far off am I after T seconds".  That
# is the right thing to average, because it is one well-defined quantity per
# window.
#
# `*_max` used to be the worst of those endpoint values across windows, and that
# is NOT a worst case a dead-reckoning user can act on: velocity and direction
# error are INSTANTANEOUS, so the endpoint samples whatever the aircraft happened
# to be doing at that one moment.  Measured on eval, flight 2026_04_07_178_47 goes
# 23.45 -> 60.83 -> 20.52 deg at 30/60/120 s on ONE continuous trajectory -- the
# 60 s spike is a moment, not an accumulation, and it vanishes by 120 s.
#
# `*_max` is now the PEAK over every frame in [0, T], which is what "the worst it
# ever got" means.  It is monotone in T by construction, since [0,T1] is contained
# in [0,T2] -- so a longer horizon can never report a smaller max.  Computed with
# a running cummax over the full window so every horizon reads the same series.
#
# Position is unaffected in practice (it accumulates, so its peak is usually its
# endpoint) but is computed the same way for consistency.

def err_series(pred, gt):
    """||pred - gt|| at EVERY frame: (B, T)."""
    n = min(pred.shape[1], gt.shape[1])
    return (pred[:, :n, :] - gt[:, :n, :]).norm(dim=-1)


def dir_series(pred_vel, gt_vel):
    """Angle between predicted and true velocity vectors at EVERY frame, degrees."""
    n = min(pred_vel.shape[1], gt_vel.shape[1])
    p, g = pred_vel[:, :n, :], gt_vel[:, :n, :]
    cos = ((p * g).sum(dim=-1)
           / (p.norm(dim=-1).clamp(min=1e-9) * g.norm(dim=-1).clamp(min=1e-9)))
    return torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0)))


def running_peak(series):
    """(B, T) -> (B, T) where [:, k] is the max over frames 0..k.

    One cummax over the longest window serves every horizon, and it is what makes
    the peak monotone in the horizon by construction rather than by luck.
    """
    return torch.cummax(series, dim=1).values


def dir_err_at(pred_vel, gt_vel, idx):
    """Angle between predicted and true VELOCITY VECTORS at one index, degrees.

    Same quantity as `endpoint_dir_err`; see that docstring for why direction is
    reported separately from ||v_pred - v_gt||.
    """
    p, g = pred_vel[:, idx, :], gt_vel[:, idx, :]
    cos = ((p * g).sum(dim=-1)
           / (p.norm(dim=-1).clamp(min=1e-9) * g.norm(dim=-1).clamp(min=1e-9)))
    return torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0)))


def run_split(network, conf, section, horizon, device, batch_size, collate_fn,
              max_flights=None, max_windows=None):
    """One forward pass over `section` at `horizon` frames.  Returns a stats dict.

    `max_flights` truncates the flight list BEFORE the dataset is built, which is
    the lever that actually makes a smoke run fast: reading the CSVs dominates,
    not the GPU.  `max_windows` caps the batches afterwards and only saves GPU
    time.  Both are for checking the plumbing -- a 3-flight number is not a
    result, and the printed window/flight count says so.
    """
    dc = copy.deepcopy(conf.dataset[section])
    for entry in dc.data_list:
        entry["window_size"] = int(horizon)
        entry["step_size"] = int(horizon)
        if max_flights:
            entry["data_drive"] = list(entry["data_drive"])[:int(max_flights)]
    ds = SeqeuncesDataset(data_set_config=dc)
    n_flights = len(set(getattr(ds, "index_map", []) and
                        [i[0] for i in ds.index_map] or []))
    if len(ds) == 0:
        return None
    # Cap the batch too, or `--max_windows 3` with batch 8 silently scores 8: the
    # limit is checked between batches, so the first one always runs whole.
    if max_windows:
        batch_size = max(1, min(int(batch_size), int(max_windows)))
    loader = Data.DataLoader(dataset=ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn)
    interval = getattr(network, "interval", 0)
    sampling = conf.train.sampling

    # keyed (arm, channel) so the model and raw arms cannot drift apart
    sq = {(a, c): 0.0 for a in ("model", "raw") for c, _, _ in REPORT}
    mx = dict.fromkeys(sq, 0.0)
    n = 0
    with torch.no_grad():
        for data, init_state, label in loader:
            if max_windows and n >= int(max_windows):
                break
            data, init_state, label = move_to([data, init_state, label], device)
            # --- model arm ---
            out = network(data, init_state)
            # --- raw arm: identical windows and init, correction removed ---
            d = dict(data)
            d["corrected_acc"] = data["acc"][:, interval:, :]
            d["corrected_gyro"] = data["gyro"][:, interval:, :]
            rout = network.integrate(init_state=init_state, data=d,
                                     cov_state={"acc_cov": None, "gyro_cov": None})
            # Same definitions as the nested path: rmse is the error AT the
            # horizon, max is the PEAK over every frame in [0, T].
            i = endpoint_index(horizon, sampling)
            for arm, o in (("model", out), ("raw", rout)):
                ser = {ch: err_series(o[ch], label[gt_key]) for ch, gt_key, _ in CHANNELS}
                ser["vel_dir"] = dir_series(o["vel"], label["gt_vel"])
                for ch, _, _ in REPORT:
                    e = ser[ch][:, i]
                    sq[(arm, ch)] += float(e.pow(2).sum())
                    mx[(arm, ch)] = max(mx[(arm, ch)],
                                        float(ser[ch][:, :i + 1].max()))
            n += int(out["vel"].shape[0])
    if n == 0:
        return None
    res = {"windows": n, "flights": n_flights}
    for ch, _, _ in REPORT:
        res["%s_rmse" % ch] = (sq[("model", ch)] / n) ** 0.5
        res["%s_max" % ch] = mx[("model", ch)]
        res["raw_%s_rmse" % ch] = (sq[("raw", ch)] / n) ** 0.5
        res["raw_%s_max" % ch] = mx[("raw", ch)]
    return res


def run_split_nested(network, conf, section, horizons, device, batch_size, collate_fn,
                     max_flights=None, max_windows=None, first_only=False, win_rows=None,
                     first_step=100):
    """ONE pass at max(horizons); every horizon read off a prefix of the same windows.

    Returns one result dict per horizon, all sharing `windows` and `flights` --
    that equality is the proof the rows are paired and is printed in the table.

    `first_only` keeps just the EARLIEST LEGAL window of each flight: the first N
    minutes of flight after the 15 s pre-window freeze has its history.  It is NOT
    frame 0 -- the freeze needs `freeze_hist_s` of aided data BEFORE the window, and
    the data-quality mask can push it later still.  Candidate starts are spaced
    `first_step` frames (default 100 = 1 s) apart for this mode only; with the step
    equal to the window, as it used to be, the next candidate after the dropped
    start 0 was a whole window later, so "the first 2 minutes" was measured 2-4
    minutes into the log.  The real start of every flight is printed.
    Off by default: non-overlapping windows give more samples for the same single
    forward pass, and every one of them is still a nested prefix from its own start.

    `win_rows`, when a list is given, is filled with one row per (window, horizon)
    so the pooled numbers can be checked leave-one-flight-out.  That check is not
    optional on this corpus: VAL is 1/3/7 usable flights per date and ONE flight
    has carried 52-64% of the pooled squared error before now.
    """
    H = int(max(horizons))
    idx_of = {h: endpoint_index(h, conf.train.sampling) for h in horizons}
    dc = copy.deepcopy(conf.dataset[section])
    for entry in dc.data_list:
        entry["window_size"] = H
        # fine spacing ONLY for first_only, so its window starts at the earliest
        # legal frame instead of one whole window after the dropped start 0
        entry["step_size"] = int(first_step) if first_only else H
        if max_flights:
            entry["data_drive"] = list(entry["data_drive"])[:int(max_flights)]
    # seq_id is assigned in this exact nested order by construct_index_map, so this
    # flattened list is the seq_id -> flight name map.
    names = [f for entry in dc.data_list for f in entry["data_drive"]]
    ds = SeqeuncesDataset(data_set_config=dc)
    if len(ds) == 0:
        return None
    if first_only:
        seen, keep = set(), []
        for e in ds.index_map:
            if e[0] not in seen:
                seen.add(e[0]); keep.append(e)
        ds.index_map = keep
        for e in keep:
            t0 = float(ds.dt[e[0]][:e[1]].sum()) if hasattr(ds, "dt") else e[1] / 100.0
            print("  [first_only] %-36s starts %6.1f s into the log (frame %d)"
                  % (names[e[0]][:36], t0, e[1]))
    imap = list(ds.index_map)
    n_flights = len(set(e[0] for e in imap))
    if max_windows:
        batch_size = max(1, min(int(batch_size), int(max_windows)))
    loader = Data.DataLoader(dataset=ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn)
    interval = getattr(network, "interval", 0)

    sq = {(a, c, h): 0.0 for a in ("model", "raw") for c, _, _ in REPORT for h in horizons}
    mx = dict.fromkeys(sq, 0.0)
    # grew[h_prev, h] counts windows whose MODEL error rose from one horizon to the
    # next -- the monotonicity claim, measured rather than assumed.
    order = sorted(horizons)
    grew = {(c, a, b): 0 for c, _, _ in REPORT for a, b in zip(order, order[1:])}
    n = 0
    with torch.no_grad():
        for data, init_state, label in loader:
            if max_windows and n >= int(max_windows):
                break
            data, init_state, label = move_to([data, init_state, label], device)
            out = network(data, init_state)
            d = dict(data)
            d["corrected_acc"] = data["acc"][:, interval:, :]
            d["corrected_gyro"] = data["gyro"][:, interval:, :]
            rout = network.integrate(init_state=init_state, data=d,
                                     cov_state={"acc_cov": None, "gyro_cov": None})
            B = int(out["vel"].shape[0])
            per, pk = {}, {}
            for arm, o in (("model", out), ("raw", rout)):
                # ONE per-frame series per channel over the whole window, then a
                # running peak; every horizon indexes into the same two arrays, so
                # the endpoint and the peak cannot disagree about which frames a
                # horizon covers.
                ser = {ch: err_series(o[ch], label[gt_key]) for ch, gt_key, _ in CHANNELS}
                ser["vel_dir"] = dir_series(o["vel"], label["gt_vel"])
                run = {c: running_peak(s) for c, s in ser.items()}
                for h in horizons:
                    i = idx_of[h]
                    for ch, _, _ in REPORT:
                        e, q = ser[ch][:, i], run[ch][:, i]
                        per[(arm, ch, h)] = e
                        pk[(arm, ch, h)] = q
                        sq[(arm, ch, h)] += float(e.pow(2).sum())
                        mx[(arm, ch, h)] = max(mx[(arm, ch, h)], float(q.max()))
            for ch, _, _ in REPORT:
                for a, b in zip(order, order[1:]):
                    grew[(ch, a, b)] += int((per[("model", ch, b)]
                                             >= per[("model", ch, a)]).sum())
            if win_rows is not None:
                for k in range(B):
                    seq_id, start, _end = imap[n + k]
                    for h in horizons:
                        win_rows.append(dict(
                            split=section, flight=names[seq_id], date=names[seq_id][:10],
                            start=int(start), horizon=int(h), tag=horizon_tag(h),
                            # `_at` is the value AT the horizon, `_peak` the worst
                            # frame in [0, T].  Both, so neither has to be guessed.
                            **{("%s_%s_at" % (a, c)): float(per[(a, c, h)][k])
                               for a in ("model", "raw") for c, _, _ in REPORT},
                            **{("%s_%s_peak" % (a, c)): float(pk[(a, c, h)][k])
                               for a in ("model", "raw") for c, _, _ in REPORT}))
            n += B
    if n == 0:
        return None
    rows = []
    for h in horizons:
        res = {"windows": n, "flights": n_flights, "horizon": int(h),
               "tag": horizon_tag(h), "nested_from": horizon_tag(H)}
        for ch, _, _ in REPORT:
            res["%s_rmse" % ch] = (sq[("model", ch, h)] / n) ** 0.5
            res["%s_max" % ch] = mx[("model", ch, h)]
            res["raw_%s_rmse" % ch] = (sq[("raw", ch, h)] / n) ** 0.5
            res["raw_%s_max" % ch] = mx[("raw", ch, h)]
        rows.append(res)
    mono = {}
    for ch, _, _ in REPORT:
        for a, b in zip(order, order[1:]):
            mono[(ch, a, b)] = grew[(ch, a, b)] / float(n)
    return rows, mono


def report_per_flight(win_rows, channel="pos"):
    """Per-flight ratios, pooled, and leave-one-flight-out -- printed, not implied.

    WHY THIS IS A FLAG AND NOT A NOTE IN THE DOCS.  The pooled `r_rmse` row is a
    ratio of RMS over windows, so it is dominated by whichever flight has the largest
    errors.  On this corpus that is not a rounding concern: measured 2026-09-08 on the
    held-out `inference` split, ONE flight (2026_02_06_143_23) carried 53% of the
    pooled squared position error at 30 s and was the only flight the model made
    worse -- pooled read 1.003 while the MEDIAN flight read 0.967.  The pooled number
    and the typical flight disagreed by 3.6 points and pointed opposite ways.

    So this prints all three: pooled, per-flight, and the pooled value recomputed with
    each flight dropped.  If `LOFO max` crosses 1.0, one flight carries the verdict.
    """
    by = {}
    for r in win_rows:
        k = (r["tag"], r["flight"])
        m = by.setdefault(k, [0.0, 0.0, 0])
        m[0] += float(r["model_%s_at" % channel]) ** 2
        m[1] += float(r["raw_%s_at" % channel]) ** 2
        m[2] += 1
    tags = sorted({t for t, _ in by}, key=lambda t: float(t[:-1]))
    for tag in tags:
        fl = {f: v for (t, f), v in by.items() if t == tag and v[1] > 0}
        if not fl:
            continue
        M = sum(v[0] for v in fl.values())
        R = sum(v[1] for v in fl.values())
        ratios = sorted((math.sqrt(v[0] / v[1]), f, v) for f, v in fl.items())
        lofo = [math.sqrt((M - v[0]) / (R - v[1])) for _, _, v in ratios if R - v[1] > 0]
        helped = sum(1 for x, _, _ in ratios if x < 1.0)
        mid = len(ratios) // 2
        med = (ratios[mid][0] if len(ratios) % 2
               else 0.5 * (ratios[mid - 1][0] + ratios[mid][0]))
        print("")
        print("=== PER-FLIGHT %s at %s -- pooled %.4f | median flight %.4f | helped %d/%d ==="
              % (channel, tag, math.sqrt(M / R), med, helped, len(ratios)))
        print("  %-36s %5s %10s %10s %8s %7s %8s"
              % ("flight", "n", "model", "raw", "ratio", "share", "LOFO"))
        for x, f, v in reversed(ratios):
            lo = math.sqrt((M - v[0]) / (R - v[1])) if R - v[1] > 0 else float("nan")
            print("  %-36s %5d %10.3f %10.3f %8.4f %6.1f%% %8.4f%s"
                  % (f[:36], v[2], math.sqrt(v[0] / v[2]), math.sqrt(v[1] / v[2]),
                     x, 100 * v[1] / R, lo, "   <-- HURT" if x >= 1.0 else ""))
        if lofo:
            print("  LOFO range %.4f .. %.4f   %s"
                  % (min(lofo), max(lofo),
                     "every value below 1.0" if max(lofo) < 1.0
                     else "** CROSSES 1.0 -- one flight is carrying the verdict **"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="the TRAINING config: supplies the network hyper-parameters "
                         "and the flight lists for every split")
    ap.add_argument("--ckpt", default=None, help="not needed with --check")
    ap.add_argument("--horizons", type=int, nargs="+", required=True,
                    help="elapsed times in FRAMES (100 Hz), e.g. 3000 6000 12000")
    ap.add_argument("--splits", nargs="+", default=["train", "eval"],
                    choices=["train", "test", "eval", "inference"],
                    help="'eval' is the VALIDATION flights in a UAV/*.conf and the "
                         "HELD-OUT TEST flights in a UAVtest/*.conf -- see command.txt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_flights", type=int, default=None,
                    help="use only the first N flights of each split.  This is the "
                         "lever for a fast smoke run -- reading the CSVs dominates.")
    ap.add_argument("--max_windows", type=int, default=None,
                    help="stop after N windows per split x horizon (GPU time only)")
    ap.add_argument("--check", action="store_true",
                    help="report flight lengths and which flights each horizon drops, "
                         "then exit.  No checkpoint is loaded and no network runs.")
    ap.add_argument("--out", default=None, help="also write this CSV")
    ap.add_argument("--nested", action="store_true",
                    help="score every horizon on a PREFIX of ONE window instead of "
                         "giving each horizon its own window set.  Identical flights, "
                         "identical start frames, one trajectory per window, so the "
                         "rows are a growth curve.  Costs the short horizons the "
                         "flights that are too short for the LONGEST one.")
    ap.add_argument("--first_only", action="store_true",
                    help="--nested only: keep just the EARLIEST LEGAL window of each "
                         "flight -- the first N minutes after the 15 s freeze history "
                         "(NOT frame 0; the real start per flight is printed)")
    ap.add_argument("--first_step", type=int, default=100,
                    help="--first_only only: spacing of candidate window starts, in "
                         "frames (100 = 1 s).  The first window starts at most this "
                         "late after the earliest legal frame.")
    ap.add_argument("--csv", default=None,
                    help="evaluate ONE flight instead of a whole split.  Give the "
                         "file name (2026_04_07_219_54_sensor_data.csv) or a path.  "
                         "Everything else -- window sizes, the 15 s freeze, gravity -- "
                         "is inherited from the --splits section, so the numbers are "
                         "directly comparable with a full-split run.  Exactly one "
                         "split may be given with this.")
    ap.add_argument("--per_flight", action="store_true",
                    help="--nested only: print per-flight ratios, the median flight, and the pooled value recomputed leave-one-flight-out.  The pooled r_rmse row is an RMS ratio dominated by the worst flight -- on the held-out split one flight carried 53%% of the squared error and was the only one the model hurt, so pooled read 1.003 where the median flight read 0.967.  Run this before quoting any pooled number.")
    ap.add_argument("--win_csv", default=None,
                    help="--nested only: one row per (window, horizon) so the pooled "
                         "numbers can be checked leave-one-flight-out")
    a = ap.parse_args()
    if (a.first_only or a.win_csv or a.per_flight) and not a.nested:
        sys.exit("--first_only, --win_csv and --per_flight only apply with --nested")

    conf = ConfigFactory.parse_file(a.config)

    # ---- SINGLE-FLIGHT MODE ---------------------------------------------------
    # Rewrite the chosen split's flight list to the one file asked for, and leave
    # every other key alone.  window_size / step_size / freeze_hist_s / gravity all
    # keep coming from that section, which is what makes a --csv run and a full-split
    # run the same measurement on a different set of flights.  Both run_split and
    # run_split_nested deepcopy conf.dataset[section], so editing it here is enough.
    if a.csv:
        if len(a.splits) != 1:
            sys.exit("--csv evaluates one flight, so give exactly one --splits "
                     "(got %s)" % a.splits)
        name = os.path.basename(a.csv)
        root = os.path.dirname(a.csv)
        sec = conf.dataset[a.splits[0]]
        for entry in sec.data_list:
            entry["data_drive"] = [name]
            if root:
                entry["data_root"] = root
        probe = os.path.join(sec.data_list[0]["data_root"], name)
        if not os.path.exists(probe):
            sys.exit("--csv: no such file: %s" % probe)
        print("[csv] single flight: %s   (split '%s' supplies window/freeze settings)"
              % (probe, a.splits[0]))
    conf.train.device = a.device
    collate_fn = (collate_fcs[conf.dataset.collate] if "collate" in conf.dataset.keys()
                  else collate_fcs["base"])

    if a.check:
        check_lengths(conf, a.splits, a.horizons)
        return
    if not a.ckpt:
        sys.exit("--ckpt is required unless --check is given")

    network = net_dict[conf.train.network](conf.train).to(a.device).float()
    ck = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    sd = ck.get("model_state_dict", ck)
    network.load_state_dict(sd)
    network.eval()
    print("[ckpt] %s (epoch %s)" % (a.ckpt, ck.get("epoch", "?")))

    rows, win_rows, monos = [], ([] if (a.win_csv or a.per_flight) else None), {}
    horizons = sorted(a.horizons)
    for split in a.splits:
        if a.nested:
            got = run_split_nested(network, conf, split, horizons, a.device,
                                   a.batch_size, collate_fn, max_flights=a.max_flights,
                                   max_windows=a.max_windows, first_only=a.first_only,
                                   first_step=a.first_step,
                                   win_rows=win_rows)
            if got is None:
                print("  %-6s no window of %s exists -- every horizon is dropped in "
                      "nested mode" % (split, horizon_tag(max(horizons))))
                continue
            hrows, mono = got
            for r in hrows:
                r.update(split=split)
                rows.append(r)
            monos[split] = mono
            continue
        for h in horizons:
            res = run_split(network, conf, split, h, a.device, a.batch_size, collate_fn,
                            max_flights=a.max_flights, max_windows=a.max_windows)
            if res is None:
                print("  %-6s %-8s no window of this length exists" % (split, horizon_tag(h)))
                continue
            res.update(split=split, horizon=h, tag=horizon_tag(h))
            rows.append(res)

    if not rows:
        return
    # One table per channel rather than one 12-column row: these get pasted into a
    # report, and a line that wraps in the terminal is unreadable in both places.
    for ch, _, unit in REPORT:
        print("\n=== %s error (%s) -- rmse AT the horizon, max = PEAK inside [0,T] ==="
              % (ch.upper(), unit))
        print("%-7s %-8s %8s | %10s %10s | %10s %10s | %7s %7s"
              % ("split", "horizon", "windows", "rmse@T", "max[0,T]",
                 "raw_rmse@T", "raw_max", "r_rmse", "r_max"))
        print("-" * 92)
        for r in rows:
            print("%-7s %-8s %8d | %10.4f %10.4f | %10.4f %10.4f | %7.3f %7.3f"
                  % (r["split"], r["tag"], r["windows"],
                     r[ch + "_rmse"], r[ch + "_max"],
                     r["raw_%s_rmse" % ch], r["raw_%s_max" % ch],
                     r[ch + "_rmse"] / max(r["raw_%s_rmse" % ch], 1e-12),
                     r[ch + "_max"] / max(r["raw_%s_max" % ch], 1e-12)))
    if a.max_flights or a.max_windows:
        print("\n*** SMOKE RUN -- max_flights=%s max_windows=%s.  A plumbing check, NOT\n"
              "*** a result: computed on a subset that represents no split."
              % (a.max_flights, a.max_windows))
    print("\npos_* are metres, vel_* are m/s and vel_dir_* are DEGREES, from a")
    print("GT-initialised state.  TWO DIFFERENT QUANTITIES PER CHANNEL:")
    print("  rmse@T    the error AT the horizon, RMS over windows -- 'how far off")
    print("            am I after T seconds'.")
    print("  max[0,T]  the WORST error at any frame inside the window, over all")
    print("            windows -- 'how bad did it ever get'.  Monotone in T by")
    print("            construction, since [0,T1] is contained in [0,T2].")
    print("The endpoint is a snapshot: vel and vel_dir error are INSTANTANEOUS, so")
    print("an endpoint max samples whatever the aircraft was doing at that moment")
    print("and can FALL as the horizon grows.  The peak cannot.")
    print("vel_dir is the angle between the predicted and true velocity VECTORS -- the")
    print("direction the aircraft is moving, NOT the attitude solution.  A model can")
    print("improve vel_rmse while degrading vel_dir, which is why it is separate: it is")
    print("direction error that curves the trajectory away.")
    print("r_rmse / r_max are model/raw -- below 1.0 means the network helped.")
    print("POSITION IS THE INTEGRAL OF THE VELOCITY ERROR, not a separate measurement:")
    print("these logs have no measured position, so the two tables are one result seen")
    print("twice and the pos row cannot corroborate the vel row.  See the module docstring.")
    if not a.nested:
        print("Window counts differ per horizon: a flight shorter than the horizon")
        print("supplies no window, so a long row is computed on fewer flights.")
        print("The rows are therefore SEPARATE EXPERIMENTS on different windows and")
        print("must not be differenced -- use --nested for a growth curve.")
    else:
        print("NESTED: every row is a PREFIX of the same %s window -- identical flights,"
              % horizon_tag(max(horizons)))
        print("identical start frames, one trajectory per window.  The equal `windows`")
        print("count across rows is the proof they are paired, so the rows CAN be")
        print("differenced.  The cost is that every horizon uses only the flights long")
        print("enough for the longest one; run --check to see which were dropped.")
        if a.first_only:
            print("--first_only: one window per flight, starting at the first legal frame "
                  "(after the freeze history and the data-quality check) -- per-flight "
                  "start times are listed above.  NOT frame 0.")
        for split, mono in monos.items():
            print("\n  monotonicity on %s -- windows whose MODEL ENDPOINT error grew"
                  % split)
            print("  with duration.  This is about rmse@T only; max[0,T] is monotone")
            print("  by construction and needs no check.")
            for ch, _, _ in REPORT:
                cells = "  ".join(
                    "%s->%s %5.1f%%" % (horizon_tag(x), horizon_tag(y), 100 * mono[(ch, x, y)])
                    for x, y in zip(horizons, horizons[1:]))
                print("    %-8s %s" % (ch, cells))
            print("    (below 100% is not a bug, and it is informative: POSITION error")
            print("     accumulates so it should be near 100%, but vel and vel_dir are")
            print("     INSTANTANEOUS at the endpoint and have no reason to grow.)")

    if win_rows and a.per_flight:
        for _ch in ("pos", "vel"):
            report_per_flight(win_rows, _ch)

    if win_rows and a.win_csv:
        with open(a.win_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(win_rows[0].keys()))
            w.writeheader()
            w.writerows(win_rows)
        print("\nper-window CSV -> %s (%d rows).  CHECK LEAVE-ONE-FLIGHT-OUT before"
              % (a.win_csv, len(win_rows)))
        print("quoting any pooled number from it.")

    if a.out:
        with open(a.out, "w", newline="") as f:
            fields = ["split", "horizon", "tag", "windows", "flights"]
            if a.nested:
                fields.append("nested_from")
            for ch, _, _ in REPORT:
                fields += ["%s_rmse" % ch, "%s_max" % ch,
                           "raw_%s_rmse" % ch, "raw_%s_max" % ch]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in w.fieldnames})
        print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
