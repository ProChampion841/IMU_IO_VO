import argparse
import os
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.utils.data as Data
import tqdm
from pyhocon import ConfigFactory


# =====================================================================
# LOADER PROGRESS BAR
# =====================================================================
# Building one SeqeuncesDataset reads every flight CSV in its config section, and
# each flight prints 2-3 long diagnostic lines.  train.py builds FIVE of them per
# run (train, conf.dataset.test rebuilt once per entry of `metric_horizons`, and
# eval), so ~93 flight loads emit ~280 lines over ~70 s before epoch 0 -- a wall
# of text with no indication of progress.  One tqdm bar per dataset replaces it.
#
# The routing rule is "IS A BAR CURRENTLY ACTIVE", never a global quiet flag.
# That is what keeps every non-bar caller byte-identical: eval.py:159 and
# inference.py:110 build one dataset PER FLIGHT through the data_path branch, and
# evaluation/evaluate_state.py plus tools/airdata_check.py, tools/error_decompose.py,
# tools/long_outage.py and tools/prewindow_align.py construct the UAV loader
# directly.  None of them ever open a bar, so for all of them loader_log IS print().
_BAR = None

# The per-flight lines are SHOWN BY DEFAULT (user's call, 2026-09-07): the bar is
# there to stop them garbling, not to hide them.  tqdm.write clears the bar, writes
# the line, and redraws, so both survive.  AIRIMU_LOADER_QUIET=1 collapses them to
# the pooled summary.  AIRIMU_NO_LOAD_BAR=1 suppresses the bar itself
# and gives back exactly today's behaviour -- useful when stdout is parsed line by
# line (tools/run_eval_when_ready.sh:42 greps this stream).
# While a dataset is being built this points at that dataset's `load_log`, so the
# verbatim text of every diagnostic is captured whether or not it is printed.  That
# is what makes suppression safe: nothing is destroyed, only not shown.
_LOG_SINK = None
_QUIET_LOAD = os.environ.get("AIRIMU_LOADER_QUIET", "") not in ("", "0", "false", "False")
_NO_BAR = os.environ.get("AIRIMU_NO_LOAD_BAR", "") not in ("", "0", "false", "False")
_HINT_SHOWN = False


def _flight_stem(data_name):
    """'data/2026_04_07_219_55_sensor_data.csv' -> '2026_04_07_219_55'.

    Only for the NEW summary lines.  The existing per-flight messages keep printing
    the full path exactly as they always have.
    """
    base = os.path.basename(str(data_name))
    for suffix in ("_sensor_data.csv", ".csv"):
        if base.endswith(suffix):
            return base[:-len(suffix)]
    return base


def _is_main_process():
    """True on rank 0, and True whenever torch.distributed was never initialised.

    A LOCAL COPY of utils/distributed.py:116 is_main(), on purpose: importing that
    runs utils/__init__.py -> utils/visualize.py -> `import matplotlib.pyplot`,
    which would drag a GUI-capable plotting stack into every `import datasets`
    (measured ~0.7 s) including headless tools and DataLoader workers.  Keep the
    two in sync; do NOT "clean this up" into `from utils.distributed import is_main`.

    The is_available()/is_initialized() guard is load-bearing: a bare
    dist.get_rank() RAISES before init_process_group, and eval.py, inference.py and
    every tools/ script build datasets without ever creating a process group.
    Both DDP entry points (train.py:1101 torchrun, train.py:718 mp.spawn) call
    ddp_setup BEFORE any dataset is constructed, so the group is live by then.
    """
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True


def loader_log(msg, important=False):
    """The single choke point for every per-flight loading diagnostic.

    Three routes, and the first is the common one:
      * no bar active         -> plain print(), i.e. today's behaviour exactly.
      * bar active, important -> tqdm.write on EVERY rank.  These are the two lines
        that say data went missing (NaN MTI, every-window-dropped); losing one
        because it surfaced on a non-main rank is worse than seeing it twice.
      * bar active, routine   -> tqdm.write on rank 0, so every line the run used to
        print is still printed, in order, with the bar redrawn underneath it.  Set
        AIRIMU_LOADER_QUIET=1 to collapse them into the pooled summary instead; the
        numbers survive either way, and EVERY line -- printed or not -- is kept
        verbatim on `dataset.load_log` for any caller or debugger that wants it.

    tqdm.write, not print, because it clears the live bar, writes, then redraws it.
    It defaults to sys.stdout while the bar defaults to sys.stderr -- the same
    stream split these messages already have, so `grep '^loaded: data'` style
    consumers keep matching.
    """
    if _LOG_SINK is not None:
        _LOG_SINK.append(msg)
    if _BAR is None:
        print(msg)
        return
    if important or (not _QUIET_LOAD and _is_main_process()):
        tqdm.tqdm.write(msg)


def mti_or_gt(seq, name=""):
    """The MTI rotation of a sequence.  RAISES if there is none.

    This used to fall back to the ground-truth rotation with a warning, because
    EuRoC / SubT / TUM / KITTI publish no MTI channel and had to keep working.
    Those loaders were removed on 2026-09-07, and with UAV the only loader the
    fallback could do exactly one thing: silently hand a config with
    ``att_source: mti`` the GPS-AIDED NAV ATTITUDE instead -- which is leakage,
    and is the failure this project has had to chase before.

    VERIFIED before the change: all 88 flights in data/ carry EulX/EulY/EulZ, so
    this raises on a malformed CSV, never in normal operation.  The name is kept
    (callers and checkpoints reference it) but the behaviour is now honest.
    """
    if "mti_orientation" in seq.data:
        return seq.data["mti_orientation"]
    raise KeyError(
        "%s%s publishes no 'mti_orientation'. Falling back to the ground-truth "
        "rotation would silently feed a GPS-aided attitude to a config that asked "
        "for the MTi, so this is an error. Check the CSV for EulX/EulY/EulZ."
        % (type(seq).__name__, (" (%s)" % name) if name else ""))


def airspeed_or_zeros(seq, name=""):
    """The pitot airspeed of a sequence, or zeros as a fallback.

    This used to return zeros for the loaders that have no pitot (EuRoC / SubT /
    TUM / KITTI).  Those were removed on 2026-09-07; with UAV the only loader, a
    missing airspeed means a malformed CSV, and a model with ``use_airspeed: True``
    silently reading a constant-zero channel is worse than an error.

    Shape ``(N, 1)``, so it concatenates onto the ``(N, 3)`` IMU channels.

    NOT LEAKAGE.  This is a pitot tube: an onboard sensor the aircraft carries at
    runtime, independent of the GPS solution that supplies the labels.  It is the
    one channel on this corpus that observes something the IMU cannot -- speed
    through the air -- which is why it is worth adding.  It is also biased and
    uncalibrated (see UAVdataset.py), so the network has to learn the scale.
    """
    if "airspeed" in seq.data:
        return seq.data["airspeed"]
    raise KeyError(
        "%s%s publishes no 'airspeed'. Check the CSV for the AirSpeed column."
        % (type(seq).__name__, (" (%s)" % name) if name else ""))


class Sequence(ABC):
    # Dictionary to keep track of subclasses
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

class SeqDataset(Data.Dataset):
    def __init__(self, root, dataname, devive = 'cpu', name='Nav', duration=200, step_size=200, mode='inference', 
                    drop_last = True, conf = {}):
        super().__init__()

        self.DataClass = Sequence.subclasses
        
        self.conf = conf
        self.seq = self.DataClass[name](root, dataname, **self.conf)
        self.data = self.seq.data
        self.seqlen = self.seq.get_length()-1
        self.gravity = conf.gravity if "gravity" in conf.keys() else 9.81007
        if duration is None: self.duration = self.seqlen
        else: self.duration = duration
        
        if step_size is None: self.step_size = self.seqlen
        else: self.step_size = step_size

        self.data['acc_cov'] = 0.08 * torch.ones_like(self.data['acc'])
        self.data['gyro_cov'] = 0.006 * torch.ones_like(self.data['gyro'])
        self.mti_ori = mti_or_gt(self.seq, dataname)

        start_frame = 0
        end_frame = self.seqlen

        self.index_map = [[i, i + self.duration] for i in range(
            0, end_frame - start_frame - self.duration, self.step_size)]
        if (self.index_map[-1][-1] < end_frame) and (not drop_last):
            self.index_map.append([self.index_map[-1][-1], end_frame])

        self.index_map = np.array(self.index_map)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, i):
        frame_id, end_frame_id = self.index_map[i]
        return {
            'dt': self.data['dt'][frame_id: end_frame_id],
            'acc': self.data['acc'][frame_id: end_frame_id],
            'gyro': self.data['gyro'][frame_id: end_frame_id],
            'rot': self.data['gt_orientation'][frame_id: end_frame_id],
            'mti_rot': self.mti_ori[frame_id: end_frame_id],
            'gt_pos': self.data['gt_translation'][frame_id+1: end_frame_id+1],
            'gt_rot': self.data['gt_orientation'][frame_id+1: end_frame_id+1],
            'gt_vel': self.data['velocity'][frame_id+1: end_frame_id+1],
            'init_pos': self.data['gt_translation'][frame_id][None, ...],
            'init_rot': self.data['gt_orientation'][frame_id: end_frame_id],
            'init_mti_rot': self.mti_ori[frame_id: end_frame_id],
            'init_vel': self.data['velocity'][frame_id][None, ...],
        }

    def get_init_value(self):
        return {'pos': self.data['gt_translation'][:1],
                'rot': self.data['gt_orientation'][:1],
                'vel': self.data['velocity'][:1]}

    def get_mask(self):
        return self.data['mask']
    
    def get_gravity(self):
        return self.gravity


class SeqInfDataset(SeqDataset):
    def __init__(self, root, dataname, inference_state, device =  'cpu', name='Nav', duration=200, step_size=200, 
                            drop_last = True, mode='inference', usecov = True, useraw = False,conf={}):
        super().__init__(root, dataname, device, name, duration, step_size, mode, drop_last, conf)
        self.data['acc'][:-1] += inference_state['correction_acc'].cpu()[0]
        self.data['gyro'][:-1] += inference_state['correction_gyro'].cpu()[0]
       
        if 'acc_cov' in inference_state.keys() and usecov:
            self.data['acc_cov'] = inference_state['acc_cov'][0]

        if 'gyro_cov' in inference_state.keys() and usecov:
            self.data['gyro_cov'] = inference_state['gyro_cov'][0]


class SeqeuncesDataset(Data.Dataset):
    """
    For the purpose of training and inferering
    1. Abandon the features of the last time frame, since there are no ground truth pose and dt
     to integrate the imu data of the last frame. So the length of the dataset is seq.get_length() - 1
    """
    def __init__(self, data_set_config, mode = None, data_path = None, data_root = None, device= "cuda:0"):
        super(SeqeuncesDataset, self).__init__()
        (
            self.ts,
            self.dt,
            self.acc,
            self.gyro,
            self.gt_pos,
            self.gt_ori,
            self.gt_velo,
            self.mti_ori,
            self.airspeed,
            self.index_map,
            self.seq_idx,
        ) = ([], [], [], [], [], [], [], [], [], [], 0)
        self.uni = torch.distributions.uniform.Uniform(-torch.ones(1), torch.ones(1))
        self.device = device
        self.conf = data_set_config
        self.gravity = data_set_config.gravity if "gravity" in data_set_config.keys() else 9.81007
        # --- optional PRE-HANDOVER BIAS FREEZE -------------------------------------
        # freeze_hist_s > 0 subtracts, from every window, the constant bias fitted on
        # the `freeze_hist_s` seconds of aided data IMMEDIATELY BEFORE that window.
        # The network then sees an already-calibrated signal and learns the RESIDUAL.
        #
        # It calls tools/prewindow_align.freeze_biases -- the same function the
        # deployed evaluator uses -- so training and inference cannot drift apart.
        # Keep freeze_hist_s / freeze_k_sub identical to the --hist / --k_sub used at
        # evaluation, or the comparison is meaningless.
        #
        # Windows with fewer than H frames of history in front of them are DROPPED:
        # there is nothing honest to fit on, and padding would invent aided data.
        self.freeze_hist_s = (data_set_config.freeze_hist_s
                              if "freeze_hist_s" in data_set_config.keys() else 0.0)
        self.freeze_k_sub = (data_set_config.freeze_k_sub
                             if "freeze_k_sub" in data_set_config.keys() else 10)
        self.freeze_b = {}
        # Per-flight diagnostics are collected even when they are not printed, so
        # the information the bar hides stays reachable programmatically:
        # `load_log` is the verbatim text, `_load_stats` / `_freeze_stats` are the
        # numbers the closing summary pools.  Plain lists of dicts -- no reference
        # to the Sequence object is retained, or all 55 full flights (~3.4M rows x
        # ~20 channels) would stay resident instead of just the sliced tensors.
        self.load_log, self._load_stats, self._freeze_stats = [], [], []
        if mode is None:
            self.mode = data_set_config.mode
        else:
            self.mode = mode

        self.DataClass = Sequence.subclasses

        ## the design of datapath provide a quick way to revisit a specific sequence, but introduce some inconsistency
        if data_path is None:
            # Flattened into a job list so one tqdm bar can span the whole section
            # and know its total up front.  This is the ONLY branch that loads many
            # flights; the two single-flight branches below are untouched, which is
            # what keeps eval.py and inference.py byte-identical.
            #
            # THE ORDER IS PRESERVED exactly: seq_id is handed out in this nested
            # order, and anything that maps seq_id back to a flight name (e.g. a
            # per-flight report) relies on it.  Reordering here would silently
            # mislabel such a report.
            jobs = [(c, c["data_root"], p)
                    for c in data_set_config.data_list for p in c.data_drive]
            self._open_bar(len(jobs))
            # try/finally: an exception mid-load must not leave the module-level
            # _BAR set, which would silence loader_log for the rest of the process.
            try:
                for conf, root, path in jobs:
                    self.construct_index_map(conf, root, path, self.seq_idx)
                    self.seq_idx += 1
                    self._tick(path)
            except Exception:
                # A crash mid-load has to stay exactly as debuggable as it is today.
                # Drop the bar, then replay every per-flight line it suppressed, so
                # the traceback still arrives with the full loading context under it
                # instead of a summary of the flights that happened to succeed.
                self._close_bar()
                for line in self.load_log:
                    print(line)
                raise
            finally:
                self._close_bar()   # idempotent; the except path already ran it
        ## the design of dataroot provide a quick way to introduce multiple sequences in eval set, but introduce some inconsistency
        elif data_root is None:
            conf = data_set_config.data_list[0]
            self.construct_index_map(conf, conf["data_root"], data_path, self.seq_idx)
            self.seq_idx += 1
        else:
            conf = data_set_config.data_list[0]
            self.construct_index_map(conf, data_root, data_path, self.seq_idx)
            self.seq_idx += 1

    def load_data(self, seq, start_frame, end_frame):
        if "time" in seq.data.keys():
            self.ts.append(seq.data["time"][start_frame:end_frame])
        self.acc.append(seq.data["acc"][start_frame:end_frame])
        self.gyro.append(seq.data["gyro"][start_frame:end_frame])
        # the groud truth state should include the init state and integrated state, thus has one more frame than imu data
        self.dt.append(seq.data["dt"][start_frame:end_frame+1])
        self.gt_pos.append(seq.data["gt_translation"][start_frame:end_frame+1])
        self.gt_ori.append(seq.data["gt_orientation"][start_frame:end_frame+1])
        self.gt_velo.append(seq.data["velocity"][start_frame:end_frame+1])
        self.mti_ori.append(mti_or_gt(seq)[start_frame:end_frame+1])
        # IMU-rate like acc/gyro (one per dt), NOT state-rate like gt_*: it is a
        # network INPUT, not a label, so it is sliced [start:end] not [start:end+1].
        self.airspeed.append(airspeed_or_zeros(seq)[start_frame:end_frame])

    def construct_index_map(self, conf, data_root, data_name, seq_id):
        seq = self.DataClass[conf.name](data_root, data_name, intepolate = True, **self.conf)
        seq_len = seq.get_length() -1 # abandon the last imu features
        window_size, step_size = conf.window_size, conf.step_size
        ## seting the starting and ending duration with different trianing mode
        start_frame, end_frame = 0, seq_len

        if self.mode == 'train_half':
            end_frame = np.floor(seq_len * 0.5).astype(int)
        elif self.mode == 'test_half':
            start_frame = np.floor(seq_len * 0.5).astype(int)
        elif self.mode == 'train_1m':
            end_frame = 12000
        elif self.mode == 'test_1m':
            start_frame = 12000
        elif self.mode == 'mini':# For the purpse of debug
            end_frame = 1000

        _duration = end_frame - start_frame
        if self.mode == "inference":
            window_size = seq_len
            step_size = seq_len
            self.index_map = [[seq_id, 0, seq_len]]
        elif self.mode == "infevaluate":
            self.index_map +=[
                [seq_id, j, j+window_size] for j in range(
                    0, _duration - window_size, step_size)
            ]
            if self.index_map[-1][2] < _duration:
                # Rerouted, not reworded: an unlabelled integer.  Reached only from
                # inference.py:107 (mode 'infevaluate'), which uses the single-path
                # branch where no bar exists -- so today this is still a plain
                # print.  Routed anyway so it cannot leak an anonymous number into
                # an otherwise clean transcript if infevaluate ever runs over a list.
                loader_log(str(self.index_map[-1][2]))
                self.index_map += [[seq_id, self.index_map[-1][2], seq_len]]
        elif self.mode == 'evaluate':
            # adding the last piece for evaluation
            self.index_map +=[
                [seq_id, j, j+window_size] for j in range(
                    0, _duration - window_size, step_size)
            ]
        elif self.mode == 'train_half_random':
            np.random.seed(1)   
            window_group_size = 3000
            selected_indices = [j for j in range(0, _duration-window_group_size, window_group_size)]
            np.random.shuffle(selected_indices)
            indices_num = len(selected_indices)
            for w in selected_indices[:np.floor(indices_num * 0.5).astype(int)]:  
                self.index_map +=[[seq_id, j, j + window_size] for j in range(w, w+window_group_size-window_size,step_size)]
        elif self.mode == 'test_half_random':
            np.random.seed(1)
            window_group_size = 3000
            selected_indices = [j for j in range(0, _duration-window_group_size, window_group_size)]
            np.random.shuffle(selected_indices)
            indices_num = len(selected_indices)
            for w in selected_indices[np.floor(indices_num * 0.5).astype(int):]:   
                self.index_map +=[[seq_id, j, j + window_size] for j in range(w, w+window_group_size-window_size,step_size)]  
        else:
            ## applied the mask if we need the training.
            self.index_map +=[
                [seq_id, j, j+window_size] for j in range(
                    0, _duration - window_size, step_size)
                    if torch.all(seq.data["mask"][j: j+window_size])
            ]
        
        ## Loading the data from each sequence into 
        self.load_data(seq, start_frame, end_frame)
        # Record the numbers the (possibly suppressed) per-flight lines carry, read
        # straight off `seq` while it is still alive -- and read-only, so no loader
        # change is needed for any of it.
        self._record_flight(seq, data_name, seq_id)
        if self.freeze_hist_s > 0:
            self._apply_freeze(seq_id, data_name)

    def _record_flight(self, seq, data_name, seq_id):
        """Snapshot one flight's diagnostics.  Never keeps `seq` itself.

        getattr defaults throughout: a future Sequence subclass that publishes
        fewer diagnostics must not be able to break dataset construction over a
        progress bar.
        """
        t = seq.data.get("time")
        dt = seq.data.get("dt")
        mask = seq.data.get("mask")
        self._load_stats.append({
            "sid": seq_id,
            "name": _flight_stem(data_name),
            "rows": int(len(t)) if t is not None else 0,
            "secs": float(t[-1] - t[0]) if t is not None and len(t) > 1 else 0.0,
            "hz": (1.0 / float(np.median(dt.numpy()))) if dt is not None and len(dt) else float("nan"),
            "usable": float(mask.double().mean()) if mask is not None and len(mask) else 1.0,
            "gaps": int(getattr(seq, "n_gaps", 0) or 0),
            "mti": "mti_orientation" in seq.data,
            "gerr": float(getattr(seq, "mti_g_error_deg", float("nan"))),
        })

    def _apply_freeze(self, seq_id, data_name):
        """Fit and cache one constant bias per window, ONCE, at construction.

        The bias depends only on data, never on training state, so it is fixed for
        the whole run and belongs off the training hot path.  Computed on CPU in
        float64 so it is identical regardless of GPU, DDP rank or batch composition.
        """
        import pypose as pp
        # Lazy: tools.prewindow_align imports datasets.UAVdataset, so a module-level
        # import here would be circular.
        from tools.prewindow_align import freeze_biases

        dt_med = float(np.median(self.dt[seq_id].numpy()))
        H = int(round(self.freeze_hist_s / dt_med))
        mine = [e for e in self.index_map if e[0] == seq_id]
        # INFERENCE mode makes ONE window spanning the whole flight, starting at
        # frame 0.  Dropping it would leave an empty dataset, so shift its start to
        # H instead: the first `freeze_hist_s` seconds are the aided interval the
        # calibration is fitted on, and the coast begins after it.  That is also the
        # real deployment semantic -- the trajectory legitimately starts H frames in.
        #
        # GATED ON THE MODE, NOT ON THE WINDOW COUNT.  A short TRAIN flight can also
        # yield exactly one window starting at 0, and shifting that one produces a
        # window of length window_size - H among windows of length window_size, which
        # custom_collate cannot stack ("got [6000, 1] ... and [4484, 1]").  Only
        # inference has variable-length windows and batch_size 1.
        if (self.mode == "inference" and len(mine) == 1 and mine[0][1] == 0
                and mine[0][2] > H + 1):
            for e in self.index_map:
                if e[0] == seq_id and e[1] == 0:
                    e[1] = H
            mine = [e for e in self.index_map if e[0] == seq_id]
        keep = [e for e in mine if e[1] >= H]
        if len(keep) < len(mine):
            self.index_map = [e for e in self.index_map
                              if e[0] != seq_id or e[1] >= H]
        if not keep:
            # important=True: this flight contributes NOTHING to the split.  It is
            # rare, it changes what the run actually trains on or scores against,
            # and it must never be hidden behind a progress bar.
            self._freeze_stats.append({"H": H, "keep": 0, "dropped": len(mine),
                                       "ba": 0.0, "bg": 0.0, "empty": True})
            loader_log("  [freeze] %s: every window dropped (needs %d frames of history)"
                       % (data_name, H), important=True)
            return
        st = [e[1] for e in keep]
        gather = lambda t: torch.stack([t[j - H:j] for j in st]).double()
        rot = pp.SO3(torch.stack([self.gt_ori[seq_id][j - H:j].tensor()
                                  for j in st])).double()
        integ = pp.module.IMUPreintegrator(reset=True, prop_cov=False,
                                           gravity=self.gravity).double()
        with torch.no_grad():
            ba, bg = freeze_biases(gather(self.acc[seq_id]), gather(self.gyro[seq_id]),
                                   gather(self.dt[seq_id]), rot,
                                   gather(self.gt_velo[seq_id]), self.gravity,
                                   k_sub=self.freeze_k_sub, integ=integ)
        dtp = self.acc[seq_id].dtype
        for i, j in enumerate(st):
            self.freeze_b[(seq_id, j)] = (ba[i].to(dtp), bg[i].to(dtp))
        # Routine: one line per flight, ~93 per run.  Same text, rerouted; its
        # numbers reappear pooled in the summary printed when the bar closes.
        _ba = float(ba.norm(dim=-1).mean())
        _bg = float(torch.rad2deg(bg.norm(dim=-1)).mean())
        self._freeze_stats.append({"H": H, "keep": len(keep),
                                   "dropped": len(mine) - len(keep),
                                   "ba": _ba, "bg": _bg, "empty": False})
        loader_log("  [freeze] %-32s %3d win (%2d dropped, H=%d) | |b_acc| %.4f m/s^2  "
                   "|b_gyro| %.4f deg/s"
                   % (data_name[:32], len(keep), len(mine) - len(keep), H, _ba, _bg))

    # ---- progress bar ----------------------------------------------------
    # The tqdm handle lives in the module-level _BAR and NEVER on self: the dataset
    # object is pickled to DataLoader workers under Windows spawn, and a live tqdm
    # handle is not safely picklable.  num_workers is 0 today (train.py:761) but
    # this must not become a landmine on the day it is not.

    def _bar_desc(self):
        """`load test      win  6000` -- mode plus window, fixed width so the five
        bars of a run line up in scrollback.

        The window size is what DISTINGUISHES them: train.py deep-copies
        conf.dataset.test once per entry of `metric_horizons` and overwrites
        window_size, so three of the five carry mode 'test' and differ only here.
        """
        try:
            wins = {int(c["window_size"]) for c in self.conf.data_list
                    if "window_size" in c}
            win = str(wins.pop()) if len(wins) == 1 else "mixed"
        except Exception:
            win = "?"
        return "load %-9s win %5s" % (str(self.mode)[:9], win)

    def _open_bar(self, n_total):
        global _BAR, _LOG_SINK
        # Reentrancy guard.  Nothing builds two datasets concurrently in one process
        # today (train.py is strictly sequential; tools/run_odometry_sweep.sh
        # parallelises at the PROCESS level), but a nested build must inherit the
        # outer one's terminal rather than orphan a bar that can never be closed.
        if _LOG_SINK is not None:
            return
        # Capture happens even when no bar is drawn, so `load_log` is always usable.
        _LOG_SINK = self.load_log
        # Skip the bar itself when there is nothing to watch: one that is full the
        # instant it appears is noise, and skipping keeps single-flight configs
        # printing exactly what they print today.
        if _NO_BAR or n_total is None or n_total <= 1:
            return
        # disable= on non-main ranks, matching train.py:543/624.  Under DDP every
        # rank builds every dataset, so without this you get N interleaved bars --
        # and because _BAR is still SET on those ranks, their routine per-flight
        # lines are suppressed too (an 8-GPU run currently emits ~2200 byte-
        # identical duplicates of them before epoch 0).
        _BAR = tqdm.tqdm(total=n_total, desc=self._bar_desc(), unit="flight",
                         # smoothing=0 -> plain elapsed/done instead of tqdm's EWMA.
                         # Flights run 21k-160k rows, so per-item cost varies ~7x and
                         # the default estimator makes the ETA jump on every big file.
                         smoothing=0, leave=True, dynamic_ncols=True, mininterval=0.5,
                         disable=not _is_main_process())

    def _tick(self, path):
        if _BAR is None:
            return
        # refresh=False: let update() do the single redraw, so a redirected log
        # collects one frame per flight instead of two.  The postfix is the running
        # window count -- the one number that says whether the split is actually
        # filling up.  Deliberately NOT the filename: measured, a 17-char stem
        # squeezes the bar to a couple of characters at ncols=80, and a failure
        # names the file in its traceback anyway.
        _BAR.set_postfix_str("%d win" % len(self.index_map), refresh=False)
        _BAR.update(1)

    def _close_bar(self):
        global _BAR, _LOG_SINK, _HINT_SHOWN
        if _LOG_SINK is not self.load_log:
            return          # a nested build; the owner closes it
        _LOG_SINK = None
        if _BAR is None:
            return          # no bar was drawn, so there is no receipt to print
        _BAR.close()
        _BAR = None
        if not _is_main_process():
            return
        for line in self._summary_lines():
            print(line)
        # Once per process, not once per dataset: the escape hatch has to be
        # discoverable from the output itself, but five copies of it is a new wall.
        if not _HINT_SHOWN and _QUIET_LOAD:
            _HINT_SHOWN = True
            print("   [per-flight lines hidden by AIRIMU_LOADER_QUIET; "
                  "they are also kept on dataset.load_log]")

    def _summary_lines(self):
        """One compact receipt per dataset, carrying the numbers the suppressed
        per-flight lines carried.

        Aggregates are POOLED, not means of means: `usable` is row-weighted and the
        freeze biases are window-weighted, because the per-flight values they come
        from are already per-flight means.  A summary that is quietly wrong would be
        worse than the wall of text it replaced.
        """
        st = self._load_stats
        if not st:
            return []
        per_seq = {}
        for e in self.index_map:
            per_seq[e[0]] = per_seq.get(e[0], 0) + 1
        rows = sum(f["rows"] for f in st)
        secs = sum(f["secs"] for f in st)
        wins = len(self.index_map)
        hz = [f["hz"] for f in st if np.isfinite(f["hz"])]
        usable = (100.0 * sum(f["usable"] * f["rows"] for f in st) / rows) if rows else 100.0
        out = ["   %d flights | %.2fM rows | %.2f h @ %.1f Hz | usable %.2f%% | "
               "%d gaps | %d windows"
               % (len(st), rows / 1e6, secs / 3600.0,
                  float(np.median(hz)) if hz else float("nan"),
                  usable, sum(f["gaps"] for f in st), wins)]
        # Flights contributing ZERO windows are invisible today unless the freeze
        # happens to drop them; with the freeze off nothing prints at all.  A split
        # silently scored on fewer flights than it names is exactly what the
        # dataconf comments tell the reader to check before trusting a long horizon.
        empty = [f["name"] for f in st if per_seq.get(f["sid"], 0) == 0]
        if empty:
            out.append("   %d flight(s) contributed NO windows: %s"
                       % (len(empty), ", ".join(empty[:4])
                          + (" ..." if len(empty) > 4 else "")))
        gerr = [f["gerr"] for f in st if f["mti"] and np.isfinite(f["gerr"])]
        if gerr:
            worst = max((f for f in st if f["mti"] and np.isfinite(f["gerr"])),
                        key=lambda f: f["gerr"])
            out.append("   mti_orientation %d/%d | g_body err %.2f deg median, "
                       "%.2f worst (%s)"
                       % (sum(1 for f in st if f["mti"]), len(st),
                          float(np.median(gerr)), worst["gerr"], worst["name"]))
        fz = self._freeze_stats
        kept = sum(f["keep"] for f in fz)
        if fz and kept:
            hs = [f["H"] for f in fz]
            out.append("   freeze %.1f s (H %d-%d) | |b_acc| %.4f m/s^2 | "
                       "|b_gyro| %.4f deg/s | %d win dropped%s"
                       % (self.freeze_hist_s, min(hs), max(hs),
                          sum(f["ba"] * f["keep"] for f in fz) / kept,
                          sum(f["bg"] * f["keep"] for f in fz) / kept,
                          sum(f["dropped"] for f in fz),
                          (", %d flight(s) emptied" % sum(1 for f in fz if f["empty"]))
                          if any(f["empty"] for f in fz) else ""))
        return out

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id, end_frame_id = self.index_map[item][0], self.index_map[item][1], self.index_map[item][2]
        acc = self.acc[seq_id][frame_id: end_frame_id]
        gyro = self.gyro[seq_id][frame_id: end_frame_id]
        if self.freeze_hist_s > 0:
            # Applied ONCE, here, before the network AND before the integrator, so
            # every downstream consumer sees the same calibrated signal.
            b_acc, b_gyro = self.freeze_b[(seq_id, frame_id)]
            acc, gyro = acc - b_acc, gyro - b_gyro
        data = {
            'dt': self.dt[seq_id][frame_id: end_frame_id],
            'acc': acc,
            'gyro': gyro,
            'rot': self.gt_ori[seq_id][frame_id: end_frame_id],
            'mti_rot': self.mti_ori[seq_id][frame_id: end_frame_id],
            'airspeed': self.airspeed[seq_id][frame_id: end_frame_id],
        }
        init_state = {
            'init_rot': self.gt_ori[seq_id][frame_id][None, ...],
            'init_mti_rot': self.mti_ori[seq_id][frame_id][None, ...],
            'init_pos': self.gt_pos[seq_id][frame_id][None, ...],
            'init_vel': self.gt_velo[seq_id][frame_id][None, ...],
        }
        label = {
            'gt_pos': self.gt_pos[seq_id][frame_id+1 : end_frame_id+1],
            'gt_rot': self.gt_ori[seq_id][frame_id+1 : end_frame_id+1],
            'gt_vel': self.gt_velo[seq_id][frame_id+1 : end_frame_id+1],
        }

        return {**data, **init_state, **label}

    def get_dtype(self):
        return self.acc[0].dtype
    


if __name__ == '__main__':
    from datasets.dataset_utils import custom_collate
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/datasets/UAV/uav_6000train_4000eval.conf', help='config file path')
    parser.add_argument("--device", type=str, default='cuda:0', help="cuda or cpu")

    args = parser.parse_args(); print(args)
    conf = ConfigFactory.parse_file(args.config)
    
    dataset = SeqeuncesDataset(data_set_config=conf.train)
    loader = Data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

    for i, (data, init, label) in enumerate(loader):
        for k in data: print(k, ":", data[k].shape)
        for k in init: print(k, ":", init[k].shape)
        for k in label: print(k, ":", label[k].shape)
