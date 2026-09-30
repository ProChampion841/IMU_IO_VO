"""Loader for the fixed-wing UAV flight logs in ``data/*_sensor_data.csv``.

Frame / unit conventions of the raw logs (established empirically -- see
``tools/verify_uav_conversion.py`` for the reproducible evidence):

======================  ==========================================================
Raw column              Meaning
======================  ==========================================================
``Time``                seconds, ~100 Hz, jittery (median dt 9.93 ms)
``AcclX/Y/Z``           specific force in **g**, body frame **FRD**
                        (x fwd, y right, z down) -> reads ``-1 g`` on z when level
``GyroX/Y/Z``           angular rate in **deg/s**, same FRD body frame
``GPSNavEulX/Y/Z``      (roll, pitch, yaw) in **radians**, intrinsic ``ZYX``,
                        rotating body(FRD) -> world(**NED**)
``GPSNavVnX/Y/Z``       velocity (V_north, V_east, V_down) in **m/s**, NED
``GPSNavAlt``           altitude in metres (up-positive)
``EulX/Y/Z``            a *second* (MTI / magnetometer) attitude solution,
                        ``(roll, pitch, yaw)`` in **degrees**, intrinsic ``ZYX``,
                        already in a **z-up** convention -- i.e. these are the
                        NWU/FLU angles directly, *not* the NED/FRD ones.  Its
                        yaw is referenced to magnetic East.  Published as
                        ``mti_orientation``; see ``MTI ATTITUDE`` below.
======================  ==========================================================

pypose's ``IMUPreintegrator`` computes ``a_body = acc - rot.Inv() @ [0,0,+g]``,
i.e. it assumes a **z-up** world in which a level, stationary sensor reads
``+1 g`` on its z axis.  The logs are z-down.  We therefore rotate both the body
and the world frame by

    T = diag(1, -1, -1)        (a 180 deg rotation about x; det = +1)

which maps body FRD -> FLU and world NED -> NWU (North-West-Up).  Because ``T``
is a proper rotation this is an exact change of basis, not a reflection:

    acc_flu   = 9.81007 * (AcclX, -AcclY, -AcclZ)          [m/s^2]
    gyro_flu  = (pi/180) * (GyroX, -GyroY, -GyroZ)         [rad/s]
    R_nwu_flu = T @ R_ned_frd @ T                          [body -> world]
    v_nwu     = (V_north, -V_east, -V_down)                [m/s, z-up]

Ground-truth position is the cumulative trapezoidal integral of ``v_nwu``.  It is
built from the velocity *on purpose*: AirIMU supervises position and velocity at
the same timestamps, so the two labels must satisfy ``p(t) = p(0) + int v dt``
exactly.  Substituting barometric altitude for the vertical channel would break
that relation and feed the network contradictory gradients.  The (slow) vertical
drift of the integrated altitude relative to ``GPSNavAlt`` is reported by
``altitude_drift`` as a diagnostic.

MTI ATTITUDE
------------
``EulX/Y/Z`` is a second, GPS-independent attitude solution.  Its convention was
established empirically over all 83 usable flights (``tools/verify_mti_attitude.py``):
building ``R_e = from_euler("ZYX", [EulZ, EulY, EulX])`` in *degrees* and
comparing against ``R_nwu_flu`` leaves a residual that is a **pure rotation about
the world z axis** (median ``|axis_z|`` 0.998-0.9998 per flight).  That is the
whole story: the roll and pitch of ``Eul`` are already NWU/FLU angles, and only
the yaw reference differs.  Concretely, over the corpus

    roll:   mean offset  +0.04 deg,  per-flight std 1.36 deg   (medians)
    pitch:  mean offset  -0.59 deg,  per-flight std 1.77 deg
    yaw:    mean offset +76.94 deg = 90 - 13.06

so the ``~13 deg`` in the original note is a magnetic declination and the ``90``
is the East reference (in NWU, East sits at yaw ``-90``).  We therefore publish

    R_mti_nwu_flu = Rz(-(90 - declination)) @ from_euler("ZYX", [EulZ,EulY,EulX])

**The yaw is not trustworthy and the caller must not pretend otherwise.**  The
declination fitted per flight has a circular std of 33.6 deg across flights, and
worse, the offset drifts *within* a flight (median within-flight circular std
11.5 deg; six flights exceed 50 deg).  Roll and pitch are fine.  This is exactly
why the network feature derived from a rotation is ``g_body = R.Inv() @ [0,0,1]``
(see ``model/attitude.py``), which is invariant to yaw: measured against the
ground-truth rotation, the MTI ``g_body`` is off by a median 2.0 deg (p95 5.9)
even on the flights whose heading is unusable.

``mti_yaw_ref`` selects the yaw handling:

    ``"fixed"``        the corpus constant above.  The historical default, and the
                       reason the heading was written off: it is wrong by 33 deg on
                       average and by 110-130 deg on two of the eleven test flights.
    ``"raw"``          leave the magnetic-East reference in place.
    ``"fit"``          circular mean of ``yaw_mti - yaw_gt`` over the flight.  An
                       ORACLE -- it leaks the GPS solution.  Diagnostics only.
    ``"wind"``         ``fit_yaw_wind`` over the whole flight.  GT-free but not
                       causal (it reads samples after the window it is used in), so
                       it is an offline relabelling, not an achievable number.
    ``"wind_causal"``  ``fit_yaw_wind_causal``, refitting on a trailing window as
                       the flight proceeds.  GT-free AND causal; this is the one to
                       quote.  Tuned by ``mti_yaw_hist_s`` (default 120 s) and
                       ``mti_yaw_hop_s``.

The yaw offset is recoverable without any attitude label because ground velocity is
air velocity plus wind: fly more than one heading and the wind triangle separates a
heading error from a crosswind.  Measured on the 11 test flights, rms heading error
on airborne frames after t0+60 s:

    fixed 54.97 deg -> wind_causal 17.19 deg, against a per-flight oracle of 16.47

i.e. the deployable estimator reaches the ceiling of what any single-number-per-
flight scheme could do, oracle included, because it also tracks the within-flight
drift.  Roll and pitch were always fine and are untouched; ``g_body`` is invariant
to yaw and so is bit-for-bit identical under every mode above.
"""

import os

import numpy as np
import pandas as pd
import pypose as pp
import torch
from scipy.spatial.transform import Rotation as _R

# loader_log: the choke point that keeps these per-flight diagnostics from
# garbling the loading progress bar.  Same module as Sequence, so this adds no
# new import edge and no cycle.
from .dataset import Sequence, loader_log

# 180 deg rotation about x: body FRD <-> FLU, world NED <-> NWU.
_T = np.diag([1.0, -1.0, -1.0])
_FLIP = np.array([1.0, -1.0, -1.0])

_IMU_COLS = ["Time", "GyroX", "GyroY", "GyroZ", "AcclX", "AcclY", "AcclZ"]
_NAV_COLS = [
    "GPSNavEulX", "GPSNavEulY", "GPSNavEulZ",
    "GPSNavVnX", "GPSNavVnY", "GPSNavVnZ",
    "GPSNavAlt", "AirSpeed",
]
REQUIRED_COLUMNS = _IMU_COLS + _NAV_COLS

# The second attitude solution.  Deliberately NOT in REQUIRED_COLUMNS: the row
# filter below is ``dropna(subset=REQUIRED_COLUMNS)``, so promoting these would
# change which frames survive and silently perturb acc/gyro/gt for every
# existing experiment.  They are read if present and ignored if not.
_MTI_COLS = ["EulX", "EulY", "EulZ"]
OPTIONAL_COLUMNS = _MTI_COLS

#: corpus-fitted magnetic declination, degrees.  Circular mean over all 83
#: usable flights of ``90 - (yaw_Eul - yaw_gt)``; 11.32 over the 55 flights whose
#: within-flight yaw offset is stable (circular std < 15 deg), median 13.03.
#: The across-flight circular std is 33.6 deg, so treat this as an order of
#: magnitude, not a calibration.
MTI_DECLINATION_DEG = 13.06


class UAV(Sequence):
    """A single fixed-wing flight log, converted to AirIMU/pypose conventions.

    Populates ``self.data`` with the keys ``SeqeuncesDataset`` /
    ``SeqDataset`` consume:

    ============================  ==================  ==========================
    key                           shape / dtype       units & frame
    ============================  ==================  ==========================
    ``time``                      ``(N,)``   f64      seconds
    ``dt``                        ``(N-1,1)`` f64     seconds
    ``acc``                       ``(N,3)``  f64      m/s^2, body FLU
    ``gyro``                      ``(N,3)``  f64      rad/s, body FLU
    ``gt_orientation``            ``(N,)``   SO3      body FLU -> world NWU
    ``gt_translation``            ``(N,3)``  f64      m, world NWU
    ``velocity``                  ``(N,3)``  f64      m/s, world NWU
    ``mask``                      ``(N,)``   bool     True where usable
    ============================  ==================  ==========================
    """

    #: gravity magnitude used for the g -> m/s^2 conversion and by the integrator
    GRAVITY = 9.81007

    def __init__(
        self,
        data_root,
        data_name,
        intepolate=True,          # accepted for API parity; logs are already
        calib=None,               # time-aligned so no interpolation is needed
        glob_coord=False,
        groundspeed_thresh=5.0,
        trim_to_airborne=True,
        dt_lo_ratio=0.4,
        dt_hi_ratio=2.5,
        gap_guard=1,
        gravity=None,
        dtype="float64",
        mti_attitude=True,
        mti_declination=None,
        mti_yaw_ref="fixed",
        mti_yaw_hist_s=120.0,
        mti_yaw_hop_s=10.0,
        mti_diagnostics=True,
        **kwargs,
    ):
        super(UAV, self).__init__()
        self.data_root, self.data_name = data_root, data_name
        self.data = dict()
        self.gravity = float(gravity) if gravity is not None else self.GRAVITY
        # All maths below is done in float64; only the published tensors are cast.
        # float32 is ~64x faster on consumer GeForce cards (which run fp64 at
        # 1/64 rate) and costs 4.7e-5 m of preintegration error over a 1000-frame
        # window against an ~11 m signal, i.e. it is numerically free here.
        tdtype = {"float32": torch.float32, "float64": torch.float64,
                  "f32": torch.float32, "f64": torch.float64}[str(dtype).lower()]

        path = self._resolve(data_root, data_name)
        df = self._read(path)

        # ---- 1. drop unusable rows -------------------------------------
        # Several logs end in a truncated all-empty record; a NaN anywhere in a
        # required column makes the whole row unusable.
        n_raw = len(df)
        df = df.dropna(subset=REQUIRED_COLUMNS)
        # Time must be strictly increasing for dt to be meaningful.
        t = df["Time"].to_numpy(dtype=np.float64)
        keep = np.ones(len(df), dtype=bool)
        keep[1:] = np.diff(t) > 0
        df = df[keep]
        n_clean = len(df)

        # ---- 2. optionally trim to the airborne segment -----------------
        # Motion is judged by GPS ground speed, NOT by AirSpeed: the pitot is
        # biased and in at least one log (2026_01_19_159_11) reads a median
        # 14.9 m/s for 89% of a flight during which |GPSNavVn| < 0.5 m/s and
        # |gyro| < 1 deg/s -- i.e. while the aircraft is provably parked.  An
        # AirSpeed criterion would admit that entire log as flight data.
        speed = np.linalg.norm(
            df[["GPSNavVnX", "GPSNavVnY", "GPSNavVnZ"]].to_numpy(dtype=np.float64), axis=1
        )
        air = speed > groundspeed_thresh
        if trim_to_airborne and air.any():
            i0, i1 = int(np.argmax(air)), int(len(air) - np.argmax(air[::-1]))
            df = df.iloc[i0:i1]
            speed = speed[i0:i1]
            air = air[i0:i1]
        df = df.reset_index(drop=True)

        # ---- 3. raw arrays in native (FRD / NED) conventions -------------
        t = df["Time"].to_numpy(dtype=np.float64)
        acc_frd_g = df[["AcclX", "AcclY", "AcclZ"]].to_numpy(dtype=np.float64)
        gyro_frd_dps = df[["GyroX", "GyroY", "GyroZ"]].to_numpy(dtype=np.float64)
        roll = df["GPSNavEulX"].to_numpy(dtype=np.float64)
        pitch = df["GPSNavEulY"].to_numpy(dtype=np.float64)
        yaw = df["GPSNavEulZ"].to_numpy(dtype=np.float64)
        v_ned = df[["GPSNavVnX", "GPSNavVnY", "GPSNavVnZ"]].to_numpy(dtype=np.float64)
        alt = df["GPSNavAlt"].to_numpy(dtype=np.float64)
        have_mti = bool(mti_attitude) and all(c in df.columns for c in _MTI_COLS)
        eul_deg = df[_MTI_COLS].to_numpy(dtype=np.float64) if have_mti else None

        # ---- 4. convert to pypose conventions ---------------------------
        acc = acc_frd_g * self.gravity * _FLIP           # m/s^2, body FLU
        gyro = np.deg2rad(gyro_frd_dps) * _FLIP          # rad/s, body FLU

        # body(FRD) -> world(NED); scipy 'ZYX' is intrinsic yaw-pitch-roll.
        R_ned_frd = _R.from_euler("ZYX", np.stack([yaw, pitch, roll], axis=1)).as_matrix()
        R_nwu_flu = _T[None] @ R_ned_frd @ _T[None]      # exact change of basis
        quat_xyzw = _R.from_matrix(R_nwu_flu).as_quat()  # scipy returns xyzw

        vel = v_ned * _FLIP                              # m/s, world NWU (z up)
        pos = _cumtrapz(vel, t)                          # m,  world NWU

        # ---- 5. validity mask -------------------------------------------
        # Five logs contain genuine recording gaps (dt up to 16 s).  Integrating
        # across one is meaningless, so the gap frames are masked; because
        # SeqeuncesDataset only emits a train/test window when *every* frame in
        # it is unmasked, masking the gap is equivalent to splitting the file
        # there.  `gap_guard` also drops a few frames on each side, since the
        # samples bracketing a dropout are usually themselves suspect.
        dt = np.diff(t)
        dt_med = float(np.median(dt))
        ok_dt = (dt > dt_lo_ratio * dt_med) & (dt < dt_hi_ratio * dt_med)
        mask = np.zeros(len(t), dtype=bool)
        mask[:-1] = ok_dt
        mask[-1] = ok_dt[-1] if len(ok_dt) else False
        if gap_guard > 0 and (~mask).any():
            bad = np.flatnonzero(~mask)
            for s in range(1, gap_guard + 1):
                mask[np.clip(bad - s, 0, len(mask) - 1)] = False
                mask[np.clip(bad + s, 0, len(mask) - 1)] = False
        mask &= air
        self.n_gaps = int((dt > 0.05).sum())

        # ---- 5b. the second (MTI) attitude solution ---------------------
        # Eul is already in the z-up convention, so -- unlike GPSNavEul -- it is
        # NOT passed through T.  Only its yaw reference is wrong; see the
        # MTI ATTITUDE section of the module docstring.
        quat_mti = None
        # `mti_declination_fit` is what this flight's data says the declination
        # is (an ORACLE: it is fitted against gt_orientation, so it is a
        # diagnostic only).  `mti_declination_applied` is the constant that was
        # actually removed from the published rotation.  They differ, often a
        # lot; that difference is the flight's heading error.
        self.mti_declination_fit = float("nan")
        self.mti_declination_applied = float("nan")
        self.mti_yaw_offset = float("nan")
        self.mti_yaw_std = float("nan")
        self.mti_roll_bias = float("nan")
        self.mti_pitch_bias = float("nan")
        self.mti_g_error_deg = float("nan")
        # `mti_yaw_cov` is the fraction of frames whose yaw offset came from a real
        # estimate rather than the fallback constant; 0.0 for every non-wind mode.
        self.mti_yaw_cov = 0.0
        self.mti_yaw_fit = None
        if have_mti and np.isfinite(eul_deg).all() and len(t):
            eul_rad = np.deg2rad(eul_deg)                 # (roll, pitch, yaw)
            R_mti = _R.from_euler(
                "ZYX", np.stack([eul_rad[:, 2], eul_rad[:, 1], eul_rad[:, 0]], axis=1)
            ).as_matrix()

            # yaw of an intrinsic-ZYX rotation is atan2(R[1,0], R[0,0])
            yaw_mti = np.arctan2(R_mti[:, 1, 0], R_mti[:, 0, 0])
            yaw_gt = np.arctan2(R_nwu_flu[:, 1, 0], R_nwu_flu[:, 0, 0])
            good = mask if mask.any() else np.ones(len(t), dtype=bool)
            off = _circ_mean(yaw_mti[good] - yaw_gt[good])          # radians
            self.mti_yaw_offset = float(np.rad2deg(off))
            self.mti_yaw_std = float(np.rad2deg(_circ_std(yaw_mti[good] - yaw_gt[good])))
            self.mti_declination_fit = float(_wrap_deg(90.0 - self.mti_yaw_offset))

            decl = MTI_DECLINATION_DEG if mti_declination is None else float(mti_declination)
            fb = np.deg2rad(90.0 - decl)
            self.mti_yaw_cov = 0.0
            if mti_yaw_ref == "fixed":
                psi = fb
            elif mti_yaw_ref == "fit":
                # ORACLE: fitted against the GPS-aided solution.  Diagnostics only.
                psi = off
            elif mti_yaw_ref == "raw":
                psi = 0.0
            elif mti_yaw_ref in ("wind", "wind_causal"):
                # GT-FREE: the wind triangle against GPS velocity.  See fit_yaw_wind.
                # "wind" is one fit over the whole flight -- it reads samples after
                # the window it will be used in, so it is an offline relabelling, not
                # something that could run online.  "wind_causal" refits as the flight
                # proceeds and only ever looks backwards; prefer it for anything whose
                # number is going to be quoted as achievable.
                vxy = vel[:, :2]
                if mti_yaw_ref == "wind":
                    r = fit_yaw_wind(yaw_mti[good], vxy[good])
                    psi = fb if not np.isfinite(r["psi"]) else r["psi"]
                    self.mti_yaw_cov = 0.0 if not np.isfinite(r["psi"]) else 1.0
                    self.mti_yaw_fit = r
                else:
                    psi, self.mti_yaw_cov = fit_yaw_wind_causal(
                        yaw_mti, vxy, t, fb,
                        hist_s=mti_yaw_hist_s, hop_s=mti_yaw_hop_s)
            else:
                raise ValueError(
                    "mti_yaw_ref must be 'fixed', 'fit', 'raw', 'wind' or "
                    "'wind_causal', got %r" % (mti_yaw_ref,)
                )
            # psi may be a scalar or per-sample; from_euler handles both, and a
            # per-sample psi gives an (N,3,3) stack that broadcasts the same way.
            self.mti_declination_applied = float(
                _wrap_deg(90.0 - np.rad2deg(np.atleast_1d(psi).mean())))
            Rz = _R.from_euler("Z", -np.atleast_1d(psi).reshape(-1, 1)).as_matrix()
            R_mti = Rz @ R_mti if Rz.shape[0] > 1 else Rz[0][None] @ R_mti
            quat_mti = _R.from_matrix(R_mti).as_quat()   # xyzw

            if mti_diagnostics:
                # angle between the two gravity directions; g_body is row 2 of R
                cosang = np.clip((R_mti[good, 2, :] * R_nwu_flu[good, 2, :]).sum(1), -1.0, 1.0)
                self.mti_g_error_deg = float(np.rad2deg(np.median(np.arccos(cosang))))
                e_mti = _R.from_matrix(R_mti[good]).as_euler("ZYX", degrees=True)
                e_gt = _R.from_matrix(R_nwu_flu[good]).as_euler("ZYX", degrees=True)
                self.mti_roll_bias = float(np.rad2deg(_circ_mean(np.deg2rad(e_mti[:, 2] - e_gt[:, 2]))))
                self.mti_pitch_bias = float(np.rad2deg(_circ_mean(np.deg2rad(e_mti[:, 1] - e_gt[:, 1]))))
        elif have_mti:
            # important=True: a channel the config may have asked for is silently
            # missing for this flight.  Never hide it behind a progress bar.
            loader_log("  MTI attitude present but contains NaN; mti_orientation not published",
                       important=True)

        # ---- 6. publish --------------------------------------------------
        self.data["time"] = torch.tensor(t, dtype=tdtype)
        self.data["dt"] = torch.tensor(dt, dtype=tdtype)[:, None]
        self.data["acc"] = torch.tensor(acc, dtype=tdtype)
        self.data["gyro"] = torch.tensor(gyro, dtype=tdtype)
        self.data["gt_orientation"] = pp.SO3(torch.tensor(quat_xyzw, dtype=tdtype))
        self.data["gt_translation"] = torch.tensor(pos, dtype=tdtype)
        self.data["velocity"] = torch.tensor(vel, dtype=tdtype)
        self.data["mask"] = torch.tensor(mask, dtype=torch.bool)
        if quat_mti is not None:
            self.data["mti_orientation"] = pp.SO3(torch.tensor(quat_mti, dtype=tdtype))
        # diagnostics (not consumed by the training pipeline)
        self.data["altitude"] = torch.tensor(alt, dtype=tdtype)
        # The logged pitot reading, published raw and UNCALIBRATED.  It is already a
        # required column (it is read for the airborne trim), it was simply never
        # exposed.  It is biased on this platform -- see the trim comment above, which
        # is why motion is judged by GPS ground speed and why `fit_yaw_wind` solves for
        # Va rather than reading this -- so anything consuming it must calibrate a
        # scale and a wind against GPS velocity first.  `tools/airdata_check.py` does
        # exactly that on the aided interval before an outage.
        self.data["airspeed"] = torch.tensor(
            df["AirSpeed"].to_numpy(dtype=np.float64), dtype=tdtype)[:, None]
        self.altitude_drift = float((pos[:, 2] - (alt - alt[0]))[-1]) if len(alt) else float("nan")

        # Routine, ~93 of these per training run.  Rerouted through loader_log so
        # the bar is not garbled; the counts reappear pooled in the per-dataset
        # summary, verbatim under AIRIMU_LOADER_VERBOSE=1, and on dataset.load_log.
        loader_log(
            "loaded: %s | %d rows (%d raw, %d after NaN/time clean) | %.1f s @ %.1f Hz "
            "| usable %.1f%% | %d gaps | alt drift %.1f m"
            % (path, len(t), n_raw, n_clean, t[-1] - t[0] if len(t) > 1 else 0.0,
               1.0 / dt_med if dt_med > 0 else float("nan"),
               100.0 * mask.mean() if len(mask) else 0.0, self.n_gaps, self.altitude_drift)
        )
        if quat_mti is not None:
            # Routine: the MTI yaw calibration line.  Same reroute; its g_body error
            # is what the summary reports as a median plus the worst flight.
            loader_log(
                "  mti_orientation | yaw_ref %s (est on %.0f%% of frames) | "
                "declination fitted %.2f / applied %.2f deg "
                "(removed %.2f deg of yaw from a raw offset of %.2f; within-flight "
                "yaw std %.2f) | roll bias %.2f, pitch bias %.2f | median g_body "
                "error vs gt %.2f deg"
                % (mti_yaw_ref, 100.0 * self.mti_yaw_cov,
                   self.mti_declination_fit, self.mti_declination_applied,
                   90.0 - self.mti_declination_applied,
                   self.mti_yaw_offset, self.mti_yaw_std,
                   self.mti_roll_bias, self.mti_pitch_bias, self.mti_g_error_deg)
            )

    # ------------------------------------------------------------------
    def get_length(self):
        # Number of IMU samples.
        # SeqeuncesDataset subtracts one itself (the last sample has no dt).
        return self.data["time"].shape[0]

    @staticmethod
    def _resolve(data_root, data_name):
        for cand in (
            os.path.join(data_root, data_name),
            os.path.join(data_root, data_name + ".csv"),
            os.path.join(data_root, data_name + "_sensor_data.csv"),
            data_name,
        ):
            if os.path.isfile(cand):
                return cand
        raise FileNotFoundError(
            "UAV log not found for root=%r name=%r" % (data_root, data_name)
        )

    @staticmethod
    def _read(path):
        wanted = set(REQUIRED_COLUMNS) | set(OPTIONAL_COLUMNS)
        df = pd.read_csv(path, usecols=lambda c: c in wanted)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError("%s is missing required columns: %s" % (path, missing))
        return df


def _wrap_deg(a):
    """Wrap degrees into (-180, 180]."""
    return (a + 180.0) % 360.0 - 180.0


def _circ_mean(a):
    """Circular mean of angles in radians.  Returns radians in (-pi, pi]."""
    if len(a) == 0:
        return float("nan")
    return float(np.angle(np.exp(1j * np.asarray(a, dtype=np.float64)).mean()))


def _circ_std(a):
    """Circular standard deviation of angles in radians (Mardia's definition)."""
    if len(a) == 0:
        return float("nan")
    r = abs(np.exp(1j * np.asarray(a, dtype=np.float64)).mean())
    r = min(max(r, 1e-12), 1.0)
    return float(np.sqrt(-2.0 * np.log(r)))


def fit_yaw_wind_causal(yaw, vel_xy, t, fallback, hist_s=120.0, hop_s=10.0,
                        backfill=True, **kw):
    """Causal per-sample yaw offset: at every hop, refit on data ALREADY SEEN.

    ``hist_s`` bounds how far back the fit looks -- ``None`` is an expanding window
    from the first sample, a number is a sliding window of that many seconds.  The
    trade is not subtle on this corpus: the true offset drifts within a flight
    (circular std 4.5 to 31 deg), which argues for a short window, but a short
    window sees less heading variation, and heading variation is the only thing
    that separates the yaw offset from the wind at all.  Measured on the 11 test
    flights, scored on airborne frames after t0+60 s (rms heading error, deg):

        fixed 54.97 | expanding 21.77 | 300 s 19.40 | 120 s 17.19 | 60 s 17.19
        whole-flight non-causal 16.39 | per-flight ORACLE 16.47

    so 120 s is the default: it is fully causal and still lands on the oracle,
    because tracking the drift buys back exactly what the shorter window loses in
    conditioning.  Expanding is clearly worse -- it averages in early-flight
    samples whose offset has since drifted away, and never forgets them.

    Before the first fit can close there is no estimate.  A real aircraft carries
    one forward from its ground alignment, but this corpus is trimmed to airborne
    so that history was thrown away, and leaving those frames on the corpus
    constant is punitive exactly where the constant is worst -- on the flights this
    is meant to rescue, the warm-up alone costs more than the fix saves (one test
    flight: 12.2 deg rms over the covered frames, 32.5 deg once ~3% of frames sit
    at the constant's 130 deg).  ``backfill=True`` therefore applies the FIRST
    in-flight estimate to those leading frames.  That is not causal, and it is
    flagged rather than hidden: it stands in for pre-takeoff data the loader
    discarded, and it touches only frames before the first fit closes.  Set
    ``backfill=False`` to see the punitive version.
    """
    psi = np.full(len(t), float(fallback), dtype=np.float64)
    n = len(t)
    if n < 2:
        return psi, 0.0
    fs = 1.0 / max(float(np.median(np.diff(t))), 1e-6)
    hop = max(int(hop_s * fs), 1)
    hist = None if hist_s is None else max(int(hist_s * fs), 1)
    last, first_idx = float(fallback), None
    e = hop
    while e < n:
        if first_idx is not None:
            psi[e:e + hop] = last                        # apply what was known, then update
        s = 0 if hist is None else max(0, e - hist)
        r = fit_yaw_wind(yaw[s:e], vel_xy[s:e], **kw)
        if np.isfinite(r["psi"]):
            last = r["psi"]
            if first_idx is None:
                first_idx = e
        e += hop
    if first_idx is None:
        return psi, 0.0
    psi[e:] = last
    cov = float((n - first_idx) / n)
    if backfill:
        psi[:first_idx] = psi[first_idx]
    return psi, cov


def fit_yaw_wind(yaw, vel_xy, min_speed=5.0, min_spread_deg=8.0, min_n=200,
                 max_cond=None):
    """Estimate the MTI yaw offset from GPS velocity alone -- NO attitude label.

    This is the deployable replacement for the corpus-constant declination.  It is
    the classical wind triangle: ground velocity is air velocity plus wind, and
    air velocity points along the *true* heading, so if the MTI heading is wrong by
    a constant ``psi`` the two disagree in a way that is separable from wind as
    soon as the aircraft flies more than one heading.

        v_x = Va*cos(y - psi) + W_x
        v_y = Va*sin(y - psi) + W_y            (NWU: x North, y West, y = MTI yaw)

    Substituting ``a = Va*cos(psi)``, ``b = Va*sin(psi)`` makes it LINEAR:

        v_x =  a*cos(y) + b*sin(y) + W_x
        v_y =  a*sin(y) - b*cos(y) + W_y

    so the whole thing is one 4-parameter least squares in ``(a, b, W_x, W_y)``
    with a closed form, and ``psi = atan2(b, a)``, ``Va = hypot(a, b)``.  No
    optimiser, no initial guess, no ground truth.

    WHY IT CAN FAIL, and why the caller is told rather than guessed at.  If the
    aircraft holds one heading, ``cos(y)`` and ``sin(y)`` are constant over the
    sample and become collinear with the two wind columns -- the design matrix
    goes rank deficient and ``psi`` and ``W`` are not separately identifiable.
    That is a real observability limit, not a numerical nuisance: a straight leg
    genuinely cannot tell a heading error from a crosswind.  We therefore refuse
    to return an estimate unless the heading actually varies, and we return the
    diagnostics needed to decide whether to trust one that is returned.

    ``Va`` is solved for rather than read from ``AirSpeed`` on purpose -- the
    logged airspeed on this corpus is not reliable.

    Args
        yaw       (N,) MTI yaw in radians, RAW (magnetic reference still in place)
        vel_xy    (N,2) ground velocity in the same z-up world frame, m/s
        min_speed drop samples slower than this (taxi, ground, hover) -- m/s
        min_spread_deg  circular std of the heading below which the fit is refused
        min_n     minimum usable samples

    Returns a dict.  ``psi`` is NaN when the fit was refused; ``reason`` says why.
    """
    yaw = np.asarray(yaw, dtype=np.float64)
    v = np.asarray(vel_xy, dtype=np.float64)
    spd = np.hypot(v[:, 0], v[:, 1])
    ok = np.isfinite(yaw) & np.isfinite(spd) & (spd >= min_speed)
    out = {"psi": float("nan"), "va": float("nan"), "wind": (float("nan"), float("nan")),
           "n": int(ok.sum()), "spread_deg": float("nan"), "cond": float("inf"),
           "resid": float("nan"), "reason": ""}
    if out["n"] < min_n:
        out["reason"] = "only %d samples above %.1f m/s" % (out["n"], min_speed)
        return out

    y = yaw[ok]
    spread = float(np.rad2deg(_circ_std(y)))
    out["spread_deg"] = spread
    if spread < min_spread_deg:
        out["reason"] = ("heading spread %.1f deg < %.1f: a straight leg cannot separate "
                         "yaw offset from wind" % (spread, min_spread_deg))
        return out

    c, s = np.cos(y), np.sin(y)
    one, zero = np.ones_like(c), np.zeros_like(c)
    # rows: [a, b, W_x, W_y]; stacked as all v_x equations then all v_y equations
    M = np.concatenate([np.stack([c, s, one, zero], axis=1),
                        np.stack([s, -c, zero, one], axis=1)], axis=0)
    rhs = np.concatenate([v[ok, 0], v[ok, 1]])
    sol, _, _, sv = np.linalg.lstsq(M, rhs, rcond=None)
    a, b, wx, wy = sol
    out["cond"] = float(sv[0] / max(sv[-1], 1e-12))
    out["resid"] = float(np.sqrt(np.mean((M @ sol - rhs) ** 2)))
    # A spread test alone is not enough: a leg can wander 10 deg and still leave
    # the heading columns nearly parallel to the wind columns.  The condition
    # number is the direct measure of that, and gating on it is what stops an
    # expanding causal fit from locking onto a confident-looking early estimate
    # made from a prefix that could not have identified psi at all.
    if max_cond is not None and out["cond"] > float(max_cond):
        out["reason"] = ("condition number %.1f > %.1f: yaw offset and wind are not "
                         "separable on this sample" % (out["cond"], float(max_cond)))
        return out
    out["psi"] = float(np.arctan2(b, a))
    out["va"] = float(np.hypot(a, b))
    out["wind"] = (float(wx), float(wy))
    return out


def _cumtrapz(v, t):
    """Cumulative trapezoidal integral of ``v`` (N,3) over ``t`` (N,), starting at 0."""
    out = np.zeros_like(v)
    if len(t) > 1:
        dt = np.diff(t)[:, None]
        out[1:] = np.cumsum(0.5 * (v[1:] + v[:-1]) * dt, axis=0)
    return out
