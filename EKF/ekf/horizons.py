"""Horizon notation shared by run_ekf.py and run_stream.py.

    30s  1m  2m  1.5m  10m  40m  1h     seconds / minutes / hours
    3000                                 a plain number = FRAMES at 100 Hz (old style)
"""
RATE_HZ = 100.0


def parse(token, plain="frames"):
    """-> frames (int).  `plain` says what a bare number means: 'frames' or 'seconds'."""
    s = str(token).strip().lower()
    for suffix, scale in (("ms", 1e-3), ("s", 1.0), ("m", 60.0), ("h", 3600.0)):
        if s.endswith(suffix) and s[:-len(suffix)].replace(".", "", 1).isdigit():
            return int(round(float(s[:-len(suffix)]) * scale * RATE_HZ))
    x = float(s)
    return int(round(x * RATE_HZ)) if plain == "seconds" else int(round(x))


def label(frames):
    s = frames / RATE_HZ
    if s >= 60 and abs(s / 60 - round(s / 60)) < 1e-9:
        return "%dm" % round(s / 60)
    return "%gs" % s


DEFAULT = ["30s", "1m", "2m", "3m", "4m", "5m", "10m", "15m", "20m", "30m", "40m"]
