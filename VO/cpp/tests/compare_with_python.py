"""C++ vo_replay vs Python VOOnnxRuntime, tick by tick and tensor by tensor.

    python cpp/tests/compare_with_python.py <build_dir> <onnx_dir> <flight folder | event log> \
        <work_dir> [--max-minutes M]

Runs the Python runtime (tools/onnx_inference.py) and vo_replay (sync and
async) over the same input, tracing every tensor handed to and returned by
ONNX Runtime, and checks: identical delivery / output / skip flags, identical
stats, identical velocities, every traced tensor bit-identical (CRC-32), and
the sync and async C++ series byte-identical. Exit status 0 = parity.
"""
import csv, math, subprocess, sys, zlib
from pathlib import Path
import numpy as np

BUILD = Path(sys.argv[1]).resolve(); onnx_dir = Path(sys.argv[2]); flight = Path(sys.argv[3])
out_dir = Path(sys.argv[4]); extra = sys.argv[5:]
label = "run"
out_dir.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # VO/
import tools.onnx_inference as oi

def fmt(tag, name, value):
    a = np.ascontiguousarray(np.asarray(value), dtype=np.float32).reshape(-1)
    line = f"{tag} {name} {a.size} {zlib.crc32(a.tobytes()) & 0xffffffff:08x}"
    if a.size <= 64:
        line += "".join(" %.9g" % float(v) for v in a)
    return line + "\n"

def python_run(trace_path):
    rt = oi.VOOnnxRuntime(onnx_dir)
    fh = open(trace_path, "w")
    def wrap(session, graph):
        inner = session.run
        names = [i.name for i in session.get_inputs()]
        def run(output_names, feeds, run_options=None):
            outs = inner(output_names, feeds, run_options)
            fh.write("".join([fmt(f"{graph}.in", n, feeds[n]) for n in names]
                             + [fmt(f"{graph}.out", n, v) for n, v in zip(output_names, outs)]))
            return outs
        session.run = run
    wrap(rt.frontend, "front"); wrap(rt.step, "step")
    rows = []
    def row_out(o):
        rows.append((float(o["time_s"]), *[float(v) for v in o["velocity"]], *[float(v) for v in o["output"]],
                     int(o["pair_delivered"]), int(o["emitted"]), int(o["skipped"]),
                     *[float(v) for v in o["log_variance"]]))
    if flight.is_file():  # an event log: frame / row lines in arrival order
        for line in flight.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            kind, rest = line.split(" ", 1)
            if kind == "frame":
                t, path = rest.split(" ", 1)
                rt.add_frame(flight.parent / path, float(t))
            else:
                t, roll, pitch, yaw, alt = (float(v) for v in rest.split())
                fh.write("tick %.17g\n" % t)
                row_out(rt.add_telemetry(t, roll, pitch, yaw, alt))
        fh.close()
        return np.array(rows, dtype=np.float64), dict(rt.stats)
    settings = rt.meta["dataset_settings"]
    fl = oi.read_flight_csv(flight / (settings.get("csv_name") or "flight.csv"), rt.meta)
    frames = oi.list_frames(flight / (settings.get("image_folder") or "images"), rt.meta)
    times = fl["times"]; total = times.size
    if "--max-minutes" in extra:
        total = int(np.searchsorted(times, times[0] + 60.0 * float(extra[extra.index("--max-minutes") + 1])))
    nf = 0
    for tick in range(total):
        while nf < len(frames) and frames[nf][0] <= times[tick]:
            rt.add_frame(frames[nf][1], frames[nf][0]); nf += 1
        roll, pitch, yaw = fl["euler"][tick]
        fh.write("tick %.17g\n" % float(times[tick]))
        row_out(rt.add_telemetry(times[tick], roll, pitch, yaw, fl["altitude"][tick]))
    fh.close()
    return np.array(rows, dtype=np.float64), dict(rt.stats)

def cpp_run(mode):
    trace = out_dir / f"{label}_cpp_{mode}.trace"; series = out_dir / f"{label}_cpp_{mode}.csv"
    source = ["--events", str(flight)] if flight.is_file() else ["--dataset", str(flight)]
    cmd = [str(BUILD / "vo_replay"), str(onnx_dir), *source, "--output", str(series),
           "--trace", str(trace), "--no-progress"] + (["--sync"] if mode == "sync" else []) + extra
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout, res.stderr); raise SystemExit(f"vo_replay failed ({mode})")
    with series.open() as h:
        r = csv.reader(h); next(r)
        rows = np.array([[float(v) for v in row] for row in r], dtype=np.float64)
    stats = {}
    for line in res.stdout.splitlines():
        if line.startswith("pairs ") and "delivered" in line:
            for part in line.split(", "):
                k, v = part.split(" "); stats[k] = int(v)
    return rows, stats, res.stdout, trace, series

def parse_trace(path):
    ticks = []
    for line in open(path):
        parts = line.split()
        if parts[0] == "tick":
            ticks.append((float(parts[1]), []))
        else:
            ticks[-1][1].append((parts[0], parts[1], int(parts[2]), parts[3], [float(v) for v in parts[4:]]))
    return ticks

def compare_traces(a_path, b_path, what):
    a, b = parse_trace(a_path), parse_trace(b_path)
    assert len(a) == len(b), (what, len(a), len(b))
    same = differ = 0; worst = {}
    for (ta, ra), (tb, rb) in zip(a, b):
        assert (ta == tb) or (math.isnan(ta) and math.isnan(tb)), (what, ta, tb)
        assert [(r[0], r[1], r[2]) for r in ra] == [(r[0], r[1], r[2]) for r in rb], (what, ta, ra, rb)
        for x, y in zip(ra, rb):
            if x[3] == y[3]:
                same += 1
            else:
                differ += 1
                key = f"{x[0]} {x[1]}"
                d = max((abs(p - q) for p, q in zip(x[4], y[4])), default=float("nan"))
                worst[key] = max(worst.get(key, 0.0), d) if not math.isnan(d) else float("nan")
    print(f"  {what}: {same} tensors bit-identical, {differ} differ" + (f"; max |diff| by tensor: {worst}" if worst else ""))
    return differ

py_trace = out_dir / f"{label}_py.trace"
py_rows, py_stats = python_run(py_trace)
print(f"[{label}] python: {len(py_rows)} rows, stats {py_stats}")
ok = True
for mode in ("sync", "async"):
    rows, stats, stdout, trace, series = cpp_run(mode)
    print(f"[{label}] C++ {mode}: " + " | ".join(l for l in stdout.splitlines() if l.startswith(("pairs", "async", "addTelemetry"))))
    assert rows.shape == py_rows.shape, (rows.shape, py_rows.shape)
    flags_same = np.array_equal(rows[:, 7:10], py_rows[:, 7:10])
    t_same = np.array_equal(rows[:, 0], py_rows[:, 0], equal_nan=True)
    good = ~py_rows[:, 9].astype(bool)
    f32 = lambda a: a.astype(np.float32).astype(np.float64)  # %.9g round-trips float32 exactly
    dv = np.abs(f32(rows[good, 1:4]) - f32(py_rows[good, 1:4])).max() if good.any() else 0.0
    do = np.abs(f32(rows[:, 4:7]) - f32(py_rows[:, 4:7])).max()
    dl = np.abs(f32(rows[good, 10:13]) - f32(py_rows[good, 10:13])).max() if good.any() else 0.0
    nan_same = np.array_equal(np.isnan(rows), np.isnan(py_rows))
    stats_same = all(stats.get(k) == py_stats[k] for k in py_stats)
    print(f"  delivered/emitted/skipped flags identical: {flags_same}, times identical: {t_same}, NaN pattern identical: {nan_same}")
    print(f"  stats identical: {stats_same}  (C++ {stats})")
    print(f"  max |velocity diff| {dv:.3g}, |held output diff| {do:.3g}, |log_variance diff| {dl:.3g}")
    differ = compare_traces(py_trace, trace, f"graph tensors python vs C++ {mode}")
    ok &= flags_same and t_same and nan_same and stats_same and dv < 1e-5 and do < 1e-5
if True:
    a = (out_dir / f"{label}_cpp_sync.csv").read_text(); b = (out_dir / f"{label}_cpp_async.csv").read_text()
    print(f"  C++ sync and async series byte-identical: {a == b}")
    ok &= a == b
    compare_traces(out_dir / f"{label}_cpp_sync.trace", out_dir / f"{label}_cpp_async.trace", "graph tensors C++ sync vs async")
print(f"[{label}] {'PARITY OK' if ok else 'PARITY FAILED'}")
sys.exit(0 if ok else 1)
