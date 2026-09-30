"""ekf/vo_onnx.py with a stub VO runtime: selection, frames, variance, cache round trip.
The real runtime is exercised by the manual end-to-end check in README (it needs a
trained VO export and images)."""
import numpy as np

from ekf import vo_onnx
from ekf.vo import load_vo_csv


class _StubRuntime:
    """Same interface as VO/tools/onnx_inference.py, deterministic outputs."""

    class VOOnnxRuntime:
        def __init__(self, onnx_dir):
            self.meta = {"dataset_settings": {}, "timing": {"frame_gap": 10, "pair_stride": 10,
                                                            "deployment_latency_s": 0.35}}
            self.stats = {"delivered": 0}
            self.n = 0

        def add_frame(self, image, t):
            pass

        def add_telemetry(self, t, roll, pitch, yaw, alt):
            fresh = self.n % 10 == 0            # 20 Hz ticks, a pair every 0.5 s
            self.n += 1
            self.stats["delivered"] += int(fresh)
            return {"velocity": np.array([20.0, 1.0, -0.5]), "log_variance": np.log([4.0, 1.0, 0.25]),
                    "pair_delivered": fresh}

    @staticmethod
    def read_flight_csv(path, meta):
        t = np.arange(0.0, 10.0, 0.05)
        return {"times": t, "euler": np.zeros((len(t), 3)), "altitude": np.full(len(t), 200.0)}

    @staticmethod
    def list_frames(folder, meta):
        return [(t, None) for t in np.arange(0.0, 10.0, 0.05)]


def test_replay_keeps_pairs_flu_variance_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(vo_onnx, "_RUNTIME", _StubRuntime)
    cache = str(tmp_path / "f_vo.csv")
    s = vo_onnx.replay("onnx", str(tmp_path), time_offset=0.1, save_csv=cache, progress=False)
    assert len(s) == 20 and np.allclose(np.diff(s.t), 0.5) and np.isclose(s.t[0], 0.1)
    assert np.allclose(s.v, [20.0, -1.0, 0.5])               # FRD -> FLU
    assert np.allclose(s.var, [4.0, 1.0, 0.25])              # exp(log_variance)
    back = load_vo_csv(cache, time_offset=0.1)               # pair_delivered selects
    assert np.allclose(back.t, s.t) and np.allclose(back.v, s.v, atol=1e-4)
    assert np.allclose(back.var, s.var, rtol=1e-4)
    assert "pair_delivered == 1" in back.source


def test_real_runtime_module_loads_by_path():
    vo_onnx._RUNTIME = None
    rt = vo_onnx.runtime_module()
    assert hasattr(rt, "VOOnnxRuntime") and hasattr(rt, "read_flight_csv")
    assert rt.__file__.replace("\\\\", "/").endswith("VO/tools/onnx_inference.py")
