"""ONNX export: the exported graph must reproduce the PyTorch network.

    python -m pytest tests/test_export_onnx.py -q
Needs `onnx` and `onnxruntime`; skipped otherwise.  CPU only, no data.
"""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from tools.export_onnx import (build, example_inputs, export, randomise_zero_heads,  # noqa: E402
                               verify)

FRAMES = 900


def _cfg(name):
    return os.path.join(ROOT, "configs/exp/UAV/%s.conf" % name)


@pytest.mark.parametrize("name", ["tilt_rotate", "tilt_aware"])
def test_export_matches_torch_any_batch(tmp_path, name):
    m = build(_cfg(name))
    randomise_zero_heads(m)
    out = str(tmp_path / "m.onnx")
    export(m, FRAMES, out)
    assert verify(m, out, FRAMES, batches=(1, 3, 8), atol=1e-4) <= 1e-4


def test_export_from_checkpoint(tmp_path):
    """--ckpt path: weights saved like train.py saves them come out identical."""
    m = build(_cfg("tilt_rotate"))
    randomise_zero_heads(m, seed=7)
    ck = str(tmp_path / "best_model.ckpt")
    torch.save({"model_state_dict": m.net.state_dict(), "epoch": 42}, ck)
    m2 = build(_cfg("tilt_rotate"), ck)
    out = str(tmp_path / "m.onnx")
    export(m2, FRAMES, out)
    xs = example_inputs(m, 2, FRAMES, seed=5)
    with torch.no_grad():
        ref = m(*xs)[0].numpy()
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    got = sess.run(["corrected_acc"], {k: v.numpy() for k, v in zip(m.input_names(), xs)})[0]
    assert np.abs(ref - got).max() < 1e-4
    assert np.abs(got - xs[0][:, 9:].numpy()).max() > 1e-3     # it really corrects


def test_export_causal_cnn(tmp_path):
    m = build(_cfg("tilt_rotate"))
    m.net.conf.put("causal_cnn", True)
    from model import net_dict
    net = net_dict["hybridnet"](m.net.conf).eval()
    from tools.export_onnx import OnnxHybrid
    m = OnnxHybrid(net).eval()
    randomise_zero_heads(m)
    out = str(tmp_path / "m.onnx")
    export(m, FRAMES, out)
    assert verify(m, out, FRAMES, batches=(1, 4)) <= 1e-4


def test_export_40s_default_window(tmp_path):
    """The shipped default: 4000 frames = 40 s, the training window."""
    m = build(_cfg("tilt_rotate"))
    randomise_zero_heads(m)
    out = str(tmp_path / "tilt_rotate_40s.onnx")
    export(m, 4000, out)
    import onnx
    g = onnx.load(out).graph
    dims = [d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
    assert dims[1:] == [4009, 3]                      # 4000 + 9 history samples
    assert verify(m, out, 4000, batches=(1, 2)) <= 1e-4
