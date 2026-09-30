"""Benchmark ``LocalCorrelation.forward`` at a given feature-map size and radius.

Reproduces the throughput claim in ``UPDATES.txt`` section 14 (the
correlation vectorization) at the TRAINER's actual default configuration -
576x1024 images / patch_size 8 -> 72x128 features, radius 4, batch 8 - not
``VisionMambaFlowFrontend``'s own smaller class default (288x384 -> 36x48),
which is what the original 2.24x number was measured at and which
``UPDATES.txt`` mislabelled as the trainer default.

    python tools/benchmark_correlation.py
    python tools/benchmark_correlation.py --height 72 --width 128 --radius 4 --batch 8 --device cuda

Reports forward-only and forward+backward wall time, and peak CUDA memory
when run on a CUDA device - torch has no equivalent figure for CPU, so that
line is skipped there rather than reported as zero. This script measures
throughput and memory only; it says nothing about model accuracy.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch

from vio.models.correlation import LocalCorrelation


def _time_ms(fn, *, device: torch.device, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) / iterations * 1000.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    correlation = LocalCorrelation(radius=args.radius, learnable_temperature=True).to(device)
    left = torch.randn(args.batch, args.channels, args.height, args.width, device=device)
    right = torch.randn(args.batch, args.channels, args.height, args.width, device=device)

    def forward_only() -> None:
        with torch.no_grad():
            correlation(left, right)

    forward_ms = _time_ms(forward_only, device=device, warmup=args.warmup, iterations=args.iterations)

    left_grad = left.clone().requires_grad_(True)

    def forward_backward() -> None:
        correlation.zero_grad(set_to_none=True)
        out = correlation(left_grad, right)
        out["flow"].sum().backward()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    forward_backward_ms = _time_ms(
        forward_backward, device=device, warmup=args.warmup, iterations=args.iterations
    )

    print(f"device: {device}")
    print(
        f"feature map: {args.batch}x{args.channels}x{args.height}x{args.width}, "
        f"radius {args.radius} ({2 * args.radius + 1}x{2 * args.radius + 1} = "
        f"{(2 * args.radius + 1) ** 2} candidates)"
    )
    print(f"forward only:       {forward_ms:8.1f} ms")
    print(f"forward + backward: {forward_backward_ms:8.1f} ms")
    if device.type == "cuda":
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"peak CUDA memory:   {peak_gib:8.2f} GiB")
    else:
        print("peak memory: not measured on CPU (no torch.cuda.max_memory_allocated equivalent)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
