#!/usr/bin/env python3
"""Multi-GPU smoke test for the reliability gate and the deterministic loss.

Run this ONCE on the real training box before committing to a long run:

    torchrun --standalone --nproc_per_node=2 tools/check_ddp_gate.py

It exercises what a single-process unit test provably cannot - the real
DistributedDataParallel reducer, with the exact ``find_unused_parameters`` and
``static_graph`` settings ``train_fixedwing_vo.py`` uses - across the cases
where those settings are most likely to break:

* every pair accepted, every pair REFUSED, and a mixture
* ``--velocity-loss simple`` (variance and concentration heads unused) and
  ``nll`` (all heads used)

Each case runs several iterations on purpose. ``static_graph`` records the
used-parameter set on the FIRST iteration and only raises on a LATER one that
disagrees, so a single step passes no matter what is wrong. A gate that
refuses every pair in one batch and not the next is exactly that situation,
which is why the refusal is implemented as a multiply-by-zero rather than as a
dropped index - see ``scatter_visual_tokens``.

``--no-distributed`` runs the same cases in one process without DDP. That
checks the cases themselves are well formed; it does NOT check the reducer,
which is the entire point of the tool, so it is a development aid rather than
a substitute.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_ROOT / "src", _ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import torch.distributed as dist  # noqa: E402
from torch.nn.parallel import DistributedDataParallel  # noqa: E402

from vio.models.vision_mamba_vo import (  # noqa: E402
    AIDING_INPUT_DIM,
    VisionMambaFlowFrontend,
    VisionMambaVO,
)
from tools.train_fixedwing_vo import VOStep, compute_velocity_loss  # noqa: E402

WINDOW_LENGTH = 8
EVENTS_PER_WINDOW = 4
VISUAL_DIM = 8
IMAGE_SIZE = (64, 96)

#: Thresholds no real correlation can satisfy, so every pair is refused.
REFUSE_EVERYTHING = {
    "max_cell_entropy": 0.001,
    "min_cell_confidence": 0.999,
    "min_reliable_cell_fraction": 0.9,
}
#: Thresholds that bite without emptying the stream, chosen so the delivered
#: count VARIES between iterations - [7, 6, 4, 5] on this fixture. That
#: variation is the whole point: static_graph records the used-parameter set on
#: iteration one and only fails on a later one that differs, so a mixed case
#: that happened to deliver the same count every step would test nothing.
#: Retune these if the fixture or the gate's scale changes, and check the
#: printed `delivered=` column really does vary.
REFUSE_SOME = {"min_cell_confidence": 0.85, "min_reliable_cell_fraction": 0.60}

CASES = (
    ("accept-all/simple", {}, "simple"),
    ("accept-all/nll", {}, "nll"),
    ("refuse-all/simple", REFUSE_EVERYTHING, "simple"),
    ("refuse-all/nll", REFUSE_EVERYTHING, "nll"),
    ("mixed/simple", REFUSE_SOME, "simple"),
    ("mixed/nll", REFUSE_SOME, "nll"),
)


def build_step(gate_settings: Dict[str, float], device: torch.device) -> VOStep:
    """A deliberately tiny model - this measures wiring, not accuracy."""

    torch.manual_seed(0)
    frontend = VisionMambaFlowFrontend(
        visual_dim=VISUAL_DIM, d_model=8, depth=1, patch_size=8,
        image_size=IMAGE_SIZE, context_grid=(4, 6), correlation_radius=2,
        token_grid=4, dropout=0.0, **gate_settings,
    )
    model = VisionMambaVO(
        visual_dim=VISUAL_DIM, aiding_dim=8, fusion_dim=8,
        dropout=0.0, frontend=frontend,
    )
    return VOStep(
        model,
        window_length=WINDOW_LENGTH,
        visual_dim=VISUAL_DIM,
        disable_visual=False,
        frontend_chunk=2,
        deployment_latency_s=0.0,
    ).to(device)


def make_batch(seed: int, device: torch.device, batch: int = 2) -> Dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    build = {
        "aiding": torch.randn(batch, WINDOW_LENGTH, AIDING_INPUT_DIM, generator=generator),
        "log_altitude": torch.full((batch, WINDOW_LENGTH), 5.0),
        "image0": torch.randint(
            0, 256, (batch, EVENTS_PER_WINDOW, 1, *IMAGE_SIZE),
            dtype=torch.uint8, generator=generator,
        ),
        "image1": torch.randint(
            0, 256, (batch, EVENTS_PER_WINDOW, 1, *IMAGE_SIZE),
            dtype=torch.uint8, generator=generator,
        ),
        "pair_dt_s": torch.full((batch, EVENTS_PER_WINDOW), 0.05),
        "body_rate": torch.zeros(batch, EVENTS_PER_WINDOW, 3),
        "offsets": torch.tensor([[0, 2, 4, 6]] * batch),
        "valid": torch.ones(batch, EVENTS_PER_WINDOW),
        "times_s": torch.arange(WINDOW_LENGTH).float().expand(
            batch, WINDOW_LENGTH
        ).contiguous(),
        "target": torch.randn(batch, WINDOW_LENGTH, 3, generator=generator),
        "mask": torch.ones(batch, WINDOW_LENGTH),
    }
    return {key: value.to(device) for key, value in build.items()}


def run_case(
    name: str,
    gate_settings: Dict[str, float],
    loss_mode: str,
    *,
    device: torch.device,
    distributed: bool,
    local_rank: int,
    iterations: int,
) -> Dict[str, object]:
    step = build_step(gate_settings, device)
    module: torch.nn.Module = step
    if distributed:
        # EXACTLY the flags train_fixedwing_vo.py sets, or this proves nothing
        # about the run it is meant to de-risk.
        module = DistributedDataParallel(
            step,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=(loss_mode == "simple"),
            static_graph=True,
            gradient_as_bucket_view=False,
        )
    optimizer = torch.optim.AdamW(step.parameters(), lr=1e-4)

    delivered: List[float] = []
    loss_value = float("nan")
    for iteration in range(iterations):
        batch = make_batch(100 + iteration, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = module(
            batch["aiding"], batch["log_altitude"],
            batch["image0"], batch["image1"], batch["pair_dt_s"],
            batch["body_rate"], batch["offsets"], batch["valid"],
            batch["times_s"],
        )
        loss, _ = compute_velocity_loss(
            prediction, batch["target"], batch["mask"],
            direction_weight=0.5, loss_mode=loss_mode, huber_delta=1.0,
        )
        loss.backward()
        optimizer.step()
        delivered.append(step._pairs_delivered)
        loss_value = float(loss.detach())

    frontend_parameters = list(step.model.frontend.parameters())
    # Two DIFFERENT questions, and only the first one is about DDP.
    #
    # "Was the parameter USED this iteration" is what the reducer tracks, and
    # it means a .grad tensor was allocated - the VALUE may be all zeros. That
    # is exactly the refuse-all case: multiplying the token by zero keeps the
    # edge, so every frontend parameter is used and static_graph sees the same
    # set every iteration. Testing for a NONZERO gradient instead would report
    # a correct run as broken, which is the mistake this comment exists to
    # stop the next reader from repeating.
    reached = sum(1 for p in frontend_parameters if p.grad is not None)
    moved = sum(
        1 for p in frontend_parameters
        if p.grad is not None and torch.any(p.grad != 0)
    )
    return {
        "loss": loss_value,
        "delivered_per_step": delivered,
        # The property the multiply-by-zero exists to preserve: even when every
        # pair is refused, every frontend parameter is still reached. If this
        # is not "all", static_graph will fail on a real run the first time a
        # batch differs from the one it recorded.
        "frontend_reached": f"{reached}/{len(frontend_parameters)}",
        "frontend_moved": f"{moved}/{len(frontend_parameters)}",
        # Under "simple" these heads are deliberately out of the loss, which is
        # what find_unused_parameters is there to tolerate.
        "variance_head_has_gradient": (
            step.model.log_variance_head.weight.grad is not None
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-test the gate and the simple loss under real DDP.",
    )
    parser.add_argument(
        "--iterations", type=int, default=4,
        help="Steps per case. Must be at least 2: static_graph only fails on "
             "an iteration AFTER the one it recorded.",
    )
    parser.add_argument(
        "--no-distributed", action="store_true",
        help="Run the cases in one process without DDP. Checks the cases are "
             "well formed; does NOT check the reducer, which is the point.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.iterations < 2:
        raise SystemExit("--iterations must be at least 2 to exercise static_graph")

    distributed = not args.no_distributed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = True

    if distributed:
        if "RANK" not in os.environ:
            raise SystemExit(
                "not running under torchrun. Launch with:\n"
                "  torchrun --standalone --nproc_per_node=2 "
                "tools/check_ddp_gate.py\n"
                "or pass --no-distributed for the single-process check."
            )
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        is_main = dist.get_rank() == 0
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )
        if is_main:
            print(f"backend {backend}, world {dist.get_world_size()}, device {device}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"single process (no DDP), device {device}")

    failures = 0
    try:
        for name, gate_settings, loss_mode in CASES:
            try:
                result = run_case(
                    name, gate_settings, loss_mode,
                    device=device, distributed=distributed,
                    local_rank=local_rank, iterations=args.iterations,
                )
                if name.startswith("mixed") and len(set(result["delivered_per_step"])) < 2:
                    failures += 1
                    if is_main:
                        print(
                            f"  FAIL  {name:20s} delivered the same count every "
                            f"iteration ({result['delivered_per_step']}), so "
                            "static_graph was never actually challenged - "
                            "retune REFUSE_SOME"
                        )
                    if distributed:
                        dist.barrier()
                    continue
                reached, total = result["frontend_reached"].split("/")
                if reached != total:
                    failures += 1
                    if is_main:
                        print(
                            f"  FAIL  {name:20s} only {result['frontend_reached']} "
                            "frontend parameters were reached; static_graph "
                            "needs the SAME set every iteration"
                        )
                elif is_main:
                    print(
                        f"  PASS  {name:20s} loss={result['loss']:9.4f}  "
                        f"delivered={result['delivered_per_step']}  "
                        f"frontend_reached={result['frontend_reached']}  "
                        f"moved={result['frontend_moved']}  "
                        f"var_grad={result['variance_head_has_gradient']}"
                    )
            except Exception as error:  # noqa: BLE001
                failures += 1
                if is_main:
                    print(f"  FAIL  {name:20s} {type(error).__name__}: {error}")
            if distributed:
                dist.barrier()
    finally:
        if distributed:
            dist.destroy_process_group()

    if is_main:
        print(f"\n{len(CASES) - failures}/{len(CASES)} cases passed")
        if failures:
            print("A failure here means a long multi-GPU run would break the "
                  "same way, usually partway through an epoch rather than at "
                  "startup.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
