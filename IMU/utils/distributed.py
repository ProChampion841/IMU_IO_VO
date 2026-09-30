"""Multi-GPU helpers: device-list parsing and DistributedDataParallel plumbing.

Why DDP and not DataParallel
----------------------------
The user's interface is ``--device 0,1,2,3,4,5,6,7,8,9``, which looks like a
``nn.DataParallel`` request.  It is not implemented that way.  DataParallel is a
single process holding the GIL that scatters and gathers every batch, it loads
GPU0 far harder than the rest, and it would have to gather this model's dict
output -- which contains pypose ``SO3`` objects and a 9x9 covariance -- back to
one device on every step.  At ten GPUs that is both slow and fragile.

So the *interface* stays as asked while the *engine* is one process per GPU with
``DistributedDataParallel``: each rank owns a device, holds a full copy of the
~930k-parameter model, and only gradients cross the interconnect.

Backend
-------
NCCL on Linux with CUDA (the A100 box), gloo otherwise.  This matters because the
development machine is Windows, where ``dist.is_nccl_available()`` is False, so
gloo is the only way to exercise the machinery here at all.

What the caller still has to get right
--------------------------------------
``reduce_metrics`` exists because a rank that logs only its own shard reports
1/world_size of the data while looking entirely normal -- the numbers silently
stop being comparable with every single-GPU baseline in this project.  And
``sampler.set_epoch(epoch)`` must be called every epoch or the shuffle repeats
identically forever, which is invisible in the loss curve.
"""

import os
import platform

import torch
import torch.distributed as dist


def parse_devices(spec):
    """'0,1,2' | '0' | 'cuda:0' | 'cuda' | 'cpu' -> (kind, [ordinals]).

    Returns ``("cpu", [])`` or ``("cuda", [0, 1, ...])``.  The bare-ordinal forms
    are the new ones: ``Tensor.to()`` rejects the string "0", which is exactly the
    error a user hits when they follow the documented --device 0,1,... syntax
    against an unpatched train.py.
    """
    s = str(spec).strip()
    if s.lower() in ("cpu", "none", ""):
        return "cpu", []

    parts = [p.strip() for p in s.split(",") if p.strip()]
    ordinals = []
    for p in parts:
        low = p.lower()
        if low.startswith("cuda:"):
            ordinals.append(int(low.split(":", 1)[1]))
        elif low == "cuda":
            ordinals.append(0)
        elif low.lstrip("+-").isdigit():
            ordinals.append(int(low))
        else:
            raise ValueError(
                "cannot parse --device %r. Use 'cpu', 'cuda:0', '0', or '0,1,2,...'" % spec)

    if not ordinals:
        raise ValueError("empty --device %r" % spec)

    n = torch.cuda.device_count()
    bad = [o for o in ordinals if o < 0 or o >= n]
    if bad:
        raise ValueError(
            "--device %r asks for GPU(s) %s but this machine has %d CUDA device(s) "
            "(valid ordinals 0..%d)" % (spec, bad, n, max(n - 1, 0)))
    if len(set(ordinals)) != len(ordinals):
        raise ValueError("--device %r repeats a GPU ordinal" % spec)
    return "cuda", ordinals


def pick_backend():
    """NCCL on Linux+CUDA, gloo otherwise (Windows has no NCCL)."""
    if platform.system() != "Windows" and torch.cuda.is_available() and dist.is_nccl_available():
        return "nccl"
    return "gloo"


def ddp_setup(rank, world_size, local_device, backend=None, port="29517"):
    """Initialise the process group and bind this rank to its GPU."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    backend = backend or pick_backend()
    # set_device BEFORE init_process_group, or NCCL can place buffers on the wrong
    # card and every rank quietly serialises through GPU0.
    if local_device is not None:
        torch.cuda.set_device(local_device)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return backend


def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def is_main():
    """True on rank 0, and always True when running single-process."""
    return get_rank() == 0


def reduce_metrics(d, device):
    """Mean-reduce a {str: float} metric dict across ranks.

    Without this, rank 0 logs its own shard only: with ten ranks metric.csv would
    report a tenth of the data and still look perfectly plausible.
    """
    if not is_dist():
        return d
    keys = sorted(k for k, v in d.items() if isinstance(v, (int, float)))
    if not keys:
        return d
    t = torch.tensor([float(d[k]) for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= get_world_size()
    out = dict(d)
    for k, v in zip(keys, t.tolist()):
        out[k] = v
    return out


def unwrap(model):
    """The bare module, so checkpoints load into every single-GPU tool unchanged.

    Saving the DDP wrapper prefixes every state_dict key with 'module.', which
    breaks inference.py, evaluation/evaluate_state.py, tools/eval_covariance.py
    and tools/eval_horizons.py -- all of which build the plain network.
    """
    return model.module if hasattr(model, "module") else model
