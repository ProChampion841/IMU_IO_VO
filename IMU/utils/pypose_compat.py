"""pypose 0.9.5 + recent torch: make `pp.cumprod` work at every length.

pypose's `cumops_` loops `for i in 2**arange(log2(L)+1): torch.arange(i, L)`, i.e.
`i` runs one power of two PAST L.  Older torch returned an empty range for
`arange(i, L)` with i > L; recent torch (2.x, seen on 2.14) raises "upper bound and
lower bound inconsistent with step sign".  Every IMUPreintegrator call whose length
+ 1 is not a power of two then crashes -- the integrator and the 15 s bias freeze.

`apply()` patches `cumops_` ONLY when the running torch shows the problem, with the
loop stopped at i >= L -- exactly what the old empty range did, so results are
unchanged.  On a torch that does not raise it is a no-op.
"""
import math

import torch


def _broken():
    try:
        torch.arange(4, 3)
        return False
    except RuntimeError:
        return True


def apply():
    import pypose.basics.ops as ops
    if getattr(ops, "_airimu_compat", False) or not _broken():
        return False

    def cumops_(input, dim, op):
        L, v = input.shape[dim], input
        for i in torch.pow(2, torch.arange(math.log2(L) + 1, device=v.device,
                                           dtype=torch.int64)):
            if int(i) >= L:
                break
            index = torch.arange(i, L, device=v.device, dtype=torch.int64)
            # argument order copied verbatim: op(earlier, later) -- it is a matrix
            # product, so swapping them would silently change every rotation.
            v.index_copy_(dim, index, op(v.index_select(dim, index - i),
                                         v.index_select(dim, index)))
        return v

    ops.cumops_ = cumops_
    ops._airimu_compat = True
    return True
