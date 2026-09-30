"""A 2-D selective state-space image encoder.

The IMU stack's :class:`CausalSelectiveStateBlock` resolves its recurrence with
a Python loop over timesteps. That is the right trade for 400 IMU ticks with an
exact streaming step, and it is unusable for an image: a 24x32 patch grid is
768 steps per scan direction, four directions per block, several blocks per
frame, two frames per pair.

So this module keeps the same selective-state mathematics and changes how the
recurrence is resolved:

* :func:`parallel_selective_scan` computes ``h_t = a_t h_{t-1} + b_t`` for all
  ``t`` at once by Hillis-Steele, in ``ceil(log2 L)`` tensor operations rather
  than ``L`` Python iterations. For a 768-step scan that is 10 rounds instead
  of 768, and it is what makes a scanned image encoder trainable at all.

* :class:`SelectiveScan2D` scans the patch grid in four directions - forward
  and backward along rows, forward and backward along columns - and sums the
  results. A single raster scan makes the top-left corner privileged: a cell
  can only be informed by cells before it in reading order. Summing the four
  restores an isotropic receptive field, which is the whole point of a 2-D
  state-space block.

One structural decision is worth stating because it is not obvious and it is
load-bearing. Correlation - the operation that actually measures motion - needs
features that TRANSLATE WITH THE IMAGE. A sequential scan does not: encoding a
shifted image is not the same as shifting the encoding, because the scan
accumulates state in a fixed order.

So the scan does not replace the convolutional stem, it modulates it. Patch
features stay convolutional and translation-equivariant; the scan runs on a
coarse grid, provides global context, and is added back as a gated residual.
The encoder is genuinely a vision Mamba - a 2-D selective scan with a global
receptive field does the representation work - without breaking the
equivariance the correspondence step depends on.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .causal_mamba import RMSNorm


def parallel_selective_scan(
    transition: torch.Tensor, drive: torch.Tensor
) -> torch.Tensor:
    """Resolve ``h_t = transition_t * h_{t-1} + drive_t`` along dim 1.

    Both tensors are ``(B, L, ...)`` and the recurrence is elementwise over the
    trailing dimensions, which is what makes the diagonal state matrix worth
    having: every channel and state component is an independent scalar
    recurrence, so the whole thing is one associative scan.

    The operator ``(a1, b1) . (a2, b2) = (a1 a2, a2 b1 + b2)`` is associative,
    so an inclusive scan gives every prefix at once. Hillis-Steele doubles the
    reach each round: after round ``k`` every position holds the exact result
    for a history of ``2^k``, and ``ceil(log2 L)`` rounds cover the sequence.

    Out-of-range positions take the identity element - drive zero, transition
    one - so the leading ``2^k`` entries are already final and are carried
    forward unchanged.
    """

    if transition.shape != drive.shape:
        raise ValueError("transition and drive must have the same shape")
    if transition.ndim < 2:
        raise ValueError("transition and drive must be at least (B, L)")
    length = transition.shape[1]
    if length == 0:
        return drive
    state = drive
    factor = transition
    step = 1
    while step < length:
        pad_state = torch.zeros_like(state[:, :step])
        pad_factor = torch.ones_like(factor[:, :step])
        shifted_state = torch.cat((pad_state, state[:, :-step]), dim=1)
        shifted_factor = torch.cat((pad_factor, factor[:, :-step]), dim=1)
        state = state + factor * shifted_state
        factor = factor * shifted_factor
        step *= 2
    return state


#: Row-major forward, row-major backward, column-major forward, column-major
#: backward. Four is the smallest set that makes every cell reachable from
#: every other cell within one block.
SCAN_DIRECTIONS = ("rows", "rows_reverse", "columns", "columns_reverse")


def _to_sequence(features: torch.Tensor, direction: str) -> torch.Tensor:
    """``(B, C, H, W)`` to ``(B, L, C)`` in one of the four scan orders."""

    if direction in {"columns", "columns_reverse"}:
        features = features.transpose(-2, -1)
    sequence = features.flatten(2).transpose(1, 2)
    if direction.endswith("_reverse"):
        sequence = sequence.flip(1)
    return sequence


def _from_sequence(
    sequence: torch.Tensor, direction: str, size: Tuple[int, int]
) -> torch.Tensor:
    """Inverse of :func:`_to_sequence`."""

    height, width = size
    if direction.endswith("_reverse"):
        sequence = sequence.flip(1)
    rows, columns = (
        (width, height)
        if direction in {"columns", "columns_reverse"}
        else (height, width)
    )
    features = sequence.transpose(1, 2).reshape(sequence.shape[0], -1, rows, columns)
    if direction in {"columns", "columns_reverse"}:
        features = features.transpose(-2, -1)
    return features


class SelectiveScan2D(nn.Module):
    """One 2-D selective state-space block over a patch grid.

    Same selective mechanism as the IMU block - per-channel ``delta`` with
    ``B`` and ``C`` shared across channels, diagonal ``A``, a ``D`` skip and a
    SiLU gate - applied four ways over the grid and summed.

    The projections are shared across the four directions rather than being
    replicated per direction as in the original SS2D. That is a deliberate
    trade for this dataset: replication quadruples the selective parameters for
    a model that has under two hours of flight to fit on, and the four scans
    already see genuinely different sequences because the ORDER differs. The
    directional asymmetry lives in the scan, not in the weights.
    """

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 8,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(d_model, d_state, expand) <= 0:
            raise ValueError("SelectiveScan2D dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(d_model * expand)

        self.norm = RMSNorm(self.d_model)
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner)
        # Depthwise 3x3 rather than the 1-D causal conv: the neighbourhood a
        # patch needs is its spatial neighbourhood, and there is no causality
        # to respect across an image.
        self.conv = nn.Conv2d(
            self.d_inner, self.d_inner, 3, padding=1, groups=self.d_inner, bias=True
        )
        self.parameter_proj = nn.Linear(
            self.d_inner, self.d_inner + 2 * self.d_state, bias=False
        )
        self.dt_bias = nn.Parameter(torch.empty(self.d_inner))
        base = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(base.log().repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Matches the IMU block: initial time constants spread over
        # [1e-3, 1e-1] so some channels start near passthrough and others
        # integrate across many patches.
        dt = torch.exp(
            torch.empty(self.d_inner).uniform_(math.log(1e-3), math.log(1e-1))
        )
        inverse_softplus = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_bias.copy_(inverse_softplus)

    def _scan(self, content: torch.Tensor, direction: str) -> torch.Tensor:
        size = (content.shape[2], content.shape[3])
        sequence = _to_sequence(content, direction)
        parameters = self.parameter_proj(sequence)
        dt_raw, input_B, output_C = torch.split(
            parameters, (self.d_inner, self.d_state, self.d_state), dim=-1
        )
        delta = F.softplus(dt_raw + self.dt_bias.to(dt_raw.dtype))
        A = -torch.exp(self.A_log.float()).to(sequence.dtype)
        transition = torch.exp(delta.unsqueeze(-1) * A)
        drive = delta.unsqueeze(-1) * input_B.unsqueeze(2) * sequence.unsqueeze(-1)
        state = parallel_selective_scan(transition, drive)
        scanned = torch.sum(state * output_C.unsqueeze(2), dim=-1)
        return _from_sequence(scanned, direction, size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != self.d_model:
            raise ValueError(f"features must have shape (B, {self.d_model}, H, W)")
        residual = features
        normalized = self.norm(features.permute(0, 2, 3, 1))
        projected, gate = (
            self.in_proj(normalized).permute(0, 3, 1, 2).chunk(2, dim=1)
        )
        content = F.silu(self.conv(projected))

        scanned = sum(self._scan(content, name) for name in SCAN_DIRECTIONS)
        scanned = scanned + self.D.reshape(1, -1, 1, 1).to(content.dtype) * content
        scanned = scanned * F.silu(gate)
        output = self.out_proj(scanned.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return residual + self.dropout(output)


class VisionMambaStem(nn.Module):
    """Patch features carrying globally scanned context.

    Two resolutions, for the reason in the module docstring:

    * ``patch_size`` sets the FINE grid the correlation runs on. It has to be
      fine enough to see the motion and it has to be translation-equivariant,
      so it is produced by a strided convolution alone.
    * ``context_grid`` sets the COARSE grid the scan runs on. Global context
      does not need patch resolution, and the scan cost grows with it - a
      12x16 context grid is 192 steps, a 36x48 one is 1728.

    The scanned context is upsampled and added as a gated residual, so features
    leaving this module still translate with the image while carrying
    information from the whole frame.
    """

    def __init__(
        self,
        *,
        input_channels: int = 1,
        d_model: int = 64,
        depth: int = 2,
        patch_size: int = 8,
        image_size: Tuple[int, int] = (288, 384),
        context_grid: Tuple[int, int] = (12, 16),
        d_state: int = 8,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(input_channels, d_model, depth, patch_size) <= 0:
            raise ValueError("VisionMambaStem dimensions must be positive")
        self.input_channels = int(input_channels)
        self.d_model = int(d_model)
        self.patch_size = int(patch_size)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.context_grid = (int(context_grid[0]), int(context_grid[1]))
        if min(self.context_grid) <= 0:
            raise ValueError("context_grid must be positive")
        self.feature_size = (
            self.image_size[0] // self.patch_size,
            self.image_size[1] // self.patch_size,
        )
        if min(self.feature_size) <= 0:
            raise ValueError("image_size must be at least patch_size in both axes")

        self.patch_embed = nn.Conv2d(
            self.input_channels,
            self.d_model,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        self.feature_norm = nn.GroupNorm(1, self.d_model)
        self.blocks = nn.ModuleList(
            [
                SelectiveScan2D(
                    self.d_model, d_state=d_state, expand=expand, dropout=dropout
                )
                for _ in range(depth)
            ]
        )
        self.context_norm = nn.GroupNorm(1, self.d_model)
        # Zero-initialised, so the stem starts as the plain convolutional
        # encoder and the scan has to earn its contribution. Without this an
        # untrained scan injects noise into the very features correspondence is
        # measured from, and early training is spent undoing it.
        self.context_gate = nn.Parameter(torch.zeros(1, self.d_model, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != self.input_channels:
            raise ValueError(f"image must have shape (B, {self.input_channels}, H, W)")
        if not torch.is_floating_point(image):
            raise ValueError("image must be a floating-point tensor")
        resized = F.interpolate(
            image, size=self.image_size, mode="bilinear", align_corners=False
        )
        features = self.feature_norm(self.patch_embed(resized))

        context = F.adaptive_avg_pool2d(features, self.context_grid)
        for block in self.blocks:
            context = block(context)
        context = self.context_norm(context)
        context = F.interpolate(
            context, size=features.shape[-2:], mode="bilinear", align_corners=False
        )
        return features + self.context_gate * context


__all__ = [
    "SCAN_DIRECTIONS",
    "SelectiveScan2D",
    "VisionMambaStem",
    "parallel_selective_scan",
]
