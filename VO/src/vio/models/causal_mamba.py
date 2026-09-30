"""Portable causal Mamba-style selective state-space encoder.

This file was called ``imu_mamba.py`` and holds no IMU code: RMSNorm, the
selective state block, and the encoder built from them are the temporal
backbone of the visual model, and this project has no rate or acceleration
sensor at all. The name was a leftover from a retired pipeline and it nearly
cost the encoder its life in an IMU cleanup, which is why it is gone.

The repository does not currently depend on ``mamba_ssm``. This module provides
an explicit reference recurrence with constant-size streaming state so causal
behavior can be implemented and tested now. It is intentionally identified as
the ``portable_reference`` backend; a fused deployment kernel must reproduce
its step/chunk behavior before replacing it.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

BlockState = Tuple[torch.Tensor, torch.Tensor]
MambaStackState = Tuple[BlockState, ...]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return (x * scale.to(x.dtype)) * self.weight.to(x.dtype)


class CausalSelectiveStateBlock(nn.Module):
    """A compact input-selective SSM block with an exact recurrent step."""

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(d_model, d_state, expand, d_conv) <= 0:
            raise ValueError("Mamba dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(d_model * expand)
        self.d_conv = int(d_conv)

        self.norm = RMSNorm(self.d_model)
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner)
        self.conv = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            bias=True,
        )
        # Per-token delta is channel-specific; B and C are shared across
        # channels, matching the grouped selective-scan formulation.
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
        # Stable initial time constants in approximately [1e-3, 1e-1].
        dt = torch.exp(
            torch.empty(self.d_inner).uniform_(math.log(1e-3), math.log(1e-1))
        )
        inverse_softplus = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_bias.copy_(inverse_softplus)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BlockState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        conv_state = torch.zeros(
            batch_size,
            self.d_inner,
            max(self.d_conv - 1, 0),
            device=device,
            dtype=dtype,
        )
        ssm_state = torch.zeros(
            batch_size,
            self.d_inner,
            self.d_state,
            device=device,
            dtype=dtype,
        )
        return conv_state, ssm_state

    def step(
        self, x: torch.Tensor, state: BlockState
    ) -> Tuple[torch.Tensor, BlockState]:
        if x.ndim != 2 or x.shape[1] != self.d_model:
            raise ValueError(f"x must have shape (B, {self.d_model})")
        conv_state, ssm_state = state
        expected_conv = (x.shape[0], self.d_inner, max(self.d_conv - 1, 0))
        expected_ssm = (x.shape[0], self.d_inner, self.d_state)
        if (
            tuple(conv_state.shape) != expected_conv
            or tuple(ssm_state.shape) != expected_ssm
        ):
            raise ValueError(
                "Streaming state shape does not match the input batch/model"
            )

        residual = x
        projected, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        window = torch.cat((conv_state, projected.unsqueeze(-1)), dim=-1)
        weight = self.conv.weight[:, 0, :].to(window.dtype)
        content = torch.sum(window * weight.unsqueeze(0), dim=-1)
        if self.conv.bias is not None:
            content = content + self.conv.bias.to(content.dtype)
        content = F.silu(content)
        next_conv = window[:, :, 1:] if self.d_conv > 1 else window[:, :, :0]

        parameters = self.parameter_proj(content)
        dt_raw, input_B, output_C = torch.split(
            parameters, (self.d_inner, self.d_state, self.d_state), dim=-1
        )
        delta = F.softplus(dt_raw + self.dt_bias.to(dt_raw.dtype))
        A = -torch.exp(self.A_log.float()).to(content.dtype)
        transition = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0))
        drive = delta.unsqueeze(-1) * input_B.unsqueeze(1) * content.unsqueeze(-1)
        next_ssm = transition * ssm_state + drive
        scanned = torch.sum(next_ssm * output_C.unsqueeze(1), dim=-1)
        scanned = scanned + self.D.to(content.dtype) * content
        scanned = scanned * F.silu(gate)
        output = residual + self.dropout(self.out_proj(scanned))
        return output, (next_conv, next_ssm)

    def forward_sequence(
        self,
        x: torch.Tensor,
        state: Optional[BlockState] = None,
    ) -> Tuple[torch.Tensor, BlockState]:
        if x.ndim != 3 or x.shape[2] != self.d_model:
            raise ValueError(f"x must have shape (B, L, {self.d_model})")
        if state is None:
            state = self.initial_state(x.shape[0], device=x.device, dtype=x.dtype)
        outputs = []
        for index in range(x.shape[1]):
            output, state = self.step(x[:, index], state)
            outputs.append(output)
        return torch.stack(outputs, dim=1), state


class CausalMambaEncoder(nn.Module):
    """Stack causal selective-state blocks for chunked and step inference."""

    backend_name = "portable_reference"

    def __init__(
        self,
        input_dim: int,
        *,
        d_model: int = 64,
        depth: int = 2,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(input_dim, d_model, depth) <= 0:
            raise ValueError("input_dim, d_model, and depth must be positive")
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.input_proj = nn.Linear(self.input_dim, self.d_model)
        self.blocks = nn.ModuleList(
            [
                CausalSelectiveStateBlock(
                    self.d_model,
                    d_state=d_state,
                    expand=expand,
                    d_conv=d_conv,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = RMSNorm(self.d_model)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> MambaStackState:
        return tuple(
            block.initial_state(batch_size, device=device, dtype=dtype)
            for block in self.blocks
        )

    @staticmethod
    def detach_state(state: MambaStackState) -> MambaStackState:
        return tuple((conv.detach(), ssm.detach()) for conv, ssm in state)

    def step(
        self,
        x: torch.Tensor,
        state: Optional[MambaStackState] = None,
    ) -> Tuple[torch.Tensor, MambaStackState]:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"x must have shape (B, {self.input_dim})")
        hidden = self.input_proj(x)
        if state is None:
            state = self.initial_state(
                hidden.shape[0], device=hidden.device, dtype=hidden.dtype
            )
        if len(state) != len(self.blocks):
            raise ValueError("State depth does not match encoder depth")
        next_state = []
        for block, block_state in zip(self.blocks, state):
            hidden, updated = block.step(hidden, block_state)
            next_state.append(updated)
        return self.output_norm(hidden), tuple(next_state)

    def forward_sequence(
        self,
        x: torch.Tensor,
        state: Optional[MambaStackState] = None,
    ) -> Tuple[torch.Tensor, MambaStackState]:
        if x.ndim != 3 or x.shape[2] != self.input_dim:
            raise ValueError(f"x must have shape (B, L, {self.input_dim})")
        if state is None:
            projected = self.input_proj(x)
            state = self.initial_state(
                x.shape[0], device=x.device, dtype=projected.dtype
            )
            # Avoid projecting each input token again in ``step``.
            hidden = projected
            next_states = []
            for block, block_state in zip(self.blocks, state):
                hidden, block_state = block.forward_sequence(hidden, block_state)
                next_states.append(block_state)
            return self.output_norm(hidden), tuple(next_states)

        outputs = []
        for index in range(x.shape[1]):
            output, state = self.step(x[:, index], state)
            outputs.append(output)
        return torch.stack(outputs, dim=1), state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_sequence(x)
        return output


__all__ = [
    "BlockState",
    "CausalMambaEncoder",
    "CausalSelectiveStateBlock",
    "MambaStackState",
    "RMSNorm",
]
