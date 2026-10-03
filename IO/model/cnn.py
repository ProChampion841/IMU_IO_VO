import math

import torch
import torch.nn as nn


def _norm1d(kind, ch):
    """Per-layer normalisation for the tokeniser.

    "batch" (default) is the original BatchNorm1d -- same parameter and buffer
    names, so every existing checkpoint still loads under strict=True.

    "group" is GroupNorm: statistics per SAMPLE, not per batch.  At batch_size 4
    BatchNorm's running mean/var are estimated from 4 windows of 1-2 flights at a
    time and then frozen for eval, where every validation flight has its own sensor
    offsets -- a train/eval mismatch GroupNorm does not have.  Up to 8 groups,
    fewer when the channel count does not divide.
    BUT GroupNorm's statistics span the WHOLE window (all time steps), so every token
    depends on frames up to the end of the window, and on the window length: a model
    trained on 60 s behaves differently on the 30 s .. 38 min evaluation windows.

    "layer" normalises each TIME STEP over its channels only.  No statistics across
    time or batch: causal, independent of window length, identical in train and eval.
    """
    if kind == "batch":
        return nn.BatchNorm1d(ch)
    if kind == "group":
        return nn.GroupNorm(math.gcd(8, ch), ch)
    if kind == "layer":
        return _TokenLayerNorm(ch)
    raise ValueError("cnn_norm must be batch|group|layer, got %r" % (kind,))


class _TokenLayerNorm(nn.LayerNorm):
    """LayerNorm over the channels of each time step of a (B, C, L) conv output."""

    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class CNNEncoder(nn.Module):
    def __init__(self, duration = 1, k_list = [7, 7, 7, 7], c_list = [6, 16, 32, 64, 128],
                        s_list = [1, 1, 1, 1], p_list = [3, 3, 3, 3], norm = "batch"):
        super(CNNEncoder, self).__init__()
        self.duration = duration
        self.k_list, self.c_list, self.s_list, self.p_list = k_list, c_list, s_list, p_list
        layers = []

        for i in range(len(self.c_list) - 1):
            layers.append(torch.nn.Conv1d(self.c_list[i], self.c_list[i+1], self.k_list[i], \
                stride=self.s_list[i], padding=self.p_list[i]))
            layers.append(_norm1d(norm, self.c_list[i+1]))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(0.1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
