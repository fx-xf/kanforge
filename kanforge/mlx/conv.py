"""KAN convolutions for MLX (channels-last, like ``mlx.nn.ConvNd``).

Same math and patch-feature order (C_in/groups, k_1, ..., k_n) as
``kanforge.torch.conv``; inputs are (N, *spatial, C).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import mlx.core as mx
from mlx import nn

from .layers import KANLinear


def _tuple(v, n: int) -> tuple[int, ...]:
    return tuple(v) if isinstance(v, Sequence) else (int(v),) * n


class _KANConvNd(nn.Module):
    n_dims = 0

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        basis="bspline",
        **kan_kwargs,
    ):
        super().__init__()
        n = self.n_dims
        if in_channels % groups or out_channels % groups:
            raise ValueError("in_channels and out_channels must be divisible by groups")
        self.in_channels, self.out_channels, self.groups = in_channels, out_channels, groups
        self.kernel_size = _tuple(kernel_size, n)
        self.stride = _tuple(stride, n)
        self.padding = _tuple(padding, n)
        self.dilation = _tuple(dilation, n)
        fan_in = in_channels // groups * math.prod(self.kernel_size)
        self.kans = [KANLinear(fan_in, out_channels // groups, basis, **kan_kwargs) for _ in range(groups)]

    def _patches(self, x: mx.array) -> mx.array:
        """(B, *L, C) -> (B, *L_out, C, prod(k)), kernel offsets in row-major order."""
        n = self.n_dims
        if any(self.padding):
            x = mx.pad(x, [(0, 0)] + [(p, p) for p in self.padding] + [(0, 0)])
        size = x.shape[1 : 1 + n]
        out = [(size[d] - self.dilation[d] * (self.kernel_size[d] - 1) - 1) // self.stride[d] + 1 for d in range(n)]
        slices = []
        for offs in itertools.product(*(range(k) for k in self.kernel_size)):
            idx = [slice(None)]
            for d, o in enumerate(offs):
                start = o * self.dilation[d]
                idx.append(slice(start, start + self.stride[d] * (out[d] - 1) + 1, self.stride[d]))
            slices.append(x[tuple(idx)])
        return mx.stack(slices, -1)

    def __call__(self, x: mx.array) -> mx.array:
        p = self._patches(x)  # (B, *L, C, K)
        cg = self.in_channels // self.groups
        outs = []
        for g, kan in enumerate(self.kans):
            pg = p[..., g * cg : (g + 1) * cg, :]
            outs.append(kan(pg.reshape(*pg.shape[:-2], -1)))
        return mx.concatenate(outs, -1)


class KANConv1d(_KANConvNd):
    n_dims = 1


class KANConv2d(_KANConvNd):
    n_dims = 2


class KANConv3d(_KANConvNd):
    n_dims = 3
