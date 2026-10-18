"""KAN convolutions (channels-first, like ``torch.nn.ConvNd``).

Every output pixel is a ``KANLinear`` of its receptive field: each element of
the kernel is a learnable edge function instead of a weight (as in
torch-conv-kan and Convolutional-KANs). Patch features are ordered like the
weights of ``nn.ConvNd`` — (in_channels/groups, k_1, ..., k_n) — so an
ordinary convolution converts exactly with ``from_conv``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

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
        padding_mode: str = "zeros",
        **kan_kwargs,
    ):
        super().__init__()
        n = self.n_dims
        if in_channels % groups or out_channels % groups:
            raise ValueError("in_channels and out_channels must be divisible by groups")
        if padding_mode not in ("zeros", "reflect", "replicate", "circular"):
            raise ValueError(f"unknown padding_mode {padding_mode!r}")
        self.in_channels, self.out_channels, self.groups = in_channels, out_channels, groups
        self.kernel_size = _tuple(kernel_size, n)
        self.stride = _tuple(stride, n)
        self.padding = _tuple(padding, n)
        self.dilation = _tuple(dilation, n)
        self.padding_mode = padding_mode
        fan_in = in_channels // groups * math.prod(self.kernel_size)
        self.kans = nn.ModuleList(KANLinear(fan_in, out_channels // groups, basis, **kan_kwargs) for _ in range(groups))

    def _patches(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, *L) -> (B, C, *L_out, *k) receptive fields."""
        n = self.n_dims
        if any(self.padding):
            pad = [p for d in reversed(range(n)) for p in (self.padding[d], self.padding[d])]
            mode = "constant" if self.padding_mode == "zeros" else self.padding_mode
            x = F.pad(x, pad, mode=mode)
        for d in range(n):
            span = self.dilation[d] * (self.kernel_size[d] - 1) + 1
            x = x.unfold(2 + d, span, self.stride[d])
        if any(dl > 1 for dl in self.dilation):
            idx = (slice(None),) * (2 + n) + tuple(slice(None, None, dl) for dl in self.dilation)
            x = x[idx]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.n_dims
        p = self._patches(x)  # (B, C, *L, *k)
        b, spatial = p.shape[0], p.shape[2 : 2 + n]
        cg = self.in_channels // self.groups
        outs = []
        for g, kan in enumerate(self.kans):
            pg = p[:, g * cg : (g + 1) * cg]
            # (B, C, *L, *k) -> (B, *L, C, *k) -> (B, *L, C*prod(k))
            perm = (0, *range(2, 2 + n), 1, *range(2 + n, 2 + 2 * n))
            feats = pg.permute(perm).reshape(b, *spatial, -1)
            outs.append(kan(feats))
        y = torch.cat(outs, -1)  # (B, *L, C_out)
        return y.permute(0, -1, *range(1, 1 + n)).contiguous()

    @classmethod
    def from_conv(cls, conv: nn.Module, activation=None, basis="bspline", **kan_kwargs):
        """Exactly ``conv``: its weights become the base branch (identity
        activation unless given), spline coefficients start at zero."""
        if not isinstance(conv.padding, tuple) or isinstance(conv.padding, str):
            raise ValueError("only numeric padding converts")
        layer = cls(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            basis,
            conv.padding_mode,
            base_activation=activation if activation is not None else nn.Identity(),
            bias=conv.bias is not None,
            coef_init="zero",
            device=conv.weight.device,
            dtype=conv.weight.dtype,
            **kan_kwargs,
        )
        og = conv.out_channels // conv.groups
        with torch.no_grad():
            for g, kan in enumerate(layer.kans):
                kan.base_weight.copy_(conv.weight[g * og : (g + 1) * og].reshape(og, -1))
                if conv.bias is not None:
                    kan.bias.copy_(conv.bias[g * og : (g + 1) * og])
        return layer

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, groups={self.groups}"
        )


class KANConv1d(_KANConvNd):
    n_dims = 1


class KANConv2d(_KANConvNd):
    n_dims = 2


class KANConv3d(_KANConvNd):
    n_dims = 3
