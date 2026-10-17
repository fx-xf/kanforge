"""PyTorch KAN layers. Run on any torch device: CUDA, ROCm, MPS, CPU.

This is the plain-ops implementation; fused kernels will plug in behind
the same modules without changing parameters or state_dict layout.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..bases import Basis, make_basis


class KANLinear(nn.Module):
    """One KAN layer: y_o = sum_i phi_oi(x_i) + sum_i w_oi * act(x_i) + b_o.

    Parameters (same names on every backend):
      ``coef``        ``(out, in, basis.size)`` — basis coefficients of every edge
      ``base_weight`` ``(out, in)`` — residual branch, absent if ``base_activation`` is None
      ``bias``        ``(out,)`` — absent if ``bias=False``
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        basis: str | dict | Basis = "bspline",
        *,
        bias: bool = True,
        base_activation: Callable[[torch.Tensor], torch.Tensor] | None = F.silu,
        coef_init_std: float = 0.1,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.basis = make_basis(basis)
        self.base_activation = base_activation
        self.coef_init_std = coef_init_std
        self.coef = nn.Parameter(torch.empty(out_features, in_features, self.basis.size, **factory))
        if base_activation is not None:
            self.base_weight = nn.Parameter(torch.empty(out_features, in_features, **factory))
        else:
            self.register_parameter("base_weight", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.coef, std=self.coef_init_std / math.sqrt(self.in_features))
        if self.base_weight is not None:
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (..., in, n) -> (..., in*n) @ (in*n, out): one GEMM, as in efficient-kan.
        y = self.basis(x).flatten(-2) @ self.coef.flatten(1).T
        if self.base_weight is not None:
            y = y + F.linear(self.base_activation(x), self.base_weight)
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, basis={self.basis!r}"


class KAN(nn.Sequential):
    """Stack of ``KANLinear`` layers, e.g. ``KAN([2, 5, 1], basis="chebyshev")``."""

    def __init__(self, widths: Sequence[int], basis: str | dict | Basis = "bspline", **layer_kwargs):
        if len(widths) < 2:
            raise ValueError("widths needs at least input and output size")
        super().__init__(
            *(KANLinear(a, b, basis, **layer_kwargs) for a, b in zip(widths[:-1], widths[1:]))
        )
        self.widths = list(widths)
