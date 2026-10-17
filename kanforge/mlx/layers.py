"""MLX KAN layers for Apple silicon.

Same parameters and math as ``kanforge.torch``; custom Metal kernels will
plug in behind these modules later.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import mlx.core as mx
from mlx import nn

from ..bases import Basis, make_basis


class KANLinear(nn.Module):
    """One KAN layer: y_o = sum_i phi_oi(x_i) + sum_i w_oi * act(x_i) + b_o."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        basis: str | dict | Basis = "bspline",
        *,
        bias: bool = True,
        base_activation: Callable[[mx.array], mx.array] | None = nn.silu,
        coef_init_std: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._basis = make_basis(basis)
        self._base_activation = base_activation
        self.coef = mx.random.normal((out_features, in_features, self._basis.size)) * (
            coef_init_std / math.sqrt(in_features)
        )
        if base_activation is not None:
            bound = 1 / math.sqrt(in_features)  # matches torch's kaiming_uniform(a=sqrt(5))
            self.base_weight = mx.random.uniform(-bound, bound, (out_features, in_features))
        if bias:
            self.bias = mx.zeros((out_features,))

    @property
    def basis(self) -> Basis:
        return self._basis

    def __call__(self, x: mx.array) -> mx.array:
        phi = self._basis(x)
        y = phi.reshape(*phi.shape[:-2], -1) @ self.coef.reshape(self.out_features, -1).T
        if "base_weight" in self:
            y = y + self._base_activation(x) @ self.base_weight.T
        if "bias" in self:
            y = y + self.bias
        return y

    def _extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, basis={self._basis!r}"


class KAN(nn.Module):
    """Stack of ``KANLinear`` layers, e.g. ``KAN([2, 5, 1], basis="chebyshev")``."""

    def __init__(self, widths: Sequence[int], basis: str | dict | Basis = "bspline", **layer_kwargs):
        super().__init__()
        if len(widths) < 2:
            raise ValueError("widths needs at least input and output size")
        self.widths = list(widths)
        self.layers = [KANLinear(a, b, basis, **layer_kwargs) for a, b in zip(widths[:-1], widths[1:])]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x
