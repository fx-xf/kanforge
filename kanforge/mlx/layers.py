"""MLX KAN layers for Apple silicon.

Same parameters and math as ``kanforge.torch``; custom Metal kernels will
plug in behind these modules later.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import mlx.core as mx
from mlx import nn

from .._calculus import silu_derivative
from .._layer import basis_param_values, coef_std, mult_widths, resolve_groups
from ..bases import Basis, Rational, make_basis


class KANLinear(nn.Module):
    """One KAN layer: y_o = sum_i s_oi * phi_oi(x_i) + sum_i w_oi * act(x_i) + b_o.

    Parameter names and shapes match ``kanforge.torch.KANLinear``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        basis: str | dict | Basis = "bspline",
        *,
        bias: bool = True,
        base_activation: Callable[[mx.array], mx.array] | None = nn.silu,
        groups: int | None = None,
        edge_scale: bool = False,
        coef_init: str = "normal",
        coef_init_std: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._basis = make_basis(basis)
        self._base_activation = base_activation
        self.groups = resolve_groups(in_features, groups)
        std = coef_std(self._basis, in_features, coef_init, coef_init_std)
        self.coef = mx.random.normal((out_features, in_features, self._basis.size)) * std
        if base_activation is not None:
            bound = 1 / math.sqrt(in_features)  # matches torch's kaiming_uniform(a=sqrt(5))
            self.base_weight = mx.random.uniform(-bound, bound, (out_features, in_features))
        if bias:
            self.bias = mx.zeros((out_features,))
        if edge_scale:
            self.edge_scale = mx.ones((out_features, in_features))
        self._basis_param_names = []
        for name, value in basis_param_values(self._basis, self.groups).items():
            setattr(self, f"basis_{name}", mx.array(value, dtype=mx.float32))
            self._basis_param_names.append(name)

    @property
    def basis(self) -> Basis:
        return self._basis

    def basis_params(self) -> dict[str, mx.array]:
        rep = self.in_features // self.groups
        out = {}
        for name in self._basis_param_names:
            p = self[f"basis_{name}"]
            out[name] = p if self.groups in (1, self.in_features) else mx.repeat(p, rep, axis=0)
        return out

    def effective_coef(self) -> mx.array:
        if "edge_scale" in self:
            return self.coef * self.edge_scale[..., None]
        return self.coef

    def __call__(self, x: mx.array) -> mx.array:
        phi = self._basis(x, self.basis_params())
        y = phi.reshape(*phi.shape[:-2], -1) @ self.effective_coef().reshape(self.out_features, -1).T
        if "base_weight" in self:
            y = y + self._base_activation(x) @ self.base_weight.T
        if "bias" in self:
            y = y + self.bias
        return y

    def edge_derivative(self, x: mx.array, order: int = 1) -> mx.array:
        """d^order phi_oi / dx_i^order for every edge, shape ``(..., out, in)``."""
        d = mx.einsum("...in,oin->...oi", self._basis.derivative(x, order, self.basis_params()), self.effective_coef())
        if "base_weight" in self:
            if self._base_activation is not nn.silu:
                raise NotImplementedError("analytic derivatives need base_activation=nn.silu or None")
            d = d + silu_derivative(x, order)[..., None, :] * self.base_weight
        return d

    def _extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, basis={self._basis!r}"


class GRKAN(KANLinear):
    """Group-rational KAN layer of KAT (see ``kanforge.torch.GRKAN``)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        groups: int = 8,
        num_degree: int = 5,
        den_degree: int = 4,
        init: str = "gelu",
        bias: bool = True,
    ):
        super().__init__(
            in_features,
            out_features,
            Rational(num_degree, den_degree, init),
            bias=bias,
            base_activation=None,
            groups=groups,
            coef_init="variance",
        )


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


class MultKANLinear(nn.Module):
    """KAN layer with addition and multiplication nodes (see ``kanforge.torch``)."""

    def __init__(self, in_features: int, n_sum: int, n_mult: int = 0, basis="bspline", *, arity: int = 2, **kwargs):
        super().__init__()
        if arity < 2:
            raise ValueError("arity must be >= 2")
        self.n_sum, self.n_mult, self.arity = n_sum, n_mult, arity
        self.kan = KANLinear(in_features, n_sum + arity * n_mult, basis, **kwargs)

    @property
    def out_features(self) -> int:
        return self.n_sum + self.n_mult

    def __call__(self, x: mx.array) -> mx.array:
        y = self.kan(x)
        if not self.n_mult:
            return y
        tail = y[..., self.n_sum :]
        prod = mx.prod(tail.reshape(*tail.shape[:-1], self.n_mult, self.arity), -1)
        return mx.concatenate([y[..., : self.n_sum], prod], -1)


class MultKAN(nn.Module):
    """``MultKAN([2, (2, 1), 1])`` (see ``kanforge.torch.MultKAN``)."""

    def __init__(self, widths: Sequence, basis: str | dict | Basis = "bspline", *, arity: int = 2, **layer_kwargs):
        super().__init__()
        ws = mult_widths(widths)
        self.widths = list(widths)
        self.layers = [
            MultKANLinear(a_sum + a_mult, b_sum, b_mult, basis, arity=arity, **layer_kwargs)
            for (a_sum, a_mult), (b_sum, b_mult) in zip(ws[:-1], ws[1:])
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x
