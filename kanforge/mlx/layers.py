"""MLX KAN layers for Apple silicon.

Same parameters and math as ``kanforge.torch``; custom Metal kernels will
plug in behind these modules later. Buffers (grids) are frozen arrays.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import mlx.core as mx
import numpy as np
from mlx import nn

from .. import regularization as reg
from .._calculus import silu_derivative
from .._layer import GridToolsMixin, basis_param_values, init_stds, mult_widths, resolve_groups
from ..bases import Basis, Rational, make_basis
from ..kernels import plan_for
from ..kernels.plan import AUTO_KINDS
from . import kernels as _kernels

BACKENDS = ("auto", "ops", "kernel")


class KANLinear(GridToolsMixin, nn.Module):
    """One KAN layer: y_o = sum_i s_oi * phi_oi(x_i) + sum_i w_oi * act(x_i) + b_o.

    Parameter names and shapes match ``kanforge.torch.KANLinear``.
    ``backend`` as in torch: ``"kernel"`` runs the fused Metal kernel on the GPU.
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
        grid_ranges=None,
        coef_init: str = "normal",
        coef_init_std: float = 0.1,
        edge_dropout: float = 0.0,
        backend: str = "auto",
    ):
        super().__init__()
        if not 0 <= edge_dropout < 1:
            raise ValueError("edge_dropout must be in [0, 1)")
        self.in_features = in_features
        self.out_features = out_features
        self._basis = make_basis(basis)
        self._base_activation = base_activation
        self.groups = resolve_groups(in_features, groups)
        self.edge_dropout = edge_dropout
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        self.backend = backend
        coef_std, base_std = init_stds(
            self._basis, in_features, out_features, coef_init, coef_init_std, base_activation is not None
        )
        self.coef = mx.random.normal((out_features, in_features, self._basis.size)) * coef_std
        if base_activation is not None:
            if base_std is None:
                bound = 1 / math.sqrt(in_features)  # matches torch's kaiming_uniform(a=sqrt(5))
                self.base_weight = mx.random.uniform(-bound, bound, (out_features, in_features))
            else:
                self.base_weight = mx.random.normal((out_features, in_features)) * base_std
        if bias:
            self.bias = mx.zeros((out_features,))
        if edge_scale:
            self.edge_scale = mx.ones((out_features, in_features))
        self._trainable_names: list[str] = []
        self._buffer_names: list[str] = []
        self._set_basis_state(*basis_param_values(self._basis, self.groups, in_features, grid_ranges))

    def _set_basis_state(self, trainable: dict, buffers: dict) -> None:
        for name in self._trainable_names + self._buffer_names:
            self.pop(f"basis_{name}", None)
        for name, value in {**trainable, **buffers}.items():
            self[f"basis_{name}"] = mx.array(np.asarray(value, dtype=np.float32))
        self._trainable_names, self._buffer_names = list(trainable), list(buffers)
        self._plan = plan_for(self._basis)
        if buffers:
            self.freeze(recurse=False, keys=[f"basis_{n}" for n in buffers])

    @property
    def basis(self) -> Basis:
        return self._basis

    def basis_params(self) -> dict[str, mx.array]:
        rep = self.in_features // self.groups
        out = {}
        for name in self._trainable_names:
            p = self[f"basis_{name}"]
            out[name] = p if self.groups in (1, self.in_features) else mx.repeat(p, rep, axis=0)
        for name in self._buffer_names:
            out[name] = self[f"basis_{name}"]
        return out

    def effective_coef(self) -> mx.array:
        if "edge_scale" in self:
            return self.coef * self.edge_scale[..., None]
        return self.coef

    def __call__(self, x: mx.array) -> mx.array:
        if self.training and self.edge_dropout > 0:
            edges = self.edge_derivative(x, 0)
            keep = mx.random.uniform(shape=edges.shape) >= self.edge_dropout
            y = (edges * keep).sum(-1) / (1 - self.edge_dropout)
            return y + self.bias if "bias" in self else y
        if self.uses_kernel():
            grid = self["basis_grid"] if self._plan.grid else None
            y = _kernels.spline_forward(x, self.effective_coef(), grid, self._plan)
        else:
            phi = self._basis(x, self.basis_params())
            y = phi.reshape(*phi.shape[:-2], -1) @ self.effective_coef().reshape(self.out_features, -1).T
        if "base_weight" in self:
            y = y + self._base_activation(x) @ self.base_weight.T
        if "bias" in self:
            y = y + self.bias
        return y

    def uses_kernel(self) -> bool:
        """Whether forward runs the fused Metal kernel (default device is the GPU)."""
        if self.backend == "ops":
            return False
        ok = self._plan is not None and mx.metal.is_available() and mx.default_device() == mx.gpu
        if self.backend == "kernel":
            if not ok:
                why = "basis" if self._plan is None else "device (needs the Metal GPU)"
                raise RuntimeError(f"no fused kernel for this {why}: {self._basis!r}; use backend='ops' or 'auto'")
            return True
        return ok and self._plan.kind in AUTO_KINDS["mlx"]

    def edge_derivative(self, x: mx.array, order: int = 1) -> mx.array:
        """d^order phi_oi / dx_i^order for every edge, shape ``(..., out, in)``."""
        d = mx.einsum("...in,oin->...oi", self._basis.derivative(x, order, self.basis_params()), self.effective_coef())
        if "base_weight" in self:
            if order == 0:
                base = self._base_activation(x)
            elif self._base_activation is nn.silu:
                base = silu_derivative(x, order)
            else:
                raise NotImplementedError("analytic derivatives need base_activation=nn.silu or None")
            d = d + base[..., None, :] * self.base_weight
        return d

    # --- grid tools (GridToolsMixin) --------------------------------------

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        return np.array(x, dtype=np.float64)

    def _export_state(self) -> dict:
        return {
            "coef": self._to_numpy(self.coef),
            "edge_scale": self._to_numpy(self.edge_scale) if "edge_scale" in self else None,
            "trainable": {k: self._to_numpy(self[f"basis_{k}"]) for k in self._trainable_names},
            "buffers": {k: self._to_numpy(self[f"basis_{k}"]) for k in self._buffer_names},
        }

    def _load_state(self, basis, coef, edge_scale, trainable, buffers) -> None:
        self._basis = basis
        self.coef = mx.array(np.asarray(coef, dtype=np.float32))
        if edge_scale is not None:
            self.edge_scale = mx.array(np.asarray(edge_scale, dtype=np.float32))
        self._set_basis_state(trainable, buffers)

    @classmethod
    def from_linear(cls, linear: nn.Linear, activation: Callable = nn.silu, basis="bspline", **kwargs) -> KANLinear:
        """Exactly ``linear(activation(x))`` with zero spline coefficients."""
        out_features, in_features = linear.weight.shape
        layer = cls(
            in_features,
            out_features,
            basis,
            bias="bias" in linear,
            base_activation=activation,
            coef_init="zero",
            **kwargs,
        )
        layer.base_weight = linear.weight
        if "bias" in linear:
            layer.bias = linear.bias
        return layer

    def _extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, basis={self._basis!r}"


def from_mlp(mlp: nn.Sequential, basis="bspline", **kwargs) -> nn.Sequential:
    """MLX version of ``kanforge.torch.from_mlp``."""
    mods = list(mlp.layers)
    if not mods or not isinstance(mods[0], nn.Linear):
        raise ValueError("the MLP must start with nn.Linear")
    out = [mods[0]]
    i = 1
    while i < len(mods):
        if i + 1 < len(mods) and isinstance(mods[i + 1], nn.Linear) and not isinstance(mods[i], nn.Linear):
            out.append(KANLinear.from_linear(mods[i + 1], mods[i], basis, **kwargs))
            i += 2
        else:
            out.append(mods[i])
            i += 1
    return nn.Sequential(*out)


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

    def update_grid(self, x: mx.array, **kwargs) -> None:
        for layer in self.layers:
            layer.update_grid(x, **kwargs)
            x = layer(x)

    def refine_grid(self, grid_size: int, **kwargs) -> None:
        for layer in self.layers:
            layer.refine_grid(grid_size, **kwargs)


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


def layer_inputs(model, x: mx.array) -> list[tuple[KANLinear, mx.array]]:
    """(KANLinear, its input) for KANLinear, MultKANLinear, KAN, MultKAN, nn.Sequential."""
    out = []

    def walk(m, h):
        if isinstance(m, KANLinear):
            out.append((m, h))
            return m(h)
        if isinstance(m, MultKANLinear):
            out.append((m.kan, h))
            return m(h)
        if isinstance(m, (KAN, MultKAN, nn.Sequential)):
            for sub in m.layers:
                h = walk(sub, h)
            return h
        return m(h)

    walk(model, x)
    return out


def regularization(model, x: mx.array | None = None, *, l1=1.0, entropy=2.0, coef_l1=0.0, coef_smooth=0.0):
    """pykan's sparsity penalty (see ``kanforge.torch.regularization``)."""
    if x is None:
        layers = [m for _, m in model.named_modules() if isinstance(m, KANLinear)]
        mags = [mx.abs(m.effective_coef()).mean(-1) for m in layers]
    else:
        pairs = layer_inputs(model, x)
        layers = [m for m, _ in pairs]
        mags = [reg.edge_magnitude(m.edge_derivative(h, 0)) for m, h in pairs]
    total = 0.0
    for m, mag in zip(layers, mags):
        total = total + reg.penalty(mag, m.coef, l1=l1, entropy=entropy, coef_l1=coef_l1, coef_smooth=coef_smooth)
    return total
