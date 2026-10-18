"""JAX KAN layers on Flax NNX. Run on CPU, CUDA/ROCm GPUs and TPUs.

Same parameter names, shapes and math as ``kanforge.torch`` and
``kanforge.mlx``. Trainable arrays are ``nnx.Param``; grids, masks and
symbolic edges are ``Buffer`` variables, which ``nnx.grad`` and optimizers
(``wrt=nnx.Param``) leave alone. Fused Pallas kernels plug in behind
``backend`` (see ``kanforge.jax.kernels``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .. import regularization as reg
from .._calculus import silu_derivative
from .._layer import GridToolsMixin, InterpretMixin, basis_param_values, init_stds, mult_widths, resolve_groups
from ..bases import Basis, Rational, make_basis
from ..kernels import plan_for
from ..kernels.plan import AUTO_KINDS

BACKENDS = ("auto", "ops", "kernel")


class Buffer(nnx.Variable):
    """Non-trainable layer state (grids, edge masks, symbolic edges)."""


class KANLinear(GridToolsMixin, InterpretMixin, nnx.Module):
    """One KAN layer: y_o = sum_i s_oi * phi_oi(x_i) + sum_i w_oi * act(x_i) + b_o.

    ``rngs`` initialises the parameters and drives ``edge_dropout``
    (stream ``dropout``); ``model.eval()`` switches dropout off.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        basis: str | dict | Basis = "bspline",
        *,
        rngs: nnx.Rngs | None = None,
        bias: bool = True,
        base_activation: Callable | None = jax.nn.silu,
        groups: int | None = None,
        edge_scale: bool = False,
        grid_ranges=None,
        coef_init: str = "normal",
        coef_init_std: float = 0.1,
        edge_dropout: float = 0.0,
        backend: str = "auto",
        dtype=jnp.float32,
    ):
        if not 0 <= edge_dropout < 1:
            raise ValueError("edge_dropout must be in [0, 1)")
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        rngs = rngs if rngs is not None else nnx.Rngs(0)
        self.in_features, self.out_features = in_features, out_features
        self._basis = make_basis(basis)
        self.base_activation = base_activation
        self.groups = resolve_groups(in_features, groups)
        self.edge_dropout = edge_dropout
        self.backend = backend
        self.dtype = dtype
        self.deterministic = False
        self.rngs = rngs
        self._init_kwargs = {
            "basis": self._basis,
            "bias": bias,
            "base_activation": base_activation,
            "groups": groups,
            "edge_scale": edge_scale,
            "coef_init": coef_init,
            "coef_init_std": coef_init_std,
            "edge_dropout": edge_dropout,
            "backend": backend,
            "dtype": dtype,
        }
        coef_std, base_std = init_stds(self._basis, in_features, out_features, coef_init, coef_init_std, base_activation is not None)
        self.coef = nnx.Param(jax.random.normal(rngs.params(), (out_features, in_features, self._basis.size), dtype) * coef_std)
        if base_activation is not None:
            if base_std is None:
                bound = 1 / math.sqrt(in_features)  # torch's kaiming_uniform(a=sqrt(5))
                w = jax.random.uniform(rngs.params(), (out_features, in_features), dtype, -bound, bound)
            else:
                w = jax.random.normal(rngs.params(), (out_features, in_features), dtype) * base_std
            self.base_weight = nnx.Param(w)
        else:
            self.base_weight = None
        self.bias = nnx.Param(jnp.zeros((out_features,), dtype)) if bias else None
        self.edge_scale = nnx.Param(jnp.ones((out_features, in_features), dtype)) if edge_scale else None
        self.edge_mask = Buffer(jnp.ones((out_features, in_features), dtype))
        self.sym_code = Buffer(jnp.full((out_features, in_features), -1, jnp.int32))
        self.sym_params = Buffer(jnp.zeros((out_features, in_features, 4), dtype))
        self._sym_codes = ()
        self._trainable_names: tuple = ()
        self._buffer_names: tuple = ()
        self._set_basis_state(*basis_param_values(self._basis, self.groups, in_features, grid_ranges))

    # --- state ------------------------------------------------------------

    def _set_basis_state(self, trainable: dict, buffers: dict) -> None:
        for name in self._trainable_names + self._buffer_names:
            delattr(self, f"basis_{name}")
        for name, value in trainable.items():
            setattr(self, f"basis_{name}", nnx.Param(jnp.asarray(value, self.dtype)))
        for name, value in buffers.items():
            setattr(self, f"basis_{name}", Buffer(jnp.asarray(value, self.dtype)))
        self._trainable_names, self._buffer_names = tuple(trainable), tuple(buffers)
        self._plan = plan_for(self._basis)

    @property
    def basis(self) -> Basis:
        return self._basis

    def basis_params(self) -> dict:
        rep = self.in_features // self.groups
        out = {}
        for name in self._trainable_names:
            p = getattr(self, f"basis_{name}")[...]
            out[name] = p if self.groups in (1, self.in_features) else jnp.repeat(p, rep, axis=0)
        for name in self._buffer_names:
            out[name] = getattr(self, f"basis_{name}")[...]
        return out

    def effective_coef(self):
        scale = self.edge_mask[...] if self.edge_scale is None else self.edge_scale[...] * self.edge_mask[...]
        return self.coef[...] * scale[..., None]

    def effective_base_weight(self):
        return None if self.base_weight is None else self.base_weight[...] * self.edge_mask[...]

    # --- computation ------------------------------------------------------

    def __call__(self, x):
        if not self.deterministic and self.edge_dropout > 0:
            edges = self.edge_derivative(x, 0)
            keep = jax.random.uniform(self.rngs.dropout(), edges.shape) >= self.edge_dropout
            y = (edges * keep).sum(-1) / (1 - self.edge_dropout)
            return y if self.bias is None else y + self.bias[...]
        if self.uses_kernel():
            from . import kernels as _kernels

            grid = self.basis_grid[...] if self._plan.grid else None
            y = _kernels.spline_forward(x, self.effective_coef(), grid, self._plan)
        else:
            phi = self._basis(x, self.basis_params())
            y = phi.reshape(*phi.shape[:-2], -1) @ self.effective_coef().reshape(self.out_features, -1).T
        if self.base_weight is not None:
            y = y + self.base_activation(x) @ self.effective_base_weight().T
        sym = self._symbolic_part(x)
        if sym is not None:
            y = y + sym.sum(-1)
        if self.bias is not None:
            y = y + self.bias[...]
        return y

    def uses_kernel(self) -> bool:
        """Whether forward runs the fused Pallas kernel (on the default JAX backend)."""
        if self.backend == "ops":
            return False
        from . import kernels as _kernels

        ok = self._plan is not None and _kernels.available()
        if self.backend == "kernel":
            if not ok:
                why = "basis" if self._plan is None else f"platform {jax.default_backend()}"
                raise RuntimeError(f"no fused kernel for this {why}: {self._basis!r}; use backend='ops' or 'auto'")
            return True
        return ok and self._plan.kind in AUTO_KINDS.get("jax-" + jax.default_backend(), ())

    def edge_derivative(self, x, order: int = 1):
        """d^order phi_oi / dx_i^order for every edge, shape ``(..., out, in)``."""
        d = jnp.einsum("...in,oin->...oi", self._basis.derivative(x, order, self.basis_params()), self.effective_coef())
        if self.base_weight is not None:
            if order == 0:
                base = self.base_activation(x)
            elif self.base_activation is jax.nn.silu:
                base = silu_derivative(x, order)
            else:
                raise NotImplementedError("analytic derivatives need base_activation=jax.nn.silu or None")
            d = d + base[..., None, :] * self.effective_base_weight()
        if self._sym_codes:
            if order:
                raise NotImplementedError("analytic derivatives of symbolic edges are not available; use jax.grad")
            d = d + self._symbolic_part(x)
        return d

    # --- mixin hooks ----------------------------------------------------------

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        if isinstance(x, nnx.Variable):
            x = x[...]
        return np.array(x, dtype=np.float64)  # a copy: views of JAX arrays are read-only

    def _get(self, name):
        return getattr(self, name)[...]

    def _set(self, name, value) -> None:
        var = getattr(self, name)
        dtype = jnp.int32 if name == "sym_code" else self.dtype
        var[...] = jnp.asarray(np.asarray(value), dtype)

    def _export_state(self) -> dict:
        n = self._to_numpy
        return {
            "coef": n(self.coef),
            "edge_scale": None if self.edge_scale is None else n(self.edge_scale),
            "trainable": {k: n(getattr(self, f"basis_{k}")) for k in self._trainable_names},
            "buffers": {k: n(getattr(self, f"basis_{k}")) for k in self._buffer_names},
        }

    def _load_state(self, basis, coef, edge_scale, trainable, buffers) -> None:
        self._basis = basis
        self._init_kwargs["basis"] = basis
        self.coef = nnx.Param(jnp.asarray(coef, self.dtype))
        if edge_scale is not None:
            self.edge_scale = nnx.Param(jnp.asarray(edge_scale, self.dtype))
        self._set_basis_state(trainable, buffers)

    def _variable_names(self) -> list[str]:
        names = ["coef", "base_weight", "bias", "edge_scale", "edge_mask", "sym_code", "sym_params"]
        return names + [f"basis_{k}" for k in self._trainable_names + self._buffer_names]

    def _export_full(self) -> dict:
        return {k: None if getattr(self, k) is None else self._to_numpy(getattr(self, k)) for k in self._variable_names()}

    def _import_full(self, state: dict) -> None:
        for name, value in state.items():
            var = getattr(self, name, None)
            if value is None or var is None:
                continue
            if tuple(var[...].shape) != np.shape(value):
                raise ValueError(f"{name}: shape {np.shape(value)} does not fit {tuple(var[...].shape)}")
            self._set(name, value)
        self._refresh_symbolic()

    def state_dict(self) -> dict:
        """NumPy copies of every variable, keyed like ``torch.nn.Module.state_dict``."""
        return {k: v for k, v in self._export_full().items() if v is not None}

    def load_state_dict(self, state: dict) -> None:
        self._import_full({k: np.asarray(v) for k, v in state.items()})

    # --- construction from MLPs -------------------------------------------------

    @classmethod
    def from_linear(cls, linear: nnx.Linear, activation: Callable = jax.nn.silu, basis="bspline", **kwargs):
        """Exactly ``linear(activation(x))`` with zero spline coefficients."""
        kernel = linear.kernel[...]  # (in, out)
        layer = cls(
            kernel.shape[0],
            kernel.shape[1],
            basis,
            bias=linear.bias is not None,
            base_activation=activation,
            coef_init="zero",
            dtype=kernel.dtype,
            **kwargs,
        )
        layer.base_weight[...] = kernel.T
        if linear.bias is not None:
            layer.bias[...] = linear.bias[...]
        return layer


class GRKAN(KANLinear):
    """Group-rational KAN layer of KAT (see ``kanforge.torch.GRKAN``)."""

    def __init__(self, in_features: int, out_features: int, *, groups: int = 8, num_degree: int = 5, den_degree: int = 4, init: str = "gelu", bias: bool = True, rngs=None, dtype=jnp.float32):
        super().__init__(
            in_features,
            out_features,
            Rational(num_degree, den_degree, init),
            rngs=rngs,
            bias=bias,
            base_activation=None,
            groups=groups,
            coef_init="variance",
            dtype=dtype,
        )


class KAN(nnx.Module):
    """Stack of ``KANLinear`` layers; ``grid_ranges`` sets the first layer's grids."""

    def __init__(
        self,
        widths: Sequence[int],
        basis: str | dict | Basis = "bspline",
        grid_ranges=None,
        *,
        rngs: nnx.Rngs | None = None,
        layers: Sequence[KANLinear] | None = None,
        **layer_kwargs,
    ):
        if len(widths) < 2:
            raise ValueError("widths needs at least input and output size")
        self.widths = list(widths)
        if layers is not None:  # existing layers (pruning)
            self.layers = nnx.List(list(layers))
            return
        rngs = rngs if rngs is not None else nnx.Rngs(0)
        self.layers = nnx.List(
            [
                KANLinear(a, b, basis, grid_ranges=grid_ranges if k == 0 else None, rngs=rngs, **layer_kwargs)
                for k, (a, b) in enumerate(zip(widths[:-1], widths[1:]))
            ]
        )

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def update_grid(self, x, **kwargs) -> None:
        for layer in self.layers:
            layer.update_grid(x, **kwargs)
            x = layer(x)

    def refine_grid(self, grid_size: int, **kwargs) -> None:
        for layer in self.layers:
            layer.refine_grid(grid_size, **kwargs)


class MultKANLinear(nnx.Module):
    """KAN layer with addition and multiplication nodes (see ``kanforge.torch``)."""

    def __init__(self, in_features: int, n_sum: int, n_mult: int = 0, basis="bspline", *, arity: int = 2, rngs=None, **kwargs):
        if arity < 2:
            raise ValueError("arity must be >= 2")
        self.n_sum, self.n_mult, self.arity = n_sum, n_mult, arity
        self.kan = KANLinear(in_features, n_sum + arity * n_mult, basis, rngs=rngs, **kwargs)

    @property
    def out_features(self) -> int:
        return self.n_sum + self.n_mult

    def __call__(self, x):
        y = self.kan(x)
        if not self.n_mult:
            return y
        tail = y[..., self.n_sum :]
        prod = jnp.prod(tail.reshape(*tail.shape[:-1], self.n_mult, self.arity), -1)
        return jnp.concatenate([y[..., : self.n_sum], prod], -1)


class MultKAN(nnx.Module):
    def __init__(self, widths: Sequence, basis: str | dict | Basis = "bspline", *, arity: int = 2, rngs=None, **layer_kwargs):
        ws = mult_widths(widths)
        rngs = rngs if rngs is not None else nnx.Rngs(0)
        self.widths = list(widths)
        self.layers = nnx.List(
            [
                MultKANLinear(a_sum + a_mult, b_sum, b_mult, basis, arity=arity, rngs=rngs, **layer_kwargs)
                for (a_sum, a_mult), (b_sum, b_mult) in zip(ws[:-1], ws[1:])
            ]
        )

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def from_mlp(mlp: nnx.Sequential, basis="bspline", **kwargs) -> nnx.Sequential:
    """JAX version of ``kanforge.torch.from_mlp`` (layers of an ``nnx.Sequential``)."""
    mods = list(mlp.layers)
    if not mods or not isinstance(mods[0], nnx.Linear):
        raise ValueError("the MLP must start with nnx.Linear")
    out = [mods[0]]
    i = 1
    while i < len(mods):
        if i + 1 < len(mods) and isinstance(mods[i + 1], nnx.Linear) and not isinstance(mods[i], nnx.Linear):
            out.append(KANLinear.from_linear(mods[i + 1], mods[i], basis, **kwargs))
            i += 2
        else:
            out.append(mods[i])
            i += 1
    return nnx.Sequential(*out)


def rebuild_kan(layers) -> KAN:
    """A ``KAN`` made of existing layers (used by pruning)."""
    return KAN([layers[0].in_features] + [layer.out_features for layer in layers], layers=layers)


def layer_inputs(model, x) -> list:
    """(KANLinear, its input) for KANLinear, MultKANLinear, KAN, MultKAN, nnx.Sequential."""
    out = []

    def walk(m, h):
        if isinstance(m, KANLinear):
            out.append((m, h))
            return m(h)
        if isinstance(m, MultKANLinear):
            out.append((m.kan, h))
            return m(h)
        if isinstance(m, (KAN, MultKAN, nnx.Sequential)):
            for sub in m.layers:
                h = walk(sub, h)
            return h
        return m(h)

    walk(model, x)
    return out


def regularization(model, x=None, *, l1=1.0, entropy=2.0, coef_l1=0.0, coef_smooth=0.0):
    """pykan's sparsity penalty (see ``kanforge.torch.regularization``)."""
    if x is None:
        layers = [m for _, m in nnx.iter_graph(model) if isinstance(m, KANLinear)]
        mags = [jnp.abs(m.effective_coef()).mean(-1) for m in layers]
    else:
        pairs = layer_inputs(model, x)
        layers = [m for m, _ in pairs]
        mags = [reg.edge_magnitude(m.edge_derivative(h, 0)) for m, h in pairs]
    total = 0.0
    for m, mag in zip(layers, mags):
        total = total + reg.penalty(mag, m.coef[...], l1=l1, entropy=entropy, coef_l1=coef_l1, coef_smooth=coef_smooth)
    return total

