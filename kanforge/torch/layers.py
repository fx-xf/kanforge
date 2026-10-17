"""PyTorch KAN layers. Run on any torch device: CUDA, ROCm, MPS, CPU.

This is the plain-ops implementation; fused kernels will plug in behind
the same modules without changing parameters or state_dict layout.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

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

    Parameters (same names on every backend):
      ``coef``           ``(out, in, basis.size)`` — basis coefficients of every edge
      ``base_weight``    ``(out, in)`` — residual branch, absent if ``base_activation`` is None
      ``bias``           ``(out,)`` — absent if ``bias=False``
      ``edge_scale``     ``(out, in)`` — per-edge scale s_oi, only with ``edge_scale=True``
      ``basis_<name>``   trainable basis parameters ``(groups, *shape)``: ``groups``
                         inputs share them (None: one set per input, 1: one per layer)
    Buffers:
      ``basis_<name>``   basis state such as grids, ``(in, *shape)``: every input
                         has its own grid (``grid_ranges``, ``update_grid``)

    ``edge_dropout`` drops individual edge activations while training (DropKAN).

    ``backend``: ``"ops"`` — plain PyTorch ops; ``"kernel"`` — fused Metal (MPS)
    or Triton (CUDA/ROCm) kernel, an error if the basis or device has none;
    ``"auto"`` — the kernel where benchmarks showed it faster (``AUTO_KINDS``).
    Both give the same values and derivatives of every order.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        basis: str | dict | Basis = "bspline",
        *,
        bias: bool = True,
        base_activation: Callable[[torch.Tensor], torch.Tensor] | None = F.silu,
        groups: int | None = None,
        edge_scale: bool = False,
        grid_ranges=None,
        coef_init: str = "normal",
        coef_init_std: float = 0.1,
        edge_dropout: float = 0.0,
        backend: str = "auto",
        device=None,
        dtype=None,
    ):
        super().__init__()
        if not 0 <= edge_dropout < 1:
            raise ValueError("edge_dropout must be in [0, 1)")
        factory = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.basis = make_basis(basis)
        self.base_activation = base_activation
        self.groups = resolve_groups(in_features, groups)
        self.coef_init = coef_init
        self.coef_init_std = coef_init_std
        self.edge_dropout = edge_dropout
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        self.backend = backend
        self._grid_ranges = grid_ranges
        init_stds(self.basis, in_features, out_features, coef_init, coef_init_std, base_activation is not None)
        self.coef = nn.Parameter(torch.empty(out_features, in_features, self.basis.size, **factory))
        if base_activation is not None:
            self.base_weight = nn.Parameter(torch.empty(out_features, in_features, **factory))
        else:
            self.register_parameter("base_weight", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory))
        else:
            self.register_parameter("bias", None)
        if edge_scale:
            self.edge_scale = nn.Parameter(torch.empty(out_features, in_features, **factory))
        else:
            self.register_parameter("edge_scale", None)
        self._trainable_names: list[str] = []
        self._buffer_names: list[str] = []
        self.reset_parameters()

    # --- parameters -------------------------------------------------------

    def _set_basis_state(self, trainable: dict, buffers: dict) -> None:
        for name in self._trainable_names + self._buffer_names:
            delattr(self, f"basis_{name}")
        like = {"dtype": self.coef.dtype, "device": self.coef.device}
        for name, value in trainable.items():
            self.register_parameter(f"basis_{name}", nn.Parameter(torch.as_tensor(value, **like)))
        for name, value in buffers.items():
            self.register_buffer(f"basis_{name}", torch.as_tensor(value, **like))
        self._trainable_names, self._buffer_names = list(trainable), list(buffers)
        self._plan = plan_for(self.basis)

    def reset_parameters(self) -> None:
        coef_std, base_std = init_stds(
            self.basis, self.in_features, self.out_features, self.coef_init, self.coef_init_std, self.base_weight is not None
        )
        nn.init.normal_(self.coef, std=coef_std) if coef_std else nn.init.zeros_(self.coef)
        if self.base_weight is not None:
            if base_std is None:
                nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
            else:
                nn.init.normal_(self.base_weight, std=base_std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if self.edge_scale is not None:
            nn.init.ones_(self.edge_scale)
        self._set_basis_state(*basis_param_values(self.basis, self.groups, self.in_features, self._grid_ranges))

    def basis_params(self) -> dict[str, torch.Tensor]:
        """Basis parameters and buffers expanded to ``(in, *shape)`` (or ``(1, *shape)``)."""
        rep = self.in_features // self.groups
        out = {}
        for name in self._trainable_names:
            p = getattr(self, f"basis_{name}")
            out[name] = p if self.groups in (1, self.in_features) else p.repeat_interleave(rep, dim=0)
        for name in self._buffer_names:
            out[name] = getattr(self, f"basis_{name}")
        return out

    def effective_coef(self) -> torch.Tensor:
        if self.edge_scale is None:
            return self.coef
        return self.coef * self.edge_scale[..., None]

    # --- computation ------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.edge_dropout > 0:
            edges = self.edge_derivative(x, 0)
            keep = torch.rand_like(edges) >= self.edge_dropout
            y = (edges * keep).sum(-1) / (1 - self.edge_dropout)
            return y if self.bias is None else y + self.bias
        if self.uses_kernel(x.device):
            grid = getattr(self, "basis_grid", None) if self._plan.grid else None
            y = _kernels.spline_forward(x, self.effective_coef(), grid, self._plan)
        else:
            # (..., in, n) -> (..., in*n) @ (in*n, out): one GEMM, as in efficient-kan.
            y = self.basis(x, self.basis_params()).flatten(-2) @ self.effective_coef().flatten(1).T
        if self.base_weight is not None:
            y = y + F.linear(self.base_activation(x), self.base_weight)
        if self.bias is not None:
            y = y + self.bias
        return y

    def uses_kernel(self, device: torch.device) -> bool:
        """Whether forward runs the fused kernel on ``device``."""
        if self.backend == "ops":
            return False
        ok = self._plan is not None and _kernels.available(device)
        if self.backend == "kernel":
            if not ok:
                why = "basis" if self._plan is None else f"device {device}"
                raise RuntimeError(f"no fused kernel for this {why}: {self.basis!r}; use backend='ops' or 'auto'")
            return True
        return ok and self._plan.kind in AUTO_KINDS.get(device.type, ())

    def edge_derivative(self, x: torch.Tensor, order: int = 1) -> torch.Tensor:
        """d^order phi_oi / dx_i^order for every edge, shape ``(..., out, in)``.

        ``order=0`` gives the edge functions themselves (bias excluded). A KAN
        layer is a sum of 1-D edge functions, so entry (o, i) is exactly
        d^m y_o / dx_i^m — the diagonal derivatives PINN losses need.
        """
        d = torch.einsum("...in,oin->...oi", self.basis.derivative(x, order, self.basis_params()), self.effective_coef())
        if self.base_weight is not None:
            if order == 0:
                base = self.base_activation(x)
            elif self.base_activation is F.silu:
                base = silu_derivative(x, order)
            else:
                raise NotImplementedError("analytic derivatives need base_activation=F.silu or None")
            d = d + base[..., None, :] * self.base_weight
        return d

    # --- grid tools (GridToolsMixin) --------------------------------------

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        return x.detach().cpu().double().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    def _export_state(self) -> dict:
        def n(t):
            return None if t is None else self._to_numpy(t)

        return {
            "coef": n(self.coef),
            "edge_scale": n(self.edge_scale),
            "trainable": {k: n(getattr(self, f"basis_{k}")) for k in self._trainable_names},
            "buffers": {k: n(getattr(self, f"basis_{k}")) for k in self._buffer_names},
        }

    @torch.no_grad()
    def _load_state(self, basis, coef, edge_scale, trainable, buffers) -> None:
        like = {"dtype": self.coef.dtype, "device": self.coef.device}
        self.basis = basis
        self.coef = nn.Parameter(torch.as_tensor(coef, **like))
        if edge_scale is not None:
            self.edge_scale = nn.Parameter(torch.as_tensor(edge_scale, **like))
        self._set_basis_state(trainable, buffers)

    # --- construction from MLPs -------------------------------------------

    @classmethod
    def from_linear(cls, linear: nn.Linear, activation: Callable = F.silu, basis="bspline", **kwargs) -> KANLinear:
        """Exactly ``linear(activation(x))``: the activation becomes the base branch,
        spline coefficients start at zero and learn corrections."""
        layer = cls(
            linear.in_features,
            linear.out_features,
            basis,
            bias=linear.bias is not None,
            base_activation=activation,
            coef_init="zero",
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            **kwargs,
        )
        with torch.no_grad():
            layer.base_weight.copy_(linear.weight)
            if linear.bias is not None:
                layer.bias.copy_(linear.bias)
        return layer

    def extra_repr(self) -> str:
        groups = "" if self.groups == self.in_features else f", groups={self.groups}"
        return f"in_features={self.in_features}, out_features={self.out_features}, basis={self.basis!r}{groups}"


def from_mlp(mlp: nn.Sequential, basis="bspline", **kwargs) -> nn.Sequential:
    """Turn ``Linear, act, Linear, act, ..., Linear`` into an equal network:
    the first Linear stays, every ``(act, Linear)`` pair becomes a ``KANLinear``
    whose base branch is that activation. Outputs match exactly at start."""
    mods = list(mlp)
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
    """Group-rational KAN layer of KAT: rational activations shared by ``groups``
    of inputs, then the layer coefficients act as the linear weight; no base branch.
    """

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
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_features,
            out_features,
            Rational(num_degree, den_degree, init),
            bias=bias,
            base_activation=None,
            groups=groups,
            coef_init="variance",
            device=device,
            dtype=dtype,
        )


class KAN(nn.Sequential):
    """Stack of ``KANLinear`` layers, e.g. ``KAN([2, 5, 1], basis="chebyshev")``."""

    def __init__(self, widths: Sequence[int], basis: str | dict | Basis = "bspline", **layer_kwargs):
        if len(widths) < 2:
            raise ValueError("widths needs at least input and output size")
        super().__init__(
            *(KANLinear(a, b, basis, **layer_kwargs) for a, b in zip(widths[:-1], widths[1:]))
        )
        self.widths = list(widths)

    def edge_derivative(self, x: torch.Tensor, order: int = 1) -> list[torch.Tensor]:
        """Per-layer edge derivatives at the activations each layer receives."""
        return [layer.edge_derivative(h, order) for layer, h in layer_inputs(self, x)]

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, **kwargs) -> None:
        """Update every layer's grid from the activations it receives (pykan)."""
        for layer in self:
            layer.update_grid(x, **kwargs)
            x = layer(x)

    def refine_grid(self, grid_size: int, **kwargs) -> None:
        for layer in self:
            layer.refine_grid(grid_size, **kwargs)


class MultKANLinear(nn.Module):
    """KAN layer with ``n_sum`` addition nodes and ``n_mult`` multiplication nodes
    (MultKAN, KAN 2.0). Every product node multiplies ``arity`` sub-nodes.
    Output: ``(..., n_sum + n_mult)``.
    """

    def __init__(self, in_features: int, n_sum: int, n_mult: int = 0, basis="bspline", *, arity: int = 2, **kwargs):
        super().__init__()
        if arity < 2:
            raise ValueError("arity must be >= 2")
        self.n_sum, self.n_mult, self.arity = n_sum, n_mult, arity
        self.kan = KANLinear(in_features, n_sum + arity * n_mult, basis, **kwargs)

    @property
    def out_features(self) -> int:
        return self.n_sum + self.n_mult

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.kan(x)
        if not self.n_mult:
            return y
        prod = y[..., self.n_sum :].unflatten(-1, (self.n_mult, self.arity)).prod(-1)
        return torch.cat([y[..., : self.n_sum], prod], -1)


class MultKAN(nn.Sequential):
    """``MultKAN([2, (2, 1), 1])``: an int is a width of addition nodes,
    ``(n_sum, n_mult)`` adds multiplication nodes to that layer."""

    def __init__(self, widths: Sequence, basis: str | dict | Basis = "bspline", *, arity: int = 2, **layer_kwargs):
        ws = mult_widths(widths)
        layers = []
        for (a_sum, a_mult), (b_sum, b_mult) in zip(ws[:-1], ws[1:]):
            layers.append(MultKANLinear(a_sum + a_mult, b_sum, b_mult, basis, arity=arity, **layer_kwargs))
        super().__init__(*layers)
        self.widths = list(widths)


def layer_inputs(model: nn.Module, x: torch.Tensor) -> list[tuple[KANLinear, torch.Tensor]]:
    """(KANLinear, the input it receives) for every KAN layer of ``model``.

    Works for KANLinear, MultKANLinear and any nn.Sequential of modules.
    """
    out = []

    def walk(m, h):
        if isinstance(m, KANLinear):
            out.append((m, h))
            return m(h)
        if isinstance(m, MultKANLinear):
            out.append((m.kan, h))
            return m(h)
        if isinstance(m, nn.Sequential):
            for sub in m:
                h = walk(sub, h)
            return h
        return m(h)

    walk(model, x)
    return out


def regularization(
    model: nn.Module,
    x: torch.Tensor | None = None,
    *,
    l1: float = 1.0,
    entropy: float = 2.0,
    coef_l1: float = 0.0,
    coef_smooth: float = 0.0,
) -> torch.Tensor:
    """pykan's sparsity penalty summed over all KAN layers.

    With ``x`` the edge magnitudes are mean |phi_oi| over the batch (pykan);
    without it, mean |coef| (efficient-kan). Add ``lamb * regularization(...)``
    to the loss.
    """
    if x is None:
        layers = [m for m in model.modules() if isinstance(m, KANLinear)]
        mags = [m.effective_coef().abs().mean(-1) for m in layers]
    else:
        pairs = layer_inputs(model, x)
        layers = [m for m, _ in pairs]
        mags = [reg.edge_magnitude(m.edge_derivative(h, 0)) for m, h in pairs]
    total = 0.0
    for m, mag in zip(layers, mags):
        total = total + reg.penalty(mag, m.coef, l1=l1, entropy=entropy, coef_l1=coef_l1, coef_smooth=coef_smooth)
    return total
