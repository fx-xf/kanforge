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

from .._calculus import silu_derivative
from .._layer import basis_param_values, coef_std, mult_widths, resolve_groups
from ..bases import Basis, Rational, make_basis


class KANLinear(nn.Module):
    """One KAN layer: y_o = sum_i s_oi * phi_oi(x_i) + sum_i w_oi * act(x_i) + b_o.

    Parameters (same names on every backend):
      ``coef``           ``(out, in, basis.size)`` — basis coefficients of every edge
      ``base_weight``    ``(out, in)`` — residual branch, absent if ``base_activation`` is None
      ``bias``           ``(out,)`` — absent if ``bias=False``
      ``edge_scale``     ``(out, in)`` — per-edge scale s_oi, only with ``edge_scale=True``
      ``basis_<name>``   ``(groups, *shape)`` — trainable basis parameters; ``groups``
                         inputs share them (None: one set per input, 1: one per layer)
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
        coef_init: str = "normal",
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
        self.groups = resolve_groups(in_features, groups)
        self.coef_init = coef_init
        self.coef_init_std = coef_init_std
        coef_std(self.basis, in_features, coef_init, coef_init_std)  # validates coef_init
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
        self._basis_param_names = []
        for name in basis_param_values(self.basis, self.groups):
            self.register_parameter(f"basis_{name}", nn.Parameter(torch.empty(0, **factory)))
            self._basis_param_names.append(name)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.coef, std=coef_std(self.basis, self.in_features, self.coef_init, self.coef_init_std))
        if self.base_weight is not None:
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if self.edge_scale is not None:
            nn.init.ones_(self.edge_scale)
        for name, value in basis_param_values(self.basis, self.groups).items():
            p = getattr(self, f"basis_{name}")
            p.data = torch.as_tensor(value, dtype=self.coef.dtype, device=self.coef.device)

    def basis_params(self) -> dict[str, torch.Tensor]:
        """Basis parameters expanded to ``(in, *shape)`` (or ``(1, *shape)``)."""
        rep = self.in_features // self.groups
        out = {}
        for name in self._basis_param_names:
            p = getattr(self, f"basis_{name}")
            out[name] = p if self.groups in (1, self.in_features) else p.repeat_interleave(rep, dim=0)
        return out

    def effective_coef(self) -> torch.Tensor:
        if self.edge_scale is None:
            return self.coef
        return self.coef * self.edge_scale[..., None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (..., in, n) -> (..., in*n) @ (in*n, out): one GEMM, as in efficient-kan.
        y = self.basis(x, self.basis_params()).flatten(-2) @ self.effective_coef().flatten(1).T
        if self.base_weight is not None:
            y = y + F.linear(self.base_activation(x), self.base_weight)
        if self.bias is not None:
            y = y + self.bias
        return y

    def edge_derivative(self, x: torch.Tensor, order: int = 1) -> torch.Tensor:
        """d^order phi_oi / dx_i^order for every edge, shape ``(..., out, in)``.

        ``order=0`` gives the edge functions themselves (bias excluded). A KAN
        layer is a sum of 1-D edge functions, so entry (o, i) is exactly
        d^m y_o / dx_i^m — the diagonal derivatives PINN losses need.
        """
        d = torch.einsum("...in,oin->...oi", self.basis.derivative(x, order, self.basis_params()), self.effective_coef())
        if self.base_weight is not None:
            if self.base_activation is not F.silu:
                raise NotImplementedError("analytic derivatives need base_activation=F.silu or None")
            d = d + silu_derivative(x, order)[..., None, :] * self.base_weight
        return d

    def extra_repr(self) -> str:
        groups = "" if self.groups == self.in_features else f", groups={self.groups}"
        return f"in_features={self.in_features}, out_features={self.out_features}, basis={self.basis!r}{groups}"


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
        out = []
        for layer in self:
            out.append(layer.edge_derivative(x, order))
            x = layer(x)
        return out


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
