"""Backend-independent pieces of the layer definitions."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .bases import Basis

COEF_INITS = ("normal", "variance")


def resolve_groups(in_features: int, groups: int | None) -> int:
    """Number of parameter groups: ``None`` means one group per input."""
    if groups is None:
        return in_features
    if groups < 1 or in_features % groups:
        raise ValueError(f"groups={groups} must divide in_features={in_features}")
    return groups


def basis_param_values(basis: Basis, groups: int) -> dict[str, np.ndarray]:
    """Initial basis parameters stored as ``(groups, *shape)``."""
    return {
        name: np.broadcast_to(spec.value, (groups, *spec.shape)).copy()
        for name, spec in basis.param_specs().items()
    }


def basis_second_moment(basis: Basis, points: int = 96) -> float:
    """E[sum_b B_b(x)^2] for x ~ N(0, 1), by Gauss–Hermite quadrature."""
    x, w = np.polynomial.hermite_e.hermegauss(points)
    values = basis(x)
    return float((w / math.sqrt(2 * math.pi)) @ (values**2).sum(-1))


def coef_std(basis: Basis, in_features: int, coef_init: str, coef_init_std: float) -> float:
    """``normal``: coef_init_std / sqrt(in). ``variance``: unit output variance
    for N(0, 1) inputs, as in KAT's variance-preserving initialisation."""
    if coef_init == "normal":
        return coef_init_std / math.sqrt(in_features)
    if coef_init == "variance":
        return 1.0 / math.sqrt(in_features * max(basis_second_moment(basis), 1e-12))
    raise ValueError(f"coef_init must be one of {COEF_INITS}")


def mult_widths(widths: Sequence) -> list[tuple[int, int]]:
    """MultKAN widths: ``3`` -> (3 sum nodes, 0 product nodes), ``(3, 2)`` as is."""
    out = []
    for w in widths:
        n_sum, n_mult = (w, 0) if isinstance(w, int) else (int(w[0]), int(w[1]))
        if n_sum < 0 or n_mult < 0 or n_sum + n_mult == 0:
            raise ValueError(f"bad MultKAN width {w!r}")
        out.append((n_sum, n_mult))
    if len(out) < 2:
        raise ValueError("widths needs at least input and output size")
    return out
