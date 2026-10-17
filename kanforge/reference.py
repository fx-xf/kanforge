"""Float64 NumPy reference for a KAN layer.

Slow and simple on purpose: every backend and every future kernel is
tested against these functions.
"""

from __future__ import annotations

import numpy as np

from .bases import Basis, make_basis


def silu(x):
    return x / (1.0 + np.exp(-x))


def kan_linear(x, coef, base_weight=None, bias=None, basis: str | dict | Basis = "bspline"):
    """y[..., o] = sum_i phi_oi(x_i) + sum_i w_oi * silu(x_i) + b_o.

    Shapes: x ``(..., in)``, coef ``(out, in, basis.size)``,
    base_weight ``(out, in)`` or None, bias ``(out,)`` or None.
    """
    basis = make_basis(basis)
    x = np.asarray(x, dtype=np.float64)
    coef = np.asarray(coef, dtype=np.float64)
    y = np.einsum("...in,oin->...o", basis(x), coef)
    if base_weight is not None:
        y = y + silu(x) @ np.asarray(base_weight, dtype=np.float64).T
    if bias is not None:
        y = y + np.asarray(bias, dtype=np.float64)
    return y
