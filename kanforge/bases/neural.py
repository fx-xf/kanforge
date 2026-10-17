"""Tiny-network edges (in the spirit of FCN-KAN): B_b(x) = act(a_b * (x - c_b)).

With the layer coefficients on top this is a one-hidden-layer MLP on every
edge whose first layer is shared by the edges leaving the same input.
"""

from __future__ import annotations

import numpy as np

from .._calculus import TANH_FACTOR, chain_polys, horner, silu_derivative
from .._xp import namespace
from .base import Basis, ParamSpec, _check_range

ACTIVATIONS = ("tanh", "silu")


class Neural(Basis):
    name = "neural"

    def __init__(self, hidden: int = 8, activation: str = "tanh", grid_range: tuple[float, float] = (-1.0, 1.0)):
        if hidden < 1:
            raise ValueError("hidden must be >= 1")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        self.hidden = int(hidden)
        self.activation = activation
        self.grid_range = _check_range(grid_range)

    @property
    def size(self) -> int:
        return self.hidden

    def param_specs(self):
        lo, hi = self.grid_range
        h = (hi - lo) / max(self.hidden - 1, 1)
        return {
            "slope": ParamSpec(np.full(self.hidden, 2.0 / h)),
            "center": ParamSpec(np.linspace(lo, hi, self.hidden) if self.hidden > 1 else np.zeros(1)),
        }

    def _eval(self, xe, order, p):
        a = p["slope"]
        u = a * (xe - p["center"])
        if self.activation == "silu":
            d = silu_derivative(u, order)
        else:
            t = namespace(xe).tanh(u)
            d = horner(list(chain_polys([0.0, 1.0], TANH_FACTOR, order)[order]), t)
        return d * a**order

    def config(self):
        return {"name": self.name, "hidden": self.hidden, "activation": self.activation, "grid_range": self.grid_range}

