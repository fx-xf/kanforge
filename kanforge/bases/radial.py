"""Bump-shaped bases: Gaussian RBF (FastKAN), RSWAF (FasterKAN) and
wavelets (Wav-KAN: Mexican hat, Morlet, derivative of Gaussian).

All Gaussian-envelope functions have the form
psi(u) = (p(u) cos(w u) + q(u) sin(w u)) * exp(-u^2 / 2), which is closed
under differentiation, so every derivative order is exact.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.polynomial import polynomial as P

from .._calculus import TANH_FACTOR, chain_polys, horner
from .._xp import namespace
from .base import Basis, ParamSpec, _check_range


def _envelope_derivatives(p, q, omega: float, m: int):
    """(p, q) polynomial pairs of psi, psi', ..., psi^(m)."""
    out = [(np.asarray(p, float), np.asarray(q, float))]
    u = np.array([0.0, 1.0])
    for _ in range(m):
        p, q = out[-1]
        new_p = P.polyadd(P.polysub(P.polyder(p), P.polymul(u, p)), omega * q)
        new_q = P.polysub(P.polysub(P.polyder(q), P.polymul(u, q)), omega * p)
        out.append((new_p, new_q))
    return out


class _Envelope(Basis):
    """psi((x - c_b) / s) on ``count`` centres spread over ``grid_range``."""

    p: tuple = (1.0,)
    q: tuple = (0.0,)
    omega = 0.0
    norm = 1.0
    u_scale = 1.0  # psi is evaluated at u_scale * (x - c) / s

    def __init__(self, count: int = 8, grid_range: tuple[float, float] = (-1.0, 1.0), learnable: bool = False):
        if count < 2:
            raise ValueError("count must be >= 2")
        self.count = int(count)
        self.grid_range = _check_range(grid_range)
        self.learnable = bool(learnable)

    @property
    def size(self) -> int:
        return self.count

    def param_specs(self):
        lo, hi = self.grid_range
        centers = np.linspace(lo, hi, self.count)
        if not self.learnable:
            # Fixed centres are a grid: per input, adaptable from data.
            return {"grid": ParamSpec(centers, trainable=False)}
        log_scale = np.array([math.log((hi - lo) / (self.count - 1))])
        return {"centers": ParamSpec(centers), "log_scale": ParamSpec(log_scale)}

    def _eval(self, xe, order, p):
        xp = namespace(xe)
        if self.learnable:
            c = p["centers"]
            k = self.u_scale * xp.exp(-p["log_scale"])
        else:
            c = p["grid"]
            k = self.u_scale * (self.count - 1) / (c[..., -1:] - c[..., :1])
        u = (xe - c) * k
        pp, qq = _envelope_derivatives(self.p, self.q, self.omega, order)[order]
        val = horner(list(pp), u)
        if self.omega:
            val = val * xp.cos(self.omega * u) + horner(list(qq), u) * xp.sin(self.omega * u)
        return self.norm * val * xp.exp(-0.5 * u * u) * k**order

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "count": self.count, "grid_range": self.grid_range, "learnable": self.learnable}


class RBF(_Envelope):
    """Gaussian RBF exp(-((x - c) / h)^2) on a uniform grid of centres (FastKAN)."""

    name = "rbf"
    u_scale = math.sqrt(2.0)


class MexicanHat(_Envelope):
    """Mexican hat wavelet 2 / (sqrt(3) pi^(1/4)) (1 - u^2) exp(-u^2 / 2)."""

    name = "mexican_hat"
    p = (1.0, 0.0, -1.0)
    norm = 2 / (math.sqrt(3) * math.pi**0.25)

    def __init__(self, count: int = 8, grid_range: tuple[float, float] = (-1.0, 1.0), learnable: bool = True):
        super().__init__(count, grid_range, learnable)


class Morlet(_Envelope):
    """Real Morlet wavelet cos(omega u) exp(-u^2 / 2)."""

    name = "morlet"

    def __init__(
        self,
        count: int = 8,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        learnable: bool = True,
        omega: float = 5.0,
    ):
        super().__init__(count, grid_range, learnable)
        self.omega = float(omega)

    def config(self):
        return {**super().config(), "omega": self.omega}


class DoG(_Envelope):
    """Derivative-of-Gaussian wavelet -u exp(-u^2 / 2)."""

    name = "dog"
    p = (0.0, -1.0)

    def __init__(self, count: int = 8, grid_range: tuple[float, float] = (-1.0, 1.0), learnable: bool = True):
        super().__init__(count, grid_range, learnable)


class RSWAF(Basis):
    """Reflectional switch activation 1 - tanh((x - c) / h)^2 (FasterKAN)."""

    name = "rswaf"

    def __init__(self, count: int = 8, grid_range: tuple[float, float] = (-1.0, 1.0)):
        if count < 2:
            raise ValueError("count must be >= 2")
        self.count = int(count)
        self.grid_range = _check_range(grid_range)

    @property
    def size(self) -> int:
        return self.count

    def param_specs(self):
        return {"grid": ParamSpec(np.linspace(*self.grid_range, self.count), trainable=False)}

    def _eval(self, xe, order, p):
        xp = namespace(xe)
        c = p["grid"]
        h = (c[..., -1:] - c[..., :1]) / (self.count - 1)
        t = xp.tanh((xe - c) / h)
        poly = chain_polys(TANH_FACTOR, TANH_FACTOR, order)[order]  # sech^2 = 1 - t^2
        return horner(list(poly), t) / h**order

    def config(self):
        return {"name": self.name, "count": self.count, "grid_range": self.grid_range}
