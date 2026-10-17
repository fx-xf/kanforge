"""Piecewise bases: B-splines, NURBS, truncated powers, ReLU-KAN."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.polynomial import legendre as L
from numpy.polynomial import polynomial as P

from .._calculus import falling, horner, quotient
from .._xp import arange_like, asarray_like, cast_like, namespace, relu, softplus
from .base import Basis, ParamSpec, _check_range


class BSpline(Basis):
    """Uniform B-splines of ``order`` on ``grid_size`` intervals over ``grid_range``.

    As in pykan, the knot vector is extended by ``order`` knots on each side,
    so there are ``grid_size + order`` functions and they sum to one inside
    ``grid_range``. Outside the extended grid all values are zero and the
    layer falls back to its base (SiLU) branch.
    """

    name = "bspline"
    local = True

    def __init__(self, grid_size: int = 5, order: int = 3, grid_range: tuple[float, float] = (-1.0, 1.0)):
        if grid_size < 1 or order < 0:
            raise ValueError("grid_size must be >= 1 and order >= 0")
        self.grid_size = int(grid_size)
        self.order = int(order)
        self.grid_range = _check_range(grid_range)

    @property
    def size(self) -> int:
        return self.grid_size + self.order

    @property
    def support(self) -> int:
        return self.order + 1

    @property
    def step(self) -> float:
        lo, hi = self.grid_range
        return (hi - lo) / self.grid_size

    def knots(self, like):
        g, k, h = self.grid_size, self.order, self.step
        return self.grid_range[0] + (arange_like(g + 2 * k + 1, like) - k) * h

    def _eval(self, xe, order, p):
        xp = namespace(xe)
        k, h = self.order, self.step
        if order > k:
            return 0 * xe * arange_like(self.size, xe)
        t = self.knots(xe)
        b = cast_like(xp.logical_and(xe >= t[:-1], xe < t[1:]), xe)
        # Cox–de Boor up to degree k - order; on a uniform grid every
        # denominator is p*h.
        for q in range(1, k - order + 1):
            b = ((xe - t[: -(q + 1)]) * b[..., :-1] + (t[q + 1 :] - xe) * b[..., 1:]) / (q * h)
        # d/dx B_{q,j} = (B_{q-1,j} - B_{q-1,j+1}) / h on a uniform grid.
        for _ in range(order):
            b = (b[..., :-1] - b[..., 1:]) / h
        return b

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "grid_size": self.grid_size, "order": self.order, "grid_range": self.grid_range}


class NURBS(Basis):
    """Rational B-splines R_j = w_j N_j / sum_k w_k N_k with trainable weights.

    Weights are stored as logs, so they stay positive. With all weights equal
    this is exactly ``BSpline``. Used by Double-KANs for signals with jumps.
    """

    name = "nurbs"
    local = True

    def __init__(self, grid_size: int = 5, order: int = 3, grid_range: tuple[float, float] = (-1.0, 1.0)):
        self._spline = BSpline(grid_size, order, grid_range)

    grid_size = property(lambda self: self._spline.grid_size)
    order = property(lambda self: self._spline.order)
    grid_range = property(lambda self: self._spline.grid_range)

    @property
    def size(self) -> int:
        return self._spline.size

    @property
    def support(self) -> int:
        return self._spline.support

    def param_specs(self):
        return {"log_weights": ParamSpec(np.zeros(self.size))}

    def _eval(self, xe, order, p):
        xp = namespace(xe)
        w = xp.exp(p["log_weights"])
        num = [w * self._spline._eval(xe, j, p) for j in range(order + 1)]
        den = [xp.sum(a, -1)[..., None] for a in num]
        den[0] = xp.where(den[0] > 0, den[0], 0 * den[0] + 1)  # zero outside the grid
        return quotient(num, den, order)

    def config(self):
        return {**self._spline.config(), "name": self.name}


class TruncatedPower(Basis):
    """TruKAN basis: Legendre P_0..P_degree plus (x - t_j)_+^degree / (hi - t_j)^degree.

    With ``learn_knots`` the interior knots are trainable per input; they stay
    ordered inside ``grid_range`` because they are cumulative sums of
    softplus increments (as in TruKAN).
    """

    name = "truncated_power"

    def __init__(
        self,
        degree: int = 3,
        num_knots: int = 6,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        learn_knots: bool = False,
    ):
        if degree < 1 or num_knots < 1:
            raise ValueError("degree and num_knots must be >= 1")
        self.degree = int(degree)
        self.num_knots = int(num_knots)
        self.grid_range = _check_range(grid_range)
        self.learn_knots = bool(learn_knots)
        self._legendre = [L.leg2poly(np.eye(self.degree + 1)[j]) for j in range(self.degree + 1)]

    @property
    def size(self) -> int:
        return self.degree + 1 + self.num_knots

    def param_specs(self):
        if not self.learn_knots:
            return {}
        inc = (self.grid_range[1] - self.grid_range[0]) / (self.num_knots + 1)
        raw = np.log(np.expm1(inc))  # softplus^-1
        return {"knot_logits": ParamSpec(np.full(self.num_knots + 1, raw))}

    def _knots(self, xe, p):
        lo, hi = self.grid_range
        if not self.learn_knots:
            return asarray_like(np.linspace(lo, hi, self.num_knots + 2)[1:-1], xe)
        xp = namespace(xe)
        inc = softplus(p["knot_logits"]) + 1e-8
        cum = xp.cumsum(inc, -1)
        return lo + (hi - lo) * cum[..., :-1] / cum[..., -1:]

    def _eval(self, xe, order, p):
        xp = namespace(xe)
        lo, hi = self.grid_range
        a = 2.0 / (hi - lo)
        s = a * xe - (lo + hi) / (hi - lo)
        polys = []
        for c in self._legendre:
            d = P.polyder(c, order) if order else c
            polys.append(horner(list(d), s) * a**order)
        t = self._knots(xe, p)
        scale = 1.0 / xp.maximum(hi - t, 0 * t + 1e-6)
        v = relu((xe - t) * scale)
        k = self.degree
        if order > k:
            trunc = 0 * v
        elif order == k:
            trunc = falling(k, order) * scale**order * cast_like(v > 0, v)
        else:
            trunc = falling(k, order) * scale**order * v ** (k - order)
        return xp.concatenate([*polys, trunc], -1)

    def config(self):
        return {
            "name": self.name,
            "degree": self.degree,
            "num_knots": self.num_knots,
            "grid_range": self.grid_range,
            "learn_knots": self.learn_knots,
        }


class ReLUKAN(Basis):
    """ReLU-KAN: R_i(x) = [relu(e_i - x) * relu(x - s_i)]^2 * 16 / (e_i - s_i)^4.

    Same layout as ``BSpline``: ``grid_size + order`` bumps of width
    ``(order + 1) * h`` starting every ``h``; each peaks at 1. Only additions,
    products and ReLU, so it is cheap on any accelerator.
    """

    name = "relu_kan"
    local = True

    def __init__(self, grid_size: int = 5, order: int = 3, grid_range: tuple[float, float] = (-1.0, 1.0)):
        if grid_size < 1 or order < 0:
            raise ValueError("grid_size must be >= 1 and order >= 0")
        self.grid_size = int(grid_size)
        self.order = int(order)
        self.grid_range = _check_range(grid_range)

    @property
    def size(self) -> int:
        return self.grid_size + self.order

    @property
    def support(self) -> int:
        return self.order + 1

    def _eval(self, xe, order, p):
        lo, hi = self.grid_range
        h = (hi - lo) / self.grid_size
        width = (self.order + 1) * h
        s = lo + (arange_like(self.size, xe) - self.order) * h
        e = s + width
        norm = 16.0 / width**4
        a, b = relu(e - xe), relu(xe - s)
        if order == 0:
            return (a * b) ** 2 * norm
        mask = cast_like(namespace(xe).logical_and(a > 0, b > 0), xe)
        q, dq, d2q = a * b, e + s - 2 * xe, -2.0
        if order == 1:
            r = 2 * q * dq
        elif order == 2:
            r = 2 * (dq * dq + q * d2q)
        elif order == 3:
            r = 6 * dq * d2q
        elif order == 4:
            r = 6 * d2q * d2q + 0 * xe
        else:
            r = 0 * xe * s
        return r * mask * norm

    def config(self):
        return {"name": self.name, "grid_size": self.grid_size, "order": self.order, "grid_range": self.grid_range}
