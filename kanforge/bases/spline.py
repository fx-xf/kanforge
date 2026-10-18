"""Piecewise bases: B-splines, NURBS, truncated powers, ReLU-KAN.

Their grid is a per-input buffer ``grid`` of increasing points; by default
it is uniform over ``grid_range`` and grid tools (``kanforge.grid``) adapt it
to the data. Splines extend it by ``order`` points on each side with the
mean grid step, as pykan does.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.polynomial import legendre as L
from numpy.polynomial import polynomial as P

from .._calculus import falling, horner, quotient
from .._xp import arange_like, cast_like, concat, namespace, relu, softplus
from .base import Basis, ParamSpec, _check_range


def extend_grid(grid, k: int):
    """Add ``k`` points on each side of ``grid`` (``(..., G+1)``) with the mean step."""
    if k == 0:
        return grid
    h = (grid[..., -1:] - grid[..., :1]) / (grid.shape[-1] - 1)
    i = arange_like(k, grid)
    left = grid[..., :1] - h * (k - i)
    right = grid[..., -1:] + h * (i + 1)
    return concat([left, grid, right], -1)


class _Gridded(Basis):
    """Shared constructor and grid spec of the spline-like bases."""

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

    def param_specs(self):
        return {"grid": ParamSpec(np.linspace(*self.grid_range, self.grid_size + 1), trainable=False)}

    def knots(self, p=None):
        """Extended knot vector; ``p`` defaults to the uniform NumPy grid."""
        grid = np.linspace(*self.grid_range, self.grid_size + 1) if p is None else p["grid"]
        return extend_grid(grid, self.order)

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "grid_size": self.grid_size, "order": self.order, "grid_range": self.grid_range}


class BSpline(_Gridded):
    """B-splines of ``order`` on ``grid_size`` intervals (pykan / efficient-kan).

    There are ``grid_size + order`` functions; they sum to one inside the
    grid. Outside the extended grid all values are zero and the layer falls
    back to its base (SiLU) branch.
    """

    name = "bspline"
    local = True

    def _eval(self, xe, order, p):
        xp = namespace(xe)
        k = self.order
        if order > k:
            return 0 * xe * arange_like(self.size, xe)
        t = self.knots(p)
        b = cast_like(xp.logical_and(xe >= t[..., :-1], xe < t[..., 1:]), xe)
        # Cox–de Boor up to degree k - order.
        for q in range(1, k - order + 1):
            left = (xe - t[..., : -(q + 1)]) / (t[..., q:-1] - t[..., : -(q + 1)])
            right = (t[..., q + 1 :] - xe) / (t[..., q + 1 :] - t[..., 1:-q])
            b = left * b[..., :-1] + right * b[..., 1:]
        # d/dx B_{q,j} = q B_{q-1,j} / (t_{j+q} - t_j) - q B_{q-1,j+1} / (t_{j+q+1} - t_{j+1}).
        for q in range(k - order + 1, k + 1):
            n = b.shape[-1] - 1
            b = q * (
                b[..., :-1] / (t[..., q : q + n] - t[..., :n]) - b[..., 1:] / (t[..., q + 1 : q + 1 + n] - t[..., 1 : 1 + n])
            )
        return b


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
        return {**self._spline.param_specs(), "log_weights": ParamSpec(np.zeros(self.size))}

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
    """TruKAN basis: Legendre P_0..P_degree of the grid-normalised input plus
    (x - t_j)_+^degree / (hi - t_j)^degree.

    The grid holds ``num_knots + 2`` points: the ends of the range and the
    interior knots. With ``learn_knots`` the interior knots are trainable per
    input; they stay ordered because they are cumulative sums of softplus
    increments (as in TruKAN).
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
        specs = {"grid": ParamSpec(np.linspace(*self.grid_range, self.num_knots + 2), trainable=False)}
        if self.learn_knots:
            inc = (self.grid_range[1] - self.grid_range[0]) / (self.num_knots + 1)
            specs["knot_logits"] = ParamSpec(np.full(self.num_knots + 1, np.log(np.expm1(inc))))
        return specs

    def _eval(self, xe, order, p):
        xp = namespace(xe)
        grid = p["grid"]
        lo, hi = grid[..., :1], grid[..., -1:]
        if self.learn_knots:
            cum = xp.cumsum(softplus(p["knot_logits"]) + 1e-8, -1)
            t = lo + (hi - lo) * cum[..., :-1] / cum[..., -1:]
        else:
            t = grid[..., 1:-1]
        a = 2.0 / (hi - lo)
        s = a * (xe - lo) - 1
        polys = [horner(list(P.polyder(c, order) if order else c), s) * a**order for c in self._legendre]
        scale = 1.0 / xp.maximum(hi - t, 0 * t + 1e-6)
        v = relu((xe - t) * scale)
        k = self.degree
        if order > k:
            trunc = 0 * v
        elif order == k:
            trunc = falling(k, order) * scale**order * cast_like(v > 0, v)
        else:
            trunc = falling(k, order) * scale**order * v ** (k - order)
        return concat([*polys, trunc], -1)

    def config(self):
        return {
            "name": self.name,
            "degree": self.degree,
            "num_knots": self.num_knots,
            "grid_range": self.grid_range,
            "learn_knots": self.learn_knots,
        }


class ReLUKAN(_Gridded):
    """ReLU-KAN: R_i(x) = [relu(e_i - x) * relu(x - s_i)]^2 * 16 / (e_i - s_i)^4.

    Same layout as ``BSpline``: bump i spans knots t_i .. t_{i+order+1} of the
    extended grid and peaks at 1 on a uniform grid. Only additions, products
    and ReLU, so it is cheap on any accelerator.
    """

    name = "relu_kan"
    local = True

    def _eval(self, xe, order, p):
        t = self.knots(p)
        n, k = self.size, self.order
        s, e = t[..., :n], t[..., k + 1 : k + 1 + n]
        norm = 16.0 / (e - s) ** 4
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
            r = 6 * d2q * d2q + 0 * q
        else:
            r = 0 * q
        return r * mask * norm
