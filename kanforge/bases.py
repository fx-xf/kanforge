"""Basis functions for KAN edges.

Every edge function is phi(x) = sum_b c_b * B_b(x). A ``Basis`` only knows
how to evaluate B_b: it maps an array of shape ``(..., in)`` to
``(..., in, size)``. Layers on every backend contract that with their
coefficients, so a new basis works everywhere without new layer code.

``local`` marks bases with compact support (at most ``support`` non-zero
values per input); fused kernels use it to pick a sparse gather instead of
a dense contraction.
"""

from __future__ import annotations

import math
from typing import Any

from ._xp import arange_like, cast_like, namespace


class Basis:
    name: str = ""
    local: bool = False

    @property
    def size(self) -> int:
        raise NotImplementedError

    @property
    def support(self) -> int:
        """Max number of non-zero basis values per input."""
        return self.size

    def __call__(self, x):
        raise NotImplementedError

    def config(self) -> dict[str, Any]:
        raise NotImplementedError

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.config().items() if k != "name")
        return f"{type(self).__name__}({args})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Basis) and self.config() == other.config()


class BSpline(Basis):
    """Uniform B-splines of ``order`` on ``grid_size`` intervals over ``grid_range``.

    As in pykan, the knot vector is extended by ``order`` knots on each side,
    so there are ``grid_size + order`` basis functions and they sum to one
    everywhere inside ``grid_range``. Outside the extended grid all values
    are zero and the layer falls back to its base (SiLU) branch.
    """

    name = "bspline"
    local = True

    def __init__(self, grid_size: int = 5, order: int = 3, grid_range: tuple[float, float] = (-1.0, 1.0)):
        if grid_size < 1 or order < 0:
            raise ValueError("grid_size must be >= 1 and order >= 0")
        lo, hi = grid_range
        if not hi > lo:
            raise ValueError("grid_range must be increasing")
        self.grid_size = int(grid_size)
        self.order = int(order)
        self.grid_range = (float(lo), float(hi))

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

    def __call__(self, x):
        xp = namespace(x)
        k, h = self.order, self.step
        t = self.knots(x)
        xe = x[..., None]
        b = cast_like(xp.logical_and(xe >= t[:-1], xe < t[1:]), x)
        # Cox–de Boor recursion; on a uniform grid every denominator is p*h.
        for p in range(1, k + 1):
            b = ((xe - t[: -(p + 1)]) * b[..., :-1] + (t[p + 1 :] - xe) * b[..., 1:]) / (p * h)
        return b

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "grid_size": self.grid_size, "order": self.order, "grid_range": self.grid_range}


class Chebyshev(Basis):
    """Chebyshev polynomials T_0..T_degree of tanh(x), as in ChebyKAN.

    tanh squashes the input into (-1, 1), where the polynomials are bounded.
    The three-term recurrence is used instead of cos(n*acos(z)) so gradients
    stay finite near |z| = 1.
    """

    name = "chebyshev"

    def __init__(self, degree: int = 5):
        if degree < 0:
            raise ValueError("degree must be >= 0")
        self.degree = int(degree)

    @property
    def size(self) -> int:
        return self.degree + 1

    def __call__(self, x):
        xp = namespace(x)
        z = xp.tanh(x)
        ts = [z * 0 + 1]
        if self.degree >= 1:
            ts.append(z)
        for _ in range(2, self.degree + 1):
            ts.append(2 * z * ts[-1] - ts[-2])
        return xp.stack(ts, -1)

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "degree": self.degree}


class Fourier(Basis):
    """cos(w_m x) and sin(w_m x) for w_m = 2*pi*m/period, m = 1..frequencies.

    The constant term is left to the layer bias, as in FourierKAN.
    """

    name = "fourier"

    def __init__(self, frequencies: int = 5, period: float = 2 * math.pi):
        if frequencies < 1 or not period > 0:
            raise ValueError("frequencies must be >= 1 and period > 0")
        self.frequencies = int(frequencies)
        self.period = float(period)

    @property
    def size(self) -> int:
        return 2 * self.frequencies

    def __call__(self, x):
        xp = namespace(x)
        w = (arange_like(self.frequencies, x) + 1) * (2 * math.pi / self.period)
        a = x[..., None] * w
        return xp.concatenate([xp.cos(a), xp.sin(a)], -1)

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "frequencies": self.frequencies, "period": self.period}


BASES: dict[str, type[Basis]] = {cls.name: cls for cls in (BSpline, Chebyshev, Fourier)}


def make_basis(spec: str | dict | Basis = "bspline", **kwargs) -> Basis:
    """Build a basis from a name, a ``config()`` dict or an existing instance."""
    if isinstance(spec, Basis):
        if kwargs:
            raise TypeError("kwargs are not allowed with a Basis instance")
        return spec
    if isinstance(spec, dict):
        spec = dict(spec)
        name = spec.pop("name")
        if "grid_range" in spec:
            spec["grid_range"] = tuple(spec["grid_range"])
        return BASES[name](**spec, **kwargs)
    try:
        cls = BASES[spec]
    except KeyError:
        raise ValueError(f"unknown basis {spec!r}; available: {sorted(BASES)}") from None
    return cls(**kwargs)
