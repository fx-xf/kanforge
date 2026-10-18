"""Polynomial bases: Chebyshev, Legendre, Jacobi, Gegenbauer, Hermite,
Laguerre, Bernstein, Gram, Taylor and fractional Jacobi (fKAN).

Every family is stored as a matrix A of Chebyshev coefficients, so values
are always computed with the stable Chebyshev recurrence:
B(z) = [T_0(z) .. T_D(z)] @ A. Derivatives in z are ``chebder(A)``; the input
map z = g(x) (tanh, identity, fractional power) is applied with Faà di Bruno.
For fused kernels A can be folded into the layer coefficients.
"""

from __future__ import annotations

import math
from functools import cached_property
from typing import Any

import numpy as np
from numpy.polynomial import chebyshev as C
from numpy.polynomial import hermite as H
from numpy.polynomial import laguerre as La
from numpy.polynomial import legendre as Le

from .._calculus import TANH_FACTOR, chain_polys, faa_di_bruno, horner
from .._xp import asarray_like, concat, log_sigmoid, namespace, sigmoid
from .base import Basis, ParamSpec

INPUT_MAPS = ("tanh", "identity", "fractional")


class PolynomialBasis(Basis):
    """Base class: subclasses define ``_family(z)`` on NumPy float64 points."""

    default_input_map = "tanh"

    def __init__(self, degree: int = 5, input_map: str | None = None):
        if degree < 0:
            raise ValueError("degree must be >= 0")
        input_map = input_map or self.default_input_map
        if input_map not in INPUT_MAPS:
            raise ValueError(f"input_map must be one of {INPUT_MAPS}")
        self.degree = int(degree)
        self.input_map = input_map

    @property
    def size(self) -> int:
        return self.degree + 1

    def _family(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @cached_property
    def _cheb(self) -> np.ndarray:
        """Chebyshev coefficients of every basis polynomial, ``(D+1, size)``."""
        d = self.degree
        nodes = np.cos(np.pi * (np.arange(d + 1) + 0.5) / (d + 1))
        return np.linalg.solve(C.chebvander(nodes, d), self._family(nodes))

    def _cheb_der(self, k: int) -> np.ndarray:
        return self._cheb if k == 0 else C.chebder(self._cheb, k, axis=0)

    def param_specs(self):
        if self.input_map == "fractional":
            return {"exponent_logit": ParamSpec(np.zeros(1))}
        return {}

    def _map(self, xe, order, p):
        """[g, g', ..., g^(order)] for the input map z = g(x)."""
        xp = namespace(xe)
        if self.input_map == "identity":
            return [xe, 0 * xe + 1] + [0] * (order - 1) if order else [xe]
        if self.input_map == "tanh":
            z = xp.tanh(xe)
            return [z] + [horner(list(q), z) for q in chain_polys([0.0, 1.0], TANH_FACTOR, order)[1:]]
        # fKAN: z = 2 * sigmoid(x)^gamma - 1 with gamma = sigmoid(logit) in (0, 1).
        # d/dx [s^g * q(s)] = s^g * (1 - s) * (g*q + s*q'), s = sigmoid(x).
        gamma = sigmoid(p["exponent_logit"])
        s = sigmoid(xe)
        h = xp.exp(gamma * log_sigmoid(xe))
        out = [2 * h - 1]
        q = [1.0]
        for _ in range(order):
            r = [(gamma + j) * c for j, c in enumerate(q)]
            q = [r[0]] + [r[j] - r[j - 1] for j in range(1, len(r))] + [-r[-1]]
            out.append(2 * h * horner(q, s))
        return out

    def _eval(self, xe, order, p):
        xp = namespace(xe)
        g = self._map(xe, order, p)
        z = g[0]
        ts = [0 * z + 1]
        if self.degree >= 1:
            ts.append(z)
        for _ in range(2, self.degree + 1):
            ts.append(2 * z * ts[-1] - ts[-2])
        t = concat(ts, -1)
        outer = []
        for k in range(order + 1):
            a = self._cheb_der(k)
            if a.shape[0] == 0:
                outer.append(0 * z)
            elif k == 0 and self._is_chebyshev:
                outer.append(t)
            else:
                outer.append(t[..., : a.shape[0]] @ asarray_like(a, xe))
        return faa_di_bruno(outer, g, order)

    _is_chebyshev = False

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "degree": self.degree, "input_map": self.input_map}


class Chebyshev(PolynomialBasis):
    """Chebyshev polynomials of the first kind T_n(tanh x), as in ChebyKAN."""

    name = "chebyshev"
    _is_chebyshev = True

    def _family(self, z):
        return C.chebvander(z, self.degree)


class Chebyshev2(PolynomialBasis):
    """Chebyshev polynomials of the second kind U_n."""

    name = "chebyshev2"

    def _family(self, z):
        u = [np.ones_like(z), 2 * z]
        for _ in range(2, self.degree + 1):
            u.append(2 * z * u[-1] - u[-2])
        return np.stack(u[: self.degree + 1], -1)


class Legendre(PolynomialBasis):
    name = "legendre"

    def _family(self, z):
        return Le.legvander(z, self.degree)


class Jacobi(PolynomialBasis):
    """Jacobi polynomials P_n^(alpha, beta), as in JacobiKAN."""

    name = "jacobi"

    def __init__(self, degree: int = 5, alpha: float = 1.0, beta: float = 1.0, input_map: str | None = None):
        if alpha <= -1 or beta <= -1:
            raise ValueError("alpha and beta must be > -1")
        super().__init__(degree, input_map)
        self.alpha, self.beta = float(alpha), float(beta)

    def _family(self, z):
        a, b = self.alpha, self.beta
        out = [np.ones_like(z), (a + 1) + (a + b + 2) * (z - 1) / 2]
        for n in range(2, self.degree + 1):
            c = 2 * n + a + b
            num = (c - 1) * (c * (c - 2) * z + a * a - b * b) * out[-1] - 2 * (n + a - 1) * (n + b - 1) * c * out[-2]
            out.append(num / (2 * n * (n + a + b) * (c - 2)))
        return np.stack(out[: self.degree + 1], -1)

    def config(self):
        return {**super().config(), "alpha": self.alpha, "beta": self.beta}


class FractionalJacobi(Jacobi):
    """fKAN: Jacobi polynomials of 2 * sigmoid(x)^gamma - 1 with trainable gamma in (0, 1)."""

    name = "fractional_jacobi"
    default_input_map = "fractional"

    def __init__(self, degree: int = 5, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(degree, alpha, beta, "fractional")

    def config(self):
        cfg = super().config()
        del cfg["input_map"]
        return cfg


class Gegenbauer(PolynomialBasis):
    """Gegenbauer (ultraspherical) polynomials C_n^(lam)."""

    name = "gegenbauer"

    def __init__(self, degree: int = 5, lam: float = 1.0, input_map: str | None = None):
        if lam <= -0.5 or lam == 0:
            raise ValueError("lam must be > -1/2 and non-zero")
        super().__init__(degree, input_map)
        self.lam = float(lam)

    def _family(self, z):
        lam = self.lam
        out = [np.ones_like(z), 2 * lam * z]
        for n in range(2, self.degree + 1):
            out.append((2 * z * (n + lam - 1) * out[-1] - (n + 2 * lam - 2) * out[-2]) / n)
        return np.stack(out[: self.degree + 1], -1)

    def config(self):
        return {**super().config(), "lam": self.lam}


class Hermite(PolynomialBasis):
    """Physicists' Hermite polynomials H_n."""

    name = "hermite"

    def _family(self, z):
        return H.hermvander(z, self.degree)


class Laguerre(PolynomialBasis):
    name = "laguerre"

    def _family(self, z):
        return La.lagvander(z, self.degree)


class Bernstein(PolynomialBasis):
    """Bernstein polynomials C(n, j) t^j (1 - t)^(n - j) of t = (z + 1) / 2."""

    name = "bernstein"

    def _family(self, z):
        t, n = (z + 1) / 2, self.degree
        return np.stack([math.comb(n, j) * t**j * (1 - t) ** (n - j) for j in range(n + 1)], -1)


class Gram(PolynomialBasis):
    """Gram polynomials: orthonormal on ``points`` equally spaced points of [-1, 1].

    Normalised so that the mean of each squared polynomial over the points is 1.
    """

    name = "gram"

    def __init__(self, degree: int = 5, points: int | None = None, input_map: str | None = None):
        super().__init__(degree, input_map)
        self.points = int(points) if points is not None else 2 * (self.degree + 1)
        if self.points <= self.degree:
            raise ValueError("points must be > degree")

    @cached_property
    def _cheb(self):
        grid = np.linspace(-1, 1, self.points)
        q, r = np.linalg.qr(C.chebvander(grid, self.degree))
        sign = np.sign(np.diag(r))
        return np.linalg.inv(r) * sign * math.sqrt(self.points)

    def config(self):
        return {**super().config(), "points": self.points}


class Taylor(PolynomialBasis):
    """Monomials 1, x, ..., x^degree (TaylorKAN); identity input map by default."""

    name = "taylor"
    default_input_map = "identity"

    def _family(self, z):
        return np.stack([z**j for j in range(self.degree + 1)], -1)
