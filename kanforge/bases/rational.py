"""Rational activation P(x) / (1 + |Q(x)|), the GR-KAN basis of KAT.

A single trainable function per input (``size == 1``); in a layer with
``groups`` the coefficients are shared inside each group of inputs, and the
layer coefficients play the role of KAT's linear weight.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .._calculus import falling, horner, quotient
from .._xp import namespace
from .base import Basis, ParamSpec

TARGETS = {
    "identity": lambda x: x,
    "silu": lambda x: x / (1 + np.exp(-x)),
    "gelu": lambda x: 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2))),
    "tanh": np.tanh,
}


@lru_cache(maxsize=None)
def fit_rational(target: str, num_degree: int, den_degree: int, lim: float = 3.0):
    """Coefficients (a, b) of P/(1+|Q|) that best fit ``target`` on [-lim, lim].

    Linear least squares for P - f*Q = f gives the start, Levenberg–Marquardt
    then minimises the true residual. Exact for ``identity``.
    """
    f = TARGETS[target]
    x = np.linspace(-lim, lim, 2001)
    y = f(x)
    vp = x[:, None] ** np.arange(num_degree + 1)
    vq = x[:, None] ** np.arange(1, den_degree + 1)
    theta = np.linalg.lstsq(np.hstack([vp, -y[:, None] * vq]), y, rcond=None)[0]
    lam = 1e-3

    def residual(th):
        s = vq @ th[num_degree + 1 :]
        return (vp @ th[: num_degree + 1]) / (1 + np.abs(s)) - y, s

    r, s = residual(theta)
    for _ in range(200):
        den = 1 + np.abs(s)
        num = vp @ theta[: num_degree + 1]
        jac = np.hstack([vp / den[:, None], (-num / den**2 * np.sign(s))[:, None] * vq])
        g = jac.T @ r
        a = jac.T @ jac
        step = np.linalg.solve(a + lam * np.diag(np.diag(a) + 1e-12), -g)
        r_new, s_new = residual(theta + step)
        if r_new @ r_new < r @ r:
            theta, r, s, lam = theta + step, r_new, s_new, lam / 3
        else:
            lam *= 4
        if lam > 1e8:
            break
    return theta[: num_degree + 1].copy(), theta[num_degree + 1 :].copy()


class Rational(Basis):
    name = "rational"

    def __init__(self, num_degree: int = 5, den_degree: int = 4, init: str = "silu"):
        if num_degree < 0 or den_degree < 1:
            raise ValueError("num_degree must be >= 0 and den_degree >= 1")
        if init not in TARGETS:
            raise ValueError(f"init must be one of {sorted(TARGETS)}")
        self.num_degree = int(num_degree)
        self.den_degree = int(den_degree)
        self.init = init

    @property
    def size(self) -> int:
        return 1

    def param_specs(self):
        a, b = fit_rational(self.init, self.num_degree, self.den_degree)
        return {"numerator": ParamSpec(a), "denominator": ParamSpec(b)}

    def _eval(self, xe, order, p):
        xp = namespace(xe)
        a, b = p["numerator"], p["denominator"]

        def poly(coef, first: int, k: int):
            # k-th derivative of sum_{j >= first} coef[j - first] * x^j
            last = first + coef.shape[-1] - 1
            terms = []
            for e in range(last - k + 1):  # power of x after differentiation
                j = e + k
                terms.append(falling(j, k) * coef[..., j - first : j - first + 1] if j >= first else 0.0)
            return horner(terms, xe)

        num = [poly(a, 0, k) for k in range(order + 1)]
        s = [poly(b, 1, k) for k in range(order + 1)]
        sign = xp.sign(s[0])
        den = [1 + xp.abs(s[0])] + [sign * sk for sk in s[1:]]
        return quotient(num, den, order)

    def config(self):
        return {"name": self.name, "num_degree": self.num_degree, "den_degree": self.den_degree, "init": self.init}
