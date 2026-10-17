"""Exact higher-order derivative machinery shared by all bases.

Polynomials with NumPy coefficients are prepared once in float64; they are
evaluated on backend arrays with Horner's rule. Compositions use Faà di
Bruno's formula, ratios the general quotient rule.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial import polynomial as P

from ._xp import sigmoid


def horner(coefs, x):
    """sum_j coefs[j] * x**j. Coefficients may be floats or backend arrays."""
    if len(coefs) == 0:
        return 0 * x
    # NumPy scalars would turn an MLX array into a NumPy one: use plain floats.
    coefs = [float(c) if isinstance(c, (np.generic, np.ndarray)) and np.ndim(c) == 0 else c for c in coefs]
    y = 0 * x + coefs[-1]
    for c in reversed(coefs[:-1]):
        y = y * x + c
    return y


def falling(n: int, k: int) -> int:
    """n * (n-1) * ... * (n-k+1)."""
    return math.perm(n, k) if 0 <= k <= n else 0


def chain_polys(start, factor, m: int) -> list[np.ndarray]:
    """[p_0 .. p_m] with p_0 = start and p_{k+1} = p_k' * factor.

    With factor = 1 - t^2 this differentiates functions of t = tanh(u); with
    factor = s - s^2 functions of s = sigmoid(u).
    """
    out = [np.asarray(start, dtype=np.float64)]
    for _ in range(m):
        out.append(P.polymul(P.polyder(out[-1]), factor))
    return out


TANH_FACTOR = np.array([1.0, 0.0, -1.0])
SIGMOID_FACTOR = np.array([0.0, 1.0, -1.0])


def faa_di_bruno(outer, inner, m: int):
    """m-th derivative of f(g(x)).

    ``outer[k]`` is f^(k) evaluated at g(x), ``inner[k]`` is g^(k)(x), k = 0..m.
    Uses partial Bell polynomials B_{n,k}(g', g'', ...).
    """
    if m == 0:
        return outer[0]
    bell = {(0, 0): 1}
    for k in range(1, m + 1):
        for n in range(k, m + 1):
            acc = None
            for i in range(1, n - k + 2):
                prev = bell.get((n - i, k - 1))
                if prev is None:
                    continue
                term = math.comb(n - 1, i - 1) * inner[i] * prev
                acc = term if acc is None else acc + term
            if acc is not None:
                bell[(n, k)] = acc
    result = None
    for k in range(1, m + 1):
        if (m, k) in bell:
            term = outer[k] * bell[(m, k)]
            result = term if result is None else result + term
    return result


def quotient(num, den, m: int):
    """m-th derivative of N/D from lists N^(0..m), D^(0..m)."""
    out = [num[0] / den[0]]
    for n in range(1, m + 1):
        acc = num[n]
        for k in range(1, n + 1):
            acc = acc - math.comb(n, k) * den[k] * out[n - k]
        out.append(acc / den[0])
    return out[m]


def silu_derivative(x, m: int):
    """d^m/dx^m of x * sigmoid(x)."""
    s = sigmoid(x)
    if m == 0:
        return x * s
    polys = chain_polys([0.0, 1.0], SIGMOID_FACTOR, m)
    return x * horner(polys[m], s) + m * horner(polys[m - 1], s)
