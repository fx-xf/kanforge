"""Library of 1-D functions for symbolic edges: c * f(a * x + b) + d.

Every function is written with operations NumPy, PyTorch and MLX share, so
the same entry evaluates symbolic edges on any backend; ``sympy`` gives its
formula and ``complexity`` follows pykan's scale (used to prefer simpler
candidates). Codes stored in layers are positions in ``NAMES``: append new
functions at the end to keep saved models valid.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .._xp import namespace, sigmoid


@dataclass(frozen=True)
class SymbolicFunction:
    name: str
    fn: Callable  # backend array -> backend array
    sympy: Callable  # sympy expression -> sympy expression
    complexity: int
    affine_inside: bool = True  # False: a, b are redundant (fixed to 1, 0)
    degree: int | None = None  # f(a x) = a^degree f(x): fits are normalised to a = 1


def _sp():
    import sympy

    return sympy


def _entries() -> list[SymbolicFunction]:
    def xp(x):
        return namespace(x)

    return [
        SymbolicFunction("0", lambda x: 0 * x, lambda u: 0, 0, False),
        SymbolicFunction("x", lambda x: x, lambda u: u, 1, False),
        SymbolicFunction("x^2", lambda x: x * x, lambda u: u**2, 2, degree=2),
        SymbolicFunction("x^3", lambda x: x * x * x, lambda u: u**3, 3, degree=3),
        SymbolicFunction("x^4", lambda x: (x * x) * (x * x), lambda u: u**4, 3, degree=4),
        SymbolicFunction("1/x", lambda x: 1 / x, lambda u: 1 / u, 2, degree=-1),
        SymbolicFunction("1/x^2", lambda x: 1 / (x * x), lambda u: u**-2, 3, degree=-2),
        SymbolicFunction("sqrt", lambda x: xp(x).sqrt(x), lambda u: _sp().sqrt(u), 2),
        SymbolicFunction("1/sqrt", lambda x: 1 / xp(x).sqrt(x), lambda u: 1 / _sp().sqrt(u), 3),
        SymbolicFunction("exp", lambda x: xp(x).exp(x), lambda u: _sp().exp(u), 2),
        SymbolicFunction("log", lambda x: xp(x).log(x), lambda u: _sp().log(u), 2),
        SymbolicFunction("sin", lambda x: xp(x).sin(x), lambda u: _sp().sin(u), 2),
        SymbolicFunction("cos", lambda x: xp(x).cos(x), lambda u: _sp().cos(u), 2),
        SymbolicFunction("tanh", lambda x: xp(x).tanh(x), lambda u: _sp().tanh(u), 3),
        SymbolicFunction("sigmoid", sigmoid, lambda u: 1 / (1 + _sp().exp(-u)), 3),
        SymbolicFunction("gaussian", lambda x: xp(x).exp(-x * x), lambda u: _sp().exp(-(u**2)), 3),
        SymbolicFunction("abs", lambda x: xp(x).abs(x), lambda u: _sp().Abs(u), 3),
        SymbolicFunction("arctan", lambda x: xp(x).arctan(x), lambda u: _sp().atan(u), 4),
    ]


LIBRARY: list[SymbolicFunction] = _entries()
NAMES: list[str] = [f.name for f in LIBRARY]
CODES: dict[str, int] = {n: i for i, n in enumerate(NAMES)}
# A value inside every function's domain: substituted on edges that do not use
# a function, so unused branches never produce NaN (also in gradients).
SAFE_INPUT = 1.0


def get(name_or_code) -> SymbolicFunction:
    if isinstance(name_or_code, str):
        if name_or_code not in CODES:
            raise ValueError(f"unknown symbolic function {name_or_code!r}; available: {NAMES}")
        return LIBRARY[CODES[name_or_code]]
    return LIBRARY[int(name_or_code)]
