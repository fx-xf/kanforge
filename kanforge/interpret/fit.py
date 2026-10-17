"""Symbolic regression of one edge: fit c * f(a * x + b) + d and rank candidates.

For each library function a coarse grid over (a, b) followed by two zoom
rounds is searched; for every (a, b) the outer c, d are the exact
least-squares solution (variable projection), so the fit is deterministic
and needs no gradient. Ranking follows pykan:
score = weight_simple * complexity + (1 - weight_simple) * log2(1 + 1e-5 - R^2),
lower is better, plus a penalty for *cancelling* fits: a function used far
from where it bends, scaled up and offset back (38 cos(0.2 x) - 38 ~ x^2) also
reaches R^2 ~ 1. Its size |c| * max|f| / (half range of y) is ~1 for a real
match and large for a Taylor-like imitation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import library as lib


@dataclass(frozen=True)
class Suggestion:
    name: str
    r2: float
    complexity: int
    params: tuple[float, float, float, float]  # a, b, c, d
    score: float

    def expr(self, u):
        """SymPy expression of the edge for input expression ``u``."""
        a, b, c, d = self.params
        return c * lib.get(self.name).sympy(a * u + b) + d


def _outer_fit(z: np.ndarray, y: np.ndarray, sst: float):
    """Exact (c, d) and R^2 for every row of ``z`` ``(K, N)``; rows with
    non-finite or constant values get R^2 = -inf."""
    z = np.atleast_2d(z)
    ok = np.all(np.isfinite(z), axis=1)
    z = np.where(ok[:, None], z, 0.0)
    zm = z.mean(1)
    zc = z - zm[:, None]
    var = (zc * zc).sum(1)
    ok &= var > 1e-12 * np.maximum(1.0, (z * z).sum(1))
    c = (zc @ (y - y.mean())) / np.where(ok, var, 1.0)
    d = y.mean() - c * zm
    r = y[None, :] - (c[:, None] * z + d[:, None])
    r2 = np.where(ok, 1.0 - (r * r).sum(1) / sst, -np.inf)
    return c, d, r2


def fit_affine(x, y, fn: lib.SymbolicFunction, a_max: float = 10.0, b_max: float = 10.0, grid: int = 41):
    """Best (a, b, c, d) and R^2 of y ~ c * fn(a * x + b) + d."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    sst = float(((y - y.mean()) ** 2).sum())
    if sst < 1e-24 * max(1.0, float(y @ y)):  # constant edge
        return (1.0, 0.0, 0.0, float(y.mean())), 1.0
    if fn.name == "0":
        return (1.0, 0.0, 0.0, float(y.mean())), 0.0
    if not fn.affine_inside:
        c, d, r2 = _outer_fit(fn.fn(x), y, sst)
        if not np.isfinite(r2[0]):
            return (1.0, 0.0, 0.0, float(y.mean())), 0.0
        return (1.0, 0.0, float(c[0]), float(d[0])), float(r2[0])

    best = None
    a_grid = np.linspace(-a_max, a_max, grid)
    b_grid = np.linspace(-b_max, b_max, grid)
    for _ in range(3):  # coarse search, then two zooms around the best point
        with np.errstate(all="ignore"):
            for a in a_grid:
                if a == 0:
                    continue
                c, d, r2 = _outer_fit(fn.fn(a * x[None, :] + b_grid[:, None]), y, sst)
                k = int(np.argmax(r2))
                if np.isfinite(r2[k]) and (best is None or r2[k] > best[4]):
                    best = (a, b_grid[k], c[k], d[k], r2[k])
        if best is None:
            break
        da, db = (a_grid[1] - a_grid[0]), (b_grid[1] - b_grid[0])
        a_grid = np.linspace(best[0] - da, best[0] + da, 21)
        b_grid = np.linspace(best[1] - db, best[1] + db, 21)
    if best is None:
        return (1.0, 0.0, 0.0, float(y.mean())), 0.0
    a, b, c, d, r2 = best
    return _canonical(fn, float(a), float(b), float(c), float(d)), float(r2)


def _canonical(fn: lib.SymbolicFunction, a: float, b: float, c: float, d: float):
    """Same function, readable parameters: a = 1 for powers, b = 0 for exp."""
    if fn.degree is not None:
        return 1.0, b / a, c * a**fn.degree, d
    if fn.name == "exp":
        return a, 0.0, c * float(np.exp(b)), d
    return a, b, c, d


def suggest_symbolic(x, y, functions=None, weight_simple: float = 0.8, topk: int = 5) -> list[Suggestion]:
    """Candidate formulas for samples (x, y) of one edge, best first."""
    names = lib.NAMES if functions is None else list(functions)
    out = []
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    half_range = max(0.5 * (y.max() - y.min()), 1e-12)
    for name in names:
        fn = lib.get(name)
        params, r2 = fit_affine(x, y, fn)
        score = weight_simple * fn.complexity + (1 - weight_simple) * np.log2(1 + 1e-5 - min(r2, 1.0))
        a, b, c, _ = params
        with np.errstate(all="ignore"):
            size = abs(c) * float(np.nanmax(np.abs(fn.fn(a * x + b)))) / half_range if c else 0.0
        if np.isfinite(size) and size > 10:
            score += weight_simple * np.log2(size / 10)
        out.append(Suggestion(name, r2, fn.complexity, params, float(score)))
    out.sort(key=lambda s: s.score)
    return out[:topk]
