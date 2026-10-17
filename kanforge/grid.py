"""Grid tools, the same for every basis and backend (NumPy float64).

* ``adaptive_grid`` — pykan's data-driven grid: quantiles of the inputs mixed
  with a uniform grid over their range (``grid_eps`` of uniform).
* ``refit`` — move a layer to a new basis or grid without changing what it
  computes: tabulate the old edge functions on samples and solve a least
  squares problem per input in the new basis. Grid update, grid extension
  (refinement) and conversion between bases (e.g. B-spline -> Chebyshev)
  are all this one operation.
"""

from __future__ import annotations

import numpy as np

from .bases import Basis


def adaptive_grid(x: np.ndarray, n_points: int, margin: float = 0.0, grid_eps: float = 0.02) -> np.ndarray:
    """Per-input grids ``(in, n_points)`` from samples ``x`` of shape ``(N, in)``."""
    if not 0 < grid_eps <= 1:
        raise ValueError("grid_eps must be in (0, 1]: some uniform part keeps the grid increasing")
    xs = np.sort(np.asarray(x, dtype=np.float64), axis=0)
    idx = np.linspace(0, len(xs) - 1, n_points).round().astype(int)
    adaptive = xs[idx].T
    lo, hi = xs[0] - margin, xs[-1] + margin
    flat = hi - lo < 1e-8
    lo, hi = np.where(flat, lo - 0.5, lo), np.where(flat, hi + 0.5, hi)
    uniform = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, n_points)
    adaptive = np.where(flat[:, None], uniform, adaptive)
    return grid_eps * uniform + (1 - grid_eps) * adaptive


def resample_grid(grid: np.ndarray, n_points: int) -> np.ndarray:
    """Interpolate grids ``(in, n)`` to ``n_points`` points, keeping their shape."""
    old = np.linspace(0, 1, grid.shape[-1])
    new = np.linspace(0, 1, n_points)
    return np.stack([np.interp(new, old, row) for row in grid])


def grid_ranges(basis: Basis, params: dict) -> np.ndarray | None:
    """Per-input ``(lo, hi)`` of the basis grid, ``(in, 2)``, or None without a grid."""
    names = basis.grid_names()
    if not names:
        return None
    g = params[names[0]]
    return np.stack([g[..., 0], g[..., -1]], -1)


def sample_inputs(ranges: np.ndarray, n: int) -> np.ndarray:
    """``n`` uniform points per input inside ``ranges`` ``(in, 2)``: ``(n, in)``."""
    t = np.linspace(0, 1, n)[:, None]
    return ranges[:, 0] + t * (ranges[:, 1] - ranges[:, 0])


def refit(
    old_basis: Basis,
    old_params: dict,
    old_coef: np.ndarray,
    new_basis: Basis,
    new_params: dict,
    x: np.ndarray,
    ridge: float = 0.0,
) -> np.ndarray:
    """Coefficients ``(out, in, new_basis.size)`` reproducing the old edge functions.

    ``x`` ``(N, in)`` are the points where they must match; params are
    expanded per input. Solved per input with ``lstsq`` (minimum-norm when a
    basis function has no samples), so a new basis that contains the old one
    reproduces it to rounding error. ``ridge`` > 0 adds Tikhonov damping,
    relative to the mean squared basis value.
    """
    x = np.asarray(x, dtype=np.float64)
    old = old_basis(x, old_params)  # (N, in, n)
    target = np.einsum("nib,oib->ino", old, old_coef)  # (in, N, out)
    b = np.moveaxis(new_basis(x, new_params), 1, 0)  # (in, N, n')
    sol = []
    for a, y in zip(b, target):
        if ridge:
            lam = np.sqrt(ridge * np.mean(a * a) * len(a))
            a = np.vstack([a, lam * np.eye(a.shape[1])])
            y = np.vstack([y, np.zeros((a.shape[1], y.shape[1]))])
        sol.append(np.linalg.lstsq(a, y, rcond=None)[0])
    return np.moveaxis(np.stack(sol), 2, 0)  # (out, in, n')
