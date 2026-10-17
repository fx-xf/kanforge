"""What a fused kernel has to compute for a basis, independent of the backend.

A ``KernelPlan`` describes the per-input *kernel features* phi_j(x) that the
generated code evaluates in registers, and how derivatives are obtained:

* ``cheb`` and ``fourier`` are *folded*: the m-th x-derivative of every basis
  function is a fixed linear combination of the kernel features
  (``phi_k^(m) = sum_j R_m[k, j] T_j``), so kernels never differentiate and
  ``R_m`` is applied to the coefficients outside the kernel. All polynomial
  families with a tanh or identity input map reduce to Chebyshev features.
* ``bspline`` and ``bump`` are *native*: the derivative order is compiled
  into the kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.polynomial import chebyshev as C

from .._calculus import TANH_FACTOR, chain_polys
from ..bases import RBF, RSWAF, BSpline, DoG, Fourier, MexicanHat, PolynomialBasis
from ..bases.radial import _envelope_derivatives

ONE_MINUS_Z2 = np.array([0.5, 0.0, -0.5])  # 1 - z^2 in the Chebyshev basis

# Plan kinds that backend="auto" sends to the kernel, per platform, chosen from
# benchmarks/bench_kernels.py on Apple silicon (see BENCHMARKS.md): only where the
# kernel was faster than plain ops for forward and forward+backward. CUDA stays
# on plain ops until the Triton kernels are measured on a GPU.
AUTO_KINDS = {"mps": ("bspline", "bump"), "mlx": ("bspline",), "cuda": ()}


@dataclass(frozen=True)
class KernelPlan:
    kind: str  # "cheb" | "fourier" | "bspline" | "bump"
    consts: tuple  # static parameters baked into the generated code
    size: int  # number of basis functions of the original basis
    fold_key: tuple = ()  # hashable description of the fold matrices
    grid: bool = False  # the kernel reads the per-input grid buffer

    @property
    def folded(self) -> bool:
        return self.kind in ("cheb", "fourier")

    def n_features(self, m: int) -> int:
        """Kernel features evaluated for derivative order m."""
        if self.kind == "cheb":
            d, input_map = self.consts
            return d + 1 + m if input_map == "tanh" else d + 1
        return self.size

    def native_order(self, m: int) -> int:
        return 0 if self.folded else m

    def fold(self, m: int) -> np.ndarray | None:
        """``R_m`` ``(size, n_features(m))`` for folded plans, None otherwise."""
        if not self.folded:
            return None
        return _fold(self.kind, self.consts, self.fold_key, m)

    def bump_poly(self, m: int) -> tuple[float, ...]:
        """Polynomial (in u, or in tanh(u)) of the m-th derivative of a bump."""
        mode, _count, _u_scale, _norm, p = self.consts
        if mode == "tanh":
            return tuple(chain_polys(TANH_FACTOR, TANH_FACTOR, m)[m])
        return tuple(_envelope_derivatives(p, (0.0,), 0.0, m)[m][0])


@lru_cache(maxsize=None)
def _fold(kind, consts, fold_key, m):
    if kind == "fourier":
        freqs, period = consts
        w = 2 * math.pi / period * np.arange(1, freqs + 1)
        c, s = math.cos(m * math.pi / 2), math.sin(m * math.pi / 2)
        r = np.zeros((2 * freqs, 2 * freqs))
        idx = np.arange(freqs)
        wm = w**m
        # cos(wx)^(m) = w^m (cos(m pi/2) cos(wx) - sin(m pi/2) sin(wx)); sin likewise
        r[idx, idx], r[idx, freqs + idx] = wm * c, -wm * s
        r[freqs + idx, idx], r[freqs + idx, freqs + idx] = wm * s, wm * c
        return r
    d, input_map = consts
    a = np.asarray(fold_key, dtype=np.float64).reshape(d + 1, -1)  # Chebyshev coefficients
    n_t = d + 1 + m if input_map == "tanh" else d + 1
    cols = []
    for k in range(a.shape[1]):
        col = a[:, k]
        for _ in range(m):
            col = C.chebder(col) if len(col) > 1 else np.zeros(1)
            if input_map == "tanh":
                col = C.chebmul(col, ONE_MINUS_Z2)
        out = np.zeros(n_t)
        out[: min(len(col), n_t)] = col[:n_t]
        cols.append(out)
    return np.stack(cols)  # (size, n_t)


def plan_for(basis) -> KernelPlan | None:
    """The kernel plan of ``basis`` or None when it needs the plain-ops path
    (trainable basis parameters or a basis without a kernel yet)."""
    if any(s.trainable for s in basis.param_specs().values()):
        return None
    if isinstance(basis, PolynomialBasis) and basis.input_map in ("tanh", "identity"):
        a = basis._cheb
        return KernelPlan("cheb", (basis.degree, basis.input_map), basis.size, tuple(a.ravel().tolist()))
    if type(basis) is Fourier:
        return KernelPlan("fourier", (basis.frequencies, basis.period), basis.size)
    if type(basis) is BSpline:
        return KernelPlan("bspline", (basis.grid_size, basis.order), basis.size, grid=True)
    if type(basis) in (RBF, MexicanHat, DoG):
        consts = ("gauss", basis.count, basis.u_scale, basis.norm, tuple(basis.p))
        return KernelPlan("bump", consts, basis.size, grid=True)
    if type(basis) is RSWAF:
        return KernelPlan("bump", ("tanh", basis.count, 1.0, 1.0, (1.0, 0.0, -1.0)), basis.size, grid=True)
    return None
