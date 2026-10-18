"""Pallas kernels for KAN layers: the F / G / P primitives of ``kanforge.kernels``.

Each primitive is a ``jax.custom_vjp`` whose backward pass is written with
the other primitives, so ``jax.grad`` nests to any order (PINN residuals).
Plans, folding of polynomial / Fourier derivatives and the feature formulas
are those of the Metal and Triton kernels. Shapes are padded to powers of two
(Pallas' Triton lowering needs them); padding carries zero weights.

Runs on GPU (Pallas -> Triton) and, with ``KANFORGE_PALLAS_INTERPRET=1``,
on any backend in Pallas' interpreter (used by the tests).
"""

from __future__ import annotations

import math
import os
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from ..kernels.plan import KernelPlan

BB = 32  # batch rows per block
BO = 32  # outputs per block
# IEEE float32 products: JAX would use TF32 on Ampere and newer GPUs by default.
HIGHEST = jax.lax.Precision.HIGHEST


def interpret_mode() -> bool:
    return os.environ.get("KANFORGE_PALLAS_INTERPRET") == "1"


def available() -> bool:
    return interpret_mode() or jax.default_backend() == "gpu"


def _pow2(n: int, floor: int = 1) -> int:
    return max(floor, 1 << (max(n, 1) - 1).bit_length())


def _ceil(n: int, m: int) -> int:
    return (n + m - 1) // m * m


# --- features: (BB, 1) inputs -> (BB, NP) kernel features ------------------------------


def _tanh(x):
    t = jnp.exp(-2.0 * jnp.abs(x))
    r = (1.0 - t) / (1.0 + t)
    return jnp.where(x >= 0, r, -r)


def _features(plan: KernelPlan, m: int, nf: int, npad: int, xv, g_ref, i):
    """Feature columns of one input ``i`` for a block of rows ``xv`` (BB, 1)."""
    cols = []
    if plan.kind == "cheb":
        _, input_map = plan.consts
        z = _tanh(xv) if input_map == "tanh" else xv
        cols = [xv * 0.0 + 1.0]
        if nf > 1:
            cols.append(z)
        for _ in range(2, nf):
            cols.append(2.0 * z * cols[-1] - cols[-2])
    elif plan.kind == "fourier":
        freqs, period = plan.consts
        w0 = 2 * math.pi / period
        angles = [(j + 1) * w0 * xv for j in range(freqs)]
        cols = [jnp.cos(a) for a in angles] + [jnp.sin(a) for a in angles]
    elif plan.kind == "bspline":
        g, k = plan.consts
        g0, gl = g_ref[i, 0], g_ref[i, g]
        h = (gl - g0) / g
        t = []
        for r in range(g + 2 * k + 1):
            if r < k:
                t.append(g0 - (k - r) * h)
            elif r <= k + g:
                t.append(g_ref[i, r - k])
            else:
                t.append(gl + (r - k - g) * h)
        if m > k:
            cols = [xv * 0.0] * plan.size
        else:
            b = [jnp.where((xv >= t[j]) & (xv < t[j + 1]), 1.0, 0.0) for j in range(len(t) - 1)]
            for q in range(1, k - m + 1):
                b = [
                    (xv - t[j]) / (t[j + q] - t[j]) * b[j] + (t[j + q + 1] - xv) / (t[j + q + 1] - t[j + 1]) * b[j + 1]
                    for j in range(len(b) - 1)
                ]
            for q in range(k - m + 1, k + 1):
                b = [q * (b[j] / (t[j + q] - t[j]) - b[j + 1] / (t[j + q + 1] - t[j + 1])) for j in range(len(b) - 1)]
            cols = b
    elif plan.kind == "bump":
        mode, count, u_scale, norm, _ = plan.consts
        poly = [float(v) for v in plan.bump_poly(m)]  # weak-typed: stay float32 under x64
        c0, cl = g_ref[i, 0], g_ref[i, count - 1]
        s = u_scale * (count - 1) / (cl - c0)
        sm = s**m
        for j in range(count):
            u = (xv - g_ref[i, j]) * s
            tt = _tanh(u) if mode == "tanh" else u
            p = poly[-1] + 0.0 * tt
            for c in reversed(poly[:-1]):
                p = p * tt + c
            env = jnp.exp(-0.5 * u * u) if mode == "gauss" else 1.0
            cols.append(norm * p * env * sm)
    else:
        raise ValueError(plan.kind)
    iota = jax.lax.broadcasted_iota(jnp.int32, (xv.shape[0], npad), 1)
    phi = jnp.zeros((xv.shape[0], npad), jnp.float32)
    for j, col in enumerate(cols):
        phi = jnp.where(iota == j, jnp.asarray(col, jnp.float32), phi)
    return phi


# --- kernels ------------------------------------------------------------------------


def _kernel_F(x_ref, w_ref, c_ref, g_ref, y_ref, *, plan, m, nf, npad, n_in):
    def body(i, acc):
        xv = x_ref[:, pl.ds(i, 1)]
        wv = w_ref[:, pl.ds(i, 1)]
        phi = _features(plan, m, nf, npad, xv, g_ref, i) * wv
        return acc + jnp.dot(phi, c_ref[pl.ds(i * npad, npad), :], preferred_element_type=jnp.float32, precision=HIGHEST)

    y_ref[...] = jax.lax.fori_loop(0, n_in, body, jnp.zeros(y_ref.shape, jnp.float32))


def _kernel_G(x_ref, w_ref, gy_ref, g_ref, out_ref, *, plan, m, nf, npad, n_rows):
    i = pl.program_id(0)

    def body(bk, acc):
        rows = pl.ds(bk * BB, BB)
        xv = x_ref[rows, pl.ds(i, 1)]
        wv = w_ref[rows, pl.ds(i, 1)]
        phi = _features(plan, m, nf, npad, xv, g_ref, i) * wv
        return acc + jnp.dot(gy_ref[rows, :].T, phi, preferred_element_type=jnp.float32, precision=HIGHEST)

    out_ref[...] = jax.lax.fori_loop(0, n_rows // BB, body, jnp.zeros(out_ref.shape, jnp.float32))


def _kernel_P(x_ref, gy_ref, c_ref, g_ref, p_ref, *, plan, m, nf, npad, n_out):
    i = pl.program_id(1)
    xv = x_ref[:, pl.ds(i, 1)]
    phi = _features(plan, m, nf, npad, xv, g_ref, i)

    def body(ok, h):
        cols = pl.ds(ok * BO, BO)
        cs = c_ref[pl.ds(i * npad, npad), cols]  # (NP, BO)
        return h + jnp.dot(gy_ref[:, cols], cs.T, preferred_element_type=jnp.float32, precision=HIGHEST)

    h = jax.lax.fori_loop(0, n_out // BO, body, jnp.zeros((BB, npad), jnp.float32))
    p_ref[...] = jnp.sum((h * phi).T, axis=0, keepdims=True)


# --- padded launches ------------------------------------------------------------------


def _pad(a, shape):
    return jnp.pad(a, [(0, s - d) for d, s in zip(a.shape, shape)])


def _grid_arg(grid, n_in_p):
    if grid is None:
        return jnp.zeros((n_in_p, 1), jnp.float32)
    gp = _pow2(grid.shape[1])
    g = jnp.pad(grid.astype(jnp.float32), [(0, n_in_p - grid.shape[0]), (0, gp - grid.shape[1])], mode="edge")
    return g


def _coef_layout(c, n_in_p, npad, n_out_p):
    """(OUT, IN, NF) -> (IN_p * NP, OUT_p) with zero padding."""
    out_f, in_f, nf = c.shape
    cp = _pad(c.astype(jnp.float32), (n_out_p, n_in_p, npad))
    return cp.transpose(1, 2, 0).reshape(n_in_p * npad, n_out_p)


def _raw_F(plan, m, x, w, ct, grid):
    b, n_in = x.shape
    n_out, _, nf = ct.shape
    npad, n_in_p, b_p, n_out_p = _pow2(nf, 16), _pow2(n_in), _ceil(b, BB), _ceil(n_out, BO)
    xp, wp = _pad(x.astype(jnp.float32), (b_p, n_in_p)), _pad(w.astype(jnp.float32), (b_p, n_in_p))
    cp, gp = _coef_layout(ct, n_in_p, npad, n_out_p), _grid_arg(grid, n_in_p)
    y = pl.pallas_call(
        partial(_kernel_F, plan=plan, m=m, nf=nf, npad=npad, n_in=n_in),
        out_shape=jax.ShapeDtypeStruct((b_p, n_out_p), jnp.float32),
        grid=(b_p // BB, n_out_p // BO),
        in_specs=[
            pl.BlockSpec((BB, n_in_p), lambda bi, oi: (bi, 0)),
            pl.BlockSpec((BB, n_in_p), lambda bi, oi: (bi, 0)),
            pl.BlockSpec((n_in_p * npad, BO), lambda bi, oi: (0, oi)),
            pl.BlockSpec(gp.shape, lambda bi, oi: (0, 0)),
        ],
        out_specs=pl.BlockSpec((BB, BO), lambda bi, oi: (bi, oi)),
        interpret=interpret_mode(),
    )(xp, wp, cp, gp)
    return y[:b, :n_out]


def _raw_G(plan, m, x, w, gy, grid, nf):
    b, n_in = x.shape
    n_out = gy.shape[1]
    npad, n_in_p, b_p, n_out_p = _pow2(nf, 16), _pow2(n_in), _ceil(b, BB), _ceil(n_out, BO)
    xp, wp = _pad(x.astype(jnp.float32), (b_p, n_in_p)), _pad(w.astype(jnp.float32), (b_p, n_in_p))
    gyp, gp = _pad(gy.astype(jnp.float32), (b_p, n_out_p)), _grid_arg(grid, n_in_p)
    tiles = n_out_p // BO
    out = pl.pallas_call(
        partial(_kernel_G, plan=plan, m=m, nf=nf, npad=npad, n_rows=b_p),
        out_shape=jax.ShapeDtypeStruct((n_in_p * n_out_p, npad), jnp.float32),
        grid=(n_in, tiles),
        in_specs=[
            pl.BlockSpec((b_p, n_in_p), lambda i, oi: (0, 0)),
            pl.BlockSpec((b_p, n_in_p), lambda i, oi: (0, 0)),
            pl.BlockSpec((b_p, BO), lambda i, oi: (0, oi)),
            pl.BlockSpec(gp.shape, lambda i, oi: (0, 0)),
        ],
        out_specs=pl.BlockSpec((BO, npad), lambda i, oi: (i * tiles + oi, 0)),
        interpret=interpret_mode(),
    )(xp, wp, gyp, gp)
    return out.reshape(n_in_p, n_out_p, npad)[:n_in, :n_out, :nf].transpose(1, 0, 2)


def _raw_P(plan, m, x, gy, ct, grid):
    b, n_in = x.shape
    n_out, _, nf = ct.shape
    npad, n_in_p, b_p, n_out_p = _pow2(nf, 16), _pow2(n_in), _ceil(b, BB), _ceil(n_out, BO)
    xp, gyp = _pad(x.astype(jnp.float32), (b_p, n_in_p)), _pad(gy.astype(jnp.float32), (b_p, n_out_p))
    cp, gp = _coef_layout(ct, n_in_p, npad, n_out_p), _grid_arg(grid, n_in_p)
    out = pl.pallas_call(
        partial(_kernel_P, plan=plan, m=m, nf=nf, npad=npad, n_out=n_out_p),
        out_shape=jax.ShapeDtypeStruct((n_in_p, b_p), jnp.float32),
        grid=(b_p // BB, n_in),
        in_specs=[
            pl.BlockSpec((BB, n_in_p), lambda bi, i: (bi, 0)),
            pl.BlockSpec((BB, n_out_p), lambda bi, i: (bi, 0)),
            pl.BlockSpec(cp.shape, lambda bi, i: (0, 0)),
            pl.BlockSpec(gp.shape, lambda bi, i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((1, BB), lambda bi, i: (i, bi)),
        interpret=interpret_mode(),
    )(xp, gyp, cp, gp)
    return out[:n_in, :b].T


# --- differentiable primitives ----------------------------------------------------------


def _fold(plan, m, a, transpose=False):
    r = plan.fold(m)
    if r is None:
        return a
    r = jnp.asarray(r, jnp.float32)
    return jnp.matmul(a, r.T if transpose else r, precision=HIGHEST)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def prim_F(x, c, w, grid, plan, m=0):
    return _raw_F(plan, plan.native_order(m), x, w, _fold(plan, m, c), grid)


def _F_fwd(x, c, w, grid, plan, m):
    return prim_F(x, c, w, grid, plan, m), (x, c, w, grid)


def _F_bwd(plan, m, res, gy):
    x, c, w, grid = res
    return w * prim_P(x, gy, c, grid, plan, m + 1), prim_G(x, gy, w, grid, plan, m), prim_P(x, gy, c, grid, plan, m), jnp.zeros_like(grid)


prim_F.defvjp(_F_fwd, _F_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def prim_G(x, g, w, grid, plan, m=0):
    gt = _raw_G(plan, plan.native_order(m), x, w, g, grid, plan.n_features(m))
    return _fold(plan, m, gt, transpose=True)


def _G_fwd(x, g, w, grid, plan, m):
    return prim_G(x, g, w, grid, plan, m), (x, g, w, grid)


def _G_bwd(plan, m, res, gg):
    x, g, w, grid = res
    return w * prim_P(x, g, gg, grid, plan, m + 1), prim_F(x, gg, w, grid, plan, m), prim_P(x, g, gg, grid, plan, m), jnp.zeros_like(grid)


prim_G.defvjp(_G_fwd, _G_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def prim_P(x, g, c, grid, plan, m=0):
    return _raw_P(plan, plan.native_order(m), x, g, _fold(plan, m, c), grid)


def _P_fwd(x, g, c, grid, plan, m):
    return prim_P(x, g, c, grid, plan, m), (x, g, c, grid)


def _P_bwd(plan, m, res, gp):
    x, g, c, grid = res
    return gp * prim_P(x, g, c, grid, plan, m + 1), prim_F(x, c, gp, grid, plan, m), prim_G(x, g, gp, grid, plan, m), jnp.zeros_like(grid)


prim_P.defvjp(_P_fwd, _P_bwd)


def spline_forward(x, coef, grid, plan: KernelPlan):
    """sum_{i,k} coef[o,i,k] phi_k(x[..., i]) through the Pallas kernel."""
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    g = jnp.zeros((1, 1), jnp.float32) if grid is None else grid
    y = prim_F(x2, coef, jnp.ones_like(x2), g, plan, 0)
    return y.astype(x.dtype).reshape(*lead, coef.shape[0])
