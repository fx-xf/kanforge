"""MLX side of the fused kernels: Metal launches and differentiable primitives.

``mx.custom_function`` gives every primitive a VJP written with the other
primitives, so gradients nest to any order (see ``kanforge.kernels``).
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

from ..kernels import metal
from ..kernels.plan import KernelPlan


@lru_cache(maxsize=256)
def _kernel(op: str, plan: KernelPlan, m: int, n_in: int, n_out: int, nf: int, chunk: int):
    ins, outs = metal.SIGNATURES[op]
    body = (
        "uint2 gid = uint2(thread_position_in_grid.xy);\n"
        "uint2 lid = uint2(thread_position_in_threadgroup.xy);\n"
        "uint2 tgid = uint2(threadgroup_position_in_grid.xy);\n"
    ) + metal.kernel_bodies(n_in, n_out, nf, chunk)[op]
    return mx.fast.metal_kernel(
        name=f"kf_{op}",
        input_names=list(ins),
        output_names=list(outs),
        source=body,
        header=metal.features_source(plan, m, nf),
    )


def _raw(op: str, plan: KernelPlan, m: int, x, a, b, grid, n_out: int, nf: int):
    batch, n_in = x.shape
    (tx, ty), (gx, gy) = metal.launch_grid(op, batch, n_in, n_out)
    shape = {"F": (batch, n_out), "G": (metal.n_chunks(batch), n_out, n_in, nf), "P": (batch, n_in)}[op]
    nb = mx.array([batch], dtype=mx.uint32)
    f32 = [v.astype(mx.float32) for v in (x, a, b, grid)]
    (out,) = _kernel(op, plan, m, n_in, n_out, nf, metal.chunk_for(batch))(
        inputs=[*f32, nb],
        grid=(tx, ty, 1),
        threadgroup=(gx, gy, 1),
        output_shapes=[shape],
        output_dtypes=[mx.float32],
    )
    return out.sum(0) if op == "G" else out


def _fold(plan: KernelPlan, m: int):
    r = plan.fold(m)
    return None if r is None else mx.array(r, dtype=mx.float32)


def _matmul(a, b):
    # Metal float32 matmul in MLX is off by ~1e-4 relative even for tiny
    # matrices; the fold matrices are small, so contract elementwise instead.
    return (a[..., :, None] * b).sum(-2)


@lru_cache(maxsize=None)
def _primitives(plan: KernelPlan, m: int):
    native = plan.native_order(m)
    r = _fold(plan, m)

    @mx.custom_function
    def F(x, c, w, grid):
        ct = c if r is None else _matmul(c, r)
        return _raw("F", plan, native, x, w, ct, grid, c.shape[0], ct.shape[-1])

    @F.vjp
    def F_vjp(primals, gy, _out):
        x, c, w, grid = primals
        up, same = _primitives(plan, m + 1), _primitives(plan, m)
        return w * up[2](x, gy, c, grid), same[1](x, gy, w, grid), same[2](x, gy, c, grid), mx.zeros_like(grid)

    @mx.custom_function
    def G(x, g, w, grid):
        gt = _raw("G", plan, native, x, w, g, grid, g.shape[1], plan.n_features(m))
        return gt if r is None else _matmul(gt, r.T)

    @G.vjp
    def G_vjp(primals, gg, _out):
        x, g, w, grid = primals
        up, same = _primitives(plan, m + 1), _primitives(plan, m)
        return w * up[2](x, g, gg, grid), same[0](x, gg, w, grid), same[2](x, g, gg, grid), mx.zeros_like(grid)

    @mx.custom_function
    def P(x, g, c, grid):
        ct = c if r is None else _matmul(c, r)
        return _raw("P", plan, native, x, g, ct, grid, c.shape[0], ct.shape[-1])

    @P.vjp
    def P_vjp(primals, gp, _out):
        x, g, c, grid = primals
        up, same = _primitives(plan, m + 1), _primitives(plan, m)
        return gp * up[2](x, g, c, grid), same[0](x, c, gp, grid), same[1](x, g, gp, grid), mx.zeros_like(grid)

    return F, G, P


def prim_F(x, c, w, grid, plan, m=0):
    return _primitives(plan, m)[0](x, c, w, grid)


def prim_G(x, g, w, grid, plan, m=0):
    return _primitives(plan, m)[1](x, g, w, grid)


def prim_P(x, g, c, grid, plan, m=0):
    return _primitives(plan, m)[2](x, g, c, grid)


def spline_forward(x, coef, grid, plan: KernelPlan):
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if grid is None:
        grid = mx.zeros((1,))
    y = prim_F(x2, coef, mx.ones(x2.shape), grid, plan, 0)
    return y.astype(x.dtype).reshape(*lead, coef.shape[0])
