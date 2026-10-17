"""PyTorch side of the fused kernels: launchers and differentiable primitives.

MPS tensors run the Metal kernels, CUDA/ROCm tensors the Triton kernels;
with ``TRITON_INTERPRET=1`` the Triton kernels also run on CPU tensors
(used by the tests). See ``kanforge.kernels`` for the math.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch

from ..kernels import metal
from ..kernels.plan import KernelPlan


def available(device: torch.device) -> bool:
    if device.type == "mps":
        return hasattr(torch.mps, "compile_shader")
    if device.type == "cuda" or (device.type == "cpu" and os.environ.get("TRITON_INTERPRET") == "1"):
        try:
            import triton  # noqa: F401
        except ImportError:
            return False
        return True
    return False


# --- raw launches (no autograd) ---------------------------------------------


@lru_cache(maxsize=256)
def _metal_lib(plan: KernelPlan, m: int, n_in: int, n_out: int, nf: int, chunk: int):
    return torch.mps.compile_shader(metal.torch_source(plan, m, n_in, n_out, nf, chunk))


def _raw(op: str, plan: KernelPlan, m: int, x, a, b, grid, n_out: int, nf: int):
    """op F: (x, w, cT) -> y; G: (x, w, g) -> GT; P: (x, g, cT) -> p."""
    batch, n_in = x.shape
    if x.device.type == "mps":
        lib = _metal_lib(plan, m, n_in, n_out, nf, metal.chunk_for(batch))
        threads, group = metal.launch_grid(op, batch, n_in, n_out)
        if op == "F":
            out = torch.empty(batch, n_out, device=x.device)
        elif op == "G":
            out = torch.empty(metal.n_chunks(batch), n_out, n_in, nf, device=x.device)
        else:
            out = torch.empty(batch, n_in, device=x.device)
        getattr(lib, f"kf_{op}")(out, x, a, b, grid, batch, threads=threads, group_size=group)
        return out.sum(0) if op == "G" else out
    from ..kernels import triton_kernels

    return triton_kernels.launch(op, plan, m, x, a, b, grid, n_out, nf)


def _prep(t):
    return t.contiguous().float()


def _fold(plan: KernelPlan, m: int, like: torch.Tensor):
    r = plan.fold(m)
    return None if r is None else torch.as_tensor(r, dtype=like.dtype, device=like.device)


# --- differentiable primitives -------------------------------------------


class _F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, c, w, grid, plan, m):
        ctx.save_for_backward(x, c, w, grid)
        ctx.plan, ctx.m = plan, m
        r = _fold(plan, m, c)
        ct = c if r is None else c @ r
        return _raw("F", plan, plan.native_order(m), _prep(x), _prep(w), _prep(ct), _prep(grid), c.shape[0], ct.shape[-1])

    @staticmethod
    def backward(ctx, gy):
        x, c, w, grid = ctx.saved_tensors
        plan, m = ctx.plan, ctx.m
        need = ctx.needs_input_grad
        gx = w * prim_P(x, gy, c, grid, plan, m + 1) if need[0] else None
        gc = prim_G(x, gy, w, grid, plan, m) if need[1] else None
        gw = prim_P(x, gy, c, grid, plan, m) if need[2] else None
        return gx, gc, gw, None, None, None


class _G(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g, w, grid, plan, m):
        ctx.save_for_backward(x, g, w, grid)
        ctx.plan, ctx.m = plan, m
        nf = plan.n_features(m)
        gt = _raw("G", plan, plan.native_order(m), _prep(x), _prep(w), _prep(g), _prep(grid), g.shape[1], nf)
        r = _fold(plan, m, gt)
        return gt if r is None else gt @ r.T

    @staticmethod
    def backward(ctx, gg_out):
        x, g, w, grid = ctx.saved_tensors
        plan, m = ctx.plan, ctx.m
        need = ctx.needs_input_grad
        gx = w * prim_P(x, g, gg_out, grid, plan, m + 1) if need[0] else None
        gg = prim_F(x, gg_out, w, grid, plan, m) if need[1] else None
        gw = prim_P(x, g, gg_out, grid, plan, m) if need[2] else None
        return gx, gg, gw, None, None, None


class _P(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g, c, grid, plan, m):
        ctx.save_for_backward(x, g, c, grid)
        ctx.plan, ctx.m = plan, m
        r = _fold(plan, m, c)
        ct = c if r is None else c @ r
        return _raw("P", plan, plan.native_order(m), _prep(x), _prep(g), _prep(ct), _prep(grid), c.shape[0], ct.shape[-1])

    @staticmethod
    def backward(ctx, gp):
        x, g, c, grid = ctx.saved_tensors
        plan, m = ctx.plan, ctx.m
        need = ctx.needs_input_grad
        gx = gp * prim_P(x, g, c, grid, plan, m + 1) if need[0] else None
        gg = prim_F(x, c, gp, grid, plan, m) if need[1] else None
        gc = prim_G(x, g, gp, grid, plan, m) if need[2] else None
        return gx, gg, gc, None, None, None


def prim_F(x, c, w, grid, plan, m=0):
    return _F.apply(x, c, w, grid, plan, m)


def prim_G(x, g, w, grid, plan, m=0):
    return _G.apply(x, g, w, grid, plan, m)


def prim_P(x, g, c, grid, plan, m=0):
    return _P.apply(x, g, c, grid, plan, m)


def spline_forward(x: torch.Tensor, coef: torch.Tensor, grid: torch.Tensor | None, plan: KernelPlan) -> torch.Tensor:
    """sum_{i,k} coef[o,i,k] phi_k(x[..., i]) through the fused kernel."""
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if grid is None:
        grid = x2.new_zeros(1)
    y = prim_F(x2.float(), coef.float(), torch.ones_like(x2, dtype=torch.float32), grid.float(), plan, 0)
    return y.to(x.dtype).reshape(*lead, coef.shape[0])
