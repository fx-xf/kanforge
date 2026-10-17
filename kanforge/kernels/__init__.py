"""Fused KAN kernels: Metal (MLX, PyTorch MPS) and Triton (CUDA, ROCm).

The spline part of a layer, y[b, o] = sum_{i,k} c[o,i,k] phi_k(x[b,i]), is
computed without materialising the (batch, in, size) basis tensor. Three
primitives close under differentiation, so autograd works to any order
(second derivatives for PINNs included):

  F_m(x, c, w)[b,o]   = sum_{i,k} w[b,i] c[o,i,k] phi_k^(m)(x[b,i])
  G_m(x, g, w)[o,i,k] = sum_b     w[b,i] g[b,o]   phi_k^(m)(x[b,i])
  P_m(x, g, c)[b,i]   = sum_{o,k} g[b,o] c[o,i,k] phi_k^(m)(x[b,i])

  dF/dc = G_m, dF/dw = P_m, dF/dx = w * P_{m+1}   (and alike for G and P)
"""

from .plan import KernelPlan, plan_for

__all__ = ["KernelPlan", "plan_for"]
