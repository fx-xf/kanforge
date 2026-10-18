# Fused kernels

The spline part of a layer can run as a fused kernel that never builds the
`(batch, in, size)` basis tensor. Three primitives close under
differentiation, so gradients of any order work (PINNs, gradient penalties):

    F_m(x, c, w)[b,o]   = Σ_{i,k} w[b,i] c[o,i,k] φ_k^(m)(x[b,i])
    G_m(x, g, w)[o,i,k] = Σ_b     w[b,i] g[b,o]   φ_k^(m)(x[b,i])
    P_m(x, g, c)[b,i]   = Σ_{o,k} g[b,o] c[o,i,k] φ_k^(m)(x[b,i])

Polynomial and Fourier derivatives are folded into the coefficients;
B-splines and bumps compile the derivative order into the kernel.

| platform | kernels | `backend="auto"` uses them for |
|---|---|---|
| PyTorch MPS | Metal | B-splines, bumps (RBF, wavelets, RSWAF) |
| MLX | Metal | B-splines |
| PyTorch CUDA | Triton (IEEE float32) | B-splines |
| JAX GPU | Pallas | never (XLA's plain path is faster) |

`backend="kernel"` forces them (error if the basis or device has none),
`backend="ops"` disables them. Bases with trainable parameters always use
plain ops. Numbers in [Benchmarks](../benchmarks.md).
