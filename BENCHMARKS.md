# Benchmarks

`benchmarks/bench_kernels.py` — one `KANLinear` layer, median of 20 runs after
warm-up, float32. "fwd+bwd" is forward plus backward into all parameters (and
into the input for PyTorch). Plain ops = basis tensor + matrix product;
kernel = fused kernel of `kanforge.kernels`.

Machine: Apple M5, 16 GB, PyTorch 2.14 (MPS), MLX 0.32 — 18.10.2026.
Timings on this machine vary by ~±15 % between runs.

## PyTorch MPS

| basis | batch x in x out | fwd ops | fwd kernel | fwd+bwd ops | fwd+bwd kernel |
|---|---|---|---|---|---|
| bspline G8 k3 | 1024x64x64 | 3.57 ms | 0.90 ms | 6.69 ms | 2.04 ms |
| bspline G8 k3 | 8192x128x128 | 109.44 ms | 19.43 ms | 204.61 ms | 69.31 ms |
| bspline G8 k3 | 32768x64x64 | 185.66 ms | 18.71 ms | 361.65 ms | 44.76 ms |
| chebyshev d8 | 1024x64x64 | 0.73 ms | 0.88 ms | 1.36 ms | 1.77 ms |
| chebyshev d8 | 8192x128x128 | 11.89 ms | 13.65 ms | 26.50 ms | 29.96 ms |
| chebyshev d8 | 32768x64x64 | 26.20 ms | 14.00 ms | 56.71 ms | 25.93 ms |
| rbf 8 | 1024x64x64 | 1.18 ms | 0.48 ms | 1.99 ms | 0.92 ms |
| rbf 8 | 8192x128x128 | 17.87 ms | 6.12 ms | 38.37 ms | 17.71 ms |
| rbf 8 | 32768x64x64 | 32.54 ms | 7.98 ms | 71.94 ms | 17.59 ms |
| fourier 4 | 1024x64x64 | 0.85 ms | 0.78 ms | 1.32 ms | 1.52 ms |
| fourier 4 | 8192x128x128 | 10.23 ms | 6.58 ms | 22.69 ms | 19.09 ms |
| fourier 4 | 32768x64x64 | 20.34 ms | 7.96 ms | 41.50 ms | 18.09 ms |

## MLX (Metal)

| basis | batch x in x out | fwd ops | fwd kernel | fwd+bwd ops | fwd+bwd kernel |
|---|---|---|---|---|---|
| bspline G8 k3 | 1024x64x64 | 1.80 ms | 0.62 ms | 1.25 ms | 1.16 ms |
| bspline G8 k3 | 8192x128x128 | 30.31 ms | 17.49 ms | 35.14 ms | 29.86 ms |
| bspline G8 k3 | 32768x64x64 | 64.72 ms | 18.08 ms | 69.33 ms | 28.09 ms |
| chebyshev d8 | 1024x64x64 | 0.57 ms | 0.78 ms | 0.64 ms | 1.04 ms |
| chebyshev d8 | 8192x128x128 | 11.12 ms | 12.66 ms | 11.77 ms | 20.27 ms |
| chebyshev d8 | 32768x64x64 | 23.74 ms | 13.67 ms | 24.83 ms | 17.15 ms |
| rbf 8 | 1024x64x64 | 0.56 ms | 0.45 ms | 0.65 ms | 0.51 ms |
| rbf 8 | 8192x128x128 | 6.48 ms | 7.87 ms | 8.81 ms | 12.96 ms |
| rbf 8 | 32768x64x64 | 15.62 ms | 7.25 ms | 19.13 ms | 10.91 ms |
| fourier 4 | 1024x64x64 | 0.51 ms | 0.60 ms | 0.63 ms | 0.91 ms |
| fourier 4 | 8192x128x128 | 4.67 ms | 5.62 ms | 7.83 ms | 12.53 ms |
| fourier 4 | 32768x64x64 | 9.61 ms | 9.08 ms | 11.46 ms | 12.59 ms |

## PyTorch CUDA (Triton)

RTX 3060 12 GB, driver 535, PyTorch 2.6.0+cu124, Triton 3.2.0, idle GPU — 18.10.2026.
`tl.dot` runs with IEEE float32 (no TF32), like the plain-ops reference.

| basis | batch x in x out | fwd ops | fwd kernel | fwd+bwd ops | fwd+bwd kernel |
|---|---|---|---|---|---|
| bspline G8 k3 | 1024x64x64 | 0.74 ms | 0.50 ms | 1.72 ms | 0.94 ms |
| bspline G8 k3 | 8192x128x128 | 8.46 ms | 8.22 ms | 21.59 ms | 17.39 ms |
| bspline G8 k3 | 32768x64x64 | 16.46 ms | 8.24 ms | 41.50 ms | 18.87 ms |
| chebyshev d8 | 1024x64x64 | 0.26 ms | 0.33 ms | 0.91 ms | 0.86 ms |
| chebyshev d8 | 8192x128x128 | 2.63 ms | 4.33 ms | 6.46 ms | 8.78 ms |
| chebyshev d8 | 32768x64x64 | 5.02 ms | 4.44 ms | 11.89 ms | 9.62 ms |
| rbf 8 | 1024x64x64 | 0.24 ms | 0.27 ms | 0.72 ms | 0.63 ms |
| rbf 8 | 8192x128x128 | 2.64 ms | 3.91 ms | 7.07 ms | 8.38 ms |
| rbf 8 | 32768x64x64 | 4.96 ms | 4.01 ms | 12.89 ms | 9.23 ms |
| fourier 4 | 1024x64x64 | 0.17 ms | 0.40 ms | 0.59 ms | 0.98 ms |
| fourier 4 | 8192x128x128 | 1.41 ms | 4.40 ms | 3.76 ms | 10.64 ms |
| fourier 4 | 32768x64x64 | 2.53 ms | 4.46 ms | 6.30 ms | 11.57 ms |

## What `backend="auto"` does with this

`kanforge.kernels.plan.AUTO_KINDS`: the kernel only where it won both columns.

- PyTorch MPS: B-splines (3–9x on fwd+bwd) and bumps — RBF, Mexican hat, DoG, RSWAF (2–4x).
- MLX: B-splines (1.1–2.5x). RBF is mixed, Chebyshev and Fourier are slower:
  MLX's own fused graph is already fast for them.
- Chebyshev-family polynomials and Fourier: plain ops by default; on MPS the
  kernel is about even for small batches and 2x faster for 32k rows, choose
  it with `backend="kernel"`.
- CUDA (Triton): B-splines (1.2–2.2x on fwd+bwd). Chebyshev and RBF win only
  for large batches, Fourier is 2–3x slower than cuBLAS + fused cos/sin.
  Correctness: the full suite passes on the RTX 3060 (and in CI under the
  Triton interpreter). ROCm is untested.
