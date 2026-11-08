# Roadmap

kanforge aims to be the KAN library you reach for whenever you would otherwise
use an MLP or search for a formula — on CUDA, ROCm, Apple silicon (MPS and
MLX), TPUs and CPUs.

## Principles

1. **One basis, everywhere.** A basis is written once and runs on NumPy,
   PyTorch, MLX and JAX. A new basis never needs a new layer.
2. **A float64 reference.** Every backend and every fused kernel is tested
   against `kanforge.reference`.
3. **Identical parameters on every backend** — `coef (out, in, n)`,
   `base_weight (out, in)`, `bias (out,)` — so weights move between frameworks
   without conversion code.
4. **Higher derivatives from day one.** Physics-informed training is the
   strongest niche of KANs; every kernel supports double backward.
5. **Interpretation is separate from computation.** Any trained layer can be
   tabulated, so pruning, symbolic fitting and plots work for every basis and
   backend.
6. **Honest benchmarks.** KANs are compared with MLPs at equal parameters and
   FLOPs, following the KANbeFair methodology.
7. **Clean licensing.** Code is only adapted from permissively licensed
   projects and credited in [NOTICE](NOTICE); everything else is written from
   the papers.

## Done — 0.8

| Phase | What shipped |
| --- | --- |
| 0 · Foundations | `Basis` contract, NumPy reference, `KANLinear` / `KAN` for PyTorch and MLX |
| 1 · Bases | 26 basis families with exact derivatives of any order, trainable basis parameters, GR-KAN, MultKAN |
| 2 · Training | per-input grids, `update_grid` / `refine_grid` / `change_basis`, regularisation, DropKAN, initialisation schemes, `from_mlp` |
| 3 · Kernels | fused Metal (MPS, MLX) and Triton kernels with derivatives of any order, benchmark-driven `backend="auto"` |
| 4 · Interpretation | tabulation, attribution, pruning, symbolic edges, `symbolic_formula`, `kanpile` |
| 5 · Blocks | KAN convolutions, KAT blocks, `kanify` for Hugging Face, graph layers, recurrent cells, PINN tools, DeepONet, FBKAN, KAN-ODE |
| 6 · JAX | Flax NNX backend with Pallas kernels |
| 7 · Deployment | weight conversion, lookup tables, int8, ONNX, Core ML, FPGA Verilog with a bit-exact model |
| 8 · Release | documentation site, 12 tutorials, equal-FLOPs benchmarks, packaging |

## Next

- **Kernels** — a sparse B-spline kernel (gather k+1 coefficients instead of
  all), simdgroup matrices on Metal, kernels for bases with trainable
  parameters, Triton autotuning, TF32 / bf16 modes for polynomial and Fourier
  bases.
- **Interpretation** — trainable affine parameters of symbolic edges,
  constant recovery (3.1416 → π), MultKAN pruning, KAN → tree conversion.
- **Blocks** — U-KAN blocks, TimeKAN, the KAN operator (KANO), RL helpers.
- **JAX** — Pallas kernels that block over several inputs to catch up with
  XLA, Mosaic / TPU kernels, ROCm validation.
- **Deployment** — interpolating FPGA lookups, HLS C++, lookup tables for
  convolutions, a direct TensorRT path (works through ONNX today).
- **Release** — PyPI, conda-forge.

Ideas and pull requests are welcome — see [CONTRIBUTING](CONTRIBUTING.md).
