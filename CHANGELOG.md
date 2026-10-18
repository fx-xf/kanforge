# Changelog

## 0.8.0 — first public release (unreleased)

Everything below was built in phases 0–8 of [PLAN.md](https://github.com/fx-xf/kanforge/blob/main/PLAN.md).

- **Bases**: 26 basis families with exact derivatives of any order, trainable
  basis parameters per input, per group or per layer.
- **Backends**: NumPy reference, PyTorch (CUDA, ROCm, MPS, CPU), MLX, JAX on
  Flax NNX — identical parameters; `kanforge.save` / `load` / `convert`.
- **Training**: per-input adaptive grids, `update_grid`, `refine_grid`,
  `change_basis`; pykan regularisation; DropKAN; LeCun / Glorot / variance
  initialisation; exact `from_mlp`.
- **Kernels**: fused Metal (MPS, MLX), Triton (CUDA) and Pallas (JAX) kernels
  with derivatives of any order; `backend="auto"` follows benchmarks.
- **Interpretation**: tabulation, attribution, pruning, symbolic regression,
  `symbolic_formula` (SymPy), `kanpile`, plots, contrastive mode.
- **Blocks**: KAN convolutions, `KANFFN`, `KATBlock`, `kanify` for Hugging
  Face models, KAGIN / KAGCN / KAGAT, KAN LSTM / GRU, PINN utilities,
  DeepONet, FBKAN, KAN-ODE.
- **Deployment**: lookup tables, int8, ONNX, Core ML, FPGA Verilog with a
  bit-exact model.
- **Docs**: guide, 12 tutorials, API reference, equal-FLOPs benchmarks.

### Fixed during development

- LBFGS failed on non-contiguous gradients (`@ W.T` in the forward pass and
  transposed parameters after grid tools).
- `KAN(grid_ranges=...)` applied the input ranges to every layer.
- `GridToolsMixin._apply` shadowed `nn.Module._apply` and broke `.to()`.
