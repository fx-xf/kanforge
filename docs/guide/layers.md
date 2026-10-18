# Layers and backends

```python
from kanforge.torch import KANLinear, KAN, GRKAN, MultKAN
layer = KANLinear(8, 4, "bspline", groups=None, edge_scale=False, coef_init="normal", backend="auto")
```

| argument | meaning |
|---|---|
| `basis` | name, config dict or `Basis` |
| `base_activation` | residual branch w_oi · act(x_i) (SiLU by default, None to drop) |
| `groups` | share trainable basis parameters within groups of inputs (GR-KAN); None = one set per input |
| `edge_scale` | per-edge scale s_oi (KANLib's `weight_spline`) |
| `grid_ranges` | per-input (lo, hi) of the grid |
| `coef_init` | `normal`, `zero` (start as the base branch), `variance`, `lecun`, `glorot` |
| `edge_dropout` | DropKAN: drop edge activations while training |
| `backend` | `auto`, `ops`, `kernel` (see [Fused kernels](kernels.md)) |

Parameters and buffers have the same names and shapes on every backend:
`coef (out, in, n)`, `base_weight (out, in)`, `bias (out,)`, `edge_scale`,
`basis_<name>` (trainable `(groups, …)` or buffers `(in, …)`), `edge_mask`,
`sym_code`, `sym_params`. `layer.edge_derivative(x, m)` returns the exact
∂ᵐy_o/∂x_iᵐ for every edge — the diagonal derivatives PINN losses need.

| backend | import | devices |
|---|---|---|
| NumPy (float64 reference) | `kanforge.reference` | CPU |
| PyTorch | `kanforge.torch` | CUDA, ROCm, MPS, CPU |
| MLX | `kanforge.mlx` | Apple silicon |
| JAX | `kanforge.jax` | CPU, GPU, TPU |

`GRKAN` is KAT's group-rational layer; `MultKANLinear` / `MultKAN` add
multiplication nodes (KAN 2.0).
