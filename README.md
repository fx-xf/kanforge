# kanforge

Kolmogorov–Arnold Networks (KAN) for every backend, with one set of basis
functions and one parameter layout:

| Backend | Import | Devices |
| --- | --- | --- |
| NumPy (float64 reference) | `kanforge.reference` | CPU |
| PyTorch | `kanforge.torch` | CUDA, ROCm, Apple MPS, CPU |
| MLX | `kanforge.mlx` | Apple silicon GPU/CPU |

Status: **phase 2** — 26 bases with exact derivatives of any order, trainable
basis parameters shared per input, per group or per layer, GR-KAN (KAT) and
MultKAN layers; per-input grids, grid update / refinement and conversion
between any bases, pykan regularisation, DropKAN, variance-preserving
initialisation and exact conversion of MLPs; fused Metal kernels (PyTorch MPS,
MLX) and Triton kernels (CUDA/ROCm) with derivatives of any order, up to 9x
faster on Apple silicon — see [BENCHMARKS.md](BENCHMARKS.md). Interpretability
and export are on the roadmap in [PLAN.md](PLAN.md).

| Family | Bases |
| --- | --- |
| Splines | `bspline`, `nurbs`, `truncated_power`, `relu_kan` |
| Polynomials | `chebyshev`, `chebyshev2`, `legendre`, `jacobi`, `fractional_jacobi`, `gegenbauer`, `hermite`, `laguerre`, `bernstein`, `gram`, `taylor` |
| Bumps and wavelets | `rbf`, `rswaf`, `mexican_hat`, `morlet`, `dog` |
| Periodic | `fourier`, `random_fourier` |
| Other | `rational`, `neural`, `concat`, `bsrbf` |

```python
import torch
from kanforge.torch import KAN

model = KAN([2, 5, 1], basis="bspline")          # or "chebyshev", "fourier"
y = model(torch.rand(64, 2))
```

```python
import torch
import kanforge as kf
from kanforge.torch import GRKAN, KANLinear, MultKAN

layer = KANLinear(8, 4, kf.Jacobi(6, alpha=0.5, beta=0.5), edge_scale=True)
d2 = layer.edge_derivative(torch.rand(32, 8), order=2)   # exact d²y_o/dx_i², (32, 4, 8)
ffn = GRKAN(768, 3072, groups=8)                         # KAT's group-rational layer
model = MultKAN([2, (2, 1), 1])                          # 2 sum + 1 product node
```

```python
import torch
import kanforge as kf
from kanforge.torch import KAN, from_mlp, regularization

model = KAN([2, 5, 1], kf.BSpline(5, 3), coef_init="lecun")
x = torch.rand(1024, 2) * 4 - 2
model.update_grid(x)                          # per-input adaptive grids (pykan)
model.refine_grid(10)                         # finer grid, same function
model[0].change_basis(kf.Chebyshev(12))       # convert a layer to another basis
loss = model(x).pow(2).mean() + 1e-3 * regularization(model, x)

mlp = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.GELU(), torch.nn.Linear(16, 1))
kan = from_mlp(mlp)                           # equals mlp exactly, then fine-tune
```

```python
import mlx.core as mx
from kanforge.mlx import KAN

model = KAN([2, 5, 1], basis={"name": "bspline", "grid_size": 8, "order": 3})
y = model(mx.random.uniform(shape=(64, 2)))
```

## Development

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[torch,mlx,test]"
.venv/bin/python -m pytest
```

## License

Apache-2.0. See [NOTICE](NOTICE) for third-party attributions.
