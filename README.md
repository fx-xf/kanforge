# kanforge

Kolmogorov–Arnold Networks (KAN) for every backend, with one set of basis
functions and one parameter layout:

| Backend | Import | Devices |
| --- | --- | --- |
| NumPy (float64 reference) | `kanforge.reference` | CPU |
| PyTorch | `kanforge.torch` | CUDA, ROCm, Apple MPS, CPU |
| MLX | `kanforge.mlx` | Apple silicon GPU/CPU |

Status: **phase 0** — plain-ops layers, three bases (B-spline, Chebyshev,
Fourier), reference tests including second derivatives. Fused kernels,
interpretability, more bases and export are on the roadmap in
[PLAN.md](PLAN.md).

```python
import torch
from kanforge.torch import KAN

model = KAN([2, 5, 1], basis="bspline")          # or "chebyshev", "fourier"
y = model(torch.rand(64, 2))
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
