# Quickstart

## A model, trained

```python
import torch
import kanforge as kf
from kanforge.torch import KAN

x = torch.rand(1000, 2) * 2 - 1
y = torch.sin(torch.pi * x[:, :1]) * x[:, 1:] ** 2

model = KAN([2, 5, 1], kf.BSpline(5, 3), coef_init="lecun")
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for step in range(1000):
    opt.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    opt.step()
    if step == 200:
        model.update_grid(x)                          # grids follow the data
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
```

## Any basis, any backend

```python
model = KAN([2, 5, 1], kf.Chebyshev(6))              # or "fourier", kf.Jacobi(6, 0.5, 0.5), ...

import mlx.core as mx
from kanforge.mlx import KAN as MLXKAN
mlx_model = MLXKAN([2, 5, 1], "bspline")

import kanforge
kanforge.save(model, "model.npz")
jax_model = kanforge.load("model.npz", backend="jax")
```

## Look inside

```python
from kanforge import interpret as ki

pruned = ki.prune(model, x)
ki.auto_symbolic(pruned, x)
print(ki.symbolic_formula(pruned, ["x", "y"]))
```

## Ship it

```python
from kanforge.export.onnx_export import export_onnx
export_onnx(model, "model.onnx", x, size=256, bits=8)
```
