"""One model, every backend: convert between PyTorch, MLX and JAX; save and load."""

import os
import tempfile

import numpy as np
import torch

import kanforge
import kanforge.convert as cv
from kanforge.torch import KAN

torch.manual_seed(0)
model = KAN([3, 5, 2], "bspline")
x = np.random.default_rng(0).uniform(-1, 1, (4, 3)).astype(np.float32)
ref = model(torch.tensor(x)).detach().numpy()

path = os.path.join(tempfile.mkdtemp(), "model.npz")
kanforge.save(model, path)
print("torch -> file -> torch:", np.abs(kanforge.load(path, "torch")(torch.tensor(x)).detach().numpy() - ref).max())
try:
    import mlx.core as mx

    with mx.stream(mx.cpu):
        print("torch -> MLX:", np.abs(np.array(cv.convert(model, "mlx")(mx.array(x))) - ref).max())
except ImportError:
    print("MLX not installed")
try:
    import jax.numpy as jnp

    print("file -> JAX: ", np.abs(np.asarray(kanforge.load(path, "jax")(jnp.asarray(x))) - ref).max())
except ImportError:
    print("JAX not installed")
