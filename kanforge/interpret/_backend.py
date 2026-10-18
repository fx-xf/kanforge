"""Find the backend module of a model and move arrays between it and NumPy."""

from __future__ import annotations

import numpy as np


def backend_of(model):
    """``kanforge.torch`` or ``kanforge.mlx`` for a model, layer or Sequential."""
    try:
        import torch

        if isinstance(model, torch.nn.Module):
            from .. import torch as backend

            return backend
    except ImportError:
        pass
    try:
        import mlx.nn

        if isinstance(model, mlx.nn.Module):
            from .. import mlx as backend

            return backend
    except ImportError:
        pass
    try:
        from flax import nnx

        if isinstance(model, nnx.Module):
            from .. import jax as backend

            return backend
    except ImportError:
        pass
    raise TypeError(f"not a PyTorch, MLX or JAX (Flax NNX) model: {type(model).__name__}")


def _unwrap(x):
    """The array inside a Flax NNX variable, else ``x``."""
    try:
        from flax import nnx

        if isinstance(x, nnx.Variable):
            return x[...]
    except ImportError:
        pass
    return x


def to_numpy(x) -> np.ndarray:
    x = _unwrap(x)
    if hasattr(x, "detach"):
        return x.detach().cpu().double().numpy()
    return np.array(x, dtype=np.float64)


def like(layer, value: np.ndarray):
    """``value`` as an array of the layer's backend, dtype and device."""
    coef = _unwrap(layer.coef)
    if hasattr(coef, "detach"):
        import torch

        return torch.as_tensor(value, dtype=coef.dtype, device=coef.device)
    if type(coef).__module__.startswith(("jax", "jaxlib")):
        import jax.numpy as jnp

        return jnp.asarray(value, dtype=coef.dtype)
    import mlx.core as mx

    return mx.array(np.asarray(value, dtype=np.float32))


def kan_layers(model):
    """[(KANLinear, MultKANLinear wrapper or None)] in order, or TypeError."""
    b = backend_of(model)
    if isinstance(model, b.KANLinear):
        return [(model, None)]
    if isinstance(model, b.MultKANLinear):
        return [(model.kan, model)]
    children = list(model) if hasattr(model, "__iter__") and not hasattr(model, "layers") else list(getattr(model, "layers", []))
    out = []
    for m in children:
        if isinstance(m, b.KANLinear):
            out.append((m, None))
        elif isinstance(m, b.MultKANLinear):
            out.append((m.kan, m))
    return out
