"""Minimal adapter over NumPy / PyTorch / MLX.

Basis functions are written once against the operations these three
libraries share (arithmetic, slicing, ``where``, ``tanh``, ``cos``, ...).
The only things that genuinely differ — picking the module and creating
a range on the same device/dtype as the input — live here.
"""

from __future__ import annotations


def namespace(x):
    """Return the array module (numpy, torch, mlx.core or jax.numpy) that owns ``x``."""
    mod = type(x).__module__
    if mod.startswith("torch"):
        import torch

        return torch
    if mod.startswith("mlx"):
        import mlx.core as mx

        return mx
    if mod.startswith(("jax", "jaxlib")):  # arrays and tracers
        import jax.numpy as jnp

        return jnp
    import numpy as np

    return np


def arange_like(n: int, x):
    """``arange(n)`` with the dtype (and device, for torch) of ``x``."""
    xp = namespace(x)
    if xp.__name__ == "torch":
        return xp.arange(n, dtype=x.dtype, device=x.device)
    return xp.arange(n, dtype=x.dtype)


def cast_like(a, x):
    """Cast ``a`` to the dtype of ``x``."""
    if namespace(x).__name__ == "torch":
        return a.to(x.dtype)
    return a.astype(x.dtype)


def asarray_like(a, x):
    """Convert a NumPy array or scalar to the backend, dtype and device of ``x``."""
    xp = namespace(x)
    if xp.__name__ == "torch":
        return xp.as_tensor(a, dtype=x.dtype, device=x.device)
    if xp.__name__ == "mlx.core":
        return xp.array(a, dtype=x.dtype)
    return xp.asarray(a, dtype=x.dtype)


def repeat_rows(a, repeats: int):
    """Repeat every row of ``a`` along axis 0, ``repeats`` times in a row."""
    xp = namespace(a)
    if xp.__name__ == "torch":
        return a.repeat_interleave(repeats, dim=0)
    return xp.repeat(a, repeats, axis=0)


def relu(x):
    return namespace(x).where(x > 0, x, 0 * x)


def sigmoid(x):
    return 0.5 * (1 + namespace(x).tanh(0.5 * x))


def softplus(x):
    xp = namespace(x)
    return relu(x) + xp.log1p(xp.exp(-xp.abs(x)))


def log_sigmoid(x):
    return -softplus(-x)
