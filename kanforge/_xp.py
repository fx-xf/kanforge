"""Minimal adapter over NumPy / PyTorch / MLX.

Basis functions are written once against the operations these three
libraries share (arithmetic, slicing, ``where``, ``tanh``, ``cos``, ...).
The only things that genuinely differ — picking the module and creating
a range on the same device/dtype as the input — live here.
"""

from __future__ import annotations


def namespace(x):
    """Return the array module (numpy, torch or mlx.core) that owns ``x``."""
    mod = type(x).__module__
    if mod.startswith("torch"):
        import torch

        return torch
    if mod.startswith("mlx"):
        import mlx.core as mx

        return mx
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
