"""Sparsity regularisation of KAN layers, written once for every backend.

pykan's penalty works on edge magnitudes |phi_oi| averaged over a batch:
L1 of the magnitudes plus the entropy of their row and column
distributions, which pushes a layer towards few important edges (needed
before pruning and symbolic regression). Optional terms penalise the
coefficients themselves and their differences (smoothness).
Without data, efficient-kan's variant uses mean |coef| as edge magnitude.
"""

from __future__ import annotations

from ._xp import namespace


def penalty(magnitude, coef, *, l1: float = 1.0, entropy: float = 2.0, coef_l1: float = 0.0, coef_smooth: float = 0.0, eps: float = 1e-8):
    """Penalty of one layer from edge magnitudes ``(out, in)`` and coefficients ``(out, in, n)``."""
    xp = namespace(coef)
    total = magnitude.sum()
    out = l1 * total
    if entropy:
        p_row = magnitude / (magnitude.sum(1)[:, None] + eps)
        p_col = magnitude / (magnitude.sum(0)[None, :] + eps)
        ent_row = -(p_row * xp.log2(p_row + eps)).sum(1).mean()
        ent_col = -(p_col * xp.log2(p_col + eps)).sum(0).mean()
        out = out + entropy * (ent_row + ent_col)
    if coef_l1:
        out = out + coef_l1 * xp.abs(coef).mean()
    if coef_smooth and coef.shape[-1] > 1:
        out = out + coef_smooth * xp.abs(coef[..., 1:] - coef[..., :-1]).mean()
    return out


def edge_magnitude(activations):
    """mean |phi_oi(x)| over all batch dimensions: ``(..., out, in)`` -> ``(out, in)``."""
    xp = namespace(activations)
    a = xp.abs(activations)
    return a.reshape(-1, *a.shape[-2:]).mean(0)
