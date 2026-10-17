"""Plots of edge functions (matplotlib, optional dependency)."""

from __future__ import annotations

import numpy as np

from ._backend import backend_of
from .analysis import attribution, tabulate


def plot_layer(layer, x=None, n: int = 201, scores=None, ax_size: float = 1.6):
    """Grid of every edge function phi_oi (rows: outputs, columns: inputs).

    Pruned edges are blank, symbolic ones are titled with their function;
    with ``scores`` (``attribution(...)[k]["edge"]``) the line opacity shows
    edge importance. Returns the matplotlib figure.
    """
    import matplotlib.pyplot as plt

    table = tabulate(layer, x, n)
    active = layer.active_edges()
    sym = layer.symbolic_edges()
    fig, axes = plt.subplots(
        layer.out_features, layer.in_features, figsize=(ax_size * layer.in_features, ax_size * layer.out_features), squeeze=False
    )
    smax = None if scores is None else max(float(np.max(scores)), 1e-12)
    for o in range(layer.out_features):
        for i in range(layer.in_features):
            ax = axes[o][i]
            ax.set_xticks([])
            ax.set_yticks([])
            if not active[o, i]:
                ax.axis("off")
                continue
            xs, ys = table.edge(o, i)
            alpha = 1.0 if smax is None else 0.15 + 0.85 * float(scores[o, i]) / smax
            ax.plot(xs, ys, color="black", alpha=alpha)
            if (o, i) in sym:
                ax.set_title(sym[(o, i)][0], fontsize=8)
            if o == layer.out_features - 1:
                ax.set_xlabel(f"in {i}", fontsize=7)
            if i == 0:
                ax.set_ylabel(f"out {o}", fontsize=7)
    fig.tight_layout()
    return fig


def plot_model(model, x, n: int = 201):
    """``plot_layer`` for every KAN layer, with attribution opacities; list of figures."""
    pairs = backend_of(model).layer_inputs(model, x)
    scores = attribution(model, x)
    return [plot_layer(layer, h, n, s["edge"]) for (layer, h), s in zip(pairs, scores)]
