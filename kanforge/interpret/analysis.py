"""Tabulation, attribution and pruning of KAN models (PyTorch or MLX)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grid import grid_ranges, sample_inputs
from ._backend import backend_of, kan_layers, like, to_numpy


@dataclass
class EdgeTable:
    """Edge functions phi_oi sampled on a per-input grid of points."""

    x: np.ndarray  # (N, in)
    values: np.ndarray  # (N, out, in)

    def edge(self, o: int, i: int) -> tuple[np.ndarray, np.ndarray]:
        return self.x[:, i], self.values[:, o, i]


def input_ranges(layer, x=None, margin: float = 0.0) -> np.ndarray:
    """Per-input (lo, hi): from samples ``x``, else the basis grid, else [-1, 1]."""
    if x is not None:
        xs = to_numpy(x).reshape(-1, layer.in_features)
        r = np.stack([xs.min(0), xs.max(0)], -1)
    else:
        st = layer._export_state()
        r = grid_ranges(layer.basis, st["buffers"])
        if r is None:
            r = np.tile([-1.0, 1.0], (layer.in_features, 1))
    pad = margin * (r[:, 1] - r[:, 0])
    return np.stack([r[:, 0] - pad, r[:, 1] + pad], -1)


def tabulate(layer, x=None, n: int = 201, margin: float = 0.0) -> EdgeTable:
    """Sample every edge of ``layer`` (any basis, any backend) on ``n`` points
    per input over the range of ``x`` (or of the grid)."""
    pts = sample_inputs(input_ranges(layer, x, margin), n)
    values = to_numpy(layer.edge_derivative(like(layer, pts), 0))
    return EdgeTable(pts, values)


def layer_activations(model, x) -> list:
    """Inputs of every KAN layer of the model, in order (backend arrays)."""
    return [h for _, h in backend_of(model).layer_inputs(model, x)]


def tabulate_model(model, x, n: int = 201) -> list[EdgeTable]:
    """Tabulate every KAN layer over the activations it receives for inputs ``x``."""
    pairs = backend_of(model).layer_inputs(model, x)
    return [tabulate(layer, h, n) for layer, h in pairs]


def attribution(model, x, eps: float = 1e-12) -> list[dict]:
    """Importance of edges and nodes, propagated from the outputs (pykan-style).

    E_oi = std over the data of phi_oi(x_i); an output node passes its score to
    its inputs in proportion to E_oi; a product node passes its full score to
    each factor. Per layer: ``edge`` (out, in), ``input`` (in,), ``output``
    (out,) scores; output nodes of the model score 1.
    """
    b = backend_of(model)
    pairs = b.layer_inputs(model, x)
    wrappers = {id(layer): wrapper for layer, wrapper in kan_layers(model)}
    scales = [to_numpy(layer.edge_derivative(h, 0)).reshape(-1, layer.out_features, layer.in_features).std(0) for layer, h in pairs]
    out = [None] * len(pairs)
    node = None
    for idx in range(len(pairs) - 1, -1, -1):
        layer = pairs[idx][0]
        wrapper = wrappers.get(id(layer))
        if node is None:
            n_nodes = wrapper.out_features if wrapper is not None else layer.out_features
            node = np.ones(n_nodes)
        if wrapper is not None and wrapper.n_mult:
            sub = np.concatenate([node[: wrapper.n_sum], np.repeat(node[wrapper.n_sum :], wrapper.arity)])
        else:
            sub = node
        e = scales[idx]
        edge = sub[:, None] * e / (e.sum(1, keepdims=True) + eps)
        out[idx] = {"edge": edge, "input": edge.sum(0), "output": sub}
        node = edge.sum(0)
    return out


def feature_importance(model, x) -> np.ndarray:
    """Importance of every model input (first-layer input scores)."""
    return attribution(model, x)[0]["input"]


def prune_edges(model, x, threshold: float = 1e-2) -> int:
    """Mask edges whose attribution score is below ``threshold``; returns how many."""
    removed = 0
    for (layer, _), scores in zip(kan_layers(model), attribution(model, x)):
        keep = (scores["edge"] >= threshold) | ~layer.active_edges()
        mask = to_numpy(layer.edge_mask) * keep
        removed += int((layer.active_edges() & ~keep).sum())
        layer.set_edge_mask(mask)
    return removed


def prune_nodes(model, x, threshold: float = 1e-2):
    """A smaller model without hidden nodes scoring below ``threshold``.

    Works for ``KAN`` and sequences of ``KANLinear``; inputs and outputs stay.
    """
    b = backend_of(model)
    layers = kan_layers(model)
    if any(w is not None for _, w in layers) or len(layers) != len(list(getattr(model, "layers", model))):
        raise TypeError("prune_nodes supports KAN models made only of KANLinear layers")
    scores = attribution(model, x)
    keeps = [np.arange(layers[0][0].in_features)]
    for idx in range(1, len(layers)):
        keep = np.nonzero(scores[idx]["input"] >= threshold)[0]
        keeps.append(keep if len(keep) else np.array([int(np.argmax(scores[idx]["input"]))]))
    keeps.append(np.arange(layers[-1][0].out_features))
    new = [layer.subset(out_idx=keeps[idx + 1], in_idx=keeps[idx]) for idx, (layer, _) in enumerate(layers)]
    return b.rebuild_kan(new)


def prune(model, x, node_threshold: float = 1e-2, edge_threshold: float = 1e-2):
    """``prune_nodes`` then ``prune_edges`` (pykan's prune)."""
    pruned = prune_nodes(model, x, node_threshold)
    prune_edges(pruned, x, edge_threshold)
    return pruned
