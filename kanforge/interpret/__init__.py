"""Interpretability: tabulation, attribution, pruning, symbolic regression,
formulas (SymPy) and the kanpiler (formula -> KAN). Backend-independent:
functions accept PyTorch or MLX models."""

from .analysis import (
    EdgeTable,
    attribution,
    feature_importance,
    layer_activations,
    prune,
    prune_edges,
    prune_nodes,
    tabulate,
    tabulate_model,
)
from .contrastive import contrastive_samples
from .fit import Suggestion, fit_affine, suggest_symbolic
from .kanpiler import kanpile
from .library import LIBRARY, NAMES, SymbolicFunction
from .plot import plot_layer, plot_model
from .symbolic import FixedEdge, auto_symbolic, fix_symbolic, suggest_edge, symbolic_formula

__all__ = [
    "LIBRARY",
    "NAMES",
    "EdgeTable",
    "FixedEdge",
    "Suggestion",
    "SymbolicFunction",
    "attribution",
    "auto_symbolic",
    "contrastive_samples",
    "feature_importance",
    "fit_affine",
    "fix_symbolic",
    "kanpile",
    "layer_activations",
    "plot_layer",
    "plot_model",
    "prune",
    "prune_edges",
    "prune_nodes",
    "suggest_edge",
    "suggest_symbolic",
    "symbolic_formula",
    "tabulate",
    "tabulate_model",
]
