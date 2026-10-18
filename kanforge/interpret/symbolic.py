"""Symbolic edges and closed-form formulas of whole models (SymPy)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import library as lib
from ._backend import backend_of, kan_layers, to_numpy
from .analysis import tabulate
from .fit import Suggestion, fit_affine, suggest_symbolic


def suggest_edge(layer, o: int, i: int, x=None, functions=None, weight_simple: float = 0.8, topk: int = 5, n: int = 201):
    """Candidate formulas for edge (o, i) of ``layer`` over the range of ``x``."""
    xs, ys = tabulate(layer, x, n).edge(o, i)
    return suggest_symbolic(xs, ys, functions, weight_simple, topk)


def fix_symbolic(layer, o: int, i: int, name: str, x=None, n: int = 201) -> float:
    """Fit c * f(a * x + b) + d to edge (o, i), make it symbolic; returns R^2."""
    xs, ys = tabulate(layer, x, n).edge(o, i)
    params, r2 = fit_affine(xs, ys, lib.get(name))
    layer.fix_symbolic(o, i, name, params)
    return r2


@dataclass
class FixedEdge:
    layer: int
    o: int
    i: int
    suggestion: Suggestion | None  # None: left as it was (below r2_threshold)


def auto_symbolic(model, x, functions=None, weight_simple: float = 0.8, r2_threshold: float = 0.9, n: int = 201):
    """Make every learned edge symbolic with its best candidate (pykan's
    auto_symbolic). Ranges come from the activations each layer receives for
    ``x``. Edges whose best R^2 is below ``r2_threshold`` stay learned, so
    ``symbolic_formula`` reports them instead of returning a wrong formula.

    The zero function is not a candidate by default — with its zero
    complexity it would win over every mediocre fit; remove negligible edges
    with ``prune_edges`` instead (or pass ``functions`` including ``"0"``).
    """
    if functions is None:
        functions = [n_ for n_ in lib.NAMES if n_ != "0"]
    b = backend_of(model)
    report = []
    for idx, (layer, h) in enumerate(b.layer_inputs(model, x)):
        table = tabulate(layer, h, n)
        learned = (to_numpy(layer.edge_mask) != 0) & (to_numpy(layer.sym_code) < 0)
        for o, i in zip(*np.nonzero(learned)):
            xs, ys = table.edge(o, i)
            best = suggest_symbolic(xs, ys, functions, weight_simple, topk=1)[0]
            if best.r2 >= r2_threshold:
                layer.fix_symbolic(int(o), int(i), best.name, best.params)
                report.append(FixedEdge(idx, int(o), int(i), best))
            else:
                report.append(FixedEdge(idx, int(o), int(i), None))
    return report


def _round(expr, digits):
    import sympy

    if digits is None:
        return expr
    return expr.xreplace({f: sympy.Float(round(float(f), digits), digits + 2) for f in expr.atoms(sympy.Float)})


def _affine(k: float, u, t: float = 0.0):
    """k * u + t without writing unit factors and zero offsets."""
    out = u if k == 1 else float(k) * u
    return out if t == 0 else out + float(t)


def _edge_expr(code: int, params, u):
    a, b, c, d = (float(v) for v in params)
    return _affine(c, lib.get(int(code)).sympy(_affine(a, u, b)), d)


def _kan_layer_exprs(layer, inputs, tol: float = 0.0):
    import sympy

    mask = to_numpy(layer.edge_mask)
    code = to_numpy(layer.sym_code).astype(int)
    sp = to_numpy(layer.sym_params)
    coef = to_numpy(layer.effective_coef())
    base = layer.effective_base_weight()
    base = None if base is None else to_numpy(base)
    bias = getattr(layer, "bias", None)
    bias = np.zeros(layer.out_features) if bias is None else to_numpy(bias)
    out, missing = [], []
    for o in range(layer.out_features):
        expr = sympy.Float(float(bias[o])) if bias[o] else sympy.Integer(0)
        for i in range(layer.in_features):
            if code[o, i] >= 0:
                expr += _edge_expr(code[o, i], sp[o, i], inputs[i])
            elif mask[o, i] != 0:
                if np.abs(coef[o, i]).max() > tol or (base is not None and abs(base[o, i]) > tol):
                    missing.append((o, i))
        out.append(expr)
    return out, missing


def symbolic_formula(model, names=None, digits: int | None = 4, simplify: bool = False, tol: float = 0.0):
    """SymPy expressions of every output.

    Every edge must be symbolic or pruned (spline and base weights zero up to
    ``tol`` count as zero); otherwise ``ValueError`` lists the remaining ones.
    Also handles MultKAN product nodes and plain ``Linear`` layers (models from
    ``from_mlp``).
    """
    import sympy

    b = backend_of(model)
    layers = [model] if isinstance(model, (b.KANLinear, b.MultKANLinear)) else list(getattr(model, "layers", model))
    first = layers[0]
    n_in = first.kan.in_features if isinstance(first, b.MultKANLinear) else getattr(first, "in_features", None)
    if n_in is None:
        n_in = to_numpy(first.weight).shape[1] if hasattr(first, "weight") else to_numpy(first.kernel).shape[0]
    names = names or [f"x_{k + 1}" for k in range(n_in)]
    exprs = [sympy.Symbol(nm) for nm in names]
    missing_all = []
    for idx, layer in enumerate(layers):
        if isinstance(layer, b.KANLinear):
            exprs, missing = _kan_layer_exprs(layer, exprs, tol)
        elif isinstance(layer, b.MultKANLinear):
            sub, missing = _kan_layer_exprs(layer.kan, exprs, tol)
            prods = [sympy.Mul(*sub[layer.n_sum + k * layer.arity : layer.n_sum + (k + 1) * layer.arity]) for k in range(layer.n_mult)]
            exprs = sub[: layer.n_sum] + prods
        elif hasattr(layer, "weight") or hasattr(layer, "kernel"):  # Linear (torch, MLX) or nnx.Linear
            w = to_numpy(layer.weight) if hasattr(layer, "weight") else to_numpy(layer.kernel).T
            bias = getattr(layer, "bias", None)
            bias = np.zeros(w.shape[0]) if bias is None else to_numpy(bias)
            exprs = [float(bias[o]) + sum(_affine(float(w[o, i]), exprs[i]) for i in range(w.shape[1])) for o in range(w.shape[0])]
            missing = []
        else:
            raise TypeError(f"no formula for layer {type(layer).__name__}")
        missing_all += [(idx, o, i) for o, i in missing]
    if missing_all:
        raise ValueError(f"edges are neither symbolic nor pruned (layer, out, in): {missing_all}")
    out = []
    for e in exprs:
        e = sympy.expand(e) if not simplify else sympy.simplify(e)
        out.append(_round(e, digits))
    return out
