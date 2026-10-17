"""kanpiler: compile SymPy formulas into MultKAN models (KAN 2.0).

The expression tree becomes a graph of nodes:

* a sum (``Add``) is an addition node, numbers go to its bias;
* a product of several factors is a binary multiplication node (``Mul`` with
  more factors is split into nested binary products);
* a function of an affine argument, ``c * f(a * u + b) + d`` with ``f`` from
  the symbolic library (powers, sqrt, exp, log, sin, ...), is an edge;
* anything else reaching an edge goes through the identity edge.

Nodes are placed on layers by depth; edges that skip layers get identity
nodes in between, and an input feeding the same node twice is routed once
through an identity node. ``mode="symbolic"`` makes every edge symbolic
(the model equals the formula); ``mode="fit"`` fits the layer's basis to
every edge by least squares over the variable ``ranges`` (a learnable
starting point, like TruKAN's compiler).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import library as lib

_POWERS = {1: None, 2: "x^2", 3: "x^3", 4: "x^4", -1: "1/x", -2: "1/x^2"}


@dataclass(eq=False)
class _Node:
    kind: str  # "input" | "sum" | "mult"
    index: int = -1  # position in its layer (set by layout)
    var: int = -1  # input nodes: variable index
    edges: list = field(default_factory=list)  # sum: [(child, fname, a, b, c, d)]
    bias: float = 0.0
    factors: list = field(default_factory=list)  # mult: two sums (single-edge nodes)
    depth: int = 0


class _Compiler:
    def __init__(self, variables):
        import sympy

        self.sp = sympy
        self.vars = list(variables)
        self.inputs = [_Node("input", var=k) for k in range(len(self.vars))]
        self.cache: dict = {}
        self.active: set = set()

    # --- expression -> graph ----------------------------------------------

    def _const(self, e) -> bool:
        return not (e.free_symbols & set(self.vars))

    def _coeff(self, e):
        """e = k * rest with a numeric constant k (pi, sqrt(2) included)."""
        k, rest = e.as_independent(*self.vars, as_Add=False)
        return float(k), rest

    def _split_affine(self, e):
        """e = alpha * g + beta with numbers alpha, beta; g None if e is constant."""
        if self._const(e):
            return 0.0, None, float(e)
        beta, rest = e.as_independent(*self.vars, as_Add=True)
        alpha, g = self._coeff(rest)
        return alpha, g, float(beta)

    def edge(self, e):
        """(child node, f, a, b, c, d) with e == c * f(a * child + b) + d."""
        sp = self.sp
        k, rest = self._coeff(e)
        if k != 1:
            child, f, a, b, c, d = self.edge(rest)
            return child, f, a, b, k * c, k * d
        e = rest
        name, arg = None, None
        if isinstance(e, sp.Pow) and self._const(e.exp):
            p = float(e.exp)
            if p in (0.5, -0.5):
                name = "sqrt" if p > 0 else "1/sqrt"
            elif p.is_integer() and int(p) in _POWERS:
                name = _POWERS[int(p)]
            else:
                raise NotImplementedError(f"power {e.exp} has no library function")
            arg = e.base
        else:
            funcs = {sp.sin: "sin", sp.cos: "cos", sp.exp: "exp", sp.log: "log", sp.tanh: "tanh", sp.Abs: "abs", sp.atan: "arctan"}
            for cls, fname in funcs.items():
                if isinstance(e, cls):
                    name, arg = fname, e.args[0]
                    break
        if name is None:  # symbol, sum, product: identity edge on its node
            if not isinstance(e, (sp.Symbol, sp.Add, sp.Mul)):
                raise NotImplementedError(f"no library function for {e} ({type(e).__name__})")
            return self.node(e), "x", 1.0, 0.0, 1.0, 0.0
        alpha, g, beta = self._split_affine(sp.expand(arg) if name not in ("exp",) else arg)
        if g is None:
            raise NotImplementedError(f"constant function argument in {e}")
        return self.node(g), name, alpha, beta, 1.0, 0.0

    def node(self, e) -> _Node:
        sp = self.sp
        if e in self.cache:
            return self.cache[e]
        if e in self.active:
            raise NotImplementedError(f"cannot compile {e}")
        self.active.add(e)
        if isinstance(e, sp.Symbol):
            if e not in self.vars:
                raise ValueError(f"symbol {e} is not among the variables {self.vars}")
            n = self.inputs[self.vars.index(e)]
        elif isinstance(e, sp.Add):
            n = _Node("sum")
            for term in e.args:
                if self._const(term):
                    n.bias += float(term)
                else:
                    self._add_edge(n, self.edge(term))
        elif isinstance(e, sp.Mul) and len([f for f in e.args if not self._const(f)]) >= 2:
            k, rest = self._coeff(e)
            factors = list(rest.args)
            left, right = sp.Mul(*factors[:-1]), factors[-1]
            n = _Node("mult")
            for part, scale in ((left, k), (right, 1.0)):
                sub = _Node("sum")
                child, f, a, b, c, d = self.edge(part)
                self._add_edge(sub, (child, f, a, b, scale * c, scale * d))
                n.factors.append(sub)
        else:
            n = _Node("sum")
            self._add_edge(n, self.edge(e))
        self.active.discard(e)
        self.cache[e] = n
        return n

    def _add_edge(self, n: _Node, edge):
        child = edge[0]
        if any(c is child for c, *_ in n.edges):  # one edge per (child, node): reroute
            child = self._identity(child)
        n.edges.append((child, *edge[1:]))

    @staticmethod
    def _identity(child: _Node) -> _Node:
        return _Node("sum", edges=[(child, "x", 1.0, 0.0, 1.0, 0.0)])

    # --- layout ---------------------------------------------------------------

    def _children(self, n: _Node):
        if n.kind == "sum":
            return [c for c, *_ in n.edges]
        if n.kind == "mult":
            return [c for sub in n.factors for c, *_ in sub.edges]
        return []

    def _set_depth(self, n: _Node, seen: set) -> int:
        if id(n) in seen:
            return n.depth
        seen.add(id(n))
        kids = self._children(n)
        n.depth = 0 if n.kind == "input" else 1 + max((self._set_depth(c, seen) for c in kids), default=0)
        return n.depth

    def _pad_edges(self, n: _Node):
        """Insert identity nodes so every edge goes from depth d-1 to d."""

        def fix(edges, depth):
            out = []
            for child, *rest in edges:
                while child.depth < depth - 1:
                    child = _Node("sum", edges=[(child, "x", 1.0, 0.0, 1.0, 0.0)], depth=child.depth + 1)
                out.append((child, *rest))
            return out

        if n.kind == "sum":
            n.edges = fix(n.edges, n.depth)
        elif n.kind == "mult":
            for sub in n.factors:
                sub.edges = fix(sub.edges, n.depth)

    def compile(self, exprs):
        exprs = [self.sp.sympify(e) for e in exprs]
        outs = [_Node("sum", bias=float(e)) if self._const(e) else self.node(e) for e in exprs]
        outs = [o if o.kind != "input" else self._identity(o) for o in outs]
        seen: set = set()
        depth = max(max(self._set_depth(o, seen) for o in outs), 1)
        final = []
        for o in outs:  # bring every output to the last layer
            while o.depth < depth:
                o = _Node("sum", edges=[(o, "x", 1.0, 0.0, 1.0, 0.0)], depth=o.depth + 1)
            final.append(o)
        # collect nodes per depth (padding creates new nodes, so iterate to a fixpoint)
        layers = [list(self.inputs)] + [[] for _ in range(depth)]
        placed = {id(n) for n in self.inputs}
        stack = list(final)
        while stack:
            n = stack.pop()
            if id(n) in placed:
                continue
            self._pad_edges(n)
            placed.add(id(n))
            layers[n.depth].append(n)
            stack.extend(self._children(n))
        for d in range(1, depth + 1):
            sums = [n for n in layers[d] if n.kind == "sum" and n not in final]
            mults = [n for n in layers[d] if n.kind == "mult" and n not in final]
            if d == depth:  # outputs keep the order of the expressions
                sums = [n for n in final if n.kind == "sum"] + sums
                mults = [n for n in final if n.kind == "mult"] + mults
            layers[d] = sums + mults
            for k, n in enumerate(layers[d]):
                n.index = k
        for k, n in enumerate(self.inputs):
            n.index = k
        return layers, final


def kanpile(
    exprs,
    variables,
    backend: str = "torch",
    basis="bspline",
    mode: str = "symbolic",
    ranges=None,
    samples: int = 1000,
    seed: int = 0,
):
    """Compile SymPy expression(s) of ``variables`` into a ``MultKAN``.

    Returns ``(model, outputs)`` where ``outputs[k]`` is the output column of
    ``exprs[k]`` (sum nodes come before product nodes in MultKAN layers).
    ``ranges`` ({variable: (lo, hi)}, default [-1, 1]) are used by
    ``mode="fit"`` to sample the edges and to place the per-input grids.
    """
    import sympy

    if mode not in ("symbolic", "fit"):
        raise ValueError("mode must be 'symbolic' or 'fit'")
    single = not isinstance(exprs, (list, tuple))
    exprs = [exprs] if single else list(exprs)
    variables = [sympy.Symbol(v) if isinstance(v, str) else v for v in variables]
    unknown = set().union(*(sympy.sympify(e).free_symbols for e in exprs)) - set(variables)
    if unknown:
        raise ValueError(f"symbols {sorted(map(str, unknown))} are not among the variables {variables}")
    comp = _Compiler(variables)
    layers, final = comp.compile(exprs)
    widths = [len(variables)] + [
        (sum(n.kind == "sum" for n in layer), sum(n.kind == "mult" for n in layer)) for layer in layers[1:]
    ]
    if backend == "torch":
        from ..torch import MultKAN
    elif backend == "mlx":
        from ..mlx import MultKAN
    else:
        raise ValueError("backend must be 'torch' or 'mlx'")

    # sample every node's value (mode "fit" needs the ranges of edge inputs)
    rng = np.random.default_rng(seed)
    ranges = ranges or {}
    pts = np.stack(
        [rng.uniform(*ranges.get(v, ranges.get(str(v), (-1.0, 1.0))), samples) for v in variables], 1
    )
    values: dict = {}

    def value(n: _Node):
        if id(n) in values:
            return values[id(n)]
        if n.kind == "input":
            v = pts[:, n.var]
        elif n.kind == "sum":
            v = n.bias + sum(c * lib.get(f).fn(a * value(ch) + b) + d for ch, f, a, b, c, d in n.edges) if n.edges else np.full(samples, n.bias)
        else:
            v = value(n.factors[0]) * value(n.factors[1])
        values[id(n)] = np.broadcast_to(np.asarray(v, dtype=np.float64), (samples,))
        return values[id(n)]

    grid_ranges_per_layer = []
    if mode == "fit":
        with np.errstate(all="ignore"):
            for d in range(1, len(layers)):
                vals = np.stack([value(n) for n in layers[d - 1]], 1)
                lo, hi = np.nanmin(vals, 0), np.nanmax(vals, 0)
                span = np.maximum(hi - lo, 1e-3)
                grid_ranges_per_layer.append(np.stack([lo - 0.05 * span, hi + 0.05 * span], 1))
    model = MultKAN(widths, basis, base_activation=None, coef_init="zero")
    mlayers = list(getattr(model, "layers", model))
    for d, mlayer in enumerate(mlayers, start=1):
        kan = mlayer.kan
        mask = np.zeros((kan.out_features, kan.in_features))
        bias = np.zeros(kan.out_features)
        coef = np.zeros((kan.out_features, kan.in_features, kan.basis.size))
        subs = []  # (kan output index, sum node)
        for n in layers[d]:
            if n.kind == "sum":
                subs.append((n.index, n))
        n_sum = sum(n.kind == "sum" for n in layers[d])
        for n in layers[d]:
            if n.kind == "mult":
                k = n.index - n_sum
                subs += [(n_sum + 2 * k, n.factors[0]), (n_sum + 2 * k + 1, n.factors[1])]
        if mode == "fit":
            r = grid_ranges_per_layer[d - 1]
            st = kan._export_state()
            for name in kan.basis.grid_names():
                npts = st["buffers"][name].shape[-1]
                kan._set(f"basis_{name}", r[:, :1] + (r[:, 1:] - r[:, :1]) * np.linspace(0, 1, npts))
            st = kan._export_state()
            params = {**{k: np.repeat(v, kan.in_features // kan.groups, 0) for k, v in st["trainable"].items()}, **st["buffers"]}
            inputs = np.stack([value(n) for n in layers[d - 1]], 1)
            design = kan.basis(inputs, params)  # (N, in, n)
        for o, sub in subs:
            bias[o] = sub.bias
            for child, f, a, b, c, dd in sub.edges:
                i = child.index
                if mode == "symbolic":
                    kan.fix_symbolic(o, i, f, (a, b, c, dd))
                else:
                    with np.errstate(all="ignore"):
                        target = c * lib.get(f).fn(a * value(child) + b) + dd
                    ok = np.isfinite(target)
                    coef[o, i] = np.linalg.lstsq(design[ok, i], target[ok], rcond=None)[0]
                    mask[o, i] = 1.0
        if mode == "fit":
            kan._set("coef", coef)
            kan.set_edge_mask(mask)
        else:
            kan.set_edge_mask(np.zeros_like(mask))
        kan._set("bias", bias)
    columns = [layers[-1].index(o) for o in final]
    return (model, columns[0]) if single else (model, columns)
