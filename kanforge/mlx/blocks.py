"""MLX versions of the architecture blocks: transformer FFN, graph layers,
recurrent cells and scientific ML tools. Same math and arguments as the
corresponding ``kanforge.torch`` modules (``transformer``, ``graph``,
``recurrent``, ``science``); see their docstrings.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence

import mlx.core as mx
from mlx import nn

from .layers import GRKAN, KAN, KANLinear

# --- transformer -------------------------------------------------------------


class KANFFN(nn.Module):
    def __init__(self, dim: int, hidden: int | None = None, kind: str = "grkan", basis="bspline", groups: int = 8, activation: str = "gelu", **kan_kwargs):
        super().__init__()
        hidden = hidden or 4 * dim
        if kind == "grkan":
            self.fc1 = GRKAN(dim, hidden, groups=groups, init="identity")
            self.fc2 = GRKAN(hidden, dim, groups=groups, init=activation)
        elif kind == "kan":
            self.fc1 = KANLinear(dim, hidden, basis, **kan_kwargs)
            self.fc2 = KANLinear(hidden, dim, basis, **kan_kwargs)
        else:
            raise ValueError("kind must be 'grkan' or 'kan'")

    def __call__(self, x):
        return self.fc2(self.fc1(x))


class KATBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, kind: str = "grkan", **ffn_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiHeadAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = KANFFN(dim, int(dim * mlp_ratio), kind, **ffn_kwargs)

    def __call__(self, x, mask=None):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, mask)
        return x + self.ffn(self.norm2(x))


# --- graphs ---------------------------------------------------------------------


def add_self_loops(edge_index, num_nodes: int):
    loops = mx.broadcast_to(mx.arange(num_nodes)[None], (2, num_nodes))
    return mx.concatenate([edge_index, loops.astype(edge_index.dtype)], 1)


def aggregate(x, edge_index, num_nodes: int, weight=None):
    src, dst = edge_index[0], edge_index[1]
    msg = x[src] if weight is None else x[src] * weight.reshape(-1, *([1] * (x.ndim - 1)))
    return mx.zeros((num_nodes, *x.shape[1:]), dtype=x.dtype).at[dst].add(msg)


def _kan(in_features, out_features, hidden, basis, kan_kwargs):
    if hidden:
        return KAN([in_features, *hidden, out_features], basis, **kan_kwargs)
    return KANLinear(in_features, out_features, basis, **kan_kwargs)


class KAGINConv(nn.Module):
    def __init__(self, in_features: int, out_features: int, basis="bspline", hidden: Sequence[int] = (), eps: float = 0.0, train_eps: bool = False, **kan_kwargs):
        super().__init__()
        self.kan = _kan(in_features, out_features, hidden, basis, kan_kwargs)
        if train_eps:
            self.eps = mx.array(float(eps))
        else:
            self._eps = float(eps)

    def __call__(self, x, edge_index):
        eps = self.eps if "eps" in self else self._eps
        return self.kan((1 + eps) * x + aggregate(x, edge_index, x.shape[0]))


class KAGCNConv(nn.Module):
    def __init__(self, in_features: int, out_features: int, basis="bspline", hidden: Sequence[int] = (), **kan_kwargs):
        super().__init__()
        self.kan = _kan(in_features, out_features, hidden, basis, kan_kwargs)

    @staticmethod
    def normalized_weights(edge_index, num_nodes: int):
        ei = add_self_loops(edge_index, num_nodes)
        deg = mx.zeros((num_nodes,)).at[ei[1]].add(mx.ones((ei.shape[1],)))
        inv = mx.rsqrt(deg)
        return ei, inv[ei[0]] * inv[ei[1]]

    def __call__(self, x, edge_index):
        ei, w = self.normalized_weights(edge_index, x.shape[0])
        return self.kan(aggregate(x, ei, x.shape[0], w))


class KAGATConv(nn.Module):
    def __init__(self, in_features: int, out_features: int, heads: int = 1, basis="bspline", concat: bool = True, negative_slope: float = 0.2, add_loops: bool = True, **kan_kwargs):
        super().__init__()
        self.heads, self.out_features, self.concat = heads, out_features, concat
        self.negative_slope, self.add_loops = negative_slope, add_loops
        self.kan = KANLinear(in_features, heads * out_features, basis, **kan_kwargs)
        bound = math.sqrt(6 / (heads + out_features))
        self.att_src = mx.random.uniform(-bound, bound, (heads, out_features))
        self.att_dst = mx.random.uniform(-bound, bound, (heads, out_features))

    def __call__(self, x, edge_index, return_attention: bool = False):
        n = x.shape[0]
        ei = add_self_loops(edge_index, n) if self.add_loops else edge_index
        src, dst = ei[0], ei[1]
        z = self.kan(x).reshape(n, self.heads, self.out_features)
        logits = nn.leaky_relu((z * self.att_src).sum(-1)[src] + (z * self.att_dst).sum(-1)[dst], self.negative_slope)
        top = mx.full((n, self.heads), -mx.inf).at[dst].maximum(logits)
        w = mx.exp(logits - top[dst])
        denom = mx.zeros((n, self.heads)).at[dst].add(w)
        alpha = w / denom[dst]
        out = mx.zeros(z.shape).at[dst].add(z[src] * alpha[..., None])
        out = out.reshape(n, -1) if self.concat else out.mean(1)
        return (out, alpha) if return_attention else out


# --- recurrent ------------------------------------------------------------------


class KANLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, basis="bspline", **kan_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.gates = KANLinear(input_size + hidden_size, 4 * hidden_size, basis, **kan_kwargs)

    def __call__(self, x, state=None):
        if state is None:
            zeros = mx.zeros((x.shape[0], self.hidden_size))
            state = (zeros, zeros)
        h, c = state
        i, f, g, o = mx.split(self.gates(mx.concatenate([x, h], -1)), 4, -1)
        c = mx.sigmoid(f) * c + mx.sigmoid(i) * mx.tanh(g)
        h = mx.sigmoid(o) * mx.tanh(c)
        return h, (h, c)


class KANGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, basis="bspline", **kan_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.gates = KANLinear(input_size + hidden_size, 2 * hidden_size, basis, **kan_kwargs)
        self.candidate = KANLinear(input_size + hidden_size, hidden_size, basis, **kan_kwargs)

    def __call__(self, x, state=None):
        h = mx.zeros((x.shape[0], self.hidden_size)) if state is None else state
        z, r = mx.split(mx.sigmoid(self.gates(mx.concatenate([x, h], -1))), 2, -1)
        n = mx.tanh(self.candidate(mx.concatenate([x, r * h], -1)))
        h = (1 - z) * n + z * h
        return h, h


class KANRNN(nn.Module):
    CELLS = {"lstm": KANLSTMCell, "gru": KANGRUCell}

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, cell: str = "lstm", basis="bspline", **kan_kwargs):
        super().__init__()
        if cell not in self.CELLS:
            raise ValueError(f"cell must be one of {sorted(self.CELLS)}")
        cls = self.CELLS[cell]
        self.cells = [cls(input_size if k == 0 else hidden_size, hidden_size, basis, **kan_kwargs) for k in range(num_layers)]

    def __call__(self, x, states=None):
        states = list(states) if states is not None else [None] * len(self.cells)
        seq = x
        for k, cell in enumerate(self.cells):
            outs, state = [], states[k]
            for t in range(seq.shape[1]):
                out, state = cell(seq[:, t], state)
                outs.append(out)
            seq = mx.stack(outs, 1)
            states[k] = state
        return seq, states


# --- scientific ML ----------------------------------------------------------------


def derivative(fn: Callable, x, order: int = 1, dim: int = 0):
    """d^order u / dx_dim^order for u = fn(x) (functional, via mx.grad)."""

    def lift(g):
        return lambda z: mx.grad(lambda zz: g(zz).sum())(z)[:, dim]

    g = lambda z: fn(z).reshape(z.shape[0])  # noqa: E731
    for _ in range(order):
        g = lift(g)
    return g(x)


def gradient(fn: Callable, x):
    return mx.grad(lambda z: fn(z).reshape(z.shape[0]).sum())(x)


def laplacian(fn: Callable, x):
    return sum(derivative(fn, x, 2, k) for k in range(x.shape[1]))


def sample_box(bounds, n: int):
    lo = mx.array([b[0] for b in bounds], dtype=mx.float32)
    hi = mx.array([b[1] for b in bounds], dtype=mx.float32)
    return lo + (hi - lo) * mx.random.uniform(shape=(n, len(bounds)))


def sample_boundary(bounds, n: int):
    import numpy as np

    x = np.array(sample_box(bounds, n))
    face = np.random.randint(0, 2 * len(bounds), n)
    for f in range(2 * len(bounds)):
        x[face == f, f // 2] = bounds[f // 2][f % 2]
    return mx.array(x)


class DeepONet(nn.Module):
    def __init__(self, branch_widths, trunk_widths, basis="bspline", **kan_kwargs):
        super().__init__()
        if branch_widths[-1] != trunk_widths[-1]:
            raise ValueError("branch and trunk must end with the same width")
        self.branch = KAN(branch_widths, basis, **kan_kwargs)
        self.trunk = KAN(trunk_widths, basis, **kan_kwargs)
        self.bias = mx.zeros(())

    def __call__(self, u, y):
        return self.branch(u) @ self.trunk(y).T + self.bias


class FBKAN(nn.Module):
    def __init__(self, widths, bounds, parts=2, overlap: float = 0.25, basis="bspline", **kan_kwargs):
        super().__init__()
        d = len(bounds)
        if widths[0] != d:
            raise ValueError("widths[0] must equal the number of dimensions")
        parts = [parts] * d if isinstance(parts, int) else list(parts)
        boxes = []
        for idx in itertools.product(*(range(p) for p in parts)):
            box = []
            for k, (lo, hi) in enumerate(bounds):
                w = (hi - lo) / parts[k]
                a, b = lo + idx[k] * w, lo + (idx[k] + 1) * w
                box.append((a - overlap * w, b + overlap * w))
            boxes.append(box)
        self._boxes = mx.array(boxes)
        # grid_ranges of a KAN set its first layer: grids on the subdomain
        self.nets = [KAN(widths, basis, grid_ranges=box, **kan_kwargs) for box in boxes]

    def windows(self, x):
        lo, hi = self._boxes[..., 0], self._boxes[..., 1]
        t = (x[:, None, :] - lo) / (hi - lo)
        inside = (t > 0) & (t < 1)
        w = mx.where(inside, mx.sin(math.pi * t) ** 2, 0.0).prod(-1)
        return w / mx.maximum(w.sum(1, keepdims=True), 1e-12)

    def __call__(self, x):
        w = self.windows(x)
        return sum(w[:, j : j + 1] * net(x) for j, net in enumerate(self.nets))


class KANODE(nn.Module):
    def __init__(self, dim: int, hidden=(8,), time_dependent: bool = False, basis="bspline", **kan_kwargs):
        super().__init__()
        self.time_dependent = time_dependent
        self.f = KAN([dim + int(time_dependent), *hidden, dim], basis, **kan_kwargs)

    def rhs(self, t, y):
        if self.time_dependent:
            return self.f(mx.concatenate([y, mx.broadcast_to(t, (y.shape[0], 1))], -1))
        return self.f(y)

    def __call__(self, y0, t, substeps: int = 1):
        ys, y = [y0], y0
        for k in range(t.shape[0] - 1):
            h = (t[k + 1] - t[k]) / substeps
            tk = t[k]
            for _ in range(substeps):
                k1 = self.rhs(tk, y)
                k2 = self.rhs(tk + h / 2, y + h / 2 * k1)
                k3 = self.rhs(tk + h / 2, y + h / 2 * k2)
                k4 = self.rhs(tk + h, y + h * k3)
                y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                tk = tk + h
            ys.append(y)
        return mx.stack(ys)
