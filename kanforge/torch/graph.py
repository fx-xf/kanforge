"""Graph KAN layers (KAGNNs): message passing with KAN transformations.

Graphs are given as ``edge_index`` ``(2, E)`` with rows (source, target), the
PyTorch Geometric convention, so the layers drop into PyG pipelines but do
not need PyG.

* ``KAGINConv`` — h_v' = KAN((1 + eps) h_v + sum_{u -> v} h_u) (GIN with a KAN).
* ``KAGCNConv`` — h' = KAN(D^-1/2 (A + I) D^-1/2 h) (GCN aggregation, then KAN).
* ``KAGATConv`` — attention over neighbours of KAN-transformed features (GAT).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .layers import KAN, KANLinear


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    loops = torch.arange(num_nodes, device=edge_index.device).repeat(2, 1)
    return torch.cat([edge_index, loops], 1)


def aggregate(x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int, weight: torch.Tensor | None = None) -> torch.Tensor:
    """out[v] = sum over edges (u -> v) of weight * x[u]."""
    src, dst = edge_index
    msg = x[src] if weight is None else x[src] * weight.reshape(-1, *([1] * (x.dim() - 1)))
    return torch.zeros(num_nodes, *x.shape[1:], dtype=x.dtype, device=x.device).index_add_(0, dst, msg)


def _kan(in_features, out_features, hidden, basis, kan_kwargs):
    if hidden:
        return KAN([in_features, *hidden, out_features], basis, **kan_kwargs)
    return KANLinear(in_features, out_features, basis, **kan_kwargs)


class KAGINConv(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, basis="bspline", hidden: Sequence[int] = (), eps: float = 0.0, train_eps: bool = False, **kan_kwargs
    ):
        super().__init__()
        self.kan = _kan(in_features, out_features, hidden, basis, kan_kwargs)
        self.eps = nn.Parameter(torch.tensor(float(eps))) if train_eps else eps

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.kan((1 + self.eps) * x + aggregate(x, edge_index, x.shape[0]))


class KAGCNConv(nn.Module):
    def __init__(self, in_features: int, out_features: int, basis="bspline", hidden: Sequence[int] = (), **kan_kwargs):
        super().__init__()
        self.kan = _kan(in_features, out_features, hidden, basis, kan_kwargs)

    @staticmethod
    def normalized_weights(edge_index: torch.Tensor, num_nodes: int, dtype=torch.float32):
        ei = add_self_loops(edge_index, num_nodes)
        deg = torch.zeros(num_nodes, dtype=dtype, device=ei.device).index_add_(0, ei[1], torch.ones(ei.shape[1], dtype=dtype, device=ei.device))
        inv = deg.rsqrt()
        return ei, inv[ei[0]] * inv[ei[1]]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        ei, w = self.normalized_weights(edge_index, x.shape[0], x.dtype)
        return self.kan(aggregate(x, ei, x.shape[0], w))


class KAGATConv(nn.Module):
    """Graph attention with a KAN feature transform; ``heads`` concatenated or averaged."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 1,
        basis="bspline",
        concat: bool = True,
        negative_slope: float = 0.2,
        add_loops: bool = True,
        **kan_kwargs,
    ):
        super().__init__()
        self.heads, self.out_features, self.concat = heads, out_features, concat
        self.negative_slope, self.add_loops = negative_slope, add_loops
        self.kan = KANLinear(in_features, heads * out_features, basis, **kan_kwargs)
        self.att_src = nn.Parameter(torch.empty(heads, out_features))
        self.att_dst = nn.Parameter(torch.empty(heads, out_features))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, return_attention: bool = False):
        n = x.shape[0]
        ei = add_self_loops(edge_index, n) if self.add_loops else edge_index
        src, dst = ei
        z = self.kan(x).view(n, self.heads, self.out_features)
        logits = F.leaky_relu((z * self.att_src).sum(-1)[src] + (z * self.att_dst).sum(-1)[dst], self.negative_slope)  # (E, H)
        top = torch.full((n, self.heads), -torch.inf, dtype=logits.dtype, device=x.device)
        top = top.scatter_reduce(0, dst[:, None].expand_as(logits), logits, reduce="amax", include_self=True)
        w = torch.exp(logits - top[dst])
        denom = torch.zeros(n, self.heads, dtype=w.dtype, device=x.device).index_add_(0, dst, w)
        alpha = w / denom[dst]
        out = torch.zeros_like(z).index_add_(0, dst, z[src] * alpha[..., None])
        out = out.reshape(n, -1) if self.concat else out.mean(1)
        return (out, alpha) if return_attention else out
