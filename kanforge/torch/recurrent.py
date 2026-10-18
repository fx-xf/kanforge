"""Recurrent KAN cells: LSTM and GRU whose gates are KAN layers (in the spirit of TKAN).

The gate pre-activations come from a ``KANLinear`` of [x_t, h_{t-1}] instead
of an affine map; the gating itself is the usual LSTM / GRU algebra.
``KANRNN`` runs a stack of cells over a sequence.
"""

from __future__ import annotations

import torch
from torch import nn

from .layers import KANLinear


class KANLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, basis="bspline", **kan_kwargs):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.gates = KANLinear(input_size + hidden_size, 4 * hidden_size, basis, **kan_kwargs)

    def forward(self, x: torch.Tensor, state=None):
        if state is None:
            zeros = x.new_zeros(x.shape[0], self.hidden_size)
            state = (zeros, zeros)
        h, c = state
        i, f, g, o = self.gates(torch.cat([x, h], -1)).chunk(4, -1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, (h, c)


class KANGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, basis="bspline", **kan_kwargs):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.gates = KANLinear(input_size + hidden_size, 2 * hidden_size, basis, **kan_kwargs)
        self.candidate = KANLinear(input_size + hidden_size, hidden_size, basis, **kan_kwargs)

    def forward(self, x: torch.Tensor, state=None):
        h = x.new_zeros(x.shape[0], self.hidden_size) if state is None else state
        z, r = torch.sigmoid(self.gates(torch.cat([x, h], -1))).chunk(2, -1)
        n = torch.tanh(self.candidate(torch.cat([x, r * h], -1)))
        h = (1 - z) * n + z * h
        return h, h


class KANRNN(nn.Module):
    """Stack of ``cell`` ("lstm" or "gru") layers over (B, T, F) sequences (batch first)."""

    CELLS = {"lstm": KANLSTMCell, "gru": KANGRUCell}

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, cell: str = "lstm", basis="bspline", **kan_kwargs):
        super().__init__()
        if cell not in self.CELLS:
            raise ValueError(f"cell must be one of {sorted(self.CELLS)}")
        cls = self.CELLS[cell]
        self.cells = nn.ModuleList(
            cls(input_size if k == 0 else hidden_size, hidden_size, basis, **kan_kwargs) for k in range(num_layers)
        )

    def forward(self, x: torch.Tensor, states=None):
        """Returns (outputs (B, T, H) of the last layer, final state of every layer)."""
        states = list(states) if states is not None else [None] * len(self.cells)
        seq = x
        for k, cell in enumerate(self.cells):
            outs, state = [], states[k]
            for t in range(seq.shape[1]):
                out, state = cell(seq[:, t], state)
                outs.append(out)
            seq = torch.stack(outs, 1)
            states[k] = state
        return seq, states
