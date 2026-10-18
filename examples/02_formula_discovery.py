"""Find the formula behind data: train sparse, prune, make edges symbolic.

The pykan workflow: train with the sparsity penalty, prune unimportant nodes
and edges, refit, then ``auto_symbolic`` picks c * f(a x + b) + d for every
edge and ``symbolic_formula`` composes them with SymPy.
"""

import os

import torch

import kanforge as kf
from kanforge import interpret as ki
from kanforge.torch import KAN, regularization

QUICK = os.environ.get("KANFORGE_QUICK") == "1"
torch.manual_seed(0)
x = torch.rand(2000, 2) * 2 - 1
y = torch.exp(torch.sin(torch.pi * x[:, :1]) + x[:, 1:] ** 2)


def fit(model, steps, lam):
    opt = torch.optim.LBFGS(model.parameters(), lr=1, history_size=10, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean() + (lam * regularization(model, x) if lam else 0)
        loss.backward()
        return loss

    for _ in range(steps):
        opt.step(closure)


model = KAN([2, 5, 1], kf.BSpline(5, 3), coef_init="lecun")
fit(model, 5 if QUICK else 60, 1e-2)
print("feature importance:", ki.feature_importance(model, x))
model = ki.prune(model, x, node_threshold=0.05, edge_threshold=0.05)
print("after pruning:", model.widths)
fit(model, 3 if QUICK else 30, 0)
ki.auto_symbolic(model, x)
try:
    print("formula:", ki.symbolic_formula(model, ["x", "y"], digits=3))
except ValueError as e:  # some edges did not fit any library function well
    print("not fully symbolic:", e)
