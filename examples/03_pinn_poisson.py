"""Physics-informed KAN: solve u'' = -pi^2 sin(pi x) on [-1, 1], u(+-1) = 0.

``derivative`` keeps the graph, so the PDE residual is part of the loss. The
exact solution is sin(pi x).
"""

import math
import os

import torch

import kanforge as kf
from kanforge.torch import KAN, derivative

QUICK = os.environ.get("KANFORGE_QUICK") == "1"
torch.manual_seed(0)
model = KAN([1, 6, 1], kf.BSpline(8, 3), dtype=torch.float64)
xi = torch.linspace(-1, 1, 128, dtype=torch.float64)[:, None]
xb = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
opt = torch.optim.LBFGS(model.parameters(), lr=1, line_search_fn="strong_wolfe", max_iter=50)


def closure():
    opt.zero_grad()
    residual = derivative(model, xi, order=2) + math.pi**2 * torch.sin(math.pi * xi[:, 0])
    loss = (residual**2).mean() + 100 * (model(xb) ** 2).mean()
    loss.backward()
    return loss


for _ in range(2 if QUICK else 20):
    opt.step(closure)
xt = torch.linspace(-1, 1, 200, dtype=torch.float64)[:, None]
print(f"max error vs sin(pi x): {(model(xt)[:, 0] - torch.sin(math.pi * xt[:, 0])).abs().max():.2e}")
