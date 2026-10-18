"""Regression with a KAN: fit, adapt the grid to the data, refine it.

KANs learn the 1-D functions on their edges. Grids follow the data
(``update_grid``) and can be refined without retraining from scratch
(``refine_grid``), which is how pykan reaches high accuracy on smooth targets.
"""

import os

import torch

import kanforge as kf
from kanforge.torch import KAN

QUICK = os.environ.get("KANFORGE_QUICK") == "1"
torch.manual_seed(0)

x = torch.rand(1000, 2) * 2 - 1
y = torch.exp(torch.sin(torch.pi * x[:, :1]) + x[:, 1:] ** 2)
model = KAN([2, 5, 1], kf.BSpline(5, 3), coef_init="lecun")


def fit(steps):
    opt = torch.optim.LBFGS(model.parameters(), lr=1, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        return loss

    for _ in range(steps):
        opt.step(closure)
    return ((model(x) - y) ** 2).mean().sqrt().item()


print(f"grid 5:  RMSE {fit(3 if QUICK else 30):.2e}")
model.update_grid(x)  # grids follow the data
model.refine_grid(10)  # finer grid, same function to start with
print(f"grid 10: RMSE {fit(3 if QUICK else 30):.2e}")
