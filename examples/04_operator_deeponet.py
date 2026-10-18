"""Operator learning: DeepONet with KAN branch and trunk learns the antiderivative.

Inputs are functions u sampled at 20 sensors; the model predicts
G(u)(y) = integral_0^y u(s) ds at query points y.
"""

import os

import torch

from kanforge.torch import DeepONet

QUICK = os.environ.get("KANFORGE_QUICK") == "1"
torch.manual_seed(0)
sensors = torch.linspace(0, 1, 20)
ys = torch.linspace(0, 1, 30)[:, None]


def batch(n):
    a, b, c = (torch.randn(n, 1) for _ in range(3))
    u = a + b * torch.cos(3 * sensors) + c * sensors  # functions at the sensors
    y = ys[:, 0]
    g = a * y + b * torch.sin(3 * y) / 3 + c * y**2 / 2  # their antiderivatives at ys
    return u, g


model = DeepONet([20, 16, 8], [1, 16, 8], "chebyshev")
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
for step in range(50 if QUICK else 2000):
    u, g = batch(64)
    opt.zero_grad()
    loss = ((model(u, ys) - g) ** 2).mean()
    loss.backward()
    opt.step()
u, g = batch(256)
print(f"test RMSE {((model(u, ys) - g) ** 2).mean().sqrt():.3e}")
