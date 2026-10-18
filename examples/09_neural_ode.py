"""Learn the dynamics of a damped oscillator with a KAN-ODE (dy/dt = KAN(y))."""

import os

import torch

from kanforge.torch import KANODE

QUICK = os.environ.get("KANFORGE_QUICK") == "1"
torch.manual_seed(0)


def true_rhs(y):
    return torch.stack([y[:, 1], -y[:, 0] - 0.2 * y[:, 1]], 1)


t = torch.linspace(0, 4, 41)
y0 = torch.randn(32, 2)
with torch.no_grad():  # reference trajectories (RK4 on the true system)
    ref = [y0]
    for k in range(len(t) - 1):
        h, y = t[k + 1] - t[k], ref[-1]
        k1 = true_rhs(y)
        k2 = true_rhs(y + h / 2 * k1)
        k3 = true_rhs(y + h / 2 * k2)
        ref.append(y + h / 6 * (k1 + 2 * k2 + 2 * k3 + true_rhs(y + h * k3)))
    ref = torch.stack(ref)

ode = KANODE(2, hidden=(8,), basis="chebyshev")
opt = torch.optim.Adam(ode.parameters(), lr=5e-3)
for _ in range(10 if QUICK else 400):
    opt.zero_grad()
    loss = ((ode(y0, t) - ref) ** 2).mean()
    loss.backward()
    opt.step()
print(f"trajectory RMSE: {loss.sqrt().item():.3e}")
