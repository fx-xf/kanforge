"""Forecasting with a recurrent KAN (LSTM gates made of KAN layers)."""

import os

import torch

from kanforge.torch import KANLinear, KANRNN

QUICK = os.environ.get("KANFORGE_QUICK") == "1"
torch.manual_seed(0)
t = torch.arange(0, 400, dtype=torch.float32) * 0.1
series = torch.sin(t) * torch.exp(-0.002 * t) + 0.3 * torch.sin(3.1 * t)
window = 30
xs = torch.stack([series[i : i + window] for i in range(len(series) - window - 1)])[..., None]
ys = series[window + 1 :][: len(xs), None]

rnn, head = KANRNN(1, 16, cell="lstm", basis="chebyshev"), KANLinear(16, 1, "chebyshev")
opt = torch.optim.Adam([*rnn.parameters(), *head.parameters()], lr=5e-3)
for _ in range(10 if QUICK else 300):
    opt.zero_grad()
    out, _ = rnn(xs[:300])
    loss = ((head(out[:, -1]) - ys[:300]) ** 2).mean()
    loss.backward()
    opt.step()
with torch.no_grad():
    out, _ = rnn(xs[300:])
    print(f"test RMSE of one-step forecasts: {((head(out[:, -1]) - ys[300:]) ** 2).mean().sqrt():.3e}")
