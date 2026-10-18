"""Image classification with KAN convolutions on 8x8 digits (scikit-learn data).

``KANConv2d`` replaces every kernel weight by a learnable edge function.
"""

import os

import torch
from sklearn.datasets import load_digits
from torch import nn

from kanforge.torch import KANConv2d, KANLinear

QUICK = os.environ.get("KANFORGE_QUICK") == "1"
torch.manual_seed(0)
d = load_digits()
x = torch.tensor(d.images / 8.0 - 1.0, dtype=torch.float32)[:, None]
y = torch.tensor(d.target)
xtr, ytr, xte, yte = x[:1400], y[:1400], x[1400:], y[1400:]

model = nn.Sequential(
    KANConv2d(1, 8, 3, padding=1, basis="chebyshev"),
    nn.MaxPool2d(2),
    KANConv2d(8, 16, 3, padding=1, basis="chebyshev"),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    KANLinear(16, 10, "chebyshev"),
)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
for epoch in range(1 if QUICK else 15):
    for i in range(0, len(xtr), 64):
        opt.zero_grad()
        nn.functional.cross_entropy(model(xtr[i : i + 64]), ytr[i : i + 64]).backward()
        opt.step()
with torch.no_grad():
    print(f"test accuracy {(model(xte).argmax(1) == yte).float().mean():.3f}")
