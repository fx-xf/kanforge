"""Node classification on a graph with KAN message passing (KAGCN / KAGIN).

Two communities with noisy features; graphs come as ``edge_index`` (PyG layout).
"""

import os

import torch

from kanforge.torch import KAGCNConv, KAGINConv

QUICK = os.environ.get("KANFORGE_QUICK") == "1"
torch.manual_seed(0)
n = 200
labels = torch.arange(n) % 2
x = labels[:, None].float() * 0.6 - 0.3 + 0.5 * torch.randn(n, 4)
src, dst = torch.randint(0, n, (2, 2000))
same = labels[src] == labels[dst]
keep = same | (torch.rand(2000) < 0.1)  # mostly edges inside a community
edge_index = torch.stack([src[keep], dst[keep]])
train = torch.rand(n) < 0.3


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn = KAGCNConv(4, 8, "chebyshev")
        self.gin = KAGINConv(8, 2, "chebyshev")

    def forward(self, x, ei):
        return self.gin(torch.tanh(self.gcn(x, ei)), ei)


model = Net()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for _ in range(20 if QUICK else 300):
    opt.zero_grad()
    torch.nn.functional.cross_entropy(model(x, edge_index)[train], labels[train]).backward()
    opt.step()
pred = model(x, edge_index).argmax(1)
print(f"accuracy on unlabelled nodes: {(pred[~train] == labels[~train]).float().mean():.3f}")
