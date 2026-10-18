"""Policy gradient (REINFORCE) with a KAN policy on a small continuous task.

The agent moves on a line and is rewarded for reaching a random target; the
Gaussian policy's mean is a KAN of (position, target). No gym needed.
"""

import os

import torch

from kanforge.torch import KAN

QUICK = os.environ.get("KANFORGE_QUICK") == "1"
torch.manual_seed(0)
policy = KAN([2, 6, 1], "chebyshev")
log_std = torch.nn.Parameter(torch.tensor(-0.5))
opt = torch.optim.Adam([*policy.parameters(), log_std], lr=1e-2)


def episode(n=64, steps=10):
    pos, target = torch.zeros(n), torch.rand(n) * 2 - 1
    logps, rewards = [], []
    for _ in range(steps):
        mean = policy(torch.stack([pos, target], 1))[:, 0].tanh() * 0.3
        dist = torch.distributions.Normal(mean, log_std.exp())
        a = dist.sample()
        logps.append(dist.log_prob(a))
        pos = pos + a
        rewards.append(-(pos - target).abs())
    returns = torch.stack(rewards).flip(0).cumsum(0).flip(0)
    return torch.stack(logps), returns, rewards[-1].mean()


for it in range(5 if QUICK else 300):
    logp, ret, final = episode()
    opt.zero_grad()
    (-(logp * (ret - ret.mean(1, keepdim=True))).mean()).backward()
    opt.step()
print(f"mean final distance to target: {-final.item():.3f}")
