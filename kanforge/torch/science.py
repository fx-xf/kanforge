"""Scientific machine learning with KANs: PINN tools, operators, ODEs.

* ``derivative`` / ``gradient`` / ``laplacian`` — derivatives of a field
  u(x) for PDE residuals, kept in the graph for training (any model;
  KAN layers additionally have exact ``edge_derivative``).
* ``sample_box`` / ``sample_boundary`` — collocation points.
* ``DeepONet`` — operator learning with KAN branch and trunk nets (DeepOKAN).
* ``FBKAN`` — finite-basis KAN: overlapping subdomains with a partition of
  unity, one KAN per subdomain with its grid on that subdomain.
* ``KANODE`` — neural ODE dy/dt = KAN(y) (optionally with t) integrated by
  fixed-step RK4, differentiable end to end (KAN-ODEs).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence

import torch
from torch import nn

from .layers import KAN


def derivative(fn: Callable, x: torch.Tensor, order: int = 1, dim: int = 0) -> torch.Tensor:
    """d^order u / dx_dim^order for u = fn(x), x ``(N, d)``, u ``(N,)`` or ``(N, 1)``."""
    x = x if x.requires_grad else x.detach().requires_grad_()
    u = fn(x).reshape(x.shape[0])
    for _ in range(order):
        (g,) = torch.autograd.grad(u.sum(), x, create_graph=True)
        u = g[:, dim]
    return u


def gradient(fn: Callable, x: torch.Tensor) -> torch.Tensor:
    """(N, d) gradient of u = fn(x)."""
    x = x if x.requires_grad else x.detach().requires_grad_()
    (g,) = torch.autograd.grad(fn(x).reshape(x.shape[0]).sum(), x, create_graph=True)
    return g


def laplacian(fn: Callable, x: torch.Tensor) -> torch.Tensor:
    """(N,) sum of second derivatives of u = fn(x)."""
    x = x if x.requires_grad else x.detach().requires_grad_()
    g = gradient(fn, x)
    return sum(torch.autograd.grad(g[:, k].sum(), x, create_graph=True)[0][:, k] for k in range(x.shape[1]))


def sample_box(bounds: Sequence[tuple[float, float]], n: int, generator=None, **kw) -> torch.Tensor:
    """n uniform points inside the box ``bounds`` [(lo, hi), ...]."""
    lo = torch.tensor([b[0] for b in bounds], **kw)
    hi = torch.tensor([b[1] for b in bounds], **kw)
    return lo + (hi - lo) * torch.rand(n, len(bounds), generator=generator, **kw)


def sample_boundary(bounds: Sequence[tuple[float, float]], n: int, generator=None, **kw) -> torch.Tensor:
    """n uniform points on the faces of the box (each face equally likely)."""
    x = sample_box(bounds, n, generator, **kw)
    d = len(bounds)
    face = torch.randint(0, 2 * d, (n,), generator=generator)
    for f in range(2 * d):
        sel = face == f
        x[sel, f // 2] = bounds[f // 2][f % 2]
    return x


class DeepONet(nn.Module):
    """G(u)(y) = sum_k branch_k(u) * trunk_k(y) + b with KAN branch and trunk."""

    def __init__(self, branch_widths: Sequence[int], trunk_widths: Sequence[int], basis="bspline", **kan_kwargs):
        super().__init__()
        if branch_widths[-1] != trunk_widths[-1]:
            raise ValueError("branch and trunk must end with the same width")
        self.branch = KAN(branch_widths, basis, **kan_kwargs)
        self.trunk = KAN(trunk_widths, basis, **kan_kwargs)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """u ``(B, m)`` sensor values, y ``(M, d)`` query points -> ``(B, M)``."""
        return self.branch(u) @ self.trunk(y).T + self.bias


class FBKAN(nn.Module):
    """Finite-basis KAN (domain decomposition, as in NeuroMANCER's FBKAN).

    The box is split into ``parts`` subdomains per dimension, each enlarged by
    ``overlap`` (fraction of its width). Window w_j(x) = prod_d cos^2 bump on
    the enlarged subdomain; u(x) = sum_j w_j(x) KAN_j(x) / sum_j w_j(x).
    """

    def __init__(
        self,
        widths: Sequence[int],
        bounds: Sequence[tuple[float, float]],
        parts: int | Sequence[int] = 2,
        overlap: float = 0.25,
        basis="bspline",
        **kan_kwargs,
    ):
        super().__init__()
        d = len(bounds)
        if widths[0] != d:
            raise ValueError("widths[0] must equal the number of dimensions")
        parts = [parts] * d if isinstance(parts, int) else list(parts)
        boxes = []
        for idx in itertools.product(*(range(p) for p in parts)):
            box = []
            for k, (lo, hi) in enumerate(bounds):
                w = (hi - lo) / parts[k]
                a, b = lo + idx[k] * w, lo + (idx[k] + 1) * w
                box.append((a - overlap * w, b + overlap * w))
            boxes.append(box)
        self.register_buffer("boxes", torch.tensor(boxes))  # (J, d, 2)
        # grid_ranges of a KAN set its first layer: grids on the subdomain
        self.nets = nn.ModuleList(KAN(widths, basis, grid_ranges=box, **kan_kwargs) for box in boxes)

    def windows(self, x: torch.Tensor) -> torch.Tensor:
        """(N, J) normalised partition of unity."""
        lo, hi = self.boxes[..., 0], self.boxes[..., 1]  # (J, d)
        t = (x[:, None, :] - lo) / (hi - lo)  # (N, J, d)
        inside = (t > 0) & (t < 1)
        w = torch.where(inside, torch.sin(torch.pi * t) ** 2, torch.zeros_like(t)).prod(-1)
        return w / w.sum(1, keepdim=True).clamp_min(1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.windows(x)
        return sum(w[:, j : j + 1] * net(x) for j, net in enumerate(self.nets))


class KANODE(nn.Module):
    """dy/dt = f(y) or f(t, y) with f a KAN, integrated by RK4."""

    def __init__(self, dim: int, hidden: Sequence[int] = (8,), time_dependent: bool = False, basis="bspline", **kan_kwargs):
        super().__init__()
        self.time_dependent = time_dependent
        self.f = KAN([dim + int(time_dependent), *hidden, dim], basis, **kan_kwargs)

    def rhs(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.time_dependent:
            return self.f(torch.cat([y, t.expand(y.shape[0], 1)], -1))
        return self.f(y)

    def forward(self, y0: torch.Tensor, t: torch.Tensor, substeps: int = 1) -> torch.Tensor:
        """Trajectory ``(T, B, dim)`` at times ``t`` ``(T,)`` starting from y0 at t[0]."""
        ys, y = [y0], y0
        for k in range(len(t) - 1):
            h = (t[k + 1] - t[k]) / substeps
            tk = t[k]
            for _ in range(substeps):
                k1 = self.rhs(tk, y)
                k2 = self.rhs(tk + h / 2, y + h / 2 * k1)
                k3 = self.rhs(tk + h / 2, y + h / 2 * k2)
                k4 = self.rhs(tk + h, y + h * k3)
                y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                tk = tk + h
            ys.append(y)
        return torch.stack(ys)
