"""Concatenation of bases, e.g. BSRBF-KAN = B-splines + Gaussian RBF."""

from __future__ import annotations

from .._xp import concat
from .base import Basis, make_basis
from .radial import RBF
from .spline import BSpline


class Concat(Basis):
    """All functions of several bases side by side.

    Parameters of part ``i`` are exposed as ``"{i}_{name}"``.
    """

    name = "concat"

    def __init__(self, bases):
        self.bases = [make_basis(b) for b in bases]
        if not self.bases:
            raise ValueError("Concat needs at least one basis")

    @property
    def size(self) -> int:
        return sum(b.size for b in self.bases)

    @property
    def local(self) -> bool:
        return all(b.local for b in self.bases)

    @property
    def support(self) -> int:
        return sum(b.support for b in self.bases)

    def param_specs(self):
        return {f"{i}_{k}": s for i, b in enumerate(self.bases) for k, s in b.param_specs().items()}

    def _eval(self, xe, order, p):
        parts = []
        for i, b in enumerate(self.bases):
            prefix = f"{i}_"
            sub = {k[len(prefix) :]: v for k, v in p.items() if k.startswith(prefix)}
            parts.append(b._eval(xe, order, sub))
        return concat(parts, -1)

    def config(self):
        return {"name": self.name, "bases": [b.config() for b in self.bases]}

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg["bases"])


class BSRBF(Concat):
    """BSRBF-KAN: B-splines and Gaussian RBF on the same grid."""

    name = "bsrbf"

    def __init__(self, grid_size: int = 5, order: int = 3, grid_range: tuple[float, float] = (-1.0, 1.0)):
        spline = BSpline(grid_size, order, grid_range)
        super().__init__([spline, RBF(spline.size, grid_range)])

    def config(self):
        s = self.bases[0]
        return {"name": self.name, "grid_size": s.grid_size, "order": s.order, "grid_range": s.grid_range}

    @classmethod
    def from_config(cls, cfg):
        return Basis.from_config.__func__(cls, cfg)
