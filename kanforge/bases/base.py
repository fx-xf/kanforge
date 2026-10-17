"""The ``Basis`` contract shared by every backend.

Every edge function is phi(x) = sum_b c_b * B_b(x; theta). A basis maps an
input of shape ``(..., in)`` to ``(..., in, size)`` and can return any
derivative order of the same shape.

``theta`` are optional basis parameters: trainable ones (rational
coefficients, wavelet scales, knot positions, ...) and fixed state such as
the grid of a spline, which differs per input and is updated from data. A basis declares them with
``param_specs()``: one default value per parameter, shaped for a *single*
input. Layers store them per input, per group of inputs (GR-KAN) or once
per layer and pass them in shaped ``(in, *shape)`` or ``(*shape)``; both
broadcast against ``x[..., None]``.

``local`` marks bases with compact support (at most ``support`` non-zero
values per input); fused kernels will use it to pick a sparse gather.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._xp import asarray_like


@dataclass(frozen=True)
class ParamSpec:
    """Default value of one basis parameter for a single input.

    ``trainable=False`` marks state such as grids: layers keep it as a buffer,
    always one copy per input, and grid tools update it from data.
    """

    value: np.ndarray
    trainable: bool = True

    @property
    def shape(self) -> tuple[int, ...]:
        return np.shape(self.value)


BASES: dict[str, type[Basis]] = {}


class Basis:
    name: str = ""
    local: bool = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("name"):
            BASES[cls.name] = cls

    @property
    def size(self) -> int:
        raise NotImplementedError

    @property
    def support(self) -> int:
        """Max number of non-zero basis values per input."""
        return self.size

    def param_specs(self) -> dict[str, ParamSpec]:
        return {}

    def grid_names(self) -> list[str]:
        """Names of the grid buffers (``"grid"``, or ``"{i}_grid"`` in ``Concat``)."""
        return [k for k, s in self.param_specs().items() if not s.trainable and k.endswith("grid")]

    def default_params(self, like) -> dict[str, Any]:
        return {k: asarray_like(s.value, like) for k, s in self.param_specs().items()}

    def __call__(self, x, params: dict | None = None):
        return self.derivative(x, 0, params)

    def derivative(self, x, order: int = 1, params: dict | None = None):
        """d^order/dx^order of every basis function, shape ``(..., in, size)``."""
        if order < 0:
            raise ValueError("order must be >= 0")
        if params is None:
            params = self.default_params(x)
        return self._eval(x[..., None], int(order), params)

    def _eval(self, xe, order: int, p: dict):
        """``xe`` is the input with a trailing axis: ``(..., in, 1)``."""
        raise NotImplementedError

    def config(self) -> dict[str, Any]:
        raise NotImplementedError

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.config().items() if k != "name")
        return f"{type(self).__name__}({args})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Basis) and self.config() == other.config()

    def __hash__(self) -> int:
        return hash(repr(self.config()))


def _check_range(grid_range) -> tuple[float, float]:
    lo, hi = float(grid_range[0]), float(grid_range[1])
    if not hi > lo:
        raise ValueError("grid_range must be increasing")
    return lo, hi


def make_basis(spec: str | dict | Basis = "bspline", **kwargs) -> Basis:
    """Build a basis from a name, a ``config()`` dict or an existing instance."""
    if isinstance(spec, Basis):
        if kwargs:
            raise TypeError("kwargs are not allowed with a Basis instance")
        return spec
    if isinstance(spec, dict):
        spec = dict(spec)
        name = spec.pop("name")
        cls = BASES.get(name)
        if cls is None:
            raise ValueError(f"unknown basis {name!r}; available: {sorted(BASES)}")
        return cls.from_config({**spec, **kwargs})
    cls = BASES.get(spec)
    if cls is None:
        raise ValueError(f"unknown basis {spec!r}; available: {sorted(BASES)}")
    return cls(**kwargs)


def _from_config(cls, cfg: dict) -> Basis:
    cfg = {k: tuple(v) if k == "grid_range" else v for k, v in cfg.items()}
    return cls(**cfg)


Basis.from_config = classmethod(_from_config)
