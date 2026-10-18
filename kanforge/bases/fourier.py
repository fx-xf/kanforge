"""Periodic bases: Fourier series (FourierKAN) and random Fourier features (KAF)."""

from __future__ import annotations

import math

import numpy as np

from .._xp import arange_like, asarray_like, concat, namespace
from .base import Basis, ParamSpec


class Fourier(Basis):
    """cos(w_m x) and sin(w_m x) for w_m = 2*pi*m/period, m = 1..frequencies.

    The constant term is left to the layer bias, as in FourierKAN. With
    ``learnable_frequency`` every input gets a trainable frequency scale.
    """

    name = "fourier"

    def __init__(self, frequencies: int = 5, period: float = 2 * math.pi, learnable_frequency: bool = False):
        if frequencies < 1 or not period > 0:
            raise ValueError("frequencies must be >= 1 and period > 0")
        self.frequencies = int(frequencies)
        self.period = float(period)
        self.learnable_frequency = bool(learnable_frequency)

    @property
    def size(self) -> int:
        return 2 * self.frequencies

    def param_specs(self):
        return {"frequency_scale": ParamSpec(np.ones(1))} if self.learnable_frequency else {}

    def _eval(self, xe, order, p):
        xp = namespace(xe)
        w = (arange_like(self.frequencies, xe) + 1) * (2 * math.pi / self.period)
        if self.learnable_frequency:
            w = w * p["frequency_scale"]
        a = xe * w + order * (math.pi / 2)  # d/dx cos(wx) = w cos(wx + pi/2)
        wm = w**order
        return concat([wm * xp.cos(a), wm * xp.sin(a)], -1)

    def config(self):
        return {
            "name": self.name,
            "frequencies": self.frequencies,
            "period": self.period,
            "learnable_frequency": self.learnable_frequency,
        }


class RandomFourier(Basis):
    """Random Fourier features cos(w_b x + phi_b), w ~ N(0, sigma^2), phi ~ U(0, 2 pi) (KAF).

    Frequencies and phases are fixed and reproducible from ``seed``.
    """

    name = "random_fourier"

    def __init__(self, features: int = 16, sigma: float = 1.0, seed: int = 0):
        if features < 1 or not sigma > 0:
            raise ValueError("features must be >= 1 and sigma > 0")
        self.features = int(features)
        self.sigma = float(sigma)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self._w = rng.normal(0.0, self.sigma, self.features)
        self._phi = rng.uniform(0.0, 2 * math.pi, self.features)

    @property
    def size(self) -> int:
        return self.features

    def _eval(self, xe, order, p):
        xp = namespace(xe)
        w, phi = asarray_like(self._w, xe), asarray_like(self._phi, xe)
        return w**order * xp.cos(xe * w + phi + order * (math.pi / 2))

    def config(self):
        return {"name": self.name, "features": self.features, "sigma": self.sigma, "seed": self.seed}
