"""Basis functions for KAN edges. See ``base.py`` for the contract."""

from .base import BASES, Basis, ParamSpec, make_basis
from .combine import BSRBF, Concat
from .fourier import Fourier, RandomFourier
from .neural import Neural
from .polynomial import (
    Bernstein,
    Chebyshev,
    Chebyshev2,
    FractionalJacobi,
    Gegenbauer,
    Gram,
    Hermite,
    Jacobi,
    Laguerre,
    Legendre,
    PolynomialBasis,
    Taylor,
)
from .radial import RBF, RSWAF, DoG, MexicanHat, Morlet
from .rational import Rational, fit_rational
from .spline import NURBS, BSpline, ReLUKAN, TruncatedPower

__all__ = [
    "BASES",
    "BSRBF",
    "Basis",
    "Bernstein",
    "BSpline",
    "Chebyshev",
    "Chebyshev2",
    "Concat",
    "DoG",
    "Fourier",
    "FractionalJacobi",
    "Gegenbauer",
    "Gram",
    "Hermite",
    "Jacobi",
    "Laguerre",
    "Legendre",
    "MexicanHat",
    "Morlet",
    "NURBS",
    "Neural",
    "ParamSpec",
    "PolynomialBasis",
    "RBF",
    "RSWAF",
    "RandomFourier",
    "Rational",
    "ReLUKAN",
    "Taylor",
    "TruncatedPower",
    "fit_rational",
    "make_basis",
]
