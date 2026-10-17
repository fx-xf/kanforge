"""kanforge — Kolmogorov–Arnold Networks for every backend.

Backends are imported explicitly so that NumPy stays the only hard dependency::

    from kanforge.torch import KAN, KANLinear   # CUDA / ROCm / MPS / CPU
    from kanforge.mlx import KAN, KANLinear     # Apple silicon
"""

from .bases import BASES, Basis, BSpline, Chebyshev, Fourier, make_basis

__version__ = "0.0.1"

__all__ = ["BASES", "Basis", "BSpline", "Chebyshev", "Fourier", "make_basis", "__version__"]
