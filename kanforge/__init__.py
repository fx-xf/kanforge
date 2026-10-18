"""kanforge — Kolmogorov–Arnold Networks for every backend.

Backends are imported explicitly so that NumPy stays the only hard dependency::

    from kanforge.torch import KAN, KANLinear   # CUDA / ROCm / MPS / CPU
    from kanforge.mlx import KAN, KANLinear     # Apple silicon
    from kanforge.jax import KAN, KANLinear     # JAX / Flax NNX: CPU, GPU, TPU
"""

from . import bases
from .bases import *  # noqa: F401,F403
from .bases import __all__ as _bases_all
from . import convert
from .convert import load, save

__version__ = "0.7.0.dev0"

__all__ = [*_bases_all, "bases", "convert", "load", "save", "__version__"]  # convert: the module
