"""FLOPs and parameters of KAN and MLP models, for comparisons at equal cost.

Counts are per sample, forward pass, following KANbeFair's convention: a
multiply-add is 2 FLOPs, elementwise arithmetic 1, a transcendental
function (exp, tanh, sin, ...) ``NONLINEAR`` FLOPs (the same constant for
KANs and MLPs, so neither side is favoured). Basis evaluation is counted
from the algorithm the plain-ops path runs.
"""

from __future__ import annotations

from .bases import (
    BSRBF,
    NURBS,
    RBF,
    RSWAF,
    BSpline,
    Concat,
    DoG,
    Fourier,
    MexicanHat,
    Morlet,
    Neural,
    PolynomialBasis,
    RandomFourier,
    Rational,
    ReLUKAN,
    TruncatedPower,
)
from .interpret._backend import backend_of, to_numpy

NONLINEAR = 5  # FLOPs charged for one exp / tanh / sin / erf ...
ACTIVATION = {"identity": 0, "relu": 1, "silu": NONLINEAR + 2, "gelu": NONLINEAR + 4, "gelu_tanh": NONLINEAR + 6, "tanh": NONLINEAR, "sigmoid": NONLINEAR + 2}


def basis_flops(basis) -> int:
    """FLOPs to evaluate all functions of ``basis`` at one input value."""
    n = basis.size
    if isinstance(basis, BSRBF) or isinstance(basis, Concat):
        return sum(basis_flops(b) for b in basis.bases)
    if isinstance(basis, NURBS):
        return basis_flops(basis._spline) + 3 * n + 1
    if isinstance(basis, BSpline):
        m = basis.grid_size + 2 * basis.order + 1  # knots
        level0 = 3 * (m - 1)  # two comparisons and an and per interval
        return level0 + sum(8 * (m - 1 - q) for q in range(1, basis.order + 1))
    if isinstance(basis, ReLUKAN):
        return 6 * n
    if isinstance(basis, TruncatedPower):
        return 4 * (basis.degree + 1) + n * (3 + basis.degree)
    if isinstance(basis, PolynomialBasis):
        cost = {"tanh": NONLINEAR, "identity": 0, "fractional": 3 * NONLINEAR + 4}[basis.input_map]
        cost += 3 * max(basis.degree - 1, 0)  # Chebyshev recurrence
        if type(basis).__name__ != "Chebyshev":
            cost += 2 * (basis.degree + 1) * n  # conversion from Chebyshev coefficients
        return cost
    if isinstance(basis, Fourier):
        return basis.frequencies * (2 + 2 * NONLINEAR)
    if isinstance(basis, RandomFourier):
        return n * (2 + NONLINEAR)
    if isinstance(basis, (RBF, MexicanHat, DoG, Morlet)):
        extra = 2 * NONLINEAR if isinstance(basis, Morlet) else 3
        return n * (4 + NONLINEAR + extra)
    if isinstance(basis, RSWAF):
        return n * (4 + NONLINEAR)
    if isinstance(basis, Rational):
        return 2 * (basis.num_degree + basis.den_degree) + 3
    if isinstance(basis, Neural):
        return n * (2 + NONLINEAR + (2 if basis.activation == "silu" else 0))
    return 10 * n


def _act_flops(act) -> int:
    from .convert import activation_name

    try:
        return ACTIVATION[activation_name(act)]
    except (ValueError, KeyError):
        return NONLINEAR


def kan_linear_flops(layer) -> int:
    i, o, n = layer.in_features, layer.out_features, layer.basis.size
    flops = i * basis_flops(layer.basis) + 2 * i * n * o
    if layer.base_weight is not None:
        flops += i * _act_flops(layer._init_kwargs["base_activation"]) + 2 * i * o
    if layer.bias is not None:
        flops += o
    return flops


def count(model) -> dict:
    """{"flops": per-sample forward FLOPs, "params": trainable parameters}.

    Handles KAN layers and models of every backend, and torch ``nn.Linear`` /
    activations / ``nn.Sequential`` MLPs for the comparison side.
    """
    flops = 0

    def walk(m):
        nonlocal flops
        name = type(m).__name__
        if name in ("KANLinear", "GRKAN"):
            flops += kan_linear_flops(m)
        elif name == "MultKANLinear":
            walk(m.kan)
            flops += m.n_mult * (m.arity - 1)
        elif name == "Linear":
            w = to_numpy(m.kernel).T if hasattr(m, "kernel") else to_numpy(m.weight)
            flops += 2 * w.size + (w.shape[0] if getattr(m, "bias", None) is not None else 0)
        elif name in ("Sequential", "KAN", "MultKAN"):
            for c in list(getattr(m, "layers", m)):
                walk(c)
        else:
            flops += _act_flops(m) * _last_width[0]
        if name in ("KANLinear", "GRKAN", "Linear"):
            _last_width[0] = m.out_features if hasattr(m, "out_features") else to_numpy(m.kernel).shape[1]
        elif name == "MultKANLinear":
            _last_width[0] = m.out_features

    _last_width = [0]
    walk(model)
    return {"flops": int(flops), "params": _params(model)}


def _params(model) -> int:
    b = backend_of(model)
    name = b.__name__.rsplit(".", 1)[-1]
    if name == "torch":
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    if name == "mlx":
        from mlx.utils import tree_flatten

        return sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    from flax import nnx

    return sum(v.size for _, v in nnx.to_flat_state(nnx.state(model, nnx.Param)))
