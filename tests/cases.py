"""One instance of every basis, with non-default trainable parameters."""

import numpy as np

import kanforge as kf

CASES = {
    "bspline": kf.BSpline(5, 3),
    "nurbs": kf.NURBS(5, 3),
    "truncated_power": kf.TruncatedPower(3, 4),
    "truncated_power_learnable": kf.TruncatedPower(3, 4, learn_knots=True),
    "relu_kan": kf.ReLUKAN(4, 3),
    "chebyshev": kf.Chebyshev(5),
    "chebyshev_identity": kf.Chebyshev(4, input_map="identity"),
    "chebyshev2": kf.Chebyshev2(5),
    "legendre": kf.Legendre(5),
    "jacobi": kf.Jacobi(5, 0.5, 1.5),
    "fractional_jacobi": kf.FractionalJacobi(4),
    "gegenbauer": kf.Gegenbauer(5, 0.75),
    "hermite": kf.Hermite(4),
    "laguerre": kf.Laguerre(4),
    "bernstein": kf.Bernstein(5),
    "gram": kf.Gram(5),
    "taylor": kf.Taylor(4),
    "rbf": kf.RBF(6),
    "rbf_learnable": kf.RBF(6, learnable=True),
    "mexican_hat": kf.MexicanHat(5),
    "morlet": kf.Morlet(5),
    "dog": kf.DoG(5),
    "rswaf": kf.RSWAF(6),
    "fourier": kf.Fourier(3),
    "fourier_learnable": kf.Fourier(3, learnable_frequency=True),
    "random_fourier": kf.RandomFourier(6, sigma=2.0, seed=3),
    "rational": kf.Rational(5, 4, "gelu"),
    "neural_tanh": kf.Neural(5, "tanh"),
    "neural_silu": kf.Neural(5, "silu"),
    "concat": kf.Concat([kf.BSpline(4, 2), kf.Fourier(2, learnable_frequency=True)]),
    "bsrbf": kf.BSRBF(4, 3),
}


def perturbed_params(basis, n_inputs, seed=0):
    """Per-input parameters = defaults + small noise, so per-input plumbing is exercised."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, spec in basis.param_specs().items():
        value = np.broadcast_to(spec.value, (n_inputs, *spec.shape)).astype(np.float64)
        out[name] = value + 0.05 * rng.standard_normal(value.shape)
    return out


def sample_x(n=40, n_inputs=3, seed=1):
    return np.random.default_rng(seed).uniform(-0.95, 0.95, size=(n, n_inputs))
