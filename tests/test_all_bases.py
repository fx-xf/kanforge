"""Every basis: shapes, backend parity and exact derivatives."""

import math

import numpy as np
import pytest
from numpy.polynomial import hermite as H
from numpy.polynomial import laguerre as La
from numpy.polynomial import legendre as Le

import kanforge as kf
from cases import CASES, perturbed_params, sample_x

IDS = list(CASES)


@pytest.mark.parametrize("name", IDS)
def test_shape_and_finite(name):
    basis = CASES[name]
    x = sample_x()
    for order in range(4):
        y = basis.derivative(x, order, perturbed_params(basis, 3))
        assert y.shape == (40, 3, basis.size)
        assert np.isfinite(y).all()


@pytest.mark.parametrize("name", IDS)
def test_config_roundtrip(name):
    basis = CASES[name]
    rebuilt = kf.make_basis(basis.config())
    assert rebuilt == basis
    x = sample_x()
    np.testing.assert_allclose(rebuilt(x), basis(x))


@pytest.mark.parametrize("name", IDS)
def test_torch_matches_numpy_and_autograd(name):
    torch = pytest.importorskip("torch")
    basis = CASES[name]
    x_np = sample_x(12, 2)
    params_np = perturbed_params(basis, 2)
    params = {k: torch.tensor(v) for k, v in params_np.items()}
    x = torch.tensor(x_np, requires_grad=True)
    values = basis(x, params)
    np.testing.assert_allclose(values.detach().numpy(), basis(x_np, params_np), rtol=1e-10, atol=1e-12)

    # Every B_b(x_i) depends on x_i only, so grad of the sum is elementwise.
    for order in (1, 2, 3):
        analytic = basis.derivative(x, order, params).detach().numpy()
        for b in range(basis.size):
            d = values[..., b]
            for _ in range(order):
                (d,) = torch.autograd.grad(d.sum(), x, create_graph=True)
            np.testing.assert_allclose(
                analytic[..., b], d.detach().numpy(), rtol=1e-7, atol=1e-8, err_msg=f"order {order}, b={b}"
            )


@pytest.mark.parametrize("name", IDS)
def test_mlx_matches_numpy(name):
    mx = pytest.importorskip("mlx.core")
    basis = CASES[name]
    x_np = sample_x(12, 2)
    params_np = perturbed_params(basis, 2)
    with mx.stream(mx.cpu):
        params = {k: mx.array(v, dtype=mx.float32) for k, v in params_np.items()}
        x = mx.array(x_np, dtype=mx.float32)
        for order in (0, 1, 2):
            got = np.array(basis.derivative(x, order, params))
            want = basis.derivative(x_np, order, params_np)
            np.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-3 * max(1.0, np.abs(want).max()))


# --- reference values of the families -------------------------------------


def test_bspline_derivative_matches_scipy():
    interp = pytest.importorskip("scipy.interpolate")
    basis = kf.BSpline(6, 3, (-1.0, 2.0))
    x = np.random.default_rng(0).uniform(-1, 2, 50)
    t = basis.knots()
    for order in (1, 2, 3):
        for j in range(basis.size):
            c = np.zeros(basis.size)
            c[j] = 1
            want = interp.BSpline(t, c, 3)(x, nu=order)
            np.testing.assert_allclose(basis.derivative(x, order)[:, j], want, atol=1e-9)


def test_orthogonal_families_match_numpy_and_scipy():
    special = pytest.importorskip("scipy.special")
    x = np.random.default_rng(0).normal(size=30)
    z = np.tanh(x)
    np.testing.assert_allclose(kf.Legendre(6)(x), Le.legvander(z, 6), atol=1e-10)
    np.testing.assert_allclose(kf.Hermite(5)(x), H.hermvander(z, 5), atol=1e-9)
    np.testing.assert_allclose(kf.Laguerre(5)(x), La.lagvander(z, 5), atol=1e-10)
    jac = np.stack([special.eval_jacobi(n, 0.5, 1.5, z) for n in range(6)], -1)
    np.testing.assert_allclose(kf.Jacobi(5, 0.5, 1.5)(x), jac, atol=1e-10)
    geg = np.stack([special.eval_gegenbauer(n, 0.75, z) for n in range(6)], -1)
    np.testing.assert_allclose(kf.Gegenbauer(5, 0.75)(x), geg, atol=1e-10)
    u2 = np.stack([special.eval_chebyu(n, z) for n in range(6)], -1)
    np.testing.assert_allclose(kf.Chebyshev2(5)(x), u2, atol=1e-10)
    np.testing.assert_allclose(kf.Taylor(4)(x), np.stack([x**j for j in range(5)], -1), atol=1e-10)


def test_bernstein_partition_of_unity():
    x = np.linspace(-3, 3, 101)
    np.testing.assert_allclose(kf.Bernstein(6)(x).sum(-1), 1.0, atol=1e-12)


def test_gram_is_orthonormal_on_its_points():
    g = kf.Gram(4, points=12, input_map="identity")
    v = g(np.linspace(-1, 1, 12))
    np.testing.assert_allclose(v.T @ v / 12, np.eye(5), atol=1e-10)


def test_fractional_jacobi_formula():
    special = pytest.importorskip("scipy.special")
    basis = kf.FractionalJacobi(3, 1.0, 1.0)
    x = np.linspace(-2, 2, 9)
    params = {"exponent_logit": np.array([0.4])}
    gamma = 1 / (1 + np.exp(-0.4))
    z = 2 * (1 / (1 + np.exp(-x))) ** gamma - 1
    want = np.stack([special.eval_jacobi(n, 1.0, 1.0, z) for n in range(4)], -1)
    np.testing.assert_allclose(basis(x, params), want, atol=1e-10)


def test_bumps():
    x = np.linspace(-1, 1, 7)
    c = np.linspace(-1, 1, 5)
    np.testing.assert_allclose(kf.RBF(5)(x), np.exp(-(((x[:, None] - c) / 0.5) ** 2)), atol=1e-12)
    u = (x[:, None] - c) / 0.5
    hat = 2 / (math.sqrt(3) * math.pi**0.25) * (1 - u**2) * np.exp(-(u**2) / 2)
    np.testing.assert_allclose(kf.MexicanHat(5)(x), hat, atol=1e-12)
    np.testing.assert_allclose(kf.Morlet(5, omega=3.0)(x), np.cos(3 * u) * np.exp(-(u**2) / 2), atol=1e-12)
    np.testing.assert_allclose(kf.DoG(5)(x), -u * np.exp(-(u**2) / 2), atol=1e-12)
    np.testing.assert_allclose(kf.RSWAF(5)(x), 1 - np.tanh(u) ** 2, atol=1e-12)


def test_relu_kan_peaks_and_support():
    basis = kf.ReLUKAN(4, 3)
    x = np.linspace(-1, 1, 401)
    v = basis(x)
    assert v.max() == pytest.approx(1.0, abs=1e-3)
    assert ((v > 0).sum(-1) <= basis.support).all()


def test_truncated_power_formula():
    basis = kf.TruncatedPower(3, 2, (-1.0, 1.0))
    x = np.linspace(-1, 1, 11)
    knots = np.array([-1 / 3, 1 / 3])
    trunc = np.maximum((x[:, None] - knots) / (1 - knots), 0) ** 3
    np.testing.assert_allclose(basis(x)[:, :4], Le.legvander(x, 3), atol=1e-12)
    np.testing.assert_allclose(basis(x)[:, 4:], trunc, atol=1e-12)


def test_nurbs_with_equal_weights_is_bspline():
    x = sample_x()
    np.testing.assert_allclose(kf.NURBS(5, 3)(x), kf.BSpline(5, 3)(x), atol=1e-12)


@pytest.mark.parametrize("target", ["identity", "silu", "gelu", "tanh"])
def test_rational_init_fits_target(target):
    from kanforge.bases.rational import TARGETS

    x = np.linspace(-3, 3, 301)
    got = kf.Rational(5, 4, target)(x)[:, 0]
    assert np.abs(got - TARGETS[target](x)).max() < 2e-2


def test_concat_routes_parameters():
    basis = CASES["concat"]
    assert set(basis.param_specs()) == {"0_grid", "1_frequency_scale"}
    assert basis.grid_names() == ["0_grid"]
    x = sample_x()
    params = {**basis.default_params(x), "1_frequency_scale": np.array([[2.0]] * 3)}
    np.testing.assert_allclose(basis(x, params)[..., 6:], kf.Fourier(2)(2 * x), atol=1e-12)


def test_unknown_names():
    with pytest.raises(ValueError, match="unknown basis"):
        kf.make_basis("nope")
    with pytest.raises(ValueError, match="unknown basis"):
        kf.make_basis({"name": "nope"})
