import math

import numpy as np
import pytest
from numpy.polynomial import chebyshev as C

from kanforge import BSpline, Chebyshev, Fourier, make_basis
from kanforge.reference import kan_linear

rng = np.random.default_rng(0)


@pytest.mark.parametrize("order", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("grid_size", [1, 4, 9])
def test_bspline_matches_scipy(order, grid_size):
    scipy_bspline = pytest.importorskip("scipy.interpolate").BSpline
    basis = BSpline(grid_size, order, (-1.5, 2.0))
    x = rng.uniform(-1.5, 2.0, size=200)
    expected = scipy_bspline.design_matrix(x, basis.knots(), order).toarray()
    np.testing.assert_allclose(basis(x), expected, atol=1e-12)


def test_bspline_partition_of_unity_and_support():
    basis = BSpline(7, 3, (-2.0, 3.0))
    x = np.linspace(-2.0, 3.0, 1001)
    b = basis(x)
    assert b.shape == (1001, basis.size)
    np.testing.assert_allclose(b.sum(-1), 1.0, atol=1e-12)
    assert (b >= -1e-15).all()
    assert ((np.abs(b) > 1e-14).sum(-1) <= basis.support).all()


def test_bspline_is_zero_outside_extended_grid():
    basis = BSpline(4, 3, (-1.0, 1.0))
    far = np.array([-10.0, -2.6, 2.6, 10.0])  # extended grid is [-2.5, 2.5)
    np.testing.assert_array_equal(basis(far), 0.0)


def test_chebyshev_matches_numpy():
    basis = Chebyshev(6)
    x = rng.normal(size=300) * 2
    expected = C.chebvander(np.tanh(x), 6)
    np.testing.assert_allclose(basis(x), expected, atol=1e-12)


def test_fourier_values():
    basis = Fourier(3, period=4.0)
    x = rng.normal(size=50)
    w = 2 * math.pi / 4.0 * np.arange(1, 4)
    expected = np.concatenate([np.cos(x[:, None] * w), np.sin(x[:, None] * w)], -1)
    np.testing.assert_allclose(basis(x), expected, atol=1e-12)


@pytest.mark.parametrize("basis", [BSpline(5, 3), Chebyshev(4), Fourier(3)])
def test_config_roundtrip(basis):
    assert make_basis(basis.config()) == basis


def test_unknown_basis():
    with pytest.raises(ValueError, match="unknown basis"):
        make_basis("nope")


def test_reference_layer_is_sum_over_edges():
    basis = BSpline(5, 3)
    x = rng.uniform(-1, 1, size=(8, 3))
    coef = rng.normal(size=(2, 3, basis.size))
    w = rng.normal(size=(2, 3))
    b = rng.normal(size=2)
    y = kan_linear(x, coef, w, b, basis)
    silu = x / (1 + np.exp(-x))
    for o in range(2):
        manual = b[o] + sum(basis(x[:, i]) @ coef[o, i] + w[o, i] * silu[:, i] for i in range(3))
        np.testing.assert_allclose(y[:, o], manual, atol=1e-12)
