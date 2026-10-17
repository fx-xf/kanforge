import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kanforge import BSpline, Chebyshev, Fourier  # noqa: E402
from kanforge.mlx import KAN, KANLinear  # noqa: E402
from kanforge.reference import kan_linear  # noqa: E402

BASES = [BSpline(5, 3), Chebyshev(4), Fourier(3)]
IDS = ["bspline", "chebyshev", "fourier"]


# Metal float32 matmul is not IEEE-exact: even a 16x4 @ 4x3 product is off by
# ~6e-4 on GPU. The exact check runs on the CPU stream, the GPU one is looser.
@pytest.mark.parametrize("basis", BASES, ids=IDS)
@pytest.mark.parametrize("device,tol", [("cpu", 1e-5), ("gpu", 2e-3)])
def test_matches_reference(basis, device, tol):
    mx.random.seed(0)
    layer = KANLinear(4, 3, basis)
    layer.bias = mx.random.normal((3,))
    x = mx.random.uniform(-1, 1, (16, 4))
    expected = kan_linear(np.array(x), np.array(layer.coef), np.array(layer.base_weight), np.array(layer.bias), basis)
    with mx.stream(getattr(mx, device)):
        y = layer(x)
    np.testing.assert_allclose(np.array(y), expected, rtol=tol, atol=tol)


def test_parameters_match_torch_names():
    layer = KANLinear(3, 2, "bspline")
    assert set(layer.parameters()) == {"coef", "base_weight", "bias"}
    bare = KANLinear(3, 2, "chebyshev", base_activation=None, bias=False)
    assert set(bare.parameters()) == {"coef"}
    assert bare(mx.zeros((4, 3))).shape == (4, 2)


def test_second_derivative_of_input():
    layer = KANLinear(1, 1, Chebyshev(4))
    f = lambda x: layer(x[None, None])[0, 0]  # noqa: E731
    x0, eps = mx.array(0.3), 1e-2
    d2 = mx.grad(mx.grad(f))(x0)
    fd = (f(x0 + eps) - 2 * f(x0) + f(x0 - eps)) / eps**2
    np.testing.assert_allclose(np.array(d2), np.array(fd), rtol=1e-2, atol=1e-3)


def test_kan_fits_1d_function():
    import mlx.nn as nn
    import mlx.optimizers as optim

    mx.random.seed(0)
    model = KAN([1, 1], BSpline(10, 3))
    x = mx.linspace(-1, 1, 256)[:, None]
    y = mx.sin(np.pi * x)
    opt = optim.Adam(learning_rate=0.05)
    loss_and_grad = nn.value_and_grad(model, lambda m: mx.mean((m(x) - y) ** 2))
    for _ in range(400):
        loss, grads = loss_and_grad(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
    assert loss.item() < 1e-3


def test_edge_derivative_matches_mx_grad():
    from cases import CASES

    mx.random.seed(0)
    layer = KANLinear(1, 1, CASES["mexican_hat"])
    f = lambda x: layer(x[None, None])[0, 0]  # noqa: E731
    x0 = mx.array(0.37)
    with mx.stream(mx.cpu):
        for order in (1, 2):
            g = f
            for _ in range(order):
                g = mx.grad(g)
            got = layer.edge_derivative(x0[None, None], order)[0, 0, 0]
            np.testing.assert_allclose(np.array(got), np.array(g(x0)), rtol=1e-4, atol=1e-5)


def test_multkan_and_grkan_shapes():
    from kanforge.mlx import GRKAN, MultKAN

    assert MultKAN([2, (2, 1), 1])(mx.zeros((5, 2))).shape == (5, 1)
    layer = GRKAN(16, 8, groups=4)
    assert layer.basis_numerator.shape == (4, 6) and "base_weight" not in layer
    assert layer(mx.random.normal((3, 16))).shape == (3, 8)
