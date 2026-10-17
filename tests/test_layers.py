"""Layer features of phase 1 on PyTorch and MLX: basis parameters, groups,
edge scale, analytic edge derivatives, GR-KAN, MultKAN."""

import numpy as np
import pytest

import kanforge as kf
from cases import CASES
from kanforge.reference import kan_linear

torch = pytest.importorskip("torch")
from kanforge.torch import GRKAN, KAN, KANLinear, MultKAN, MultKANLinear  # noqa: E402

LEARNABLE = [n for n, b in CASES.items() if b.param_specs()]  # trainable params or grids


def expanded_params(layer):
    return {k: v.detach().double().expand(layer.in_features, *v.shape[1:]).numpy() for k, v in layer.basis_params().items()}


@pytest.mark.parametrize("name", LEARNABLE)
@pytest.mark.parametrize("groups", [None, 2, 1])
def test_learnable_basis_layer_matches_reference(name, groups):
    torch.manual_seed(0)
    layer = KANLinear(4, 3, CASES[name], groups=groups, edge_scale=True, dtype=torch.float64)
    with torch.no_grad():
        for p in layer.parameters():
            p.add_(0.05 * torch.randn_like(p))
    rows = {None: 4, 2: 2, 1: 1}[groups]
    for k, spec in layer.basis.param_specs().items():
        assert getattr(layer, f"basis_{k}").shape[0] == (rows if spec.trainable else 4)
    x = torch.rand(10, 4, dtype=torch.float64) * 1.8 - 0.9
    want = kan_linear(
        x.numpy(),
        layer.coef.detach().numpy(),
        layer.base_weight.detach().numpy(),
        layer.bias.detach().numpy(),
        layer.basis,
        expanded_params(layer),
        layer.edge_scale.detach().numpy(),
    )
    np.testing.assert_allclose(layer(x).detach().numpy(), want, rtol=1e-10, atol=1e-12)


def test_groups_share_parameters_and_get_gradients():
    layer = KANLinear(6, 2, kf.Rational(), groups=3, base_activation=None)
    assert layer.basis_numerator.shape == (3, 6)
    layer(torch.randn(5, 6)).sum().backward()
    assert layer.basis_numerator.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="must divide"):
        KANLinear(5, 2, "rational", groups=2)


@pytest.mark.parametrize("name", ["bspline", "chebyshev", "rational", "mexican_hat", "fractional_jacobi", "neural_silu"])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_edge_derivative_matches_autograd(name, order):
    torch.manual_seed(0)
    layer = KANLinear(3, 2, CASES[name], groups=1 if name == "rational" else None, dtype=torch.float64)
    x = (torch.rand(7, 3, dtype=torch.float64) * 1.6 - 0.8).requires_grad_()
    analytic = layer.edge_derivative(x, order)
    assert analytic.shape == (7, 2, 3)
    if order == 0:
        np.testing.assert_allclose(analytic.sum(-1).detach() + layer.bias.detach(), layer(x).detach(), atol=1e-12)
        return
    for o in range(2):
        d = layer(x)[..., o]
        for _ in range(order):
            (d,) = torch.autograd.grad(d.sum(), x, create_graph=True)
        np.testing.assert_allclose(analytic[:, o, :].detach().numpy(), d.detach().numpy(), rtol=1e-8, atol=1e-10)


def test_edge_derivative_rejects_unknown_activation():
    layer = KANLinear(2, 2, base_activation=torch.tanh)
    with pytest.raises(NotImplementedError):
        layer.edge_derivative(torch.zeros(1, 2))


def test_kan_edge_derivative_per_layer():
    model = KAN([2, 3, 1], "chebyshev")
    out = model.edge_derivative(torch.rand(4, 2), 2)
    assert [o.shape for o in out] == [(4, 3, 2), (4, 1, 3)]


def test_grkan_matches_kat_layout_and_keeps_variance():
    torch.manual_seed(0)
    layer = GRKAN(64, 64, groups=8)
    assert layer.base_weight is None
    assert layer.basis_numerator.shape == (8, 6) and layer.basis_denominator.shape == (8, 4)
    x = torch.randn(4096, 64)
    std = layer(x).std().item()
    assert 0.7 < std < 1.4
    first_group = {k: v[:1] for k, v in layer.basis_params().items()}
    z = torch.linspace(-3, 3, 301)[:, None]  # the range the rationals were fitted on
    with torch.no_grad():
        acts = layer.basis(z, first_group)[..., 0]
    assert (acts - torch.nn.functional.gelu(z)).abs().max() < 2e-2  # rationals start as GELU


def test_multkan_products():
    torch.manual_seed(0)
    layer = MultKANLinear(3, n_sum=2, n_mult=2, arity=2)
    x = torch.rand(5, 3)
    y, inner = layer(x), layer.kan(x)
    assert y.shape == (5, 4)
    torch.testing.assert_close(y[:, :2], inner[:, :2])
    torch.testing.assert_close(y[:, 2], inner[:, 2] * inner[:, 3])
    torch.testing.assert_close(y[:, 3], inner[:, 4] * inner[:, 5])


def test_multkan_learns_product():
    torch.manual_seed(0)
    model = MultKAN([2, (0, 1), 1], kf.BSpline(5, 3))
    x = torch.rand(512, 2) * 2 - 1
    y = (x[:, 0] * x[:, 1])[:, None]
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(300):
        opt.zero_grad()
        loss = torch.mean((model(x) - y) ** 2)
        loss.backward()
        opt.step()
    assert loss.item() < 1e-3
    with pytest.raises(ValueError):
        MultKAN([2])


def test_state_dict_moves_between_torch_and_mlx():
    mx = pytest.importorskip("mlx.core")
    from kanforge.mlx import KANLinear as MLXLinear

    t = KANLinear(4, 3, CASES["rational"], groups=2, edge_scale=True)
    m = MLXLinear(4, 3, CASES["rational"], groups=2, edge_scale=True)
    m.update({k: mx.array(v.detach().numpy()) for k, v in t.state_dict().items()})
    x = np.random.default_rng(0).uniform(-1, 1, (6, 4)).astype(np.float32)
    with mx.stream(mx.cpu):
        got = np.array(m(mx.array(x)))
    np.testing.assert_allclose(got, t(torch.tensor(x)).detach().numpy(), rtol=1e-4, atol=1e-5)
    assert set(dict(m.parameters())) == set(t.state_dict())
