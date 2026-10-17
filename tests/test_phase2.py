"""Phase 2: per-input grids, grid update / refinement / basis conversion,
regularisation, DropKAN, initialisation schemes, exact MLP conversion."""

import numpy as np
import pytest

import kanforge as kf
from kanforge.grid import adaptive_grid, resample_grid

torch = pytest.importorskip("torch")
from kanforge.torch import KAN, KANLinear, MultKAN, from_mlp, regularization  # noqa: E402

rng = np.random.default_rng(0)


def random_grid(n_points, lo=-1.0, hi=1.0):
    steps = rng.uniform(0.3, 1.0, n_points - 1)
    return lo + (hi - lo) * np.concatenate([[0.0], np.cumsum(steps) / steps.sum()])


# --- non-uniform grids ---------------------------------------------------


@pytest.mark.parametrize("order", [1, 2, 3])
def test_nonuniform_bspline_matches_scipy(order):
    interp = pytest.importorskip("scipy.interpolate")
    basis = kf.BSpline(6, order)
    grid = random_grid(7)
    params = {"grid": grid}
    t = basis.knots(params)
    x = rng.uniform(-1, 1, 300)
    np.testing.assert_allclose(basis(x, params), interp.BSpline.design_matrix(x, t, order).toarray(), atol=1e-12)
    for m in range(1, order + 1):
        for j in range(basis.size):
            c = np.eye(basis.size)[j]
            want = interp.BSpline(t, c, order)(x, nu=m)
            np.testing.assert_allclose(basis.derivative(x, m, params)[:, j], want, atol=1e-8)


def test_grid_ranges_per_input():
    layer = KANLinear(3, 2, kf.BSpline(4, 3), grid_ranges=[(-1, 1), (0, 5), (-10, -2)])
    g = layer.basis_grid
    assert g.shape == (3, 5) and not g.requires_grad
    np.testing.assert_allclose(g[:, [0, -1]].numpy(), [[-1, 1], [0, 5], [-10, -2]])
    assert "basis_grid" in dict(layer.named_buffers())


def test_adaptive_grid_properties():
    x = np.concatenate([rng.normal(0, 0.1, (1000, 1)), rng.uniform(2, 3, (1000, 1)), np.full((1000, 1), 7.0)], 1)
    g = adaptive_grid(x, 9, grid_eps=0.05)
    assert g.shape == (3, 9)
    assert (np.diff(g, axis=1) > 0).all()
    np.testing.assert_allclose(g[:2, [0, -1]], [[x[:, 0].min(), x[:, 0].max()], [x[:, 1].min(), x[:, 1].max()]])
    np.testing.assert_allclose(g[2, [0, -1]], [6.5, 7.5])  # constant input gets a unit window
    with pytest.raises(ValueError):
        adaptive_grid(x, 9, grid_eps=0.0)
    np.testing.assert_allclose(resample_grid(np.array([[0.0, 1.0, 3.0]]), 5), [[0, 0.5, 1, 2, 3]])


# --- moving a layer to a new grid or basis keeps its function ------------------


def trained_layer(basis, seed=0, **kw):
    torch.manual_seed(seed)
    layer = KANLinear(3, 2, basis, dtype=torch.float64, **kw)
    with torch.no_grad():
        layer.coef.normal_(0, 0.5)
    return layer


def test_update_grid_moves_grid_to_data_and_keeps_function():
    layer = trained_layer(kf.BSpline(8, 3))
    x = torch.rand(2000, 3, dtype=torch.float64) * 1.6 - 0.8
    before = layer(x)
    layer.update_grid(x)
    g = layer.basis_grid
    np.testing.assert_allclose(g[:, 0].numpy(), x.min(0).values.numpy(), atol=0.02)
    np.testing.assert_allclose(g[:, -1].numpy(), x.max(0).values.numpy(), atol=0.02)
    err = (layer(x) - before).abs().max() / before.abs().max()
    assert err < 0.1  # approximation of a spline on another grid, as in pykan


def test_update_grid_on_data_outside_the_grid_is_exact():
    layer = trained_layer(kf.BSpline(5, 3))
    x = torch.rand(500, 3, dtype=torch.float64) * 2 + 3  # spline part is zero here
    before = layer(x)
    layer.update_grid(x)
    torch.testing.assert_close(layer(x), before, rtol=1e-9, atol=1e-9)
    assert layer.basis_grid.min() > 2.9


@pytest.mark.parametrize("edge_scale", [False, True])
def test_refine_grid_is_exact_for_nested_bspline_grids(edge_scale):
    layer = trained_layer(kf.BSpline(5, 3), edge_scale=edge_scale)
    if edge_scale:
        with torch.no_grad():
            layer.edge_scale.uniform_(0.5, 2)
    x = torch.linspace(-1, 1, 301, dtype=torch.float64)[:, None].repeat(1, 3)
    before = layer(x)
    layer.refine_grid(10)
    assert layer.basis.grid_size == 10 and layer.coef.shape == (2, 3, 13)
    torch.testing.assert_close(layer(x), before, rtol=1e-9, atol=1e-9)


def test_refine_keeps_adapted_grid_shape():
    layer = trained_layer(kf.BSpline(4, 3))
    layer.update_grid(torch.randn(1000, 3, dtype=torch.float64) * 0.3 + 0.5)
    old = layer.basis_grid.numpy().copy()
    layer.refine_grid(8)
    np.testing.assert_allclose(layer.basis_grid.numpy()[:, ::2], old)


@pytest.mark.parametrize(
    "new,tol",
    [(kf.BSpline(5, 3), 1e-10), (kf.BSpline(10, 3), 1e-10), (kf.Chebyshev(14), 2e-2), (kf.Legendre(14, input_map="identity"), 2e-2), (kf.RBF(16), 6e-2)],
    ids=["same", "nested", "chebyshev", "legendre", "rbf"],
)
def test_change_basis_preserves_function(new, tol):
    layer = trained_layer(kf.BSpline(5, 3))
    x = torch.linspace(-1, 1, 201, dtype=torch.float64)[:, None].repeat(1, 3)
    before = layer(x)
    layer.change_basis(new)
    assert layer.basis == new and layer.coef.shape[-1] == new.size
    err = (layer(x) - before).abs().max() / before.abs().max()
    assert err < tol


def test_change_basis_without_grid_needs_samples():
    layer = trained_layer(kf.Chebyshev(4))
    with pytest.raises(ValueError, match="pass samples"):
        layer.change_basis(kf.Legendre(4))
    x = torch.rand(400, 3, dtype=torch.float64) * 2 - 1
    before = layer(x)
    layer.change_basis(kf.Legendre(4), x)  # same polynomial space: exact
    torch.testing.assert_close(layer(x), before, rtol=1e-8, atol=1e-8)
    with pytest.raises(TypeError, match="no grid"):
        layer.update_grid(x)


def test_trainable_basis_params_survive_grid_update():
    layer = trained_layer(kf.NURBS(5, 3), seed=1)
    with torch.no_grad():
        layer.basis_log_weights.uniform_(-0.3, 0.3)
    w = layer.basis_log_weights.clone()
    layer.update_grid(torch.rand(800, 3, dtype=torch.float64) - 0.5)
    torch.testing.assert_close(layer.basis_log_weights, w)


def test_kan_update_grid_uses_each_layer_inputs():
    torch.manual_seed(0)
    model = KAN([2, 3, 1], kf.BSpline(5, 3))
    x = torch.rand(300, 2) * 4 - 2
    model.update_grid(x)
    h = model[0](x)
    np.testing.assert_allclose(model[1].basis_grid[:, 0].numpy(), h.min(0).values.detach().numpy(), atol=0.05)


# --- regularisation and DropKAN ------------------------------------------


def test_regularization_matches_pykan_formula_and_has_gradients():
    torch.manual_seed(0)
    model = KAN([3, 4, 2], "chebyshev")
    x = torch.rand(64, 3)
    total = regularization(model, x, l1=1.0, entropy=2.0, coef_l1=0.1, coef_smooth=0.1)
    total.backward()
    assert model[0].coef.grad.abs().sum() > 0
    # recompute layer 0 by hand
    edges = model[0].edge_derivative(x, 0).abs().mean(0)
    p_row = edges / edges.sum(1, keepdim=True)
    p_col = edges / edges.sum(0, keepdim=True)
    ent = -(p_row * torch.log2(p_row + 1e-8)).sum(1).mean() - (p_col * torch.log2(p_col + 1e-8)).sum(0).mean()
    coef = model[0].coef
    manual = edges.sum() + 2 * ent + 0.1 * coef.abs().mean() + 0.1 * (coef[..., 1:] - coef[..., :-1]).abs().mean()
    from kanforge import regularization as reg

    torch.testing.assert_close(
        reg.penalty(reg.edge_magnitude(model[0].edge_derivative(x, 0)), coef, coef_l1=0.1, coef_smooth=0.1), manual
    )


def test_regularization_prefers_sparse_layers():
    dense = KANLinear(4, 4, "chebyshev", base_activation=None)
    sparse = KANLinear(4, 4, "chebyshev", base_activation=None)
    with torch.no_grad():
        dense.coef.fill_(0.25)
        sparse.coef.zero_()
        sparse.coef[0, 0] = 4.0  # same total magnitude on one edge
    assert regularization(sparse, l1=0.0) < regularization(dense, l1=0.0)


def test_regularization_walks_multkan_and_sequential():
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), MultKAN([3, (2, 1), 1]))
    assert regularization(model, torch.rand(8, 3)).ndim == 0


def test_edge_dropout_is_unbiased_and_off_in_eval():
    torch.manual_seed(0)
    layer = KANLinear(6, 3, "bspline", edge_dropout=0.3)
    x = torch.rand(1, 6) * 2 - 1
    layer.eval()
    clean = layer(x)
    layer.train()
    samples = torch.stack([layer(x) for _ in range(20000)])
    assert not torch.allclose(samples[0], clean)
    torch.testing.assert_close(samples.mean(0), clean, atol=0.02, rtol=0.05)


# --- initialisation ----------------------------------------------------------


@pytest.mark.parametrize("basis", ["bspline", "chebyshev", "rbf", "fourier", "legendre"])
@pytest.mark.parametrize("scheme", ["lecun", "glorot"])
def test_variance_preserving_init(basis, scheme):
    stds = []
    for seed in range(8):
        torch.manual_seed(seed)
        layer = KANLinear(64, 64, basis, coef_init=scheme)
        stds.append(layer(torch.randn(2048, 64)).std().item())
    assert 0.8 < np.mean(stds) < 1.25


def test_zero_init_starts_as_base_branch():
    layer = KANLinear(5, 3, coef_init="zero")
    x = torch.randn(4, 5)
    torch.testing.assert_close(layer(x), torch.nn.functional.silu(x) @ layer.base_weight.T + layer.bias)
    with pytest.raises(ValueError, match="coef_init"):
        KANLinear(2, 2, coef_init="nope")


# --- exact conversion from MLPs ------------------------------------------


@pytest.mark.parametrize("act", [torch.nn.SiLU(), torch.nn.GELU(), torch.nn.Tanh()])
def test_from_mlp_is_exact_and_trainable(act):
    torch.manual_seed(0)
    mlp = torch.nn.Sequential(torch.nn.Linear(4, 16), act, torch.nn.Linear(16, 8), act, torch.nn.Linear(8, 2))
    kan = from_mlp(mlp, kf.BSpline(5, 3))
    assert [type(m).__name__ for m in kan] == ["Linear", "KANLinear", "KANLinear"]
    x = torch.randn(32, 4)
    torch.testing.assert_close(kan(x), mlp(x), rtol=0, atol=1e-6)
    kan(x).pow(2).mean().backward()
    assert kan[1].coef.grad.abs().sum() > 0


def test_from_mlp_keeps_trailing_activation_and_rejects_bad_start():
    mlp = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Linear(3, 1), torch.nn.Sigmoid())
    kan = from_mlp(mlp)
    x = torch.randn(5, 2)
    torch.testing.assert_close(kan(x), mlp(x), rtol=0, atol=1e-6)
    with pytest.raises(ValueError):
        from_mlp(torch.nn.Sequential(torch.nn.ReLU()))


# --- MLX gives the same results ------------------------------------------------


def test_mlx_grid_tools_match_torch():
    mx = pytest.importorskip("mlx.core")
    from kanforge.mlx import KANLinear as MLXLinear

    t = trained_layer(kf.BSpline(5, 3))
    m = MLXLinear(3, 2, kf.BSpline(5, 3))
    m.update({k: mx.array(v.float().numpy()) for k, v in t.state_dict().items()})
    x = np.random.default_rng(1).uniform(-0.8, 0.8, (500, 3))
    t.update_grid(torch.tensor(x))
    m.update_grid(mx.array(x.astype(np.float32)))
    t.refine_grid(9)
    m.refine_grid(9)
    np.testing.assert_allclose(np.array(m.basis_grid), t.basis_grid.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.array(m.coef), t.coef.detach().numpy(), rtol=1e-3, atol=1e-4)
    assert set(m.trainable_parameters()) == {"coef", "base_weight", "bias"}


def test_mlx_from_mlp_and_regularization():
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as mnn

    from kanforge.mlx import KAN as MLXKAN
    from kanforge.mlx import from_mlp as mlx_from_mlp
    from kanforge.mlx import regularization as mlx_reg

    mlp = mnn.Sequential(mnn.Linear(3, 8), mnn.GELU(), mnn.Linear(8, 2))
    kan = mlx_from_mlp(mlp)
    x = mx.random.normal((16, 3))
    with mx.stream(mx.cpu):
        np.testing.assert_allclose(np.array(kan(x)), np.array(mlp(x)), atol=1e-6)
    model = MLXKAN([3, 4, 2], "chebyshev")
    loss, grads = mnn.value_and_grad(model, lambda mdl: mlx_reg(mdl, x, coef_l1=0.1))(model)
    assert loss.item() > 0 and np.abs(np.array(grads["layers"][0]["coef"])).sum() > 0


def test_layer_moves_between_devices_and_dtypes():
    # GridToolsMixin once shadowed nn.Module._apply and broke .to()/.double().
    layer = KANLinear(3, 2, kf.BSpline(4, 3), edge_scale=True).double()
    assert layer.coef.dtype == torch.float64 and layer.basis_grid.dtype == torch.float64
    layer = layer.to("cpu", torch.float32)
    x = torch.rand(8, 3, dtype=torch.float64) - 0.5
    layer.double().update_grid(x)
    assert layer.basis_grid.dtype == torch.float64
