"""Phase 6: the JAX backend (Flax NNX) and the Pallas kernels."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
nnx = pytest.importorskip("flax.nnx")
torch = pytest.importorskip("torch")

import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

import kanforge as kf  # noqa: E402
import kanforge.jax as J  # noqa: E402
from cases import CASES, perturbed_params, sample_x  # noqa: E402
from kanforge import interpret as ki  # noqa: E402
from kanforge.kernels import plan_for  # noqa: E402
from kanforge.torch import KANLinear as TorchLinear  # noqa: E402

IDS = list(CASES)


@pytest.mark.parametrize("name", IDS)
def test_bases_in_jax_match_numpy_and_jax_grad(name):
    basis = CASES[name]
    x = sample_x(12, 2)
    p = perturbed_params(basis, 2)
    pj = {k: jnp.asarray(v) for k, v in p.items()}
    for m in range(3):
        np.testing.assert_allclose(np.asarray(basis.derivative(jnp.asarray(x), m, pj)), basis.derivative(x, m, p), rtol=1e-10, atol=1e-10)
    for k in range(basis.size):
        g = jax.grad(lambda xx, k=k: basis(xx, pj)[..., k].sum())(jnp.asarray(x))
        np.testing.assert_allclose(np.asarray(g), basis.derivative(x, 1, p)[..., k], rtol=1e-8, atol=1e-9)


def _pair(basis, **kw):
    torch.manual_seed(0)
    t = TorchLinear(4, 3, basis, dtype=torch.float64, **kw)
    with torch.no_grad():
        for p in t.parameters():
            p.add_(0.05 * torch.randn_like(p))
    j = J.KANLinear(4, 3, basis, dtype=jnp.float64, **kw)
    j.load_state_dict({k: v.numpy() for k, v in t.state_dict().items()})
    return t, j


@pytest.mark.parametrize("name", IDS)
def test_layer_matches_torch(name):
    t, j = _pair(CASES[name], edge_scale=True)
    x = np.random.default_rng(0).uniform(-0.9, 0.9, (9, 4))
    np.testing.assert_allclose(np.asarray(j(jnp.asarray(x))), t(torch.tensor(x)).detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(j.edge_derivative(jnp.asarray(x), 2)), t.edge_derivative(torch.tensor(x), 2).detach().numpy(), atol=1e-9
    )


def test_state_dict_names_match_torch_and_buffers_are_not_params():
    t, j = _pair(kf.Rational(), groups=2, edge_scale=True, base_activation=None)
    assert set(j.state_dict()) == set(t.state_dict())
    params = {".".join(map(str, k)) for k, _ in nnx.to_flat_state(nnx.state(j, nnx.Param))}
    assert params == {"coef", "bias", "edge_scale", "basis_numerator", "basis_denominator"}
    grads = nnx.grad(lambda m: m(jnp.ones((2, 4))).sum())(j)
    assert {".".join(map(str, k)) for k, _ in nnx.to_flat_state(grads)} == params


def test_grid_tools_match_torch():
    t, j = _pair(kf.BSpline(5, 3))
    x = np.random.default_rng(1).uniform(-0.8, 0.8, (500, 4))
    t.update_grid(torch.tensor(x))
    j.update_grid(jnp.asarray(x))
    t.refine_grid(9)
    j.refine_grid(9)
    np.testing.assert_allclose(np.asarray(j.basis_grid[...]), t.basis_grid.numpy(), atol=1e-12)
    np.testing.assert_allclose(np.asarray(j.coef[...]), t.coef.detach().numpy(), atol=1e-9)
    j.change_basis(kf.Chebyshev(10))
    assert j.coef[...].shape == (3, 4, 11)


def test_models_train_under_jit():
    optax = pytest.importorskip("optax")
    model = J.KAN([1, 1], kf.BSpline(10, 3), rngs=nnx.Rngs(0), dtype=jnp.float64)
    x = jnp.linspace(-1, 1, 256)[:, None]
    y = jnp.sin(jnp.pi * x)
    opt = nnx.Optimizer(model, optax.adam(0.05), wrt=nnx.Param)

    @nnx.jit
    def step(m, opt):
        loss, g = nnx.value_and_grad(lambda m: ((m(x) - y) ** 2).mean() + 1e-4 * J.regularization(m, x))(m)
        opt.update(m, g)
        return loss

    for _ in range(400):
        loss = step(model, opt)
    assert float(loss) < 2e-3


def test_grkan_multkan_dropout_and_from_mlp():
    layer = J.GRKAN(16, 8, groups=4)
    assert layer.basis_numerator[...].shape == (4, 6) and layer.base_weight is None
    assert J.MultKAN([2, (2, 1), 1])(jnp.ones((5, 2))).shape == (5, 1)
    d = J.KANLinear(6, 3, edge_dropout=0.5, rngs=nnx.Rngs(0, dropout=1))
    x = jnp.ones((2, 6)) * 0.3
    train = d(x)
    d.eval()
    assert not np.allclose(np.asarray(train), np.asarray(d(x)))
    mlp = nnx.Sequential(nnx.Linear(3, 8, rngs=nnx.Rngs(0)), jax.nn.gelu, nnx.Linear(8, 2, rngs=nnx.Rngs(1)))
    kan = J.from_mlp(mlp)
    xs = jax.random.normal(jax.random.key(0), (16, 3))
    np.testing.assert_allclose(np.asarray(kan(xs)), np.asarray(mlp(xs)), atol=1e-12)


def test_interpretation_tools_work_on_jax_models():
    model = J.KAN([2, 3, 1], "chebyshev", dtype=jnp.float64)
    x = jax.random.uniform(jax.random.key(0), (64, 2), minval=-1, maxval=1)
    assert len(ki.tabulate_model(model, x)) == 2
    assert ki.attribution(model, x)[0]["edge"].shape == (3, 2)
    pruned = ki.prune(model, x, 0.0, 0.0)
    assert pruned.widths == [2, 3, 1]
    ki.auto_symbolic(pruned, x, r2_threshold=0.0)
    (expr,) = ki.symbolic_formula(pruned, digits=None)
    import sympy

    f = sympy.lambdify(sympy.symbols("x_1 x_2"), expr, "numpy")
    np.testing.assert_allclose(f(*np.asarray(x).T), np.asarray(pruned(x))[:, 0], atol=1e-6)


# --- Pallas kernels (interpreter on CPU) ---------------------------------------------

KERNEL_BASES = {"bspline": kf.BSpline(5, 3), "chebyshev": kf.Chebyshev(5), "rbf": kf.RBF(6), "fourier": kf.Fourier(3)}


@pytest.mark.parametrize("name", list(KERNEL_BASES))
def test_pallas_kernel_matches_ops_with_nested_grads(name, monkeypatch):
    monkeypatch.setenv("KANFORGE_PALLAS_INTERPRET", "1")
    basis = KERNEL_BASES[name]
    ops = J.KANLinear(3, 4, basis, backend="ops", base_activation=None)
    ker = J.KANLinear(3, 4, basis, backend="kernel", base_activation=None)
    ker.load_state_dict(ops.state_dict())
    x = jax.random.uniform(jax.random.key(1), (21, 3), minval=-0.9, maxval=0.9, dtype=jnp.float32)
    w = jnp.linspace(-1, 1, 4)

    def results(layer):
        f = lambda v: (layer(v) * w).sum()  # noqa: E731
        g1 = jax.grad(f)
        g2 = jax.grad(lambda v: g1(v).sum())
        gc = nnx.grad(lambda m: (m(x) ** 2).sum())(layer)["coef"][...]
        return layer(x), g1(x), g2(x), gc

    for a, b in zip(results(ker), results(ops)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-4 * max(1.0, float(np.abs(b).max())))


def test_kernel_selection(monkeypatch):
    monkeypatch.delenv("KANFORGE_PALLAS_INTERPRET", raising=False)
    layer = J.KANLinear(2, 2, "bspline")
    if jax.default_backend() == "cpu":
        assert not layer.uses_kernel()
        with pytest.raises(RuntimeError):
            J.KANLinear(2, 2, "bspline", backend="kernel")(jnp.ones((1, 2)))
    with pytest.raises(RuntimeError, match="basis"):
        J.KANLinear(2, 2, kf.Rational(), backend="kernel", base_activation=None)(jnp.ones((1, 2)))
    assert plan_for(kf.NURBS()) is None
