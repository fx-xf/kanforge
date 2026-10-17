"""Phase 4: tabulation, attribution, pruning, symbolic regression, formulas, kanpiler."""

import numpy as np
import pytest

sympy = pytest.importorskip("sympy")
torch = pytest.importorskip("torch")

import kanforge as kf  # noqa: E402
from kanforge import interpret as ki  # noqa: E402
from kanforge.interpret import library as lib  # noqa: E402
from kanforge.torch import KAN, KANLinear, MultKAN  # noqa: E402

X, Y, Z = sympy.symbols("x y z")
rng = np.random.default_rng(0)


def mlx_or_skip():
    return pytest.importorskip("mlx.core")


# --- library and fitting -----------------------------------------------------


@pytest.mark.parametrize("name", lib.NAMES)
def test_library_backends_and_sympy_agree(name):
    f = lib.get(name)
    u = np.linspace(0.2, 1.7, 9)  # inside every domain
    want = f.fn(u)
    np.testing.assert_allclose(f.fn(torch.tensor(u)).numpy(), want, rtol=1e-12, atol=1e-12)
    s = sympy.Symbol("u")
    np.testing.assert_allclose(np.broadcast_to(sympy.lambdify(s, f.sympy(s), "numpy")(u), u.shape), want, rtol=1e-12, atol=1e-12)
    try:
        mx = __import__("mlx.core", fromlist=["core"])
    except ImportError:
        return
    with mx.stream(mx.cpu):
        np.testing.assert_allclose(np.array(f.fn(mx.array(u.astype(np.float32)))), want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "name,fn,params",
    [
        ("x^2", lambda x: 2 * (x - 0.3) ** 2, (1.0, -0.3, 2.0, 0.0)),
        ("sin", lambda x: 1.5 * np.sin(2 * x + 0.5) - 0.3, None),
        ("exp", lambda x: np.exp(1.5 * x), (1.5, 0.0, 1.0, 0.0)),
        ("1/x", lambda x: 1 / (x + 2), (1.0, 2.0, 1.0, 0.0)),
        ("tanh", lambda x: np.tanh(3 * x), None),
        ("x", lambda x: 3 * x + 1, (1.0, 0.0, 3.0, 1.0)),
    ],
)
def test_suggest_symbolic_recovers_functions(name, fn, params):
    x = np.linspace(-1, 1, 201)
    best = ki.suggest_symbolic(x, fn(x))[0]
    assert best.name == name and best.r2 > 1 - 1e-8
    if params is not None:
        np.testing.assert_allclose(best.params, params, atol=1e-3)
    u = sympy.Symbol("u")
    np.testing.assert_allclose(sympy.lambdify(u, best.expr(u), "numpy")(x), fn(x), atol=1e-5)


def test_cancelling_fits_lose_to_the_real_function():
    x = np.linspace(-1, 1, 201)
    ranked = [s.name for s in ki.suggest_symbolic(x, 0.8 * x**2 + 0.1, topk=18)]
    assert ranked[0] == "x^2" and ranked.index("cos") > 0


def test_fit_handles_constants_and_domains():
    x = np.linspace(-1, 1, 51)
    params, r2 = ki.fit_affine(x, np.full_like(x, 2.5), lib.get("sin"))
    assert r2 == 1.0 and params == (1.0, 0.0, 0.0, 2.5)
    params, r2 = ki.fit_affine(x, np.log(x + 3), lib.get("log"))  # needs b > 1
    assert r2 > 1 - 1e-8 and params[1] > 1


# --- symbolic edges and masks in layers --------------------------------------


def test_fixed_edges_compute_their_formula_and_stay_finite():
    torch.manual_seed(0)
    layer = KANLinear(2, 2, kf.BSpline(4, 3), dtype=torch.float64)
    layer.fix_symbolic(0, 0, "log", (1.0, 3.0, 2.0, 0.5))
    layer.fix_symbolic(1, 1, "sqrt", (1.0, 0.0, 1.0, 0.0))
    x = torch.tensor([[-0.5, 0.25], [0.5, 0.81]], dtype=torch.float64, requires_grad=True)
    edges = layer.edge_derivative(x, 0)
    torch.testing.assert_close(edges[:, 0, 0], 2 * torch.log(x[:, 0] + 3) + 0.5)
    torch.testing.assert_close(edges[:, 1, 1], torch.sqrt(x[:, 1]))
    y = layer(x)
    torch.testing.assert_close(y, edges.sum(-1) + layer.bias)
    (g,) = torch.autograd.grad(y.sum(), x)
    assert torch.isfinite(g).all()  # sqrt is never evaluated at x_0 = -0.5
    assert layer.symbolic_edges() == {(0, 0): ("log", (1.0, 3.0, 2.0, 0.5)), (1, 1): ("sqrt", (1.0, 0.0, 1.0, 0.0))}
    with pytest.raises(NotImplementedError):
        layer.edge_derivative(x, 1)
    layer.unfix_symbolic(0, 0)
    assert list(layer.symbolic_edges()) == [(1, 1)] and layer.active_edges().all()


def test_symbolic_state_survives_state_dict_and_mlx():
    layer = KANLinear(2, 1, kf.Chebyshev(3))
    layer.fix_symbolic(0, 1, "sin", (2.0, 0.1, 1.5, 0.0))
    fresh = KANLinear(2, 1, kf.Chebyshev(3))
    fresh.load_state_dict(layer.state_dict())
    x = torch.rand(5, 2)
    torch.testing.assert_close(fresh(x), layer(x))
    mx = pytest.importorskip("mlx.core")
    from kanforge.mlx import KANLinear as MLXLinear

    m = MLXLinear(2, 1, kf.Chebyshev(3))
    m.update({k: mx.array(v.numpy()) for k, v in layer.state_dict().items()})
    assert m.symbolic_edges().keys() == {(0, 1)}
    with mx.stream(mx.cpu):
        np.testing.assert_allclose(np.array(m(mx.array(x.numpy()))), layer(x).detach().numpy(), atol=1e-5)


def test_masks_apply_on_the_kernel_path():
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip("needs MPS or CUDA")
    dev = "mps" if torch.backends.mps.is_available() else "cuda"
    try:
        torch.ones(1, device=dev).cpu()
    except RuntimeError:
        pytest.skip("device not usable")
    ops = KANLinear(3, 2, kf.BSpline(5, 3), backend="ops").to(dev)
    ker = KANLinear(3, 2, kf.BSpline(5, 3), backend="kernel").to(dev)
    ker.load_state_dict(ops.state_dict())
    for layer in (ops, ker):
        layer.set_edge_mask([[1, 0, 1], [0, 1, 1]])
        layer.fix_symbolic(1, 0, "x^2", (1.0, 0.0, 2.0, 0.0))
    x = torch.rand(9, 3, device=dev)
    torch.testing.assert_close(ker(x), ops(x), atol=1e-5, rtol=1e-5)


# --- tabulation, attribution, pruning -----------------------------------------


def test_tabulate_matches_edge_functions_and_ranges():
    layer = KANLinear(3, 2, kf.BSpline(4, 3), grid_ranges=[(-1, 1), (0, 2), (-3, -1)])
    table = ki.tabulate(layer, n=11)
    np.testing.assert_allclose(table.x[[0, -1]], [[-1, 0, -3], [1, 2, -1]])
    want = layer.edge_derivative(torch.tensor(table.x, dtype=torch.float32), 0).detach().numpy()
    np.testing.assert_allclose(table.values, want, rtol=1e-6, atol=1e-6)
    xs, ys = table.edge(1, 2)
    assert xs.shape == ys.shape == (11,)


def _two_branch_model():
    """KAN [2, 4, 1] where hidden nodes 2 and 3 do not reach the output."""
    torch.manual_seed(0)
    model = KAN([2, 4, 1], kf.Chebyshev(3), dtype=torch.float64)
    with torch.no_grad():
        model[1].coef[:, 2:] = 0
        model[1].base_weight[:, 2:] = 0
    return model


def test_attribution_and_pruning():
    model = _two_branch_model()
    x = torch.rand(256, 2, dtype=torch.float64) * 2 - 1
    scores = ki.attribution(model, x)
    assert np.allclose(scores[1]["input"][2:], 0) and (scores[1]["input"][:2] > 0.01).all()
    assert scores[0]["input"].sum() == pytest.approx(1.0)
    pruned = ki.prune_nodes(model, x, threshold=0.01)
    assert pruned.widths == [2, 2, 1]
    torch.testing.assert_close(pruned(x), model(x))
    removed = ki.prune_edges(pruned, x, threshold=0.2)
    assert removed >= 1 and not pruned[0].active_edges().all()


def test_feature_importance_finds_unused_inputs():
    model, _ = ki.kanpile(sympy.sin(X) + X**2, [X, Y])
    imp = ki.feature_importance(model, torch.rand(128, 2) * 2 - 1)
    assert imp[1] == 0 and imp[0] > 0.99


def test_subset_keeps_the_selected_function():
    torch.manual_seed(0)
    layer = KANLinear(4, 3, kf.BSpline(4, 3), edge_scale=True)
    layer.fix_symbolic(2, 1, "sin")
    sub = layer.subset(out_idx=[0, 2], in_idx=[0, 1, 3])
    x = torch.rand(7, 4)
    with torch.no_grad():
        full = layer.edge_derivative(x, 0)[:, [0, 2]][:, :, [0, 1, 3]].sum(-1) + layer.bias[[0, 2]]
    torch.testing.assert_close(sub(x[:, [0, 1, 3]]), full)
    assert sub.symbolic_edges().keys() == {(1, 1)}


# --- auto_symbolic and formulas ------------------------------------------------


def test_auto_symbolic_and_formula_recover_a_trained_function():
    torch.manual_seed(0)
    x = torch.rand(512, 1, dtype=torch.float64) * 2 - 1
    y = 2 * torch.sin(3 * x) + 1
    model = KAN([1, 1], kf.BSpline(10, 3), dtype=torch.float64)
    opt = torch.optim.LBFGS(model.parameters(), lr=1, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        return loss

    for _ in range(20):
        opt.step(closure)
    report = ki.auto_symbolic(model, x)
    assert [r.suggestion.name for r in report] == ["sin"]
    (expr,) = ki.symbolic_formula(model, ["x"], digits=None)
    f = sympy.lambdify(sympy.Symbol("x"), expr, "numpy")
    np.testing.assert_allclose(f(x.numpy()[:, 0]), y.numpy()[:, 0], atol=1e-3)


def test_formula_lists_learned_edges_and_handles_linear_layers():
    model = KAN([2, 1], "chebyshev")
    with pytest.raises(ValueError, match="neither symbolic nor pruned"):
        ki.symbolic_formula(model)
    lin = torch.nn.Linear(2, 2)
    kan = KANLinear(2, 1, "bspline", base_activation=None)
    kan.fix_symbolic(0, 0, "exp")
    kan.fix_symbolic(0, 1, "x", (1.0, 0.0, -2.0, 0.0))
    seq = torch.nn.Sequential(lin, kan)
    (expr,) = ki.symbolic_formula(seq, ["a", "b"], digits=None)
    pts = np.random.default_rng(1).uniform(-1, 1, (20, 2))
    f = sympy.lambdify(sympy.symbols("a b"), expr, "numpy")
    np.testing.assert_allclose(f(*pts.T), seq(torch.tensor(pts, dtype=torch.float32)).detach().numpy()[:, 0], atol=1e-5)


# --- kanpiler ------------------------------------------------------------------

FORMULAS = [
    sympy.exp(sympy.sin(sympy.pi * X) + Y**2),
    X * Y,
    X * Y * Z + sympy.sin(X),
    sympy.sqrt(X**2 + Y**2 + 1),
    2 * X + 3,
    X / (Y + 2),
    sympy.exp(-(X**2)) * sympy.cos(2 * Y),
    sympy.sin(X) + sympy.cos(X),
    X,
]


def _vars(e):
    return [v for v in (X, Y, Z) if v in e.free_symbols]


@pytest.mark.parametrize("expr", FORMULAS, ids=[str(e) for e in FORMULAS])
def test_kanpile_is_exact_and_formula_round_trips(expr):
    vs = _vars(expr)
    model, col = ki.kanpile(expr, vs)
    pts = rng.uniform(-1, 1, (100, len(vs)))
    want = np.broadcast_to(sympy.lambdify(vs, expr, "numpy")(*pts.T), (100,))
    got = model(torch.tensor(pts, dtype=torch.float32))[:, col].detach().numpy()
    np.testing.assert_allclose(got, want, atol=2e-5 * max(1, np.abs(want).max()))
    back = ki.symbolic_formula(model, [str(v) for v in vs], digits=None)[col]
    np.testing.assert_allclose(np.broadcast_to(sympy.lambdify(vs, back, "numpy")(*pts.T), (100,)), want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("expr", FORMULAS[:6], ids=[str(e) for e in FORMULAS[:6]])
def test_kanpile_fit_mode_learns_the_edges(expr):
    vs = _vars(expr)
    model, col = ki.kanpile(expr, vs, basis={"name": "bspline", "grid_size": 20, "order": 3}, mode="fit")
    assert not any(layer.kan.symbolic_edges() for layer in model)
    pts = rng.uniform(-1, 1, (100, len(vs)))
    want = np.broadcast_to(sympy.lambdify(vs, expr, "numpy")(*pts.T), (100,))
    got = model(torch.tensor(pts, dtype=torch.float32))[:, col].detach().numpy()
    np.testing.assert_allclose(got, want, atol=1e-3 * max(1, np.abs(want).max()))


def test_kanpile_multiple_outputs_mlx_and_errors():
    model, cols = ki.kanpile([X * Y, X + Y], [X, Y])
    pts = torch.rand(10, 2)
    out = model(pts)
    torch.testing.assert_close(out[:, cols[0]], pts[:, 0] * pts[:, 1])
    torch.testing.assert_close(out[:, cols[1]], pts[:, 0] + pts[:, 1])
    with pytest.raises(NotImplementedError):
        ki.kanpile(sympy.gamma(X), [X])
    with pytest.raises(ValueError):
        ki.kanpile(X + Z, [X])
    mx = mlx_or_skip()
    m, col = ki.kanpile(sympy.sin(X) * Y, [X, Y], backend="mlx")
    p = np.random.default_rng(2).uniform(-1, 1, (10, 2)).astype(np.float32)
    with mx.stream(mx.cpu):
        np.testing.assert_allclose(np.array(m(mx.array(p)))[:, col], np.sin(p[:, 0]) * p[:, 1], atol=1e-5)


def test_mlx_interpretation_tools():
    mx = mlx_or_skip()
    from kanforge.mlx import KAN as MLXKAN

    model = MLXKAN([2, 3, 1], "chebyshev")
    x = mx.random.uniform(-1, 1, (64, 2))
    assert len(ki.tabulate_model(model, x)) == 2
    assert ki.attribution(model, x)[0]["edge"].shape == (3, 2)
    pruned = ki.prune(model, x, 0.0, 0.0)
    assert pruned.widths == [2, 3, 1]
    ki.auto_symbolic(pruned, x, r2_threshold=0.0)
    assert len(ki.symbolic_formula(pruned)) == 1


# --- plots and contrastive mode ------------------------------------------------


def test_plots():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    model = KAN([2, 3, 1], "bspline")
    model[0].fix_symbolic(0, 0, "sin")
    model[0].set_edge_mask(np.array([[0, 1], [1, 1], [1, 0]]))
    figs = ki.plot_model(model, torch.rand(64, 2))
    assert len(figs) == 2 and len(figs[0].axes) == 6


def test_contrastive_samples_keep_marginals():
    x = rng.normal(size=(200, 3))
    x[:, 2] = x[:, 0] * x[:, 1]
    inputs, labels = ki.contrastive_samples(x)
    assert inputs.shape == (400, 3) and labels.sum() == 200
    fake = inputs[labels == 0]
    np.testing.assert_allclose(np.sort(fake, 0), np.sort(x, 0))
    assert abs(np.corrcoef(fake[:, 2], fake[:, 0] * fake[:, 1])[0, 1]) < 0.3
