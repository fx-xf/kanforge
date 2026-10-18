"""Phase 5: convolutions, transformer FFNs and HF surgery, graphs, recurrent cells, scientific ML."""

import copy
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

import kanforge as kf  # noqa: E402
import kanforge.torch as T  # noqa: E402


def _mlx():
    return pytest.importorskip("mlx.core")


def _load_into_mlx(module, state_dict):
    mx = _mlx()
    from mlx.utils import tree_unflatten

    module.update(tree_unflatten([(k, mx.array(v.detach().numpy())) for k, v in state_dict.items()]))


# --- convolutions --------------------------------------------------------------

CONVS = [(T.KANConv1d, nn.Conv1d, (2, 4, 17)), (T.KANConv2d, nn.Conv2d, (2, 4, 9, 11)), (T.KANConv3d, nn.Conv3d, (1, 4, 5, 6, 7))]
CONV_KW = [{}, {"stride": 2, "padding": 1}, {"dilation": 2, "padding": 2, "groups": 2}, {"padding": 1, "padding_mode": "reflect"}]


@pytest.mark.parametrize("kan_cls,conv_cls,shape", CONVS, ids=["1d", "2d", "3d"])
@pytest.mark.parametrize("kw", CONV_KW, ids=["plain", "stride", "dilation_groups", "reflect"])
def test_from_conv_is_exact(kan_cls, conv_cls, shape, kw):
    torch.manual_seed(0)
    conv = conv_cls(4, 6, 3, **kw)
    kan = kan_cls.from_conv(conv)
    x = torch.randn(*shape)
    torch.testing.assert_close(kan(x), conv(x), atol=1e-5, rtol=1e-5)


def test_kan_conv_trains_and_matches_mlx():
    torch.manual_seed(0)
    conv = T.KANConv2d(3, 5, 3, stride=2, padding=1, groups=1, basis="chebyshev")
    x = torch.randn(2, 3, 8, 8)
    y = conv(x)
    assert y.shape == (2, 5, 4, 4)
    y.square().mean().backward()
    assert conv.kans[0].coef.grad.abs().sum() > 0
    mx = _mlx()
    from kanforge.mlx import KANConv2d as M2

    m = M2(3, 5, 3, stride=2, padding=1, basis="chebyshev")
    _load_into_mlx(m.kans[0], conv.kans[0].state_dict())
    with mx.stream(mx.cpu):
        ym = np.array(m(mx.array(x.numpy().transpose(0, 2, 3, 1))))
    np.testing.assert_allclose(ym.transpose(0, 3, 1, 2), y.detach().numpy(), atol=1e-5)


def test_conv_argument_checks():
    with pytest.raises(ValueError):
        T.KANConv2d(3, 4, 3, groups=2)
    with pytest.raises(ValueError):
        T.KANConv1d(2, 2, 3, padding_mode="mirror")


# --- transformers ----------------------------------------------------------------


@pytest.mark.parametrize("kind", ["grkan", "kan"])
def test_kan_ffn_and_kat_block(kind):
    torch.manual_seed(0)
    block = T.KATBlock(16, 4, kind=kind)
    x = torch.randn(2, 5, 16)
    y = block(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert block.ffn.fc1.coef.grad is not None


def test_grkan_ffn_starts_as_gelu_mlp():
    ffn = T.KANFFN(8, 32, kind="grkan")
    lin1, lin2 = nn.Linear(8, 32), nn.Linear(32, 8)
    with torch.no_grad():
        ffn.fc1.coef.copy_(lin1.weight[..., None])
        ffn.fc1.bias.copy_(lin1.bias)
        ffn.fc2.coef.copy_(lin2.weight[..., None])
        ffn.fc2.bias.copy_(lin2.bias)
    x = torch.randn(64, 8)
    torch.testing.assert_close(ffn(x), lin2(nn.functional.gelu(lin1(x))), atol=2e-2, rtol=2e-2)


def _tiny_models():
    tr = pytest.importorskip("transformers")
    torch.manual_seed(0)
    return {
        "gpt2": (tr.GPT2LMHeadModel(tr.GPT2Config(n_layer=2, n_embd=32, n_head=4, vocab_size=100, n_positions=32, bos_token_id=0, eos_token_id=0)), "ids"),
        "bert": (tr.BertModel(tr.BertConfig(num_hidden_layers=2, hidden_size=32, num_attention_heads=4, intermediate_size=64, vocab_size=100)), "ids"),
        "llama": (
            tr.LlamaForCausalLM(tr.LlamaConfig(num_hidden_layers=2, hidden_size=32, num_attention_heads=4, num_key_value_heads=2, intermediate_size=64, vocab_size=100)),
            "ids",
        ),
        "vit": (tr.ViTModel(tr.ViTConfig(num_hidden_layers=2, hidden_size=32, num_attention_heads=4, intermediate_size=64, image_size=16, patch_size=8)), "pixels"),
    }


def _run(model, kind):
    torch.manual_seed(1)
    inputs = {"pixel_values": torch.randn(2, 3, 16, 16)} if kind == "pixels" else {"input_ids": torch.randint(0, 100, (2, 7))}
    out = model(**inputs)
    return out.logits if hasattr(out, "logits") else out.last_hidden_state


@pytest.mark.parametrize("name", ["gpt2", "bert", "llama", "vit"])
def test_kanify_exact_keeps_outputs_and_trains(name):
    model, kind = _tiny_models()[name]
    model.eval()
    with torch.no_grad():
        ref = _run(model, kind)
    done = T.kanify(model, mode="exact")
    assert len(done) == 2
    with torch.no_grad():
        torch.testing.assert_close(_run(model, kind), ref, rtol=0, atol=1e-6)
    kans = [m for m in model.modules() if isinstance(m, T.KANLinear)]
    assert len(kans) == 2
    _run(model, kind).square().mean().backward()
    assert all(k.coef.grad is not None and k.coef.grad.abs().sum() > 0 for k in kans)


@pytest.mark.parametrize("name", ["gpt2", "bert", "vit"])
def test_kanify_grkan_transfers_weights(name):
    model, kind = _tiny_models()[name]
    model.eval()
    with torch.no_grad():
        ref = _run(model, kind)
    converted = copy.deepcopy(model)
    assert len(T.kanify(converted, mode="grkan")) == 2
    with torch.no_grad():
        got = _run(converted, kind)
    assert (got - ref).abs().max() / ref.abs().max() < 1e-2
    assert any(isinstance(m, T.GRKAN) for m in converted.modules())


def test_kanify_rejects_unknown_mode():
    with pytest.raises(ValueError):
        T.kanify(nn.Sequential(), mode="fast")


# --- graphs ---------------------------------------------------------------------

EDGES = torch.tensor([[0, 1, 2, 3, 4, 5, 1], [1, 2, 3, 4, 5, 0, 0]])


def test_graph_layers_match_dense_formulas():
    torch.manual_seed(0)
    n, x = 6, torch.randn(6, 4)
    a = torch.zeros(n, n)
    a[EDGES[1], EDGES[0]] = 1
    gin = T.KAGINConv(4, 3, "chebyshev", hidden=[5], eps=0.3)
    torch.testing.assert_close(gin(x, EDGES), gin.kan(1.3 * x + a @ x))
    gcn = T.KAGCNConv(4, 3, "chebyshev")
    ah = a + torch.eye(n)
    d = ah.sum(1)
    torch.testing.assert_close(gcn(x, EDGES), gcn.kan(ah / torch.sqrt(d[:, None] * d[None, :]) @ x))
    gat = T.KAGATConv(4, 3, heads=2, basis="chebyshev")
    out, alpha = gat(x, EDGES, return_attention=True)
    assert out.shape == (6, 6)
    targets = torch.cat([EDGES[1], torch.arange(n)])
    torch.testing.assert_close(torch.zeros(n, 2).index_add_(0, targets, alpha), torch.ones(n, 2))


def test_graph_layers_match_mlx():
    mx = _mlx()
    import kanforge.mlx as M

    torch.manual_seed(0)
    x = torch.randn(6, 4)
    for tcls, mcls, kw in [(T.KAGCNConv, M.KAGCNConv, {}), (T.KAGINConv, M.KAGINConv, {}), (T.KAGATConv, M.KAGATConv, {"heads": 2})]:
        t, m = tcls(4, 3, basis="chebyshev", **kw), mcls(4, 3, basis="chebyshev", **kw)
        _load_into_mlx(m, t.state_dict())
        with mx.stream(mx.cpu):
            ym = np.array(m(mx.array(x.numpy()), mx.array(EDGES.numpy())))
        np.testing.assert_allclose(ym, t(x, EDGES).detach().numpy(), atol=1e-5)


# --- recurrent ------------------------------------------------------------------


def test_lstm_cell_follows_lstm_algebra():
    torch.manual_seed(0)
    cell = T.KANLSTMCell(3, 4, "chebyshev")
    x, h, c = torch.randn(2, 3), torch.randn(2, 4), torch.randn(2, 4)
    i, f, g, o = cell.gates(torch.cat([x, h], -1)).chunk(4, -1)
    c2 = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
    h2, (h3, c3) = cell(x, (h, c))
    torch.testing.assert_close(c3, c2)
    torch.testing.assert_close(h2, torch.sigmoid(o) * torch.tanh(c2))


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_kan_rnn_sequences_and_gradients(cell):
    torch.manual_seed(0)
    rnn = T.KANRNN(3, 5, num_layers=2, cell=cell)
    x = torch.randn(2, 7, 3, requires_grad=True)
    y, states = rnn(x)
    assert y.shape == (2, 7, 5) and len(states) == 2
    y[:, -1].sum().backward()
    assert x.grad[:, 0].abs().sum() > 0  # gradient flows back through time
    mx = _mlx()
    import kanforge.mlx as M

    ym, _ = M.KANRNN(3, 5, num_layers=2, cell=cell)(mx.random.normal((2, 7, 3)))
    assert ym.shape == (2, 7, 5)


# --- scientific ML ------------------------------------------------------------------


def test_derivatives_of_known_field():
    x = torch.rand(10, 2, dtype=torch.float64)

    def f(z):
        return torch.sin(z[:, 0]) * z[:, 1] ** 3

    torch.testing.assert_close(T.derivative(f, x, 1, 0), torch.cos(x[:, 0]) * x[:, 1] ** 3)
    torch.testing.assert_close(T.derivative(f, x, 2, 1), 6 * torch.sin(x[:, 0]) * x[:, 1])
    torch.testing.assert_close(T.laplacian(f, x), -torch.sin(x[:, 0]) * x[:, 1] ** 3 + 6 * torch.sin(x[:, 0]) * x[:, 1])
    torch.testing.assert_close(T.gradient(f, x)[:, 1], 3 * torch.sin(x[:, 0]) * x[:, 1] ** 2)
    mx = _mlx()
    import kanforge.mlx as M

    xm = mx.array(x.numpy().astype(np.float32))
    np.testing.assert_allclose(np.array(M.laplacian(lambda z: mx.sin(z[:, 0]) * z[:, 1] ** 3, xm)), T.laplacian(f, x).detach().numpy(), atol=1e-5)


def test_pinn_solves_poisson():
    torch.manual_seed(0)
    model = T.KAN([1, 6, 1], kf.BSpline(8, 3), dtype=torch.float64)
    xi = torch.linspace(-1, 1, 128, dtype=torch.float64)[:, None]
    xb = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
    opt = torch.optim.LBFGS(model.parameters(), lr=1, line_search_fn="strong_wolfe", max_iter=50)

    def closure():
        opt.zero_grad()
        r = T.derivative(model, xi, 2) + math.pi**2 * torch.sin(math.pi * xi[:, 0])
        loss = (r**2).mean() + 100 * (model(xb) ** 2).mean()
        loss.backward()
        return loss

    for _ in range(20):
        opt.step(closure)
    xt = torch.linspace(-1, 1, 200, dtype=torch.float64)[:, None]
    assert (model(xt)[:, 0] - torch.sin(math.pi * xt[:, 0])).abs().max() < 2e-3


def test_sampling():
    g = torch.Generator().manual_seed(0)
    x = T.sample_box([(0, 1), (-2, 2)], 100, g)
    assert x.shape == (100, 2) and (x[:, 1] >= -2).all() and (x[:, 1] <= 2).all()
    b = T.sample_boundary([(0, 1), (-2, 2)], 100, g)
    on = (b[:, 0] == 0) | (b[:, 0] == 1) | (b[:, 1] == -2) | (b[:, 1] == 2)
    assert on.all()


def test_fbkan_partition_of_unity_and_local_grids():
    fb = T.FBKAN([2, 3, 1], [(-1, 1), (0, 2)], parts=(3, 2), overlap=0.25)
    assert len(fb.nets) == 6
    w = fb.windows(T.sample_box([(-1, 1), (0, 2)], 200))
    torch.testing.assert_close(w.sum(1), torch.ones(200))
    assert fb(torch.rand(5, 2)).shape == (5, 1)
    g = fb.nets[0][0].basis_grid
    assert g[0, 0] == pytest.approx(-1 - 0.25 * 2 / 3) and g[1, -1] == pytest.approx(1 + 0.25)
    assert fb.nets[0][1].basis_grid.shape[0] == 3  # hidden layer keeps its own grids


def test_deeponet_shapes():
    op = T.DeepONet([10, 8, 6], [1, 8, 6])
    assert op(torch.rand(3, 10), torch.rand(20, 1)).shape == (3, 20)
    with pytest.raises(ValueError):
        T.DeepONet([10, 4], [1, 5])


def test_kan_ode_rk4_is_accurate():
    class Decay(T.KANODE):
        def rhs(self, t, y):
            return -y

    ode = Decay(1)
    t = torch.linspace(0, 1, 11, dtype=torch.float64)
    traj = ode(torch.ones(1, 1, dtype=torch.float64), t, substeps=4)
    torch.testing.assert_close(traj[:, 0, 0], torch.exp(-t), atol=1e-7, rtol=0)
    learn = T.KANODE(2, time_dependent=True)
    out = learn(torch.rand(4, 2), torch.linspace(0, 1, 5))
    out.sum().backward()
    assert out.shape == (5, 4, 2) and learn.f[0].coef.grad is not None
