"""Fused kernels: fold matrices, Metal (MPS, MLX) and Triton against plain ops,
including derivatives of every order through the closed F/G/P family."""

import os

import numpy as np
import pytest
from numpy.polynomial import chebyshev as C

import kanforge as kf
from kanforge.kernels import plan_for
from kanforge.kernels.triton_kernels import module_source

SUPPORTED = {
    "bspline": kf.BSpline(5, 3),
    "bspline_linear": kf.BSpline(4, 1),
    "chebyshev": kf.Chebyshev(5),
    "legendre_identity": kf.Legendre(4, input_map="identity"),
    "jacobi": kf.Jacobi(4, 0.5, 1.5),
    "gram": kf.Gram(4),
    "taylor": kf.Taylor(3),
    "fourier": kf.Fourier(3),
    "rbf": kf.RBF(6),
    "mexican_hat": kf.MexicanHat(5, learnable=False),
    "dog": kf.DoG(5, learnable=False),
    "rswaf": kf.RSWAF(6),
}
UNSUPPORTED = [kf.Rational(), kf.NURBS(), kf.MexicanHat(5), kf.Fourier(3, learnable_frequency=True), kf.BSRBF(), kf.ReLUKAN()]


@pytest.mark.parametrize("name", [n for n in SUPPORTED if plan_for(SUPPORTED[n]).folded])
@pytest.mark.parametrize("m", [0, 1, 2, 3])
def test_fold_matrices_reproduce_derivatives(name, m):
    basis = SUPPORTED[name]
    plan = plan_for(basis)
    x = np.linspace(-1.2, 1.2, 41)
    r = plan.fold(m)
    if plan.kind == "cheb":
        z = np.tanh(x) if plan.consts[1] == "tanh" else x
        feats = C.chebvander(z, plan.n_features(m) - 1)
    else:
        w = 2 * np.pi / plan.consts[1] * np.arange(1, plan.consts[0] + 1)
        feats = np.concatenate([np.cos(np.outer(x, w)), np.sin(np.outer(x, w))], 1)
    np.testing.assert_allclose(feats @ r.T, basis.derivative(x, m), atol=1e-9 * max(1, np.abs(feats @ r.T).max()))


def test_unsupported_bases_have_no_plan():
    assert all(plan_for(b) is None for b in UNSUPPORTED)


@pytest.mark.parametrize("name", list(SUPPORTED))
def test_triton_sources_are_valid_python(name):
    plan = plan_for(SUPPORTED[name])
    for m in range(4):
        compile(module_source(plan, plan.native_order(m), plan.n_features(m)), name, "exec")


# --- PyTorch: MPS (Metal) and Triton (CUDA, or CPU interpreter) ---------------

torch = pytest.importorskip("torch")
from kanforge.torch import KANLinear  # noqa: E402
from kanforge.torch import kernels as tk  # noqa: E402


def _torch_devices():
    out = []
    if torch.backends.mps.is_available():
        try:
            torch.ones(1, device="mps").cpu()
            out.append("mps")
        except RuntimeError:
            pass
    if torch.cuda.is_available() and tk.available(torch.device("cuda")):
        out.append("cuda")
    if os.environ.get("TRITON_INTERPRET") == "1" and tk.available(torch.device("cpu")):
        out.append("cpu")
    return out


DEVICES = _torch_devices()
needs_device = pytest.mark.skipif(not DEVICES, reason="no MPS, CUDA or Triton interpreter")


def _pair(basis, device, n_in=4, n_out=5, seed=0):
    torch.manual_seed(seed)
    ref = KANLinear(n_in, n_out, basis, backend="ops", edge_scale=True)
    with torch.no_grad():
        ref.coef.normal_(0, 0.5)
        ref.edge_scale.uniform_(0.5, 1.5)
        ref.bias.normal_()
    ker = KANLinear(n_in, n_out, basis, backend="kernel", edge_scale=True)
    ker.load_state_dict(ref.state_dict())
    return ref.to(device), ker.to(device)


def _close(a, b, tol):
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    scale = max(1.0, b.abs().max().item())
    assert (a - b).abs().max().item() <= tol * scale, (a - b).abs().max().item()


@needs_device
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", list(SUPPORTED))
def test_torch_kernel_matches_ops_with_all_gradients(name, device):
    ref, ker = _pair(SUPPORTED[name], device)
    x0 = (torch.rand(37, 4) * 1.8 - 0.9).to(device)
    outs = []
    for layer in (ref, ker):
        x = x0.clone().requires_grad_()
        y = layer(x)
        loss = (y * torch.linspace(-1, 1, 5, device=device)).sum()
        (gx,) = torch.autograd.grad(loss, x, create_graph=True)
        (gxx,) = torch.autograd.grad(gx.sum(), x, create_graph=True)
        (gxxx,) = torch.autograd.grad(gxx.sum(), x, create_graph=True)
        # gradient penalty: double backward into the coefficients
        (gc,) = torch.autograd.grad((gx**2).sum(), layer.coef, retain_graph=True)
        gs = torch.autograd.grad(loss, [layer.coef, layer.edge_scale, layer.base_weight])
        outs.append((y, gx, gxx, gxxx, gc, *gs))
    for a, b in zip(outs[1], outs[0]):
        _close(a, b, 2e-4)


@needs_device
@pytest.mark.parametrize("device", DEVICES)
def test_torch_kernel_extra_batch_dims_and_dtype(device):
    ref, ker = _pair(SUPPORTED["bspline"], device)
    x = (torch.rand(2, 3, 7, 4) * 2 - 1).to(device)
    _close(ker(x), ref(x), 1e-5)
    if device != "cpu":  # half layers: the kernel computes in float32 and casts back
        y = ker.half()(x.half())
        assert y.dtype == torch.float16
        _close(y, ref.half()(x.half()), 1e-2)


@needs_device
@pytest.mark.parametrize("device", DEVICES)
def test_torch_kernel_after_grid_update(device):
    ref, ker = _pair(SUPPORTED["bspline"], device)
    x = (torch.randn(500, 4) * 0.4 + 0.2).to(device)
    ref.update_grid(x)
    ker.update_grid(x)
    _close(ker(x), ref(x), 1e-5)


def test_backend_selection():
    layer = KANLinear(3, 2, kf.Rational(), base_activation=None)
    assert not layer.uses_kernel(torch.device("mps"))  # auto falls back for this basis
    with pytest.raises(RuntimeError, match="basis"):
        KANLinear(3, 2, kf.Rational(), backend="kernel")(torch.zeros(1, 3))
    with pytest.raises(ValueError):
        KANLinear(3, 2, backend="fast")
    plain = KANLinear(3, 2, "bspline", backend="ops")
    assert not plain.uses_kernel(torch.device("mps"))
    # auto follows the benchmarks: splines on MPS yes, Chebyshev no, CUDA not yet
    if tk.available(torch.device("mps")):
        assert KANLinear(3, 2, "bspline").uses_kernel(torch.device("mps"))
        assert not KANLinear(3, 2, "chebyshev").uses_kernel(torch.device("mps"))
        assert KANLinear(3, 2, "chebyshev", backend="kernel").uses_kernel(torch.device("mps"))


# --- MLX -----------------------------------------------------------------------


def _mlx():
    try:
        import mlx.core as mx
    except ImportError:
        return None
    return mx if mx.metal.is_available() else None


@pytest.mark.skipif(_mlx() is None, reason="needs MLX with Metal")
@pytest.mark.parametrize("name", list(SUPPORTED))
def test_mlx_kernel_matches_ops_with_nested_gradients(name):
    import mlx.core as mx
    import mlx.nn as nn

    from kanforge.mlx import KANLinear as MLXLinear

    basis = SUPPORTED[name]
    mx.random.seed(0)
    # No base branch: its x @ W.T is MLX's inexact Metal matmul, not the kernel.
    ref = MLXLinear(4, 5, basis, backend="ops", base_activation=None)
    ref.coef = mx.random.normal(ref.coef.shape) * 0.5
    ker = MLXLinear(4, 5, basis, backend="kernel", base_activation=None)
    ker.update(ref.parameters())
    x = mx.random.uniform(-0.9, 0.9, (37, 4))
    wts = mx.linspace(-1, 1, 5)

    def results(layer):
        f = lambda v: (layer(v) * wts).sum()  # noqa: E731
        g1 = mx.grad(f)
        g2 = mx.grad(lambda v: g1(v).sum())
        g3 = mx.grad(lambda v: g2(v).sum())
        gc = nn.value_and_grad(layer, lambda mdl: (mx.grad(lambda v: (mdl(v) * wts).sum())(x) ** 2).sum())(layer)[1]
        return layer(x), g1(x), g2(x), g3(x), gc["coef"]

    with mx.stream(mx.cpu):
        want = results(ref)  # CPU stream: exact float32 matmul for the reference
    got = results(ker)
    for a, b in zip(got, want):
        a, b = np.array(a), np.array(b)
        assert np.abs(a - b).max() <= 2e-4 * max(1.0, np.abs(b).max())
