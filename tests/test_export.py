"""Phase 7: weight conversion, lookup tables, int8, ONNX, Core ML, FPGA."""

import shutil
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import kanforge as kf  # noqa: E402
from kanforge import convert as cv  # noqa: E402
from kanforge.export import fpga, quantize, to_lut  # noqa: E402
from kanforge.torch import GRKAN, KAN, MultKAN, from_mlp  # noqa: E402


def _models():
    torch.manual_seed(0)
    kan = KAN([3, 5, 2], kf.BSpline(6, 3), edge_scale=True)
    sym = KAN([2, 2], kf.Fourier(3))
    sym[0].fix_symbolic(0, 1, "sin", (2.0, 0.1, 1.5, 0.0))
    sym[0].set_edge_mask([[1, 0], [0, 1]])
    models = {
        "kan": (kan, 3),
        "multkan": (MultKAN([3, (2, 2), 1], "chebyshev"), 3),
        "grkan": (GRKAN(8, 4, groups=2), 8),
        "mlp": (from_mlp(torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.GELU(), torch.nn.Linear(6, 2))), 3),
        "symbolic": (sym, 2),
    }
    with torch.no_grad():
        for m, _ in models.values():
            for p in m.parameters():
                p.add_(0.2 * torch.randn_like(p))
    return models


def _run(model, x, backend):
    if backend == "torch":
        return model(torch.tensor(x)).detach().numpy()
    if backend == "mlx":
        mx = pytest.importorskip("mlx.core")
        with mx.stream(mx.cpu):
            return np.array(model(mx.array(x)))
    jnp = pytest.importorskip("jax.numpy")
    return np.asarray(model(jnp.asarray(x)))


# --- conversion -----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["kan", "multkan", "grkan", "mlp", "symbolic"])
@pytest.mark.parametrize("target", ["mlx", "jax", "torch"])
def test_convert_between_backends(name, target):
    if target == "mlx":
        pytest.importorskip("mlx.core")
    if target == "jax":
        pytest.importorskip("flax.nnx")
    model, n_in = _models()[name]
    x = np.random.default_rng(0).uniform(-0.9, 0.9, (7, n_in)).astype(np.float32)
    converted = cv.convert(model, target)
    np.testing.assert_allclose(_run(converted, x, target), _run(model, x, "torch"), atol=2e-6)
    back = cv.convert(converted, "torch")
    np.testing.assert_allclose(_run(back, x, "torch"), _run(model, x, "torch"), atol=2e-6)


def test_save_load_round_trip_and_dtype(tmp_path):
    model = KAN([2, 3, 1], kf.BSpline(4, 3), dtype=torch.float64)
    model[0].update_grid(torch.randn(100, 2, dtype=torch.float64) * 0.4)
    path = tmp_path / "m.npz"
    cv.save(model, path)
    back = cv.load(path, "torch")
    assert back[0].coef.dtype == torch.float64
    x = torch.rand(5, 2, dtype=torch.float64)
    torch.testing.assert_close(back(x), model(x), rtol=0, atol=0)
    torch.testing.assert_close(back[0].basis_grid, model[0].basis_grid)


def test_activation_names():
    import torch.nn.functional as F

    assert cv.activation_name(F.silu) == "silu"
    assert cv.activation_name(torch.nn.GELU()) == "gelu"
    assert cv.activation_name(torch.nn.GELU(approximate="tanh")) == "gelu_tanh"
    assert cv.activation_name(torch.nn.Identity()) == "identity"
    with pytest.raises(ValueError):
        cv.activation_name(torch.nn.Softplus())
    x = torch.linspace(-3, 3, 11)
    for name in cv.ACTIVATIONS:
        torch.testing.assert_close(cv.make_activation(name, "torch")(x), cv.make_activation(name, "torch")(x))


# --- lookup tables -------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["kan", "multkan", "mlp", "symbolic"])
def test_lut_accuracy_and_int8(name):
    model, n_in = _models()[name]
    x = torch.rand(3000, n_in) * 2 - 1
    xin = (torch.rand(500, n_in) * 1.8 - 0.9).numpy()
    ref = _run(model, xin.astype(np.float32), "torch")
    scale = np.abs(ref).max()
    errs = [np.abs(to_lut(model, x, size)(xin) - ref).max() / scale for size in (64, 256, 1024)]
    assert errs[2] < 1e-4 and errs[2] < errs[0]
    q = quantize(to_lut(model, x, 256))
    assert np.abs(q(xin) - ref).max() / scale < 3e-2
    assert q.nbytes() < to_lut(model, x, 256).nbytes() / 3


def test_lut_extrapolation_modes():
    model = KAN([1, 1], kf.BSpline(5, 3))
    lut = to_lut(model, torch.rand(500, 1) * 2 - 1, 256, margin=0.0)
    far = np.array([[3.0]])
    layer = lut.layers[0]
    slope = layer.slope_hi[0, 0] / layer.step[0]
    lin = lut(far)
    lut.extrapolate = "clamp"
    clamp = lut(far)
    np.testing.assert_allclose(clamp, lut(np.array([[layer.hi[0]]])), rtol=1e-12)
    np.testing.assert_allclose(lin - clamp, slope * (3.0 - layer.hi[0]), rtol=1e-9)


def test_lut_of_mlx_and_jax_models_match_torch():
    model, _ = _models()["kan"]
    x = np.random.default_rng(3).uniform(-1, 1, (400, 3)).astype(np.float32)
    want = to_lut(model, torch.tensor(x), 128)
    for target in ("mlx", "jax"):
        try:
            other = cv.convert(model, target)
        except ImportError:
            continue
        if target == "mlx":
            # CPU stream: MLX's GPU matmul would shift the hidden ranges the tables are built on
            mx = __import__("mlx.core", fromlist=["core"])
            with mx.stream(mx.cpu):
                got = to_lut(other, mx.array(x), 128)
        else:
            got = to_lut(other, __import__("jax.numpy", fromlist=["numpy"]).asarray(x), 128)
        for a, b in zip(got.layers, want.layers):
            np.testing.assert_allclose(a.lo, b.lo, atol=1e-6)
            np.testing.assert_allclose(a.tables, b.tables, atol=5e-5)


# --- ONNX -------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["kan", "multkan", "mlp"])
@pytest.mark.parametrize("bits", [None, 8])
def test_onnx_lut_runs_in_onnxruntime(name, bits, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from kanforge.export.onnx_export import export_onnx

    model, n_in = _models()[name]
    x = torch.rand(2000, n_in) * 2 - 1
    lut = export_onnx(model, tmp_path / "m.onnx", x, 256, bits)
    xt = (np.random.default_rng(1).uniform(-1.2, 1.2, (300, n_in))).astype(np.float32)
    y = ort.InferenceSession(str(tmp_path / "m.onnx")).run(None, {"x": xt})[0]
    np.testing.assert_allclose(y, lut(xt), rtol=1e-5, atol=1e-5 * np.abs(lut(xt)).max())


@pytest.mark.parametrize("basis", [kf.BSpline(5, 3), kf.Chebyshev(4), kf.RBF(6), kf.Fourier(3)], ids=["bspline", "chebyshev", "rbf", "fourier"])
def test_onnx_exact_export(basis, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from kanforge.export.onnx_export import export_onnx_exact

    model = KAN([3, 4, 2], basis, backend="ops")
    export_onnx_exact(model, tmp_path / "e.onnx", torch.rand(5, 3))
    xt = np.random.default_rng(2).uniform(-1.2, 1.2, (64, 3)).astype(np.float32)
    y = ort.InferenceSession(str(tmp_path / "e.onnx")).run(None, {"x": xt})[0]
    np.testing.assert_allclose(y, model(torch.tensor(xt)).detach().numpy(), atol=1e-5)


# --- Core ML ----------------------------------------------------------------------------

coreml = pytest.mark.skipif(sys.platform != "darwin", reason="Core ML runs on macOS")


@coreml
@pytest.mark.parametrize("name", ["kan", "multkan", "mlp"])
@pytest.mark.parametrize("bits", [None, 8])
def test_coreml_lut(name, bits, tmp_path):
    ct = pytest.importorskip("coremltools")
    from kanforge.export.coreml import export_coreml

    model, n_in = _models()[name]
    lut = export_coreml(model, tmp_path / "m.mlpackage", torch.rand(2000, n_in) * 2 - 1, 256, bits)
    xt = np.random.default_rng(1).uniform(-1.2, 1.2, (64, n_in)).astype(np.float32)
    y = ct.models.MLModel(str(tmp_path / "m.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY).predict({"x": xt})["y"]
    np.testing.assert_allclose(y, lut(xt), rtol=1e-5, atol=1e-5 * np.abs(lut(xt)).max())


@coreml
@pytest.mark.parametrize("basis", [kf.BSpline(5, 3), kf.Chebyshev(4)], ids=["bspline", "chebyshev"])
def test_coreml_exact(basis, tmp_path):
    pytest.importorskip("coremltools")
    from kanforge.export.coreml import export_coreml_exact

    model = KAN([3, 4, 2], basis, backend="ops")
    ml = export_coreml_exact(model, tmp_path / "e.mlpackage", torch.rand(5, 3))
    xt = np.random.default_rng(2).uniform(-1.2, 1.2, (64, 3)).astype(np.float32)
    np.testing.assert_allclose(ml.predict({"x": xt})["y"], model(torch.tensor(xt)).detach().numpy(), atol=2e-5)


# --- FPGA -------------------------------------------------------------------------------


def _fpga_model():
    model, _ = _models()["kan"]
    return model, torch.rand(3000, 3) * 2 - 1


def test_fpga_design_accuracy_improves_with_bits():
    model, x = _fpga_model()
    ref = model(x).detach().numpy()
    errs = []
    for bits in (6, 8, 10):
        d = fpga.design(model, x, bits, bits)
        _, y = fpga.simulate(d, x.numpy())
        errs.append(np.abs(y - ref).max() / np.abs(ref).max())
    assert errs[0] > errs[1] > errs[2] and errs[1] < 6e-2


def test_fpga_rejects_unsupported_layers():
    with pytest.raises(ValueError):
        fpga.design(MultKAN([2, (1, 1), 1]), torch.rand(100, 2))


def test_fpga_files_encode_the_design(tmp_path):
    model, x = _fpga_model()
    d = fpga.export_fpga(model, tmp_path, x, 8, 8, test_vectors=16)
    names = {p.name for p in tmp_path.iterdir()}
    assert {"kan_top.v", "kan_layer0.v", "kan_layer1.v", "tb_kan.v", "design.json", "l0_in0.mem"} <= names
    rows = (tmp_path / "l1_in3.mem").read_text().split()
    words = [[(int(r, 16) >> (o * 8)) & 0xFF for o in range(2)] for r in rows]
    decoded = np.array(words, dtype=np.int64)
    decoded[decoded > 127] -= 256
    np.testing.assert_array_equal(decoded, d.layers[1].q[3])
    exp = (tmp_path / "tb_expected.mem").read_text().split()
    first = int(exp[0], 16)
    acc0 = [(first >> (64 * o)) & (2**64 - 1) for o in range(2)]
    acc0 = [v - 2**64 if v >= 2**63 else v for v in acc0]
    xs = x.numpy()[np.linspace(0, len(x) - 1, 16).astype(int)]
    np.testing.assert_array_equal(acc0, fpga.simulate(d, xs)[0][0])


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="needs Icarus Verilog")
def test_fpga_rtl_matches_bit_exact_model(tmp_path):
    model, x = _fpga_model()
    fpga.export_fpga(model, tmp_path, x, 8, 8, test_vectors=64)
    sources = sorted(str(p.name) for p in tmp_path.glob("*.v"))
    subprocess.run(["iverilog", "-g2012", "-o", "sim", *sources], cwd=tmp_path, check=True)
    out = subprocess.run(["vvp", "sim"], cwd=tmp_path, check=True, capture_output=True, text=True).stdout
    assert "PASS 64 vectors" in out, out
