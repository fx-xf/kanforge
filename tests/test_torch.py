import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kanforge import BSpline, Chebyshev, Fourier  # noqa: E402
from kanforge.reference import kan_linear  # noqa: E402
from kanforge.torch import KAN, KANLinear  # noqa: E402

BASES = [BSpline(5, 3), Chebyshev(4), Fourier(3)]
IDS = ["bspline", "chebyshev", "fourier"]


def devices():
    out = ["cpu"]
    if torch.cuda.is_available():
        out.append("cuda")
    if torch.backends.mps.is_available():
        out.append("mps")
    return out


@pytest.mark.parametrize("basis", BASES, ids=IDS)
@pytest.mark.parametrize("device", devices())
def test_matches_reference(basis, device):
    torch.manual_seed(0)
    layer = KANLinear(4, 3, basis, device=device)
    x = torch.rand(16, 4, device=device) * 2 - 1
    expected = kan_linear(
        x.cpu().numpy(),
        layer.coef.detach().cpu().numpy(),
        layer.base_weight.detach().cpu().numpy(),
        layer.bias.detach().cpu().numpy(),
        basis,
    )
    np.testing.assert_allclose(layer(x).detach().cpu().numpy(), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("basis", BASES, ids=IDS)
def test_first_and_second_derivatives(basis):
    # PINN-style losses differentiate the output twice w.r.t. the input.
    torch.manual_seed(0)
    layer = KANLinear(2, 2, basis, dtype=torch.float64)
    with torch.no_grad():
        layer.bias.normal_()
    x = (torch.rand(5, 2, dtype=torch.float64) * 1.6 - 0.8).requires_grad_()
    assert torch.autograd.gradcheck(layer, (x,))
    assert torch.autograd.gradgradcheck(layer, (x,))


def test_without_base_branch_and_bias():
    layer = KANLinear(3, 2, "chebyshev", base_activation=None, bias=False)
    assert layer.base_weight is None and layer.bias is None
    assert set(dict(layer.named_parameters())) == {"coef"}
    assert layer(torch.zeros(4, 3)).shape == (4, 2)


def test_half_precision_dtype_is_preserved():
    layer = KANLinear(3, 2, "bspline", dtype=torch.bfloat16)
    assert layer(torch.rand(4, 3, dtype=torch.bfloat16)).dtype == torch.bfloat16


def test_extra_batch_dims():
    layer = KANLinear(3, 2, "fourier")
    assert layer(torch.rand(2, 5, 7, 3)).shape == (2, 5, 7, 2)


def test_kan_fits_1d_function():
    torch.manual_seed(0)
    model = KAN([1, 1], BSpline(10, 3))
    x = torch.linspace(-1, 1, 256)[:, None]
    y = torch.sin(torch.pi * x)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        loss = torch.mean((model(x) - y) ** 2)
        loss.backward()
        opt.step()
    assert loss.item() < 1e-3


def test_kan_requires_two_widths():
    with pytest.raises(ValueError):
        KAN([3])
