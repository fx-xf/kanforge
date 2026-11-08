<p align="center">
  <img src="assets/hero.svg" width="100%" alt="kanforge — Kolmogorov–Arnold Networks for every backend">
</p>

<p align="center">
  <a href="https://github.com/fx-xf/kanforge/actions/workflows/ci.yml"><img alt="tests" src="https://img.shields.io/github/actions/workflow/status/fx-xf/kanforge/ci.yml?branch=main&style=flat-square&label=tests&labelColor=0E0E14&color=FF8A2E"></a>
  <a href="https://fx-xf.github.io/kanforge/"><img alt="docs" src="https://img.shields.io/badge/docs-mkdocs-FFC65C?style=flat-square&labelColor=0E0E14"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.10%20%E2%80%93%203.13-F2464F?style=flat-square&labelColor=0E0E14">
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-B23CF0?style=flat-square&labelColor=0E0E14"></a>
  <img alt="tests count" src="https://img.shields.io/badge/583%20tests-float64%20reference-6A4CFF?style=flat-square&labelColor=0E0E14">
</p>

<p align="center">
  <a href="https://fx-xf.github.io/kanforge/"><b>Documentation</b></a> ·
  <a href="https://fx-xf.github.io/kanforge/quickstart/">Quickstart</a> ·
  <a href="https://fx-xf.github.io/kanforge/tutorials/">Tutorials</a> ·
  <a href="BENCHMARKS.md">Kernel benchmarks</a> ·
  <a href="ROADMAP.md">Roadmap</a> ·
  <a href="CHANGELOG.md">Changelog</a>
</p>

<br>

A Kolmogorov–Arnold Network replaces every weight of a neural network with a
small learnable function. **kanforge** gives you those layers with
26 families of edge functions, on PyTorch, Apple MLX and JAX — one parameter
layout everywhere, fused GPU kernels, tools that turn trained curves back into
formulas, and exporters that take a model all the way down to an FPGA.

<p align="center">
  <img src="assets/bento.svg" width="100%" alt="Fused kernels, interpretation, transformer surgery, scientific ML, deployment, one file format">
</p>

<img src="assets/rule-install.svg" width="100%" alt="01 Install">

```bash
pip install "kanforge[torch] @ git+https://github.com/fx-xf/kanforge"
```

Combine extras as you need them: `torch`, `mlx` (Apple silicon), `jax`
(Flax NNX, Python ≥ 3.11), `symbolic`, `plot`, `hf`, `export` (ONNX),
`coreml`. The core depends on NumPy only.

| Backend | Import | Devices |
| :-- | :-- | :-- |
| PyTorch | `kanforge.torch` | CUDA · ROCm · Apple MPS · CPU |
| MLX | `kanforge.mlx` | Apple silicon GPU / CPU |
| JAX · Flax NNX | `kanforge.jax` | GPU · TPU · CPU |
| NumPy, float64 | `kanforge.reference` | CPU — the reference every test checks against |

<img src="assets/rule-tour.svg" width="100%" alt="02 A short tour">

**A KAN in three lines.** Any basis, any backend, the same parameters.

```python
import torch
from kanforge.torch import KAN

model = KAN([2, 5, 1], basis="bspline")          # or "chebyshev", "fourier", "morlet", …
y = model(torch.rand(64, 2))
```

**Layers with character.** Exact derivatives of any order, the group-rational
layer of KAT, multiplication nodes of KAN 2.0.

```python
import kanforge as kf
from kanforge.torch import GRKAN, KANLinear, MultKAN

layer = KANLinear(8, 4, kf.Jacobi(6, alpha=0.5, beta=0.5), edge_scale=True)
d2 = layer.edge_derivative(torch.rand(32, 8), order=2)   # exact d²y/dx², (32, 4, 8)
ffn = GRKAN(768, 3072, groups=8)                         # KAT's group-rational FFN
model = MultKAN([2, (2, 1), 1])                          # two sum nodes, one product node
```

**Grids that follow your data.** Adaptive per-input grids, refinement that
keeps the function, conversion between bases, and MLPs turned into KANs
without changing a single output.

```python
from kanforge.torch import KAN, from_mlp, regularization

model = KAN([2, 5, 1], kf.BSpline(5, 3), coef_init="lecun")
x = torch.rand(1024, 2) * 4 - 2
model.update_grid(x)                          # per-input adaptive grids
model.refine_grid(10)                         # finer grid, same function
model[0].change_basis(kf.Chebyshev(12))       # convert a layer to another basis
loss = model(x).pow(2).mean() + 1e-3 * regularization(model, x)

mlp = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.GELU(), torch.nn.Linear(16, 1))
kan = from_mlp(mlp)                           # equals the MLP exactly — now fine-tune the splines
```

**From curves to formulas — and back.**

```python
import sympy
from kanforge import interpret as ki

pruned = ki.prune(model, x)                   # drop unimportant nodes and edges
ki.auto_symbolic(pruned, x)                   # fit c·f(a·x + b) + d to every edge
print(ki.symbolic_formula(pruned, ["x", "y"]))

X, Y = sympy.symbols("x y")
kan, _ = ki.kanpile(sympy.exp(sympy.sin(sympy.pi * X) + Y**2), [X, Y])   # formula → MultKAN
```

**Bigger models.** KAN convolutions, transformer surgery, derivatives for
physics-informed training.

```python
import transformers
from kanforge.torch import KANConv2d, kanify, derivative

gpt2 = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
kanify(gpt2)                                  # FFN projections → KANLinear, identical outputs
conv = KANConv2d(3, 16, 3, padding=1, basis="chebyshev")
u_xx = derivative(net, x, order=2)            # PDE residuals for PINNs
```

**Anywhere it needs to run.**

```python
import kanforge
from kanforge.export import fpga
from kanforge.export.onnx_export import export_onnx

kanforge.save(model, "model.npz")                         # from any backend …
jax_model = kanforge.load("model.npz", backend="jax")     # … into any other
mlx_model = kanforge.convert.convert(model, "mlx")
export_onnx(model, "model.onnx", x, size=256, bits=8)     # int8 lookup tables
fpga.export_fpga(model, "rtl/", x, address_bits=8, value_bits=8)   # Verilog + bit-exact model
```

<img src="assets/rule-bases.svg" width="100%" alt="03 Bases">

<p align="center">
  <img src="assets/bases.svg" width="100%" alt="All 26 basis families, plotted on [-1, 1]">
</p>

Every curve above is computed by `kanforge.reference`. Every basis provides
derivatives of any order in closed form, shares its trainable parameters per
input, per group or per layer, and runs through the same fused kernels.

<img src="assets/rule-bench.svg" width="100%" alt="04 When to use a KAN">

<p align="center">
  <img src="assets/benchmark.svg" width="100%" alt="Equal-FLOPs comparison of KANs and MLPs">
</p>

KANs are not a free win. At equal FLOPs they are **5–50× more accurate** on
smooth, compositional problems — physical laws, symbolic formulas, scientific
tabular data — and slightly behind MLPs on raw pixels. The full tables are in
[`benchmarks/fair_results.md`](benchmarks/fair_results.md); kernel timings on
Apple M5 and RTX 3060 are in [BENCHMARKS.md](BENCHMARKS.md).

<img src="assets/rule-more.svg" width="100%" alt="05 Documentation">

| | |
| :-- | :-- |
| [**Guide**](https://fx-xf.github.io/kanforge/guide/bases/) | bases · layers and backends · training · fused kernels · interpretation · architecture blocks · JAX · deployment |
| [**Tutorials**](https://fx-xf.github.io/kanforge/tutorials/) | regression · formula discovery · PINNs · operator learning · images · transformers · graphs · time series · neural ODEs · reinforcement learning · deployment · backends |
| [**API reference**](https://fx-xf.github.io/kanforge/api/bases/) | generated from the docstrings |
| [**Examples**](examples/) | the twelve tutorials as runnable scripts |

<img src="assets/rule-contrib.svg" width="100%" alt="06 Contributing">

Issues and pull requests are welcome — read [CONTRIBUTING](CONTRIBUTING.md)
and the [roadmap](ROADMAP.md) first. kanforge is released under the
[Apache-2.0](LICENSE) licence; ideas and code from other KAN projects are
credited in [NOTICE](NOTICE). If kanforge helps your research, please cite it
([CITATION.cff](CITATION.cff)) together with the original KAN paper
([arXiv:2404.19756](https://arxiv.org/abs/2404.19756)).

<br>
<p align="center"><sub>The figures in this README are drawn by <code>assets/src/build.py</code> — every curve in them is computed by kanforge.</sub></p>
