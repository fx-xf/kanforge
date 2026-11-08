# Installation

```bash
pip install "kanforge @ git+https://github.com/fx-xf/kanforge"          # NumPy reference only
pip install "kanforge[torch] @ git+https://github.com/fx-xf/kanforge"   # + PyTorch layers (CUDA, ROCm, MPS, CPU)
```

Once the package is on PyPI, `pip install "kanforge[torch]"` will do the same.

Optional extras, combine as needed (`pip install "kanforge[torch,export] @ git+https://github.com/fx-xf/kanforge"`):

| extra | adds |
|---|---|
| `torch` | PyTorch layers, kernels (Metal on MPS; Triton on CUDA needs `triton`) |
| `mlx` | MLX layers and Metal kernels (Apple silicon) |
| `jax` | JAX / Flax NNX layers and Pallas kernels (Python ≥ 3.11) |
| `symbolic` | SymPy, for formulas and the kanpiler |
| `plot` | matplotlib, for edge plots |
| `hf` | transformers, for `kanify` |
| `export` | onnx, onnxruntime |
| `coreml` | coremltools (macOS) |
| `test` | everything the test suite uses |

Requirements: Python ≥ 3.10, NumPy ≥ 1.26. Kernels: PyTorch ≥ 2.6 for MPS
(`torch.mps.compile_shader`), Triton ≥ 3.0 for CUDA.
