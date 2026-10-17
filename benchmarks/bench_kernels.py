"""Fused kernels vs plain ops: median time of forward and forward+backward.

    .venv/bin/python benchmarks/bench_kernels.py [--quick]

Runs on PyTorch MPS / CUDA and MLX (Metal) when available.
"""

from __future__ import annotations

import argparse
import statistics
import time

import kanforge as kf

BASES = {"bspline G8 k3": kf.BSpline(8, 3), "chebyshev d8": kf.Chebyshev(8), "rbf 8": kf.RBF(8), "fourier 4": kf.Fourier(4)}


def timeit(fn, sync, reps):
    for _ in range(3):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def bench_torch(shapes, reps):
    import torch

    from kanforge.torch import KANLinear

    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None
    if dev is None:
        return
    sync = torch.cuda.synchronize if dev == "cuda" else torch.mps.synchronize
    print(f"\n## PyTorch {dev}")
    print("| basis | batch x in x out | fwd ops | fwd kernel | fwd+bwd ops | fwd+bwd kernel |\n|---|---|---|---|---|---|")
    for name, basis in BASES.items():
        for b, i, o in shapes:
            x = torch.rand(b, i, device=dev) * 2 - 1
            row = []
            for mode in ("fwd", "bwd"):
                for backend in ("ops", "kernel"):
                    layer = KANLinear(i, o, basis, backend=backend, device=dev)
                    xx = x.clone().requires_grad_(mode == "bwd")

                    def run():
                        y = layer(xx)
                        if mode == "bwd":
                            y.sum().backward()

                    row.append(timeit(run, sync, reps))
            print(f"| {name} | {b}x{i}x{o} | " + " | ".join(f"{t:.2f} ms" for t in row) + " |")


def bench_mlx(shapes, reps):
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError:
        return
    from kanforge.mlx import KANLinear

    print("\n## MLX (Metal)")
    print("| basis | batch x in x out | fwd ops | fwd kernel | fwd+bwd ops | fwd+bwd kernel |\n|---|---|---|---|---|---|")
    for name, basis in BASES.items():
        for b, i, o in shapes:
            x = mx.random.uniform(-1, 1, (b, i))
            row = []
            for mode in ("fwd", "bwd"):
                for backend in ("ops", "kernel"):
                    layer = KANLinear(i, o, basis, backend=backend)
                    grad = nn.value_and_grad(layer, lambda m: m(x).sum())

                    def run():
                        if mode == "fwd":
                            mx.eval(layer(x))
                        else:
                            mx.eval(grad(layer))

                    row.append(timeit(run, lambda: None, reps))
            print(f"| {name} | {b}x{i}x{o} | " + " | ".join(f"{t:.2f} ms" for t in row) + " |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    shapes = [(1024, 64, 64)] if args.quick else [(1024, 64, 64), (8192, 128, 128), (32768, 64, 64)]
    reps = 5 if args.quick else 20
    bench_torch(shapes, reps)
    bench_mlx(shapes, reps)
