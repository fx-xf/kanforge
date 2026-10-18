"""Pallas kernels vs plain jax.numpy (both jit-compiled): spline part of one layer.

    XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/bench_jax.py [--quick]

fwd+bwd differentiates a loss with respect to the coefficients and the input.
Plain ops use IEEE float32 products like the kernels (no TF32).
"""

from __future__ import annotations

import argparse
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

import kanforge as kf
from kanforge.jax.kernels import spline_forward
from kanforge.kernels import plan_for

BASES = {"bspline G8 k3": kf.BSpline(8, 3), "chebyshev d8": kf.Chebyshev(8), "rbf 8": kf.RBF(8), "fourier 4": kf.Fourier(4)}


def timeit(fn, reps):
    for _ in range(3):
        jax.block_until_ready(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def main(shapes, reps):
    print(f"\n## JAX {jax.__version__} on {jax.devices()[0].device_kind}")
    print("| basis | batch x in x out | fwd ops | fwd kernel | fwd+bwd ops | fwd+bwd kernel |\n|---|---|---|---|---|---|")
    for name, basis in BASES.items():
        plan = plan_for(basis)
        for b, i, o in shapes:
            key = jax.random.key(0)
            x = jax.random.uniform(key, (b, i), minval=-1, maxval=1)
            c = jax.random.normal(key, (o, i, basis.size)) * 0.1
            params = {k: jnp.asarray(np.broadcast_to(s.value, (i, *s.shape)), jnp.float32) for k, s in basis.param_specs().items()}
            grid = params.get("grid")

            def ops(x, c):
                phi = basis(x, params)
                return jnp.einsum("bin,oin->bo", phi, c, precision=jax.lax.Precision.HIGHEST)

            def ker(x, c):
                return spline_forward(x, c, grid, plan)

            row = []
            for f in (ops, ker):
                fwd = jax.jit(f)
                row.append(timeit(lambda fwd=fwd: fwd(x, c), reps))
            for f in (ops, ker):
                bwd = jax.jit(jax.grad(lambda x, c, f=f: (f(x, c) ** 2).sum(), argnums=(0, 1)))
                row.append(timeit(lambda bwd=bwd: bwd(x, c), reps))
            print(f"| {name} | {b}x{i}x{o} | " + " | ".join(f"{t:.2f} ms" for t in row) + " |", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    main([(1024, 64, 64)] if args.quick else [(1024, 64, 64), (8192, 128, 128), (32768, 64, 64)], 5 if args.quick else 20)
