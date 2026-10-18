"""KAN vs MLP at equal FLOPs (KANbeFair methodology).

    .venv/bin/python benchmarks/fair_comparison.py [--quick] [--out results.md]

Every model is trained with the same optimiser (Adam, lr 3e-3), steps and
data; B-spline KANs additionally adapt their grids to the data at the start
and after a quarter of training (their standard recipe, ``update_grid``). FLOPs and
parameters come from ``kanforge.flops``. Reported: test RMSE (regression,
lower is better) or accuracy (classification, higher is better), and the
best model of each family within a FLOP budget.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
from torch import nn

import kanforge as kf
from kanforge.flops import count
from kanforge.torch import KAN

torch.set_num_threads(4)


# --- tasks --------------------------------------------------------------------------


def _split(x, y, n_test, seed=0):
    idx = np.random.default_rng(seed).permutation(len(x))
    te, tr = idx[:n_test], idx[n_test:]
    f = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    return f(x[tr]), f(y[tr]), f(x[te]), f(y[te])


def formula_task(name, fn, n_in, n=3000):
    x = np.random.default_rng(1).uniform(-1, 1, (n, n_in))
    return {"name": name, "kind": "regression", "data": _split(x, fn(x)[:, None], 1000)}


def friedman_task(n=3000):
    from sklearn.datasets import make_friedman1

    x, y = make_friedman1(n, n_features=10, noise=0.0, random_state=0)
    y = (y - y.mean()) / y.std()
    return {"name": "Friedman-1 (10 inputs)", "kind": "regression", "data": _split(2 * x - 1, y[:, None], 1000)}


def digits_task():
    from sklearn.datasets import load_digits

    d = load_digits()
    x = d.data / 8.0 - 1.0
    return {"name": "digits (64 inputs, 10 classes)", "kind": "classification", "data": _split(x, d.target, 500)}


TASKS = [
    lambda: formula_task("exp(sin(pi x1) + x2^2)", lambda x: np.exp(np.sin(np.pi * x[:, 0]) + x[:, 1] ** 2), 2),
    lambda: formula_task("x1 * x2", lambda x: x[:, 0] * x[:, 1], 2),
    lambda: formula_task("sin(x1) exp(-x2^2) + x3 / (1 + x4^2)", lambda x: np.sin(x[:, 0]) * np.exp(-x[:, 1] ** 2) + x[:, 2] / (1 + x[:, 3] ** 2), 4),
    friedman_task,
    digits_task,
]


# --- models -------------------------------------------------------------------------


def mlp(n_in, width, n_out):
    return nn.Sequential(nn.Linear(n_in, width), nn.SiLU(), nn.Linear(width, n_out))


def families(n_in, n_out, quick):
    mlp_w = [8, 32, 128] if quick else [4, 8, 16, 32, 64, 128, 256]
    kan_w = [2, 8] if quick else [1, 2, 4, 8, 16, 32]
    return {
        "MLP": [(w, mlp(n_in, w, n_out)) for w in mlp_w],
        "KAN B-spline G5 k3": [(w, KAN([n_in, w, n_out], kf.BSpline(5, 3))) for w in kan_w],
        "KAN Chebyshev d4": [(w, KAN([n_in, w, n_out], kf.Chebyshev(4))) for w in kan_w],
    }


def train(model, task, steps, is_kan):
    """``is_kan``: a KAN with grids, which follows its grid-update recipe."""
    xtr, ytr, xte, yte = task["data"]
    clf = task["kind"] == "classification"
    loss_fn = nn.CrossEntropyLoss() if clf else nn.MSELoss()
    torch.manual_seed(0)

    def opt_for(m):
        return torch.optim.Adam(m.parameters(), lr=3e-3)

    if is_kan:
        model.update_grid(xtr)
    opt = opt_for(model)
    bs = 256 if clf else len(xtr)
    g = torch.Generator().manual_seed(0)
    for step in range(steps):
        if is_kan and step == steps // 4:
            with torch.no_grad():
                model.update_grid(xtr)
            opt = opt_for(model)
        idx = torch.randint(0, len(xtr), (bs,), generator=g) if clf else slice(None)
        opt.zero_grad()
        out = model(xtr[idx])
        loss = loss_fn(out, ytr[idx].long() if clf else ytr[idx])
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = model(xte)
        if clf:
            return float((out.argmax(1) == yte.long()).float().mean())
        return float(torch.sqrt(((out - yte) ** 2).mean()))


def run(quick: bool):
    steps = 300 if quick else 3000
    sections = []
    for make in TASKS:
        task = make()
        xtr, ytr, _, _ = task["data"]
        n_in = xtr.shape[1]
        n_out = 10 if task["kind"] == "classification" else 1
        rows = []
        for fam, models in families(n_in, n_out, quick).items():
            for width, model in models:
                c = count(model)
                t0 = time.perf_counter()
                gridded = fam.startswith("KAN") and bool(model[0].basis.grid_names())
                metric = train(model, task, steps, gridded)
                rows.append((fam, width, c["params"], c["flops"], metric, time.perf_counter() - t0))
                print(f"  {task['name']:40s} {fam:20s} w={width:<4d} flops={c['flops']:>8d} metric={metric:.4g}", flush=True)
        sections.append((task, rows))
    return sections, steps


def markdown(sections, steps) -> str:
    out = [f"Adam lr 3e-3, {steps} steps, seed 0; metric: test RMSE (lower is better) or accuracy (higher).", ""]
    for task, rows in sections:
        clf = task["kind"] == "classification"
        better = max if clf else min
        out += [f"### {task['name']}", "", "| model | width | params | FLOPs | " + ("accuracy" if clf else "RMSE") + " |", "|---|---|---|---|---|"]
        for fam, w, p, f, m, _ in rows:
            out.append(f"| {fam} | {w} | {p} | {f} | {m:.4g} |")
        budgets = sorted({10 ** math.ceil(math.log10(f)) for *_, f, _, _ in [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows]})
        out += ["", "Best within a FLOP budget:", "", "| FLOPs <= | " + " | ".join(sorted({r[0] for r in rows})) + " |", "|---" * (1 + len({r[0] for r in rows})) + "|"]
        for b in budgets:
            cells = []
            for fam in sorted({r[0] for r in rows}):
                cand = [r[4] for r in rows if r[0] == fam and r[3] <= b]
                cells.append(f"{better(cand):.4g}" if cand else "—")
            out.append(f"| {b:.0e} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    sections, steps = run(args.quick)
    md = markdown(sections, steps)
    print(md)
    if args.out:
        with open(args.out, "w") as f:
            f.write(md)
