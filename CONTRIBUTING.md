# Contributing to kanforge

Thank you for helping. Bug reports, new bases, kernels, benchmarks and
documentation fixes are all welcome.

## Setting up

```bash
git clone https://github.com/fx-xf/kanforge
cd kanforge
python3.12 -m venv .venv
.venv/bin/pip install -e ".[torch,test]"      # add mlx, jax, export, coreml as needed
.venv/bin/python -m pytest
```

The documentation builds with `pip install -e ".[docs]" && mkdocs serve`.

## Ground rules

- **Every backend agrees with the reference.** New bases and kernels come with
  tests against `kanforge.reference` (float64), including derivatives.
- **Same parameters everywhere.** A layer stores `coef (out, in, n)`,
  `base_weight (out, in)` and `bias (out,)` on every backend.
- **Licences matter.** Only adapt code from MIT, BSD or Apache-2.0 projects
  and credit it in `NOTICE`. For projects without a licence or under GPL,
  implement from the paper instead.
- **Benchmarks are fair.** Compare at equal parameters or FLOPs and publish
  the losing cases too.

## Pull requests

1. Open an issue first for larger changes, so we can agree on the design.
2. Keep pull requests focused; add a line to `CHANGELOG.md`.
3. Make sure `pytest` and `mkdocs build --strict` pass.

By contributing you agree that your work is released under the Apache-2.0
licence of the project.
