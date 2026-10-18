# PDEs with PINNs

Solve a Poisson equation with the PDE residual in the loss; `derivative` keeps the graph for second derivatives.

```bash
python examples/03_pinn_poisson.py
```

Set `KANFORGE_QUICK=1` for a few-second run (what the test suite does).

```python
--8<-- "examples/03_pinn_poisson.py"
```
