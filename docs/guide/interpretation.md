# Interpretation

`kanforge.interpret` works on PyTorch, MLX and JAX models.

```python
from kanforge import interpret as ki

tables = ki.tabulate_model(model, x)             # every edge sampled on its input range
scores = ki.attribution(model, x)                # edge and node importance
pruned = ki.prune(model, x, node_threshold=0.05, edge_threshold=0.05)
ki.suggest_symbolic(xs, ys)                      # ranked c*f(a x + b) + d candidates
pruned[0].fix_symbolic(0, 1, "sin", (3.14, 0, 1, 0))
ki.auto_symbolic(pruned, x)                      # best candidate for every edge (R² ≥ 0.9)
ki.symbolic_formula(pruned, ["x", "y"])          # SymPy expressions

import sympy
X, Y = sympy.symbols("x y")
kan, col = ki.kanpile(sympy.exp(sympy.sin(sympy.pi * X) + Y**2), [X, Y])   # formula -> MultKAN
```

Symbolic edges, masks and fits are stored in the layer, so they survive
`state_dict`, conversion and export. Edges that are neither symbolic nor
pruned are listed by `symbolic_formula` instead of being approximated
silently. `ki.contrastive_samples` sets up pykan's unsupervised mode.
