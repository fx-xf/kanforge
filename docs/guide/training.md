# Training

**Grids.** Every input has its own grid (a buffer). Keep them on the data:

```python
model.update_grid(x)          # pykan's adaptive grid: quantiles + a uniform share
model.refine_grid(10)         # finer grid, same function (exact for nested B-spline grids)
model[0].change_basis(kf.Chebyshev(12))   # any basis to any other, by least squares
```

All three refit the coefficients so the function is preserved; parameters
are replaced, so create the optimiser again afterwards.

**Regularisation** (pykan): `loss + lam * regularization(model, x)` — L1 and
entropy of edge magnitudes, optionally L1 and smoothness of coefficients;
without `x` it uses |coef| (efficient-kan). Needed before pruning.

**Initialisation**: `coef_init="lecun"` / `"glorot"` give unit output
variance for N(0, 1) inputs, accounting for the basis and the SiLU branch.

**From an MLP**: `from_mlp(nn.Sequential(Linear, act, Linear, ...))` and
`KANLinear.from_linear(linear, act)` reproduce the MLP exactly; the splines
start at zero and learn corrections.

**Optimisers**: everything works with Adam; LBFGS (pykan's default for small
problems) converges much faster on smooth regression and PDE tasks.
