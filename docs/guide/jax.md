# JAX

`kanforge.jax` implements the layers on Flax NNX (`flax>=0.12`):

```python
from flax import nnx
import optax
from kanforge.jax import KAN, regularization

model = KAN([2, 5, 1], "bspline", rngs=nnx.Rngs(0))
opt = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)

@nnx.jit
def step(model, opt, x, y):
    loss, grads = nnx.value_and_grad(lambda m: ((m(x) - y) ** 2).mean())(model)
    opt.update(model, grads)
    return loss
```

Trainable state is `nnx.Param`; grids, masks and symbolic edges are
`Buffer` variables, which gradients and optimisers skip. `load_state_dict`
accepts PyTorch state dicts. Pallas kernels exist (`backend="kernel"`) but
XLA's fused plain path is faster today.
