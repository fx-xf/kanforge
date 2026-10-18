"""Move models between PyTorch, MLX and JAX, and save / load them anywhere.

A model is described by a JSON-able *spec* (architecture, basis configs,
constructor options, activation names) plus a flat dict of NumPy arrays
named like torch's ``state_dict``. Supported: ``KANLinear``, ``GRKAN``,
``KAN``, ``MultKANLinear``, ``MultKAN`` and sequences of those with
``Linear`` layers and standard activations (models from ``from_mlp``).

    mlx_model = convert(torch_model, "mlx")
    save(model, "model.npz");  model = load("model.npz", backend="jax")
"""

from __future__ import annotations

import functools
import json

import numpy as np

from .interpret._backend import backend_of, to_numpy

ACTIVATIONS = ("identity", "silu", "gelu", "gelu_tanh", "tanh", "relu", "sigmoid")


# --- activations ------------------------------------------------------------------


def activation_name(act) -> str | None:
    """Portable name of a base activation / activation layer (None = absent)."""
    if act is None:
        return None
    if isinstance(act, functools.partial):
        name = getattr(act.func, "__name__", "")
        if name == "gelu":
            approx = act.keywords.get("approximate", None)
            return "gelu_tanh" if approx in (True, "tanh") else "gelu"
    cls = type(act).__name__.lower()
    fname = getattr(act, "__name__", "").lower()
    mod = getattr(act, "__module__", "") or type(act).__module__
    for name in (cls, fname):
        if name in ("identity", "_identity"):
            return "identity"
        if name in ("silu", "swish"):
            return "silu"
        if name in ("tanh",):
            return "tanh"
        if name in ("relu",):
            return "relu"
        if name in ("sigmoid",):
            return "sigmoid"
    if "gelu" in cls or "gelu" in fname:
        if mod.startswith("jax") and fname == "gelu":
            return "gelu_tanh"  # jax.nn.gelu defaults to approximate=True
        approx = getattr(act, "approximate", None) or getattr(act, "approx", None)
        if approx in ("tanh", "precise", True) or "new" in cls or "fast" in cls:
            return "gelu_tanh"
        return "gelu"
    raise ValueError(f"cannot convert activation {act!r}")


def make_activation(name: str | None, backend: str):
    """Callable for ``name`` on ``backend`` ("torch", "mlx", "jax")."""
    if name is None:
        return None
    if backend == "torch":
        import torch.nn.functional as F
        from torch import nn

        table = {
            "identity": nn.Identity(),
            "silu": F.silu,
            "gelu": F.gelu,
            "gelu_tanh": functools.partial(F.gelu, approximate="tanh"),
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
        }
    elif backend == "mlx":
        import mlx.core as mx
        from mlx import nn

        table = {
            "identity": nn.Identity(),
            "silu": nn.silu,
            "gelu": nn.gelu,
            "gelu_tanh": nn.GELU(approx="precise"),
            "tanh": mx.tanh,
            "relu": nn.relu,
            "sigmoid": mx.sigmoid,
        }
    elif backend == "jax":
        import jax

        table = {
            "identity": lambda x: x,
            "silu": jax.nn.silu,
            "gelu": functools.partial(jax.nn.gelu, approximate=False),
            "gelu_tanh": functools.partial(jax.nn.gelu, approximate=True),
            "tanh": jax.numpy.tanh,
            "relu": jax.nn.relu,
            "sigmoid": jax.nn.sigmoid,
        }
    else:
        raise ValueError("backend must be 'torch', 'mlx' or 'jax'")
    return table[name]


# --- model -> (spec, arrays) ----------------------------------------------------------


def _backend_name(model) -> str:
    return backend_of(model).__name__.rsplit(".", 1)[-1]


def _dtype_name(array) -> str:
    """"float32" / "float64" / ... of a torch, MLX, JAX or NumPy array."""
    from .interpret._backend import _unwrap

    return str(_unwrap(array).dtype).replace("torch.", "").replace("mlx.core.", "")


def _kan_linear_spec(layer, b):
    kw = dict(layer._init_kwargs)
    spec = {
        "type": "GRKAN" if isinstance(layer, b.GRKAN) else "KANLinear",
        "in": layer.in_features,
        "out": layer.out_features,
        "basis": layer.basis.config(),
        "bias": bool(kw["bias"]),
        "base_activation": activation_name(kw["base_activation"]),
        "groups": kw["groups"],
        "edge_scale": bool(kw["edge_scale"]),
        "edge_dropout": float(kw["edge_dropout"]),
        "backend": kw["backend"],
        "dtype": _dtype_name(layer.coef),
    }
    arrays = {k: v for k, v in layer._export_full().items() if v is not None}
    return spec, arrays


def to_spec(model):
    """(spec, arrays) of a model of any backend."""
    b = backend_of(model)
    arrays: dict[str, np.ndarray] = {}

    def walk(m, prefix):
        if isinstance(m, b.KANLinear):
            spec, arr = _kan_linear_spec(m, b)
            arrays.update({prefix + k: v for k, v in arr.items()})
            return spec
        if isinstance(m, b.MultKANLinear):
            return {"type": "MultKANLinear", "n_sum": m.n_sum, "n_mult": m.n_mult, "arity": m.arity, "kan": walk(m.kan, prefix + "kan.")}
        if isinstance(m, (b.KAN, b.MultKAN)):
            children = list(getattr(m, "layers", m))
            return {"type": type(m).__name__, "widths": json.loads(json.dumps(m.widths)), "layers": [walk(c, f"{prefix}{k}.") for k, c in enumerate(children)]}
        kind = type(m).__name__
        if kind == "Sequential":
            children = list(getattr(m, "layers", m))
            return {"type": "Sequential", "layers": [walk(c, f"{prefix}{k}.") for k, c in enumerate(children)]}
        if kind == "Linear":
            if hasattr(m, "kernel"):  # flax nnx: kernel (in, out)
                w, bias = to_numpy(m.kernel).T, getattr(m, "bias", None)
            else:
                w, bias = to_numpy(m.weight), getattr(m, "bias", None) if "bias" in dir(m) or hasattr(m, "bias") else None
            arrays[prefix + "weight"] = w
            if bias is not None:
                arrays[prefix + "bias"] = to_numpy(bias)
            dtype = _dtype_name(m.kernel if hasattr(m, "kernel") else m.weight)
            return {"type": "Linear", "in": int(w.shape[1]), "out": int(w.shape[0]), "bias": bias is not None, "dtype": dtype}
        return {"type": "Activation", "name": activation_name(m)}

    return walk(model, ""), arrays


# --- (spec, arrays) -> model --------------------------------------------------------------


def _build(spec, arrays, backend: str, prefix: str = ""):
    import importlib

    b = importlib.import_module(f"kanforge.{backend}")
    t = spec["type"]
    if t in ("KANLinear", "GRKAN"):
        kw = {
            "bias": spec["bias"],
            "base_activation": make_activation(spec["base_activation"], backend),
            "groups": spec["groups"],
            "edge_scale": spec["edge_scale"],
            "edge_dropout": spec["edge_dropout"],
            "backend": spec["backend"],
        }
        dtype = spec.get("dtype", "float32")
        if backend == "torch":
            kw["dtype"] = _torch_dtype(dtype)
        elif backend == "jax":
            import jax.numpy as jnp

            kw["dtype"] = jnp.dtype(dtype)
        layer = b.KANLinear(spec["in"], spec["out"], spec["basis"], **kw)
        state = {k[len(prefix) :]: v for k, v in arrays.items() if k.startswith(prefix) and "." not in k[len(prefix) :]}
        layer._import_full(state)
        return layer
    if t == "MultKANLinear":
        m = b.MultKANLinear(spec["kan"]["in"], spec["n_sum"], spec["n_mult"], arity=spec["arity"])
        m.kan = _build(spec["kan"], arrays, backend, prefix + "kan.")
        return m
    if t in ("KAN", "MultKAN", "Sequential"):
        children = [_build(c, arrays, backend, f"{prefix}{k}.") for k, c in enumerate(spec["layers"])]
        if t == "KAN":
            return b.rebuild_kan(children)
        if t == "MultKAN":
            widths = [tuple(w) if isinstance(w, list) else w for w in spec["widths"]]
            model = b.MultKAN(widths, arity=children[0].arity if children else 2)
            if backend == "torch":
                for k, c in enumerate(children):
                    model[k] = c
            else:
                model.layers = type(model.layers)(children) if backend == "jax" else children
            return model
        return _sequential(children, backend)
    if t == "Linear":
        return _linear(spec, arrays, backend, prefix)
    if t == "Activation":
        act = make_activation(spec["name"], backend)
        if backend == "torch" and not hasattr(act, "parameters"):
            return _TorchLambda(act)
        return act
    raise ValueError(f"unknown layer type {t}")


def _torch_dtype(name: str):
    import torch

    return getattr(torch, name)


def _torch_lambda_cls():
    from torch import nn

    class TorchLambda(nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, x):
            return self.fn(x)

    return TorchLambda


def _TorchLambda(fn):  # noqa: N802
    return _torch_lambda_cls()(fn)


def _linear(spec, arrays, backend, prefix):
    w = arrays[prefix + "weight"]
    bias = arrays.get(prefix + "bias")
    if backend == "torch":
        import torch
        from torch import nn

        lin = nn.Linear(spec["in"], spec["out"], bias=spec["bias"], dtype=_torch_dtype(spec.get("dtype", "float32")))
        with torch.no_grad():
            lin.weight.copy_(torch.as_tensor(w))
            if bias is not None:
                lin.bias.copy_(torch.as_tensor(bias))
        return lin
    if backend == "mlx":
        import mlx.core as mx
        from mlx import nn

        lin = nn.Linear(spec["in"], spec["out"], bias=spec["bias"])
        lin.weight = mx.array(w.astype(np.float32))
        if bias is not None:
            lin.bias = mx.array(bias.astype(np.float32))
        return lin
    import jax.numpy as jnp
    from flax import nnx

    dtype = jnp.dtype(spec.get("dtype", "float32"))
    lin = nnx.Linear(spec["in"], spec["out"], use_bias=spec["bias"], rngs=nnx.Rngs(0), param_dtype=dtype, dtype=dtype)
    lin.kernel[...] = jnp.asarray(w.T, dtype)
    if bias is not None:
        lin.bias[...] = jnp.asarray(bias, dtype)
    return lin


def _sequential(children, backend):
    if backend == "torch":
        from torch import nn

        return nn.Sequential(*children)
    if backend == "mlx":
        from mlx import nn

        return nn.Sequential(*children)
    from flax import nnx

    return nnx.Sequential(*children)


def from_spec(spec, arrays, backend: str):
    """Build the model described by ``spec`` on ``backend`` and load ``arrays``."""
    return _build(spec, arrays, backend)


def convert(model, backend: str):
    """The same model (architecture and weights) on another backend."""
    spec, arrays = to_spec(model)
    return from_spec(spec, arrays, backend)


def save(model, path) -> None:
    """Backend-independent file: arrays and the JSON spec in one ``.npz``."""
    spec, arrays = to_spec(model)
    np.savez(path, __spec__=np.array(json.dumps({"format": 1, "source": _backend_name(model), "model": spec})), **arrays)


def load(path, backend: str = "torch"):
    """Load a file written by ``save`` on any backend."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["__spec__"]))
        arrays = {k: data[k] for k in data.files if k != "__spec__"}
    return from_spec(meta["model"], arrays, backend)
