"""Lookup-table form of KAN models: the deployable representation.

Every edge phi_oi of a layer is tabulated at ``size`` uniform points over
its input range; a layer then computes

    t = (x_i - lo_i) / step_i,  j = clip(floor(t), 0, size - 2),  f = t - j
    y_o = b_o + sum_i T[i, j, o] + f * (T[i, j+1, o] - T[i, j, o])

i.e. gather + lerp + sum — operations every runtime (ONNX, Core ML, FPGA
logic) has. Outside the range every edge continues linearly with its end
slope (``extrapolate="linear"``, which follows the SiLU base branch; the
slopes are float and estimated over the last 1/32 of the table, so
quantisation noise does not leak into them) or is held (``"clamp"``). A
table is only as good as its range: choose ``margin`` / samples so that
inputs stay inside. Works for models of any backend; MultKAN product nodes,
``Linear`` layers and standard activations are kept as such.

``quantize(lut, bits)`` stores the tables as signed integers with one scale
per output and a per-edge offset folded into the bias, so a layer sums
integers and multiplies once per output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..convert import activation_name
from ..interpret._backend import backend_of, like, to_numpy
from ..interpret.analysis import input_ranges

ACT_NUMPY = {
    "identity": lambda x: x,
    "silu": lambda x: x / (1 + np.exp(-x)),
    "relu": lambda x: np.maximum(x, 0),
    "tanh": np.tanh,
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "gelu": lambda x: 0.5 * x * (1 + np.vectorize(__import__("math").erf)(x / np.sqrt(2))),
    "gelu_tanh": lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))),
}


@dataclass
class LUTLayer:
    """Tables ``(in, size, out)`` over per-input ranges [lo, hi]."""

    tables: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    bias: np.ndarray
    n_sum: int | None = None  # MultKAN: sum nodes, then products of ``arity`` outputs
    n_mult: int = 0
    arity: int = 2
    scale: np.ndarray | None = None  # set when quantized: tables are integers, y = scale * sum + bias
    bits: int | None = None
    slope_lo: np.ndarray | None = None  # (in, out) per index step, real units
    slope_hi: np.ndarray | None = None

    kind = "kan"

    @property
    def size(self) -> int:
        return self.tables.shape[1]

    @property
    def step(self) -> np.ndarray:
        return (self.hi - self.lo) / (self.size - 1)

    def __call__(self, x: np.ndarray, extrapolate: str = "linear") -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        t_raw = (x - self.lo) / self.step
        t = np.clip(t_raw, 0, self.size - 1)
        j = np.clip(np.floor(t), 0, self.size - 2).astype(int)
        f = t - j
        i = np.arange(x.shape[-1])
        t0 = self.tables[i, j]  # (B, in, out)
        t1 = self.tables[i, j + 1]
        acc = (t0 + f[..., None] * (t1 - t0)).sum(-2)
        y = acc * self.scale + self.bias if self.scale is not None else acc + self.bias
        if extrapolate == "linear" and self.slope_lo is not None:
            below = np.minimum(t_raw, 0.0)[..., None]
            above = np.maximum(t_raw - (self.size - 1), 0.0)[..., None]
            y = y + (below * self.slope_lo + above * self.slope_hi).sum(-2)
        if self.n_mult:
            tail = y[..., self.n_sum :].reshape(*y.shape[:-1], self.n_mult, self.arity).prod(-1)
            y = np.concatenate([y[..., : self.n_sum], tail], -1)
        return y


@dataclass
class DenseLayer:
    weight: np.ndarray  # (out, in)
    bias: np.ndarray | None
    kind = "linear"

    def __call__(self, x, extrapolate="linear"):
        y = np.asarray(x, dtype=np.float64) @ self.weight.T
        return y if self.bias is None else y + self.bias


@dataclass
class ActivationLayer:
    name: str
    kind = "activation"

    def __call__(self, x, extrapolate="linear"):
        return ACT_NUMPY[self.name](np.asarray(x, dtype=np.float64))


@dataclass
class LUTModel:
    layers: list = field(default_factory=list)
    extrapolate: str = "linear"

    def __call__(self, x) -> np.ndarray:
        h = np.asarray(x, dtype=np.float64)
        for layer in self.layers:
            h = layer(h, self.extrapolate)
        return h

    @property
    def n_inputs(self) -> int:
        first = self.layers[0]
        return first.tables.shape[0] if first.kind == "kan" else first.weight.shape[1]

    def nbytes(self) -> int:
        return int(sum(layer.tables.nbytes for layer in self.layers if layer.kind == "kan"))


def end_slopes(tables: np.ndarray):
    """(slope_lo, slope_hi) per index step, each ``(in, out)``, from float tables."""
    size = tables.shape[1]
    k = max(1, (size - 1) // 32)
    return (tables[:, k] - tables[:, 0]) / k, (tables[:, -1] - tables[:, -1 - k]) / k


def _table(layer, lo, hi, size):
    pts = lo + (hi - lo) * np.linspace(0, 1, size)[:, None]  # (size, in)
    edges = to_numpy(layer.edge_derivative(like(layer, pts), 0))  # (size, out, in)
    return np.ascontiguousarray(edges.transpose(2, 0, 1))  # (in, size, out)


def to_lut(model, x=None, size: int = 256, margin: float = 0.05, extrapolate: str = "linear") -> LUTModel:
    """Tabulate a model (any backend). Ranges come from the activations every
    layer receives for samples ``x`` (widened by ``margin``), or from the grids."""
    if size < 2:
        raise ValueError("size must be >= 2")
    b = backend_of(model)
    children = [model] if isinstance(model, (b.KANLinear, b.MultKANLinear)) else list(getattr(model, "layers", model))
    out = LUTModel(extrapolate=extrapolate)
    h = x
    for m in children:
        kan, wrapper = (m, None) if isinstance(m, b.KANLinear) else (m.kan, m) if isinstance(m, b.MultKANLinear) else (None, None)
        if kan is not None:
            r = input_ranges(kan, h, margin)
            bias = getattr(kan, "bias", None)
            bias = np.zeros(kan.out_features) if bias is None else to_numpy(bias)
            tables = _table(kan, r[:, 0], r[:, 1], size)
            s_lo, s_hi = end_slopes(tables)
            layer = LUTLayer(tables, r[:, 0], r[:, 1], bias, slope_lo=s_lo, slope_hi=s_hi)
            if wrapper is not None and wrapper.n_mult:
                layer.n_sum, layer.n_mult, layer.arity = wrapper.n_sum, wrapper.n_mult, wrapper.arity
            out.layers.append(layer)
        elif type(m).__name__ == "Linear":
            w = to_numpy(m.kernel).T if hasattr(m, "kernel") else to_numpy(m.weight)
            bias = getattr(m, "bias", None)
            out.layers.append(DenseLayer(w, None if bias is None else to_numpy(bias)))
        else:
            out.layers.append(ActivationLayer(activation_name(m)))
        h = None if h is None else m(h)
    return out


def quantize(lut: LUTModel, bits: int = 8) -> LUTModel:
    """Integer tables: T[i, :, o] = c_io + s_o * q[i, :, o], |q| <= 2^(bits-1) - 1."""
    if not 2 <= bits <= 16:
        raise ValueError("bits must be in [2, 16]")
    qmax = 2 ** (bits - 1) - 1
    layers = []
    for layer in lut.layers:
        if layer.kind != "kan":
            layers.append(layer)
            continue
        t = layer.tables
        center = 0.5 * (t.max(1) + t.min(1))  # (in, out)
        dev = np.abs(t - center[:, None, :]).max((0, 1))  # (out,)
        scale = np.where(dev > 0, dev / qmax, 1.0)
        q = np.round((t - center[:, None, :]) / scale).astype(np.int8 if bits <= 8 else np.int16)
        bias = layer.bias + center.sum(0)
        layers.append(
            LUTLayer(
                q, layer.lo, layer.hi, bias, layer.n_sum, layer.n_mult, layer.arity,
                scale=scale, bits=bits, slope_lo=layer.slope_lo, slope_hi=layer.slope_hi,
            )
        )
    return LUTModel(layers, lut.extrapolate)
