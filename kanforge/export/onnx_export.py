"""ONNX export.

* ``export_onnx(model_or_lut, path, x)`` — builds the graph directly from the
  lookup-table form (gather + lerp + sum per layer, int8 tables with a scale
  per output, product nodes, Linear layers, activations). Independent of the
  training backend: PyTorch, MLX and JAX models export the same way. Runs in
  onnxruntime and any ONNX runtime with opset 17.
* ``export_onnx_exact(torch_model, path, example)`` — the exact network
  (basis functions as arithmetic) through ``torch.onnx.export``.
"""

from __future__ import annotations

import numpy as np

from .lut import LUTModel, quantize, to_lut

OPSET = 17
IR_VERSION = 8


class _Graph:
    def __init__(self):
        from onnx import helper

        self.h = helper
        self.nodes, self.inits = [], []
        self.n = 0

    def name(self, base="t"):
        self.n += 1
        return f"{base}{self.n}"

    def const(self, value, dtype=np.float32, base="c"):
        from onnx import numpy_helper

        nm = self.name(base)
        self.inits.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), nm))
        return nm

    def op(self, op_type, *inputs, **attrs):
        out = self.name(op_type.lower())
        self.nodes.append(self.h.make_node(op_type, list(inputs), [out], **attrs))
        return out


def _lut_layer(g: _Graph, x: str, layer, extrapolate: str) -> str:
    from onnx import TensorProto

    n_in, size, n_out = layer.tables.shape
    traw = g.op("Mul", g.op("Sub", x, g.const(layer.lo)), g.const(1.0 / layer.step))
    t = g.op("Clip", traw, g.const(0.0), g.const(size - 1.0))
    j = g.op("Clip", g.op("Floor", t), g.const(0.0), g.const(size - 2.0))
    f = g.op("Sub", t, j)
    idx = g.op("Add", g.op("Cast", j, to=TensorProto.INT64), g.const(np.arange(n_in) * size, np.int64))
    idx1 = g.op("Add", idx, g.const(1, np.int64))
    quant = layer.scale is not None
    table = g.const(layer.tables.reshape(n_in * size, n_out), layer.tables.dtype if quant else np.float32, "table")
    t0, t1 = g.op("Gather", table, idx, axis=0), g.op("Gather", table, idx1, axis=0)
    if quant:
        t0, t1 = g.op("Cast", t0, to=TensorProto.FLOAT), g.op("Cast", t1, to=TensorProto.FLOAT)
    axis_last, axis_in = g.const([-1], np.int64), g.const([1], np.int64)
    v = g.op("Add", t0, g.op("Mul", g.op("Unsqueeze", f, axis_last), g.op("Sub", t1, t0)))
    acc = g.op("ReduceSum", v, axis_in, keepdims=0)
    if quant:
        acc = g.op("Mul", acc, g.const(layer.scale))
    y = g.op("Add", acc, g.const(layer.bias))
    if extrapolate == "linear" and layer.slope_lo is not None:
        below = g.op("Unsqueeze", g.op("Min", traw, g.const(0.0)), axis_last)
        above = g.op("Unsqueeze", g.op("Max", g.op("Sub", traw, g.const(size - 1.0)), g.const(0.0)), axis_last)
        ext = g.op("Add", g.op("Mul", below, g.const(layer.slope_lo)), g.op("Mul", above, g.const(layer.slope_hi)))
        y = g.op("Add", y, g.op("ReduceSum", ext, axis_in, keepdims=0))
    if layer.n_mult:
        sums = g.op("Slice", y, g.const([0], np.int64), g.const([layer.n_sum], np.int64), axis_in)
        tail = g.op("Slice", y, g.const([layer.n_sum], np.int64), g.const([n_out], np.int64), axis_in)
        tail = g.op("Reshape", tail, g.const([-1, layer.n_mult, layer.arity], np.int64))
        prod = g.op("ReduceProd", tail, axes=[2], keepdims=0)
        y = g.op("Concat", sums, prod, axis=1)
    return y


def _activation(g: _Graph, x: str, name: str) -> str:
    if name == "identity":
        return g.op("Identity", x)
    if name == "silu":
        return g.op("Mul", x, g.op("Sigmoid", x))
    if name in ("relu", "tanh", "sigmoid"):
        return g.op({"relu": "Relu", "tanh": "Tanh", "sigmoid": "Sigmoid"}[name], x)
    if name == "gelu":
        erf = g.op("Erf", g.op("Mul", x, g.const(1 / np.sqrt(2))))
        return g.op("Mul", g.op("Mul", x, g.const(0.5)), g.op("Add", erf, g.const(1.0)))
    if name == "gelu_tanh":
        inner = g.op("Mul", g.op("Add", x, g.op("Mul", g.op("Mul", g.op("Mul", x, x), x), g.const(0.044715))), g.const(np.sqrt(2 / np.pi)))
        return g.op("Mul", g.op("Mul", x, g.const(0.5)), g.op("Add", g.op("Tanh", inner), g.const(1.0)))
    raise ValueError(f"no ONNX activation for {name}")


def onnx_model(lut: LUTModel):
    """ONNX ``ModelProto`` of a lookup-table model; input "x" (batch, n_in) float32."""
    from onnx import TensorProto, checker

    g = _Graph()
    h = "x"
    for layer in lut.layers:
        if layer.kind == "kan":
            h = _lut_layer(g, h, layer, lut.extrapolate)
        elif layer.kind == "linear":
            h = g.op("MatMul", h, g.const(layer.weight.T))
            if layer.bias is not None:
                h = g.op("Add", h, g.const(layer.bias))
        else:
            h = _activation(g, h, layer.name)
    g.nodes.append(g.h.make_node("Identity", [h], ["y"]))
    n_out = _out_features(lut)
    graph = g.h.make_graph(
        g.nodes,
        "kanforge",
        [g.h.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", lut.n_inputs])],
        [g.h.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", n_out])],
        g.inits,
    )
    model = g.h.make_model(graph, opset_imports=[g.h.make_opsetid("", OPSET)], producer_name="kanforge")
    model.ir_version = IR_VERSION  # what opset 17 needs; newer onnx writes versions older runtimes reject
    checker.check_model(model)
    return model


def _out_features(lut: LUTModel) -> int:
    for layer in reversed(lut.layers):
        if layer.kind == "kan":
            return layer.n_sum + layer.n_mult if layer.n_mult else layer.tables.shape[2]
        if layer.kind == "linear":
            return layer.weight.shape[0]
    raise ValueError("model has no layers")


def export_onnx(model, path, x=None, size: int = 256, bits: int | None = None, margin: float = 0.05, extrapolate: str = "linear"):
    """Write ``model`` (any backend, or a ``LUTModel``) as an ONNX lookup-table network."""
    import onnx

    lut = model if isinstance(model, LUTModel) else to_lut(model, x, size, margin, extrapolate)
    if bits is not None:
        lut = quantize(lut, bits)
    onnx.save(onnx_model(lut), str(path))
    return lut


def export_onnx_exact(model, path, example, opset: int = OPSET):
    """The exact PyTorch network as ONNX (``torch.onnx.export``, TorchScript exporter)."""
    import torch

    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            (example,),
            str(path),
            input_names=["x"],
            output_names=["y"],
            dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
            opset_version=opset,
            dynamo=False,
        )
