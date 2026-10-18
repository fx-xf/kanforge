"""Core ML export (``coremltools``): Apple devices, Neural Engine / GPU / CPU.

* ``export_coreml(model_or_lut, path, x)`` — the lookup-table network as an
  ML Program built op by op (gather + lerp + sum; int8 tables stored as int8
  with ``constexpr_affine_dequantize``), for models of any backend.
  Needs macOS 13 / iOS 16 or newer.
* ``export_coreml_exact(torch_model, path, example)`` — the exact network
  through ``torch.jit.trace`` and ``coremltools.convert``.
"""

from __future__ import annotations

import numpy as np

from .lut import LUTModel, quantize, to_lut


def _lut_layer(mb, x, layer, extrapolate):
    n_in, size, n_out = layer.tables.shape
    f32 = np.float32
    traw = mb.mul(x=mb.sub(x=x, y=layer.lo.astype(f32)), y=(1.0 / layer.step).astype(f32))
    t = mb.clip(x=traw, alpha=f32(0.0), beta=f32(size - 1))
    j = mb.clip(x=mb.floor(x=t), alpha=f32(0.0), beta=f32(size - 2))
    f = mb.sub(x=t, y=j)
    idx = mb.add(x=mb.cast(x=j, dtype="int32"), y=(np.arange(n_in) * size).astype(np.int32))
    idx1 = mb.add(x=idx, y=np.int32(1))
    flat = layer.tables.reshape(n_in * size, n_out)
    if layer.scale is not None and layer.bits <= 8:
        # int8 in the file, one scale per output column: exactly q * s_o
        table = mb.constexpr_affine_dequantize(
            quantized_data=flat.astype(np.int8), zero_point=np.int8(0), scale=layer.scale.astype(f32), axis=1
        )
    elif layer.scale is not None:
        table = mb.const(val=(flat * layer.scale).astype(f32))
    else:
        table = mb.const(val=flat.astype(f32))
    t0, t1 = mb.gather(x=table, indices=idx, axis=0), mb.gather(x=table, indices=idx1, axis=0)
    v = mb.add(x=t0, y=mb.mul(x=mb.expand_dims(x=f, axes=[-1]), y=mb.sub(x=t1, y=t0)))
    acc = mb.reduce_sum(x=v, axes=[1], keep_dims=False)
    y = mb.add(x=acc, y=layer.bias.astype(f32))
    if extrapolate == "linear" and layer.slope_lo is not None:
        below = mb.expand_dims(x=mb.minimum(x=traw, y=f32(0.0)), axes=[-1])
        above = mb.expand_dims(x=mb.maximum(x=mb.sub(x=traw, y=f32(size - 1)), y=f32(0.0)), axes=[-1])
        ext = mb.add(x=mb.mul(x=below, y=layer.slope_lo.astype(f32)), y=mb.mul(x=above, y=layer.slope_hi.astype(f32)))
        y = mb.add(x=y, y=mb.reduce_sum(x=ext, axes=[1], keep_dims=False))
    if layer.n_mult:
        sums = mb.slice_by_index(x=y, begin=[0, 0], end=[0, layer.n_sum], begin_mask=[True, False], end_mask=[True, False])
        tail = mb.slice_by_index(x=y, begin=[0, layer.n_sum], end=[0, n_out], begin_mask=[True, False], end_mask=[True, False])
        tail = mb.reshape(x=tail, shape=[-1, layer.n_mult, layer.arity])
        y = mb.concat(values=[sums, mb.reduce_prod(x=tail, axes=[2], keep_dims=False)], axis=1)
    return y


def _activation(mb, x, name):
    if name == "identity":
        return mb.identity(x=x)
    if name == "silu":
        return mb.silu(x=x)
    if name == "gelu":
        return mb.gelu(x=x, mode="EXACT")
    if name == "gelu_tanh":
        return mb.gelu(x=x, mode="TANH_APPROXIMATION")
    return {"relu": mb.relu, "tanh": mb.tanh, "sigmoid": mb.sigmoid}[name](x=x)


def coreml_model(lut: LUTModel, max_batch: int = 65536):
    """``coremltools.models.MLModel`` of a lookup-table model (input "x", output "y")."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import get_new_symbol

    @mb.program(input_specs=[mb.TensorSpec(shape=(get_new_symbol(), lut.n_inputs))], opset_version=ct.target.macOS13)
    def prog(x):
        h = x
        for layer in lut.layers:
            if layer.kind == "kan":
                h = _lut_layer(mb, h, layer, lut.extrapolate)
            elif layer.kind == "linear":
                h = mb.matmul(x=h, y=layer.weight.T.astype(np.float32))
                if layer.bias is not None:
                    h = mb.add(x=h, y=layer.bias.astype(np.float32))
            else:
                h = _activation(mb, h, layer.name)
        return mb.identity(x=h, name="y")

    batch = ct.RangeDim(1, max_batch)
    return ct.convert(
        prog,
        inputs=[ct.TensorType(name="x", shape=(batch, lut.n_inputs))],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS13,
    )


def export_coreml(model, path, x=None, size: int = 256, bits: int | None = None, margin: float = 0.05, extrapolate: str = "linear"):
    """Write ``model`` (any backend, or a ``LUTModel``) as a Core ML lookup-table network."""
    lut = model if isinstance(model, LUTModel) else to_lut(model, x, size, margin, extrapolate)
    if bits is not None:
        lut = quantize(lut, bits)
    coreml_model(lut).save(str(path))
    return lut


def export_coreml_exact(model, path, example, max_batch: int = 65536):
    """The exact PyTorch network as Core ML (traced)."""
    import coremltools as ct
    import torch

    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    shape = ct.Shape(shape=(ct.RangeDim(1, max_batch), example.shape[1]))
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=shape)],
        outputs=[ct.TensorType(name="y")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
    )
    ml.save(str(path))
    return ml
