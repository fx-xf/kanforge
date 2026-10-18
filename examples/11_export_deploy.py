"""Deploy a KAN: lookup tables, int8, ONNX (onnxruntime) and FPGA Verilog."""

import os
import tempfile

import numpy as np
import torch

import kanforge as kf
from kanforge.export import fpga, quantize, to_lut
from kanforge.torch import KAN

QUICK = os.environ.get("KANFORGE_QUICK") == "1"
torch.manual_seed(0)
x = torch.rand(2000, 3) * 2 - 1
model = KAN([3, 6, 2], kf.BSpline(6, 3))
xt = (torch.rand(500, 3) * 1.8 - 0.9).numpy()
ref = model(torch.tensor(xt)).detach().numpy()

lut = to_lut(model, x, size=256)
q = quantize(lut, bits=8)
print(f"LUT 256 points: rel error {np.abs(lut(xt) - ref).max() / np.abs(ref).max():.1e}, {lut.nbytes()} bytes")
print(f"int8:           rel error {np.abs(q(xt) - ref).max() / np.abs(ref).max():.1e}, {q.nbytes()} bytes")

out = tempfile.mkdtemp()
try:
    import onnxruntime as ort

    from kanforge.export.onnx_export import export_onnx

    export_onnx(q, os.path.join(out, "model.onnx"))
    y = ort.InferenceSession(os.path.join(out, "model.onnx")).run(None, {"x": xt.astype(np.float32)})[0]
    print(f"onnxruntime matches the int8 LUT: {np.abs(y - q(xt)).max():.1e}")
except ImportError:
    print("onnx / onnxruntime not installed: skipping ONNX")

design = fpga.export_fpga(model, os.path.join(out, "rtl"), x, address_bits=8, value_bits=8)
_, y = fpga.simulate(design, xt)
print(f"FPGA integer pipeline: rel error {np.abs(y - ref).max() / np.abs(ref).max():.1e}; RTL in {out}/rtl")
