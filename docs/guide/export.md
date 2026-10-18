# Deployment

**Between frameworks**: `kanforge.save(model, "m.npz")` /
`kanforge.load("m.npz", backend=...)`, `kanforge.convert.convert(model, "mlx")`.

**Lookup tables**: `to_lut(model, x, size)` tabulates every edge on its
input range; a layer becomes gather + lerp + sum. Error falls as 1/size²
inside the range (1024 points: ~1e-5); outside, edges continue linearly.
`quantize(lut, 8)` stores int8 tables (one scale per output, ~1 % error).

**ONNX**: `export_onnx(model_or_lut, path, x, size, bits)` (any backend,
opset 17) and `export_onnx_exact(torch_model, path, example)`.

**Core ML**: `export_coreml(...)` (ML Program, int8 tables stay int8; macOS
13+/iOS 16+) and `export_coreml_exact(...)`.

**FPGA**: `fpga.export_fpga(model, dir, x, address_bits, value_bits)` writes
Verilog (ROM per input, adders, requantisation by multiply-and-shift, one
register per layer), ROM images, a self-checking testbench and
`design.json`; `fpga.simulate` is the bit-exact model the RTL is tested
against.
