"""FPGA generator: KAN layers as integer lookup tables in Verilog (KANELÉ-style).

Every edge becomes a ROM indexed by its quantised input, so a layer is
memory + adders, no multipliers in the datapath:

* inputs of a layer are ``A``-bit addresses: a_i = clamp(round((x_i - lo_i) / step_i))
* edge (i, o) stores q[i, a, o] = round((phi_oi(lo_i + a step_i) - c_io) / s_o), ``W`` bits
* acc_o = sum_i q[i, a_i, o]                                  (adder per output)
* real output y_o = s_o * acc_o + b_o   (b_o includes sum_i c_io)
* next layer's address: a'_o = clamp((acc_o * M_o + K_o) >>> SH, 0, 2^A - 1)
  with integers M_o, K_o chosen so that a'_o = round((y_o - lo'_o) / step'_o)

One register stage per layer; the last layer outputs the accumulators and
``design.json`` holds s_o and b_o to turn them into real values.
``simulate(design, x)`` is the bit-exact NumPy model of the generated logic;
the testbench compares the RTL against vectors from it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .lut import LUTModel, to_lut

SHIFT = 24  # fixed-point fraction bits of the requantisation multipliers
ACC_BITS = 64  # accumulator width in the RTL (synthesis trims unused bits)


@dataclass
class FPGALayer:
    q: np.ndarray  # (in, 2^A, out) signed integers
    scale: np.ndarray  # (out,)
    bias: np.ndarray  # (out,)
    mult: np.ndarray | None  # (out,) requantisation to the next layer, None for the last
    offset: np.ndarray | None


@dataclass
class FPGADesign:
    address_bits: int
    value_bits: int
    lo: np.ndarray  # first layer input range
    step: np.ndarray
    layers: list

    @property
    def n_inputs(self) -> int:
        return self.layers[0].q.shape[0]

    @property
    def n_outputs(self) -> int:
        return self.layers[-1].q.shape[2]


def design(model, x=None, address_bits: int = 8, value_bits: int = 8, margin: float = 0.05) -> FPGADesign:
    """Integer design of a KAN (any backend, or a float ``LUTModel`` with 2^A points)."""
    size = 2**address_bits
    lut = model if isinstance(model, LUTModel) else to_lut(model, x, size, margin)
    if any(layer.kind != "kan" or layer.n_mult or layer.scale is not None for layer in lut.layers):
        raise ValueError("FPGA export supports float lookup tables of plain KAN layers (no MultKAN, Linear, activations)")
    if any(layer.size != size for layer in lut.layers):
        raise ValueError(f"tables must have 2^address_bits = {size} points")
    qmax = 2 ** (value_bits - 1) - 1
    layers = []
    for k, layer in enumerate(lut.layers):
        t = layer.tables
        center = 0.5 * (t.max(1) + t.min(1))  # (in, out)
        dev = np.abs(t - center[:, None, :]).max((0, 1))
        scale = np.where(dev > 0, dev / qmax, 1.0)
        q = np.round((t - center[:, None, :]) / scale).astype(np.int64)
        bias = layer.bias + center.sum(0)
        mult = offset = None
        if k + 1 < len(lut.layers):
            nxt = lut.layers[k + 1]
            r = scale / nxt.step
            mult = np.round(r * 2**SHIFT).astype(np.int64)
            offset = np.round(((bias - nxt.lo) / nxt.step + 0.5) * 2**SHIFT).astype(np.int64)
        layers.append(FPGALayer(q, scale, bias, mult, offset))
    return FPGADesign(address_bits, value_bits, lut.layers[0].lo, lut.layers[0].step, layers)


def quantize_input(d: FPGADesign, x) -> np.ndarray:
    """A-bit addresses of real inputs (what the host feeds the FPGA)."""
    a = np.round((np.asarray(x, dtype=np.float64) - d.lo) / d.step)
    return np.clip(a, 0, 2**d.address_bits - 1).astype(np.int64)


def simulate(d: FPGADesign, x=None, addresses=None):
    """Bit-exact model of the RTL: (last-layer accumulators, real outputs)."""
    a = quantize_input(d, x) if addresses is None else np.asarray(addresses, dtype=np.int64)
    amax = 2**d.address_bits - 1
    for layer in d.layers:
        i = np.arange(a.shape[-1])
        acc = layer.q[i, a].sum(-2)  # (B, out)
        if layer.mult is None:
            return acc, acc * layer.scale + layer.bias
        a = np.clip((acc * layer.mult + layer.offset) >> SHIFT, 0, amax)
    raise AssertionError("unreachable")


# --- Verilog ----------------------------------------------------------------------------


def _hex(value: int, bits: int) -> str:
    return format(value & ((1 << bits) - 1), f"0{(bits + 3) // 4}x")


def _sv(v: int) -> str:
    """A signed 64-bit Verilog literal."""
    return f"-64'sd{-v}" if v < 0 else f"64'sd{v}"


def _layer_module(k: int, layer: FPGALayer, A: int, W: int) -> str:
    n_in, size, n_out = layer.q.shape
    last = layer.mult is None
    lines = [
        f"// KAN layer {k}: {n_in} inputs x {n_out} outputs, ROMs of {size} x {n_out * W} bits per input",
        f"module kan_layer{k} (",
        "    input  wire clk,",
        f"    input  wire [{n_in * A - 1}:0] addr_in,",
        f"    output reg  [{n_out * ACC_BITS - 1}:0] acc_out" if last else f"    output reg  [{n_out * A - 1}:0] addr_out",
        ");",
    ]
    for i in range(n_in):
        lines += [
            f"    reg [{n_out * W - 1}:0] rom{i} [0:{size - 1}];",
            f'    initial $readmemh("l{k}_in{i}.mem", rom{i});',
            f"    wire [{n_out * W - 1}:0] row{i} = rom{i}[addr_in[{i * A} +: {A}]];",
        ]
    for o in range(n_out):
        terms = " + ".join(f"$signed(row{i}[{o * W} +: {W}])" for i in range(n_in))
        lines.append(f"    wire signed [{ACC_BITS - 1}:0] acc{o} = {terms};")
        if not last:
            m, c = int(layer.mult[o]), int(layer.offset[o])
            lines.append(f"    wire signed [{ACC_BITS - 1}:0] r{o} = (acc{o} * {_sv(m)} + {_sv(c)}) >>> {SHIFT};")
    lines.append("    always @(posedge clk) begin")
    for o in range(n_out):
        if last:
            lines.append(f"        acc_out[{o * ACC_BITS} +: {ACC_BITS}] <= acc{o};")
        else:
            amax = 2**A - 1
            lines.append(
                f"        addr_out[{o * A} +: {A}] <= (r{o} < 0) ? {A}'d0 : (r{o} > {amax}) ? {A}'d{amax} : r{o}[{A - 1}:0];"
            )
    lines += ["    end", "endmodule", ""]
    return "\n".join(lines)


def _top_module(d: FPGADesign) -> str:
    A = d.address_bits
    n = len(d.layers)
    lines = [
        f"// kanforge KAN: {d.n_inputs} inputs -> {d.n_outputs} outputs, {n} layers, latency {n} cycles",
        "module kan_top (",
        "    input  wire clk,",
        f"    input  wire [{d.n_inputs * A - 1}:0] addr_in,",
        f"    output wire [{d.n_outputs * ACC_BITS - 1}:0] acc_out",
        ");",
    ]
    prev = "addr_in"
    for k, layer in enumerate(d.layers):
        if layer.mult is None:
            lines.append(f"    kan_layer{k} u{k} (.clk(clk), .addr_in({prev}), .acc_out(acc_out));")
        else:
            width = layer.q.shape[2] * A
            lines.append(f"    wire [{width - 1}:0] a{k + 1};")
            lines.append(f"    kan_layer{k} u{k} (.clk(clk), .addr_in({prev}), .addr_out(a{k + 1}));")
            prev = f"a{k + 1}"
    lines += ["endmodule", ""]
    return "\n".join(lines)


def _testbench(d: FPGADesign, n: int) -> str:
    A, n_layers = d.address_bits, len(d.layers)
    in_w, out_w = d.n_inputs * A, d.n_outputs * ACC_BITS
    return f"""`timescale 1ns/1ps
module tb_kan;
    reg clk = 0;
    always #5 clk = ~clk;
    reg  [{in_w - 1}:0] inputs [0:{n - 1}];
    reg  [{out_w - 1}:0] expected [0:{n - 1}];
    reg  [{in_w - 1}:0] addr_in = 0;
    wire [{out_w - 1}:0] acc_out;
    integer k, errors;
    kan_top dut (.clk(clk), .addr_in(addr_in), .acc_out(acc_out));
    initial begin
        $readmemh("tb_inputs.mem", inputs);
        $readmemh("tb_expected.mem", expected);
        errors = 0;
        for (k = 0; k < {n + n_layers}; k = k + 1) begin
            if (k < {n}) addr_in = inputs[k];
            @(posedge clk); #1;
            if (k >= {n_layers - 1} && k - {n_layers - 1} < {n}) begin
                if (acc_out !== expected[k - {n_layers - 1}]) begin
                    errors = errors + 1;
                    if (errors <= 5) $display("mismatch at vector %0d", k - {n_layers - 1});
                end
            end
        end
        if (errors == 0) $display("PASS {n} vectors");
        else $display("FAIL %0d of {n} vectors", errors);
        $finish;
    end
endmodule
"""


def write_verilog(d: FPGADesign, out_dir, test_inputs=None) -> Path:
    """Write RTL, ROM images, a self-checking testbench and ``design.json``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    A, W = d.address_bits, d.value_bits
    for k, layer in enumerate(d.layers):
        n_in, size, n_out = layer.q.shape
        for i in range(n_in):
            rows = []
            for a in range(size):
                word = 0
                for o in range(n_out):
                    word |= (int(layer.q[i, a, o]) & ((1 << W) - 1)) << (o * W)
                rows.append(_hex(word, n_out * W))
            (out / f"l{k}_in{i}.mem").write_text("\n".join(rows) + "\n")
        (out / f"kan_layer{k}.v").write_text(_layer_module(k, layer, A, W))
    (out / "kan_top.v").write_text(_top_module(d))
    if test_inputs is not None:
        addrs = quantize_input(d, test_inputs)
        acc, _ = simulate(d, addresses=addrs)
        ins = [_hex(sum(int(a) << (i * A) for i, a in enumerate(row)), d.n_inputs * A) for row in addrs]
        exp = [_hex(sum((int(v) & ((1 << ACC_BITS) - 1)) << (o * ACC_BITS) for o, v in enumerate(row)), d.n_outputs * ACC_BITS) for row in acc]
        (out / "tb_inputs.mem").write_text("\n".join(ins) + "\n")
        (out / "tb_expected.mem").write_text("\n".join(exp) + "\n")
        (out / "tb_kan.v").write_text(_testbench(d, len(addrs)))
    meta = {
        "address_bits": A,
        "value_bits": W,
        "shift": SHIFT,
        "accumulator_bits": ACC_BITS,
        "latency_cycles": len(d.layers),
        "input_lo": d.lo.tolist(),
        "input_step": d.step.tolist(),
        "output_scale": d.layers[-1].scale.tolist(),
        "output_bias": d.layers[-1].bias.tolist(),
        "layers": [list(layer.q.shape) for layer in d.layers],
        "rom_bits": int(sum(layer.q.size * W for layer in d.layers)),
    }
    (out / "design.json").write_text(json.dumps(meta, indent=2))
    return out


def export_fpga(model, out_dir, x=None, address_bits: int = 8, value_bits: int = 8, margin: float = 0.05, test_vectors: int = 64):
    """``design`` + ``write_verilog`` with a testbench on ``test_vectors`` samples of ``x``."""
    d = design(model, x, address_bits, value_bits, margin)
    tests = None
    if x is not None and test_vectors:
        from ..interpret._backend import to_numpy

        xs = to_numpy(x)
        tests = xs[np.linspace(0, len(xs) - 1, min(test_vectors, len(xs))).astype(int)]
    write_verilog(d, out_dir, tests)
    return d
