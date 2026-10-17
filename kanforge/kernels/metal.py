"""Metal Shading Language source for the three KAN primitives.

The same code runs under MLX (``mx.fast.metal_kernel``, which writes the
signature itself) and PyTorch MPS (``torch.mps.compile_shader``, full
kernels). Shapes are compiled in; the batch size comes in ``nb[0]``.

Buffers (all float32, row-major):
  x (B, IN), w (B, IN), c (OUT, IN, NF), g (B, OUT), grid (IN, GN)
Kernels:
  F: y[b, o]            = sum_i w[b,i] sum_k c[o,i,k] f_k(x[b,i])
  G: part[ch, o, i, k]  = sum_{b in chunk ch} w[b,i] g[b,o] f_k(x[b,i])
  P: p[b, i]            = sum_o g[b,o] sum_k c[o,i,k] f_k(x[b,i])
"""

from __future__ import annotations

from .plan import KernelPlan

OT = 4  # outputs per thread in F and G
GX = 16  # threads along outputs (F, G) / inputs per step (F)
TB = 8  # batch rows per threadgroup step



def _f(v: float) -> str:
    return f"{float(v)!r}f".replace("inf", "INFINITY")


def features_source(plan: KernelPlan, m: int, nf: int) -> str:
    """``kf_features(xv, i, grid, f)`` filling f[0 .. nf-1]; ``m`` is the native order."""
    head = "template <typename P>\ninline void kf_features(float xv, uint i, P grid, thread float* f) {\n"
    if plan.kind == "cheb":
        _, input_map = plan.consts
        z = "metal::precise::tanh(xv)" if input_map == "tanh" else "xv"
        body = f"    float z = {z};\n    f[0] = 1.0f;\n"
        if nf > 1:
            body += "    f[1] = z;\n"
        body += f"    for (uint j = 2; j < {nf}; ++j) f[j] = 2.0f * z * f[j - 1] - f[j - 2];\n"
    elif plan.kind == "fourier":
        freqs, period = plan.consts
        body = (
            f"    for (uint j = 0; j < {freqs}; ++j) {{\n"
            f"        float a = float(j + 1) * {_f(6.283185307179586 / period)} * xv;\n"
            f"        f[j] = metal::precise::cos(a);\n"
            f"        f[{freqs} + j] = metal::precise::sin(a);\n    }}\n"
        )
    elif plan.kind == "bspline":
        g, k = plan.consts
        nk = g + 2 * k + 1
        body = (
            f"    float g0 = grid[i * {g + 1}], gl = grid[i * {g + 1} + {g}];\n"
            f"    float h = (gl - g0) / {_f(g)};\n"
            f"    float t[{nk}];\n"
            f"    for (uint r = 0; r < {nk}; ++r)\n"
            f"        t[r] = r < {k} ? g0 - float({k} - int(r)) * h\n"
            f"             : (r <= {k + g} ? grid[i * {g + 1} + r - {k}] : gl + float(int(r) - {k + g}) * h);\n"
            f"    float b[{nk - 1}];\n"
            f"    for (uint j = 0; j < {nk - 1}; ++j) b[j] = (xv >= t[j] && xv < t[j + 1]) ? 1.0f : 0.0f;\n"
        )
        if m > k:
            body += f"    for (uint j = 0; j < {plan.size}; ++j) f[j] = 0.0f;\n}}\n"
            return head + body
        for q in range(1, k - m + 1):  # Cox–de Boor up to degree k - m
            body += (
                f"    for (uint j = 0; j < {nk - 1 - q}; ++j)\n"
                f"        b[j] = (xv - t[j]) / (t[j + {q}] - t[j]) * b[j]"
                f" + (t[j + {q + 1}] - xv) / (t[j + {q + 1}] - t[j + 1]) * b[j + 1];\n"
            )
        for q in range(k - m + 1, k + 1):  # derivative steps
            n = nk - 1 - q
            body += (
                f"    for (uint j = 0; j < {n}; ++j)\n"
                f"        b[j] = {_f(q)} * (b[j] / (t[j + {q}] - t[j]) - b[j + 1] / (t[j + {q + 1}] - t[j + 1]));\n"
            )
        body += f"    for (uint j = 0; j < {plan.size}; ++j) f[j] = b[j];\n"
    elif plan.kind == "bump":
        mode, count, u_scale, norm, _ = plan.consts
        poly = plan.bump_poly(m)
        horner = _f(poly[-1])
        for c in reversed(poly[:-1]):
            horner = f"({horner} * t + {_f(c)})"
        t = "metal::precise::tanh(u)" if mode == "tanh" else "u"
        env = " * metal::precise::exp(-0.5f * u * u)" if mode == "gauss" else ""
        body = (
            f"    float c0 = grid[i * {count}], cl = grid[i * {count} + {count - 1}];\n"
            f"    float s = {_f(u_scale * (count - 1))} / (cl - c0);\n"
            f"    float sm = metal::precise::pow(s, {_f(m)});\n"
            f"    for (uint j = 0; j < {count}; ++j) {{\n"
            f"        float u = (xv - grid[i * {count} + j]) * s;\n"
            f"        float t = {t};\n"
            f"        f[j] = {_f(norm)} * {horner}{env} * sm;\n    }}\n"
        )
    else:
        raise ValueError(plan.kind)
    return head + body + "}\n"


def kernel_bodies(n_in: int, n_out: int, nf: int, chunk: int) -> dict[str, str]:
    """Bodies of F, G, P. Expect ``gid``, ``lid``, ``tgid`` (uint2) and the buffers.

    F: a threadgroup of GX x TB threads covers TB batch rows and GX*OT outputs;
       inputs are processed GX at a time, each thread computing the features
       of one (row, input) into threadgroup memory, shared by all outputs.
    G: a threadgroup covers one input, GX*OT outputs and ``chunk`` batch rows;
       each step all GX*TB threads compute the features of GX*TB rows, then
       every thread accumulates its outputs over GX of them; partial sums are
       reduced over TB in threadgroup memory.
    P: one thread per (row, input).
    """
    og = (n_out + GX * OT - 1) // (GX * OT)
    f_body = f"""
    uint tx = lid.x, ty = lid.y;
    uint b = tgid.y * {TB} + ty;
    uint o0 = (tgid.x * {GX} + tx) * {OT};
    bool row = b < nb[0];
    threadgroup float feat[{TB}][{GX}][{nf}];
    float acc[{OT}] = {{0.0f}};
    float ff[{nf}];
    for (uint i0 = 0; i0 < {n_in}; i0 += {GX}) {{
        uint i = i0 + tx;
        if (row && i < {n_in}) {{
            kf_features(x[b * {n_in} + i], i, grid, ff);
            float wv = w[b * {n_in} + i];
            for (uint k = 0; k < {nf}; ++k) feat[ty][tx][k] = ff[k] * wv;
        }} else {{
            for (uint k = 0; k < {nf}; ++k) feat[ty][tx][k] = 0.0f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint ni = min(uint({GX}), uint({n_in}) - i0);
        for (uint t = 0; t < {OT}; ++t) {{
            uint o = o0 + t;
            if (o >= {n_out}) break;
            float s = 0.0f;
            for (uint ii = 0; ii < ni; ++ii) {{
                uint base = (o * {n_in} + i0 + ii) * {nf};
                for (uint k = 0; k < {nf}; ++k) s += c[base + k] * feat[ty][ii][k];
            }}
            acc[t] += s;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (row) for (uint t = 0; t < {OT}; ++t) {{
        uint o = o0 + t;
        if (o < {n_out}) y[b * {n_out} + o] = acc[t];
    }}
"""
    rows = GX * TB  # rows whose features one step computes, one per thread
    g_body = f"""
    uint tx = lid.x, ty = lid.y, lt = ty * {GX} + tx;
    uint i = tgid.x / {og}, o0 = ((tgid.x % {og}) * {GX} + tx) * {OT}, ch = tgid.y;
    threadgroup float feat[{rows}][{nf}];
    threadgroup float red[{TB}][{GX}][{nf}];
    float acc[{OT}][{nf}];
    for (uint t = 0; t < {OT}; ++t) for (uint k = 0; k < {nf}; ++k) acc[t][k] = 0.0f;
    float ff[{nf}];
    uint b_end = min(nb[0], (ch + 1) * {chunk});
    for (uint b0 = ch * {chunk}; b0 < b_end; b0 += {rows}) {{
        uint bf = b0 + lt;
        if (bf < b_end) {{
            kf_features(x[bf * {n_in} + i], i, grid, ff);
            float wv = w[bf * {n_in} + i];
            for (uint k = 0; k < {nf}; ++k) feat[lt][k] = ff[k] * wv;
        }} else {{
            for (uint k = 0; k < {nf}; ++k) feat[lt][k] = 0.0f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // thread (tx, ty) accumulates rows ty*GX .. ty*GX+GX-1 of this step
        for (uint r = ty * {GX}; r < ty * {GX} + {GX}; ++r) {{
            uint b = b0 + r;
            if (b >= b_end) break;
            for (uint t = 0; t < {OT}; ++t) {{
                uint o = o0 + t;
                if (o >= {n_out}) break;
                float gv = g[b * {n_out} + o];
                for (uint k = 0; k < {nf}; ++k) acc[t][k] += gv * feat[r][k];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    for (uint t = 0; t < {OT}; ++t) {{
        for (uint k = 0; k < {nf}; ++k) red[ty][tx][k] = acc[t][k];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint o = o0 + t;
        if (ty == 0 && o < {n_out}) {{
            uint base = ((ch * {n_out} + o) * {n_in} + i) * {nf};
            for (uint k = 0; k < {nf}; ++k) {{
                float s = 0.0f;
                for (uint r = 0; r < {TB}; ++r) s += red[r][tx][k];
                part[base + k] = s;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
"""
    p_body = f"""
    uint i = gid.x, b = gid.y;
    if (b >= nb[0] || i >= {n_in}) return;
    float f[{nf}];
    kf_features(x[b * {n_in} + i], i, grid, f);
    float h[{nf}] = {{0.0f}};
    for (uint o = 0; o < {n_out}; ++o) {{
        float gv = g[b * {n_out} + o];
        uint base = (o * {n_in} + i) * {nf};
        for (uint k = 0; k < {nf}; ++k) h[k] += gv * c[base + k];
    }}
    float s = 0.0f;
    for (uint k = 0; k < {nf}; ++k) s += h[k] * f[k];
    p[b * {n_in} + i] = s;
"""
    return {"F": f_body, "G": g_body, "P": p_body}


# Inputs and outputs of each kernel, in call order.
SIGNATURES = {
    "F": (("x", "w", "c", "grid", "nb"), ("y",)),
    "G": (("x", "w", "g", "grid", "nb"), ("part",)),
    "P": (("x", "g", "c", "grid", "nb"), ("p",)),
}


def chunk_for(batch: int) -> int:
    """Batch rows per G threadgroup: small batches need more groups to fill the GPU."""
    return 128 if batch <= 8192 else 512


def launch_grid(op: str, batch: int, n_in: int, n_out: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """((threads x, threads y), (group x, group y)) — dispatchThreads semantics."""
    og = (n_out + GX * OT - 1) // (GX * OT)
    if op == "F":
        return (og * GX, (batch + TB - 1) // TB * TB), (GX, TB)
    if op == "G":
        return (n_in * og * GX, n_chunks(batch) * TB), (GX, TB)
    gx = max(1, min(n_in, 32))
    return (n_in, batch), (gx, max(1, min(batch, 256 // gx)))


def n_chunks(batch: int) -> int:
    c = chunk_for(batch)
    return (batch + c - 1) // c


def torch_source(plan: KernelPlan, m: int, n_in: int, n_out: int, nf: int, chunk: int) -> str:
    """Complete MSL library with kernels kf_F, kf_G, kf_P for torch.mps.compile_shader."""
    bodies = kernel_bodies(n_in, n_out, nf, chunk)
    src = "#include <metal_stdlib>\nusing namespace metal;\n" + features_source(plan, m, nf)
    for op, (ins, outs) in SIGNATURES.items():
        args, n = [], 0
        for name in outs:
            args.append(f"device float* {name} [[buffer({n})]]")
            n += 1
        for name in ins:
            kind = "constant uint&" if name == "nb" else "device const float*"
            args.append(f"{kind} {name}_ [[buffer({n})]]" if name == "nb" else f"{kind} {name} [[buffer({n})]]")
            n += 1
        args.append("uint2 gid [[thread_position_in_grid]]")
        args.append("uint2 lid [[thread_position_in_threadgroup]]")
        args.append("uint2 tgid [[threadgroup_position_in_grid]]")
        body = bodies[op].replace("nb[0]", "nb_")
        src += f"kernel void kf_{op}({', '.join(args)}) {{{body}}}\n"
    return src
