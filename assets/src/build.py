#!/usr/bin/env python3
"""Draws every figure of the README into assets/*.svg.

Every curve in these pictures is computed by kanforge itself: the basis
specimens call the NumPy reference, the edge functions of the hero network are
random B-spline combinations, and the benchmark chart reads the numbers of
``benchmarks/fair_results.md``.

Fonts (SIL Open Font License, see ``fonts/OFL-*.txt``) are subset to the glyphs
a figure uses and embedded as WOFF2, so the typography survives GitHub's image
proxy without external requests. The figures are self-contained dark plates:
one file reads well on both GitHub themes.

    pip install -e . fonttools brotli
    python assets/src/build.py
"""
from __future__ import annotations

import base64
import html
import io
import math
import re
from pathlib import Path

import numpy as np
from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

import kanforge as kf
from kanforge.bases import make_basis

HERE = Path(__file__).resolve().parent
FONTS = HERE / "fonts"
OUT = HERE.parent
ROOT = OUT.parent

# ─── palette ────────────────────────────────────────────────────────────────
INK = "#08080C"        # page of the plate
PLATE = "#0E0E14"      # plate
TILE = "#14141C"       # tile on the plate
LINE = "#24242F"       # hairline
LINE2 = "#33333F"
WHITE = "#F2EEE6"      # warm paper white
GREY = "#A19DA8"
DIM = "#66636F"
FAINT = "#3A3844"
# molten metal: white heat → amber → ember → rose → violet
HEAT = ["#FFF3D1", "#FFC65C", "#FF8A2E", "#F2464F", "#B23CF0", "#6A4CFF"]


def heat(t: float) -> str:
    """Colour of the molten ramp at t in [0, 1]."""
    t = min(max(t, 0.0), 1.0) * (len(HEAT) - 1)
    i = min(int(t), len(HEAT) - 2)
    f = t - i
    a = [int(HEAT[i][k:k + 2], 16) for k in (1, 3, 5)]
    b = [int(HEAT[i + 1][k:k + 2], 16) for k in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * f):02X}" for x, y in zip(a, b))


# ─── fonts ──────────────────────────────────────────────────────────────────
FACES = {  # class: (family, weight, style, file, variable weight)
    "s":  ("IS", 400, "normal", "InstrumentSerif-Regular.ttf", None),
    "si": ("IS", 400, "italic", "InstrumentSerif-Italic.ttf", None),
    "t":  ("IT", 400, "normal", "InterTight.ttf", 400),
    "tm": ("IT", 500, "normal", "InterTight.ttf", 500),
    "tb": ("IT", 650, "normal", "InterTight.ttf", 650),
    "m":  ("JB", 400, "normal", "JetBrainsMono.ttf", 400),
    "mb": ("JB", 600, "normal", "JetBrainsMono.ttf", 600),
}
FALLBACK = {"IS": "Georgia,serif", "IT": "Helvetica,Arial,sans-serif", "JB": "Menlo,monospace"}
_bytes: dict[str, bytes] = {}
_fonts: dict[str, TTFont] = {}


def font(cls: str) -> TTFont:
    if cls not in _fonts:
        *_, file, wght = FACES[cls]
        f = TTFont(FONTS / file)
        if wght is not None and "fvar" in f:
            f = instancer.instantiateVariableFont(f, {"wght": wght})
        buf = io.BytesIO()
        f.save(buf)
        _bytes[cls] = buf.getvalue()
        _fonts[cls] = TTFont(io.BytesIO(_bytes[cls]))
    return _fonts[cls]


def measure(s: str, cls: str, size: float, ls: float = 0) -> float:
    f = font(cls)
    cmap, hmtx, upm = f.getBestCmap(), f["hmtx"], f["head"].unitsPerEm
    s = html.unescape(re.sub(r"<[^>]+>", "", s))
    adv = sum(hmtx[cmap.get(ord(c), cmap[32])][0] for c in s)
    return adv * size / upm + ls * max(len(s) - 1, 0)


def woff2(cls: str, chars: str) -> str:
    font(cls)
    f = TTFont(io.BytesIO(_bytes[cls]))
    opts = subset.Options()
    opts.flavor = "woff2"
    opts.layout_features = ["kern", "liga", "calt", "tnum"]
    sub = subset.Subsetter(opts)
    sub.populate(text=chars + " ")
    sub.subset(f)
    buf = io.BytesIO()
    f.save(buf)
    return base64.b64encode(buf.getvalue()).decode()


# ─── svg primitives ─────────────────────────────────────────────────────────
def esc(s: str) -> str:
    return html.escape(s, quote=False)


def text(x, y, s, cls="t", size=14, fill=WHITE, anchor=None, ls=None, extra=""):
    a = f' text-anchor="{anchor}"' if anchor else ""
    l = f' letter-spacing="{ls}"' if ls else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" font-size="{size}" fill="{fill}"{a}{l}{extra}>'
            f"{s}</text>")


def rect(x, y, w, h, fill="none", rx=0, stroke=None, sw=1, extra=""):
    s = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{s}{extra}/>'


def line(x1, y1, x2, y2, stroke=LINE, sw=1, extra=""):
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
            f'stroke-width="{sw}"{extra}/>')


def poly(pts, stroke, sw=1.5, extra=""):
    d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    return (f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-linecap="round" stroke-linejoin="round"{extra}/>')


_uid = [0]


def uid(p="g"):
    _uid[0] += 1
    return f"{p}{_uid[0]}"


def heat_gradient(gid, x1="0", y1="0", x2="1", y2="0", stops=None, units=""):
    stops = stops or [(i / (len(HEAT) - 1), c) for i, c in enumerate(HEAT)]
    s = "".join(f'<stop offset="{o:.3f}" stop-color="{c}"/>' for o, c in stops)
    u = f' gradientUnits="{units}"' if units else ""
    return f'<linearGradient id="{gid}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"{u}>{s}</linearGradient>'


BASE_CSS = """
.flow{stroke-dasharray:2 14;animation:flow 2.4s linear infinite}
@keyframes flow{to{stroke-dashoffset:-32}}
.breathe{animation:breathe 4.8s ease-in-out infinite;transform-box:fill-box;transform-origin:center}
@keyframes breathe{0%,100%{opacity:.55}50%{opacity:1}}
.draw{stroke-dasharray:1200;stroke-dashoffset:0;animation:draw 7s ease-in-out infinite}
@keyframes draw{0%{stroke-dashoffset:1200}35%,100%{stroke-dashoffset:0}}
.blink{animation:blink 1.1s steps(1) infinite}
@keyframes blink{50%{opacity:0}}
@media (prefers-reduced-motion:reduce){.flow,.breathe,.draw,.blink{animation:none}}
"""


def svg(w, h, body, label, defs=""):
    classes = set()
    for m in re.finditer(r'class="([^"]+)"', body):
        classes.update(m.group(1).split())
    chars = "".join(sorted(set(html.unescape("".join(re.findall(r">([^<]+)<", body))))))
    rules = []
    for cls, (fam, wt, style, *_r) in FACES.items():
        if cls in classes:
            rules.append(
                f"@font-face{{font-family:{fam}{cls};src:url(data:font/woff2;base64,{woff2(cls, chars)}) "
                f"format('woff2')}}.{cls}{{font-family:{fam}{cls},{FALLBACK[fam]};font-weight:{wt};"
                f"font-style:{style}}}")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'role="img" aria-label="{esc(label)}"><title>{esc(label)}</title>'
            f'<defs>{defs}</defs><style>{"".join(rules)}{BASE_CSS}</style>{body}</svg>\n')


def plate(w, h, rx=28):
    """The dark plate every figure sits on, with a fine knot grid."""
    gid, pid = uid("pg"), uid("pat")
    defs = (f'<radialGradient id="{gid}" cx="0.12" cy="1.05" r="0.9">'
            f'<stop offset="0" stop-color="#FF8A2E" stop-opacity="0.10"/>'
            f'<stop offset="0.45" stop-color="#B23CF0" stop-opacity="0.035"/>'
            f'<stop offset="1" stop-color="{PLATE}" stop-opacity="0"/></radialGradient>'
            f'<pattern id="{pid}" width="32" height="32" patternUnits="userSpaceOnUse">'
            f'<circle cx="1" cy="1" r="0.9" fill="#FFFFFF" fill-opacity="0.055"/></pattern>')
    body = (rect(0.5, 0.5, w - 1, h - 1, PLATE, rx, LINE, 1)
            + rect(0.5, 0.5, w - 1, h - 1, f"url(#{pid})", rx)
            + rect(0.5, 0.5, w - 1, h - 1, f"url(#{gid})", rx))
    return defs, body


def kicker(x, y, s, fill=GREY, dot=True):
    out = ""
    if dot:
        out += f'<circle cx="{x + 3:.1f}" cy="{y - 4:.1f}" r="3" fill="{HEAT[2]}"/>'
        x += 14
    return out + text(x, y, s, "m", 13, fill, ls=1.6)


# ─── curves from kanforge ───────────────────────────────────────────────────
def spline_curve(rng, n=120, grid=6):
    """A random edge function phi(x) = sum_k c_k B_k(x), as kanforge evaluates it."""
    b = kf.BSpline(grid, 3)
    x = np.linspace(-1, 1, n)
    B = b(x[:, None])[:, 0, :]
    c = rng.normal(size=b.size)
    c = np.cumsum(c) * 0.6 + rng.normal(size=b.size) * 0.5   # smooth, characterful
    y = B @ c + 0.35 * x * rng.normal()
    return x, y


def fit(xs, ys, x0, y0, w, h, pad=0.1, lo=None, hi=None):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    lo = np.nanmin(ys) if lo is None else lo
    hi = np.nanmax(ys) if hi is None else hi
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    span = hi - lo
    lo, hi = lo - pad * span, hi + pad * span
    px = x0 + (xs - xs.min()) / (xs.max() - xs.min()) * w
    py = y0 + h - (np.clip(ys, lo, hi) - lo) / (hi - lo) * h
    return list(zip(px, py))


# ═══ 1. hero ════════════════════════════════════════════════════════════════
def hero():
    W, H = 1280, 640
    defs, body = plate(W, H)
    g_title, g_edge = uid("ht"), uid("he")
    defs += heat_gradient(g_title, "0", "0", "1", "0.3",
                          [(0, HEAT[0]), (0.35, HEAT[1]), (0.7, HEAT[2]), (1, HEAT[3])])
    defs += heat_gradient(g_edge, "0", "0", "1", "0", [(0, HEAT[1]), (0.5, HEAT[3]), (1, HEAT[5])])
    glow = uid("glow")
    defs += (f'<filter id="{glow}" x="-50%" y="-50%" width="200%" height="200%">'
             f'<feGaussianBlur stdDeviation="6"/></filter>')

    # left column: the words
    x0 = 72
    body += kicker(x0, 92, "KOLMOGOROV–ARNOLD NETWORKS")
    body += text(x0 + 268, 92, "v0.8 · APACHE-2.0", "m", 13, DIM, ls=1.6)
    body += text(x0 - 4, 228, "kan", "s", 150, WHITE, ls=-3)
    kw = measure("kan", "s", 150, -3)
    body += text(x0 - 4 + kw + 2, 228, "forge", "si", 150, f"url(#{g_title})", ls=-3)
    body += text(x0, 288, "Learnable functions on every edge —", "t", 25, WHITE, ls=-0.2)
    body += text(x0, 322, "one library, every backend.", "t", 25, GREY, ls=-0.2)

    # the theorem, typeset
    fy = 418
    body += line(x0, fy - 44, x0 + 470, fy - 44, LINE2)
    body += text(x0, fy - 58 + 32, "THE KOLMOGOROV–ARNOLD REPRESENTATION", "m", 12.5, DIM, ls=1.6)
    formula = ('f(x<tspan font-size="15" dx="2" dy="6">1</tspan><tspan dx="3" dy="-6">, …, x</tspan>'
               '<tspan font-size="15" dx="2" dy="6">n</tspan><tspan dx="3" dy="-6">) = Σ</tspan>'
               '<tspan font-size="15" dx="2" dy="7">q</tspan><tspan dx="3" dy="-7"> Φ</tspan>'
               '<tspan font-size="15" dx="2" dy="7">q</tspan><tspan dx="3" dy="-7">( Σ</tspan>'
               '<tspan font-size="15" dx="2" dy="7">p</tspan><tspan dx="3" dy="-7"> φ</tspan>'
               '<tspan font-size="15" dx="2" dy="7">q,p</tspan><tspan dx="3" dy="-7">(x</tspan>'
               '<tspan font-size="15" dx="2" dy="7">p</tspan><tspan dx="3" dy="-7">) )</tspan>')
    body += text(x0, fy + 26, formula, "si", 30, WHITE)

    # backends row
    by = 530
    chips = [("PyTorch", "CUDA · ROCm · MPS"), ("MLX", "Apple silicon"), ("JAX", "Flax NNX · TPU"),
             ("NumPy", "float64 reference")]
    cx = x0
    for name, sub in chips:
        w = max(measure(name, "tb", 15), measure(sub, "m", 12.5, 0.4)) + 34
        body += rect(cx, by - 26, w, 58, TILE, 14, LINE2)
        body += text(cx + 17, by - 2, name, "tb", 15, WHITE)
        body += text(cx + 17, by + 17, sub, "m", 12.5, DIM, ls=0.4)
        cx += w + 10

    # right column: a [3, 4, 2] KAN whose edges carry their functions
    rng = np.random.default_rng(7)
    L = [3, 4, 2]
    xs = [700, 940, 1180]
    top, bot = 120, 560
    nodes = []
    for li, n in enumerate(L):
        span = (bot - top) * (0.55 + 0.15 * (n - 2))
        c = (top + bot) / 2
        ys = [c] if n == 1 else [c - span / 2 + i * span / (n - 1) for i in range(n)]
        nodes.append([(xs[li], y) for y in ys])

    edges, minis = "", ""
    placed = []
    k = 0
    for li in range(len(L) - 1):
        nb = len(nodes[li + 1])
        na = len(nodes[li])
        for ai, a in enumerate(nodes[li]):
            for bi, b in enumerate(nodes[li + 1]):
                dx = b[0] - a[0]
                c1, c2 = (a[0] + dx * 0.5, a[1]), (b[0] - dx * 0.5, b[1])
                d = (f"M{a[0]:.1f} {a[1]:.1f} C{c1[0]:.1f} {c1[1]:.1f} {c2[0]:.1f} {c2[1]:.1f} "
                     f"{b[0]:.1f} {b[1]:.1f}")
                edges += f'<path d="{d}" fill="none" stroke="{LINE2}" stroke-width="1"/>'
                col = heat(0.1 + 0.8 * k / 19)
                delay = f"{(k * 0.37) % 2.4:.2f}s"
                edges += (f'<path d="{d}" fill="none" stroke="{col}" stroke-width="1.6" class="flow" '
                          f'stroke-linecap="round" style="animation-delay:-{delay}" opacity="0.9"/>')
                # mini plot of phi, staggered along the edge so no two collide
                t0 = 0.16 + 0.68 * (bi * na + ai) / max(na * nb - 1, 1)
                bw, bh = 44, 28

                def at(t):
                    u = 1 - t
                    return (u**3 * a[0] + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t**3 * b[0],
                            u**3 * a[1] + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t**3 * b[1])

                # nearest spot along the edge that does not touch an earlier plot
                for t in sorted(np.linspace(0.12, 0.88, 77), key=lambda v: abs(v - t0)):
                    mx, my = at(t)
                    if all(abs(mx - px) > bw + 5 or abs(my - py) > bh + 5 for px, py in placed):
                        break
                placed.append((mx, my))
                bx, byy = mx - bw / 2, my - bh / 2
                xx, yy = spline_curve(rng)
                pts = fit(xx, yy, bx + 5, byy + 4, bw - 10, bh - 8, pad=0.05)
                minis += rect(bx, byy, bw, bh, INK, 8, col, 0.8, ' stroke-opacity="0.55"')
                minis += poly(pts, col, 1.5)
                k += 1
    body += edges + minis
    for li, layer in enumerate(nodes):
        for (x, y) in layer:
            body += (f'<circle cx="{x}" cy="{y:.1f}" r="16" fill="{HEAT[2]}" opacity="0.35" '
                     f'filter="url(#{glow})" class="breathe"/>')
            body += f'<circle cx="{x}" cy="{y:.1f}" r="9" fill="{INK}" stroke="{WHITE}" stroke-width="1.4"/>'
            body += f'<circle cx="{x}" cy="{y:.1f}" r="3" fill="{WHITE}"/>'
    labels = [("x", "input"), ("Σ φ", "hidden"), ("y", "output")]
    for (x, _), (sym, cap) in zip([n[0] for n in nodes], labels):
        body += text(x, bot + 52, cap.upper(), "m", 12, DIM, "middle", 1.6)
    body += text(1180, 100, "every edge is a learned curve φ(x)", "si", 19, GREY, "end")
    return svg(W, H, body, "kanforge — Kolmogorov–Arnold Networks for every backend", defs)


# ═══ 2. basis specimen sheet ═══════════════════════════════════════════════
FAMILIES = [
    ("SPLINES", ["bspline", "nurbs", "truncated_power", "relu_kan"]),
    ("POLYNOMIALS", ["chebyshev", "chebyshev2", "legendre", "jacobi", "fractional_jacobi", "gegenbauer",
                     "hermite", "laguerre", "bernstein", "gram", "taylor"]),
    ("BUMPS & WAVELETS", ["rbf", "rswaf", "mexican_hat", "morlet", "dog"]),
    ("PERIODIC", ["fourier", "random_fourier"]),
    ("COMPOSITE", ["rational", "neural", "concat", "bsrbf"]),
]


def basis_values(name):
    if name == "concat":
        b = kf.Concat([kf.BSpline(3, 3), kf.Fourier(2)])
    elif name == "rational":
        b = make_basis("rational")
    else:
        b = make_basis(name)
    x = np.linspace(-1, 1, 110)
    if name == "rational":
        rng = np.random.default_rng(3)
        p = b.default_params(x[:, None])
        out = []
        for _ in range(5):
            q = {k: np.asarray(v) + rng.normal(scale=0.6, size=np.shape(v)) for k, v in p.items()}
            out.append(b(x[:, None], q)[:, 0, 0])
        return x, np.stack(out, 1)
    return x, b(x[:, None])[:, 0, :]


def specimen():
    cols, tw, th, gap = 5, 224, 170, 12
    x0, y0 = 56, 150
    W = x0 * 2 + cols * tw + (cols - 1) * gap
    names = [(fam, n) for fam, ns in FAMILIES for n in ns]
    rows = math.ceil((len(names) + 1) / cols)
    H = y0 + rows * th + (rows - 1) * gap + 64
    defs, body = plate(W, H)
    body += kicker(x0, 64, "TWENTY-SIX BASES · ONE CONTRACT")
    body += text(x0, 112, "A specimen of edge functions", "s", 44, WHITE, ls=-0.5)
    tx = x0 + measure("A specimen of edge functions", "s", 44, -0.5) + 14
    body += text(tx, 112, "— exact derivatives of any order, on every backend.", "si", 22, GREY)

    for i, (fam, name) in enumerate(names):
        r, c = divmod(i, cols)
        x, y = x0 + c * (tw + gap), y0 + r * (th + gap)
        body += rect(x, y, tw, th, TILE, 16, LINE)
        body += text(x + 14, y + 24, name, "mb", 13.5, WHITE)
        tag = fam.split()[0]
        if measure(name, "mb", 13.5) + measure(tag, "m", 10.5, 1.2) + 40 < tw:
            body += text(x + tw - 14, y + 24, tag, "m", 10.5, DIM, "end", 1.2)
        xs, Y = basis_values(name)
        Y = np.asarray(Y, float)
        lo, hi = np.nanpercentile(Y, 1), np.nanpercentile(Y, 99)
        px, py, pw, ph = x + 14, y + 38, tw - 28, th - 54
        body += line(px, py + ph, px + pw, py + ph, LINE2, 0.8)
        zero = py + ph - (0 - lo) / (hi - lo + 1e-12) * ph if lo < 0 < hi else None
        if zero:
            body += line(px, zero, px + pw, zero, FAINT, 0.6, ' stroke-dasharray="2 3"')
        n = Y.shape[1]
        cid = uid("clip")
        defs += f'<clipPath id="{cid}"><rect x="{px - 2}" y="{py - 2}" width="{pw + 4}" height="{ph + 4}"/></clipPath>'
        curves = ""
        for j in range(n):
            pts = fit(xs, Y[:, j], px, py, pw, ph, pad=0.06, lo=lo, hi=hi)
            curves += poly(pts, heat(0.05 + 0.9 * j / max(n - 1, 1)), 1.35, ' opacity="0.95"')
        body += f'<g clip-path="url(#{cid})">{curves}</g>'
    # closing tile: the code
    i = len(names)
    r, c = divmod(i, cols)
    x, y = x0 + c * (tw + gap), y0 + r * (th + gap)
    w = W - x0 - x
    body += rect(x, y, w, th, INK, 16, LINE2)
    body += text(x + 18, y + 34, "KANLinear(64, 32,", "m", 13, WHITE)
    body += text(x + 18, y + 56, '  basis="morlet")', "m", 13, HEAT[1])
    body += text(x + 18, y + 96, "or any basis above, its derivatives,", "t", 13, GREY)
    body += text(x + 18, y + 116, "per-input grids and fused kernels.", "t", 13, GREY)
    return svg(W, H, body, "The 26 basis families of kanforge, plotted by kanforge", defs)


# ═══ 3. equal-FLOPs benchmark ══════════════════════════════════════════════
def read_fair():
    """Best metric within 1e4 FLOPs for the MLP and the best KAN, per task."""
    txt = (ROOT / "benchmarks" / "fair_results.md").read_text()
    out = []
    for block in re.split(r"^### ", txt, flags=re.M)[1:]:
        title = block.splitlines()[0].strip()
        m = re.search(r"Best within a FLOP budget:\s*\n\s*\n\|([^\n]+)\|\n\|[-| ]+\|\n((?:\|[^\n]+\|\n?)+)", block)
        head = [h.strip() for h in m.group(1).split("|")]
        rows = {r.split("|")[1].strip(): [v.strip() for v in r.split("|")[2:-1]]
                for r in m.group(2).strip().splitlines()}
        row = rows["1e+04"]
        vals = {h: (float(v) if v not in ("—", "") else None) for h, v in zip(head[1:], row)}
        mlp = vals.pop("MLP")
        kan_name, kan = (min if "accuracy" not in block else max)(
            ((k, v) for k, v in vals.items() if v is not None), key=lambda kv: kv[1])
        out.append((title, "accuracy" in block, mlp, kan, kan_name))
    return out


def benchmark():
    W = 1280
    data = read_fair()
    row_h, y0 = 74, 196
    H = y0 + row_h * len(data) + 84
    defs, body = plate(W, H)
    x0 = 56
    body += kicker(x0, 64, "HONEST BENCHMARKS · EQUAL FLOPs · KANbeFair METHOD")
    body += text(x0, 112, "Where a KAN earns its keep", "s", 44, WHITE, ls=-0.5)
    body += text(x0, 146, "Error of the best model within 10⁴ FLOPs. Bars to the right: KAN is more accurate.",
                 "t", 17, GREY)
    axis = 700
    scale = 300 / math.log10(60)
    body += line(axis, y0 - 26, axis, y0 + row_h * len(data) - 18, LINE2, 1)
    for t, lab in [(1, "1×"), (10, "10×"), (50, "50×")]:
        xx = axis + math.log10(t) * scale
        if t > 1:
            body += line(xx, y0 - 26, xx, y0 + row_h * len(data) - 18, LINE, 1, ' stroke-dasharray="2 4"')
        body += text(xx, y0 - 34, lab, "m", 12.5, DIM, "middle")
    xx = axis - math.log10(2) * scale
    body += line(xx, y0 - 26, xx, y0 + row_h * len(data) - 18, LINE, 1, ' stroke-dasharray="2 4"')
    body += text(xx, y0 - 34, "2×", "m", 12.5, DIM, "middle")

    nice = {"exp(sin(pi x1) + x2^2)": "exp(sin πx₁ + x₂²)", "x1 * x2": "x₁ · x₂",
            "sin(x1) exp(-x2^2) + x3 / (1 + x4^2)": "sin x₁ · exp(−x₂²) + x₃ / (1 + x₄²)",
            "Friedman-1 (10 inputs)": "Friedman-1, 10 inputs", "digits (64 inputs, 10 classes)": "digits, 64 px → 10 classes"}
    gpos, gneg = uid("bp"), uid("bn")
    defs += heat_gradient(gpos, stops=[(0, HEAT[1]), (0.5, HEAT[2]), (1, HEAT[3])])
    defs += heat_gradient(gneg, "1", "0", "0", "0", stops=[(0, "#6E6A78"), (1, "#3A3844")])
    for i, (title, acc, mlp, kan, kname) in enumerate(data):
        y = y0 + i * row_h
        e_m, e_k = (1 - mlp, 1 - kan) if acc else (mlp, kan)
        ratio = e_m / e_k
        body += text(x0, y + 6, nice.get(title, title), "si", 23, WHITE)
        metric = "error rate" if acc else "test RMSE"
        body += text(x0, y + 28, f"{metric}: MLP {e_m:.3g} · {kname} {e_k:.3g}", "m", 12.5, DIM, ls=0.3)
        bw = math.log10(ratio) * scale
        bh = 22
        if bw >= 0:
            body += rect(axis, y - 9, bw, bh, f"url(#{gpos})", 6)
            body += text(axis + bw + 12, y + 7, f"{ratio:.1f}×", "tb", 17, WHITE)
            body += text(axis + bw + 12 + measure(f"{ratio:.1f}×", "tb", 17) + 8, y + 7,
                         "KAN", "m", 12.5, HEAT[1], ls=1)
        else:
            body += rect(axis + bw, y - 9, -bw, bh, f"url(#{gneg})", 6)
            body += text(axis + bw - 12, y + 7, f"{1 / ratio:.1f}× MLP", "tb", 15, GREY, "end")
    fy = y0 + row_h * len(data) + 18
    body += line(x0, fy, W - x0, fy, LINE)
    body += text(x0, fy + 36, "KANs win on smooth, compositional structure and lose a little on raw pixels — "
                 "reproduce with benchmarks/fair_comparison.py", "t", 15, GREY)
    return svg(W, H, body, "Equal-FLOPs benchmark of KAN against MLP", defs)


# ═══ 4. capabilities bento ═════════════════════════════════════════════════
def bento():
    W, H = 1280, 760
    defs, body = plate(W, H)
    x0, y0, gap = 40, 40, 16
    cw = (W - 2 * x0 - 2 * gap) / 3
    ch = (H - 2 * y0 - gap) / 2
    cards = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
    g = uid("bg")
    defs += heat_gradient(g, stops=[(0, HEAT[0]), (0.4, HEAT[1]), (1, HEAT[3])])

    def card(i, num, kick, title, sub):
        cx, cy = cards[i]
        x, y = x0 + cx * (cw + gap), y0 + cy * (ch + gap)
        s = rect(x, y, cw, ch, TILE, 22, LINE)
        s += text(x + 26, y + 40, num, "m", 12.5, HEAT[2], ls=1.4)
        s += text(x + 26 + measure(num, "m", 12.5, 1.4) + 10, y + 40, kick, "m", 12.5, DIM, ls=1.4)
        s += text(x + 26, y + ch - 56, title, "s", 33, WHITE, ls=-0.3)
        s += text(x + 26, y + ch - 28, sub, "t", 15.5, GREY)
        return s, x, y

    # 01 speed
    s, x, y = card(0, "01", "FUSED KERNELS", "Up to 9.9× faster", "Metal on Apple silicon, Triton on CUDA, Pallas on JAX")
    body += s
    bars = [("ops", 185.66, GREY), ("kernel", 18.71, None)]
    for j, (lab, ms, col) in enumerate(bars):
        by = y + 82 + j * 46
        bw = (cw - 120) * ms / 185.66
        body += text(x + 26, by + 14, lab, "m", 12.5, DIM)
        body += rect(x + 84, by, cw - 120, 20, INK, 6, LINE)
        body += rect(x + 84, by, bw, 20, col or f"url(#{g})", 6, extra=(' class="breathe"' if col is None else ""))
        if j:
            body += text(x + 84 + bw + 10, by + 15, f"{ms:.1f} ms", "mb", 13, WHITE)
        else:
            body += text(x + 84 + bw - 10, by + 15, f"{ms:.1f} ms", "mb", 13, INK, "end")
    body += text(x + 26, y + 196, "B-spline 32768×64×64 · PyTorch MPS · Apple M5", "m", 12, DIM)

    # 02 symbolic
    s, x, y = card(1, "02", "INTERPRETATION", "Curves become formulas", "prune · auto_symbolic · SymPy · kanpile")
    body += s
    rng = np.random.default_rng(1)
    xs = np.linspace(-1, 1, 160)
    pts = fit(xs, np.sin(np.pi * xs) + 0.04 * rng.normal(size=xs.size), x + 26, y + 70, cw - 52, 90)
    for (px, py) in pts[::5]:
        body += f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.8" fill="{GREY}" opacity="0.6"/>'
    pts = fit(xs, np.sin(np.pi * xs), x + 26, y + 70, cw - 52, 90)
    body += poly(pts, f"url(#{g})", 2.2, ' class="draw"')
    body += text(x + cw - 26, y + 188, "φ(x) → 1.00·sin(3.14·x)", "si", 21, HEAT[1], "end")

    # 03 surgery
    s, x, y = card(2, "03", "TRANSFORMERS", "kanify(model)", "Hugging Face FFNs → KANs, bit-exact outputs")
    body += s
    for j, lab in enumerate(["attention", "FFN → KANLinear", "attention", "FFN → KANLinear"]):
        by = y + 66 + j * 31
        hot = "KAN" in lab
        body += rect(x + 26, by, cw - 52, 24, INK if not hot else "#1C1410", 7, HEAT[2] if hot else LINE2)
        body += text(x + 40, by + 16, lab, "m", 12.5, HEAT[1] if hot else DIM)
        if hot:
            mini = fit(xs, np.tanh(3 * xs) + 0.3 * np.sin(5 * xs), x + cw - 120, by + 4, 70, 16)
            body += poly(mini, HEAT[1], 1.3)

    # 04 PINN
    s, x, y = card(3, "04", "SCIENCE", "PDEs, operators, ODEs", "derivatives of any order · DeepONet · KAN-ODE")
    body += s
    nx, ny = 46, 12
    for a in range(ny):
        yy = np.linspace(-1, 1, nx)
        u = np.sin(np.pi * yy) * np.exp(-a / 5) * np.cos(a / 2.2)
        pts = fit(yy, u, x + 26, y + 60 + a * 6.5, cw - 52, 36, lo=-1.2, hi=1.2)
        body += poly(pts, heat(a / ny), 1.1, f' opacity="{1 - a / (ny + 4):.2f}"')

    # 05 deploy
    s, x, y = card(4, "05", "DEPLOYMENT", "From GPU to FPGA", "ONNX · Core ML · int8 lookup tables · Verilog")
    body += s
    steps = ["torch", "LUT int8", "ONNX", "Verilog"]
    sx = x + 26
    sw_ = (cw - 52 - 3 * 18) / 4
    for j, st in enumerate(steps):
        bx = sx + j * (sw_ + 18)
        body += rect(bx, y + 92, sw_, 44, INK, 10, HEAT[2] if j == 3 else LINE2)
        body += text(bx + sw_ / 2, y + 119, st, "m", 12.5, WHITE if j == 3 else GREY, "middle")
        if j < 3:
            body += text(bx + sw_ + 9, y + 119, "→", "m", 12, DIM, "middle")
    for j in range(16):
        h = 6 + 18 * abs(math.sin(j * 0.7))
        body += rect(x + 26 + j * 9, y + 176 - h, 6, h, heat(j / 16), 1, extra=' opacity="0.8"')

    # 06 one format
    s, x, y = card(5, "06", "ONE FORMAT", "Same weights everywhere", "coef (out, in, n) · save · load · convert")
    body += s
    names = ["torch", "mlx", "jax", "numpy"]
    cx_, cy_ = x + cw / 2, y + 118
    pts = [(cx_ + 92 * math.cos(a), cy_ + 52 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 5)[:-1] + math.pi / 4]
    for a in range(4):
        for b in range(a + 1, 4):
            body += line(*pts[a], *pts[b], LINE2, 1)
            body += line(*pts[a], *pts[b], HEAT[2], 1.2, f' class="flow" style="animation-delay:-{a * .5 + b * .3:.1f}s"')
    for (px, py), nm in zip(pts, names):
        w = measure(nm, "m", 12.5) + 22
        body += rect(px - w / 2, py - 13, w, 26, INK, 13, LINE2)
        body += text(px, py + 4, nm, "m", 12.5, WHITE, "middle")
    body += text(cx_, cy_ + 5, "npz", "si", 17, HEAT[1], "middle")
    return svg(W, H, body, "What kanforge does: kernels, interpretation, transformers, science, deployment, one format", defs)


# ═══ 5. thin section rule ══════════════════════════════════════════════════
def rule(name, num, title):
    W, H = 1280, 96
    defs, body = "", ""
    g = uid("rg")
    nx = 2 + measure(num, "m", 13, 2) + 18
    lx = nx + measure(title, "s", 38, -0.3) + 24
    # a horizontal line has a zero-height box: the gradient must use user space
    defs += heat_gradient(g, f"{lx:.0f}", "0", f"{W:.0f}", "0", [(0, HEAT[1]), (0.5, HEAT[3]), (1, HEAT[5])],
                          units="userSpaceOnUse")
    body += text(2, 58, num, "m", 13, HEAT[2], ls=2)
    body += text(nx, 62, title, "s", 38, "#8E8A96", ls=-0.3)
    body += line(lx, 50, W - 2, 50, f"url(#{g})", 1)
    body += f'<circle cx="{W - 4}" cy="50" r="3" fill="{HEAT[5]}"/>'
    (OUT / f"rule-{name}.svg").write_text(svg(W, H, body, f"{num} {title}", defs))


# ═══ social preview (1280×640, rendered to PNG separately) ═════════════════
def social():
    W, H = 1280, 640
    defs, body = plate(W, H, rx=0)
    g = uid("sg")
    defs += heat_gradient(g, "0", "0", "1", "0.3", [(0, HEAT[0]), (0.35, HEAT[1]), (0.7, HEAT[2]), (1, HEAT[3])])
    rng = np.random.default_rng(11)
    for k in range(26):
        xx, yy = spline_curve(rng, 200, 8)
        pts = fit(xx, yy, -40, 330 + (k - 13) * 9, W + 80, 220)
        body += poly(pts, heat(k / 25), 1.2, f' opacity="{0.15 + 0.5 * (1 - abs(k - 13) / 13):.2f}"')
    tw = measure("kan", "s", 190, -4) + measure("forge", "si", 190, -4) + 4
    sx = (W - tw) / 2
    body += text(sx, 330, "kan", "s", 190, WHITE, ls=-4)
    body += text(sx + measure("kan", "s", 190, -4) + 4, 330, "forge", "si", 190, f"url(#{g})", ls=-4)
    body += text(W / 2, 420, "Kolmogorov–Arnold Networks for PyTorch, MLX and JAX", "t", 28, GREY, "middle")
    body += text(W / 2, 560, "26 BASES · FUSED KERNELS · SYMBOLIC FORMULAS · ONNX / CORE ML / FPGA", "m", 14, DIM,
                 "middle", 2)
    (OUT / "social-preview.svg").write_text(svg(W, H, body, "kanforge", defs))


def main():
    (OUT / "hero.svg").write_text(hero())
    (OUT / "bases.svg").write_text(specimen())
    (OUT / "benchmark.svg").write_text(benchmark())
    (OUT / "bento.svg").write_text(bento())
    for name, num, title in [("install", "01", "Install"), ("tour", "02", "A short tour"),
                             ("bases", "03", "Bases"), ("bench", "04", "When to use a KAN"),
                             ("more", "05", "Documentation"), ("contrib", "06", "Contributing")]:
        rule(name, num, title)
    social()
    docs = ROOT / "docs" / "assets"
    docs.mkdir(exist_ok=True)
    for name in ("hero.svg", "bases.svg", "benchmark.svg"):
        (docs / name).write_text((OUT / name).read_text())
    for p in sorted(OUT.glob("*.svg")):
        print(f"{p.name:24s} {p.stat().st_size / 1024:7.1f} KB")


if __name__ == "__main__":
    main()
