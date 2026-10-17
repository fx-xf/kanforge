"""Backend-independent pieces of the layer definitions."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from . import grid as gridlib
from .bases import Basis, make_basis
from ._xp import namespace
from .reference import silu

COEF_INITS = ("normal", "zero", "variance", "lecun", "glorot")


def resolve_groups(in_features: int, groups: int | None) -> int:
    """Number of parameter groups: ``None`` means one group per input."""
    if groups is None:
        return in_features
    if groups < 1 or in_features % groups:
        raise ValueError(f"groups={groups} must divide in_features={in_features}")
    return groups


def basis_param_values(
    basis: Basis, groups: int, in_features: int, grid_ranges=None
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Initial basis state: trainable ``(groups, *shape)`` and buffers ``(in, *shape)``.

    ``grid_ranges`` ``(in, 2)`` places every input's grid uniformly on its own range.
    """
    trainable, buffers = {}, {}
    for name, spec in basis.param_specs().items():
        if spec.trainable:
            trainable[name] = np.broadcast_to(spec.value, (groups, *spec.shape)).copy()
            continue
        value = np.broadcast_to(spec.value, (in_features, *spec.shape)).astype(np.float64)
        if grid_ranges is not None and name in basis.grid_names():
            r = np.broadcast_to(np.asarray(grid_ranges, dtype=np.float64), (in_features, 2))
            value = r[:, :1] + (r[:, 1:] - r[:, :1]) * np.linspace(0, 1, spec.shape[-1])
        buffers[name] = value.copy()
    return trainable, buffers


def _gauss_hermite(points: int = 96):
    x, w = np.polynomial.hermite_e.hermegauss(points)
    return x, w / math.sqrt(2 * math.pi)


def basis_second_moment(basis: Basis) -> float:
    """E[sum_b B_b(x)^2] for x ~ N(0, 1) with default parameters."""
    x, w = _gauss_hermite()
    return float(w @ (basis(x) ** 2).sum(-1))


SILU_SECOND_MOMENT = float(_gauss_hermite()[1] @ silu(_gauss_hermite()[0]) ** 2)


def init_stds(
    basis: Basis, in_features: int, out_features: int, coef_init: str, coef_init_std: float, has_base: bool
) -> tuple[float, float | None]:
    """(coef std, base weight std or None for torch's default kaiming-uniform).

    * ``normal``  — coef ~ N(0, coef_init_std^2 / in) (efficient-kan style)
    * ``zero``    — coef = 0: the layer starts as its SiLU branch (pykan style)
    * ``variance``— spline branch alone gives unit output variance (KAT)
    * ``lecun`` / ``glorot`` — unit output variance for N(0, 1) inputs, split
      evenly between spline and base branch, with fan-in or fan-average.
    """
    if coef_init not in COEF_INITS:
        raise ValueError(f"coef_init must be one of {COEF_INITS}")
    if coef_init == "normal":
        return coef_init_std / math.sqrt(in_features), None
    if coef_init == "zero":
        return 0.0, None
    m2 = max(basis_second_moment(basis), 1e-12)
    if coef_init == "variance":
        return 1.0 / math.sqrt(in_features * m2), None
    fan = in_features if coef_init == "lecun" else (in_features + out_features) / 2
    share = 0.5 if has_base else 1.0
    return math.sqrt(share / (fan * m2)), math.sqrt(share / (fan * SILU_SECOND_MOMENT)) if has_base else None


def mult_widths(widths: Sequence) -> list[tuple[int, int]]:
    """MultKAN widths: ``3`` -> (3 sum nodes, 0 product nodes), ``(3, 2)`` as is."""
    out = []
    for w in widths:
        n_sum, n_mult = (w, 0) if isinstance(w, int) else (int(w[0]), int(w[1]))
        if n_sum < 0 or n_mult < 0 or n_sum + n_mult == 0:
            raise ValueError(f"bad MultKAN width {w!r}")
        out.append((n_sum, n_mult))
    if len(out) < 2:
        raise ValueError("widths needs at least input and output size")
    return out


def _resize_key(basis: Basis) -> str:
    cfg = basis.config()
    for key in ("grid_size", "count", "num_knots"):
        if key in cfg:
            return key
    raise TypeError(f"{type(basis).__name__} has no grid size to refine")


class GridToolsMixin:
    """Grid update, grid refinement and basis conversion for KAN layers.

    Backends implement ``_export_state()`` (NumPy float64 copies of
    ``coef``, ``edge_scale``, trainable basis parameters, buffers) and
    ``_load_state(basis, coef, edge_scale, trainable, buffers)``. Shapes of
    the parameters can change: re-create the optimizer afterwards.
    """

    def _expanded(self, trainable: dict, buffers: dict) -> dict:
        rep = self.in_features // self.groups
        out = {k: np.repeat(v, rep, axis=0) for k, v in trainable.items()}
        return {**out, **buffers}

    def _refit_to(self, new_basis: Basis, new_trainable: dict, new_buffers: dict, x: np.ndarray, ridge: float):
        st = self._export_state()
        coef_eff = st["coef"] if st["edge_scale"] is None else st["coef"] * st["edge_scale"][..., None]
        coef = gridlib.refit(
            self.basis,
            self._expanded(st["trainable"], st["buffers"]),
            coef_eff,
            new_basis,
            self._expanded(new_trainable, new_buffers),
            x,
            ridge,
        )
        scale = st["edge_scale"]
        if scale is not None:
            safe = np.where(np.abs(scale) > 1e-12, scale, 1.0)
            coef = coef / safe[..., None]
            scale = np.where(np.abs(scale) > 1e-12, scale, 1.0)
        self._load_state(new_basis, coef, scale, new_trainable, new_buffers)

    def _samples(self, x, basis: Basis, buffers: dict, n: int):
        if x is not None:
            return np.asarray(self._to_numpy(x), dtype=np.float64).reshape(-1, self.in_features)
        ranges = gridlib.grid_ranges(basis, buffers)
        if ranges is None:
            raise ValueError(f"{type(basis).__name__} has no grid: pass samples x")
        return gridlib.sample_inputs(ranges, n)

    def update_grid(self, x, *, margin: float = 0.0, grid_eps: float = 0.02, ridge: float = 0.0):
        """Adapt every grid to the inputs ``x`` (pykan's update_grid_from_samples).

        The edge functions are refitted on ``x``; a spline on the new grid can
        only approximate the old one, so like in pykan this is not exact.
        """
        if x is None:
            raise ValueError("update_grid needs input samples x")
        names = self.basis.grid_names()
        if not names:
            raise TypeError(f"{type(self.basis).__name__} has no grid")
        st = self._export_state()
        xs = self._samples(x, self.basis, st["buffers"], 0)
        buffers = dict(st["buffers"])
        for name in names:
            buffers[name] = gridlib.adaptive_grid(xs, buffers[name].shape[-1], margin, grid_eps)
        self._refit_to(self.basis, st["trainable"], buffers, xs, ridge)

    def refine_grid(self, grid_size: int, x=None, *, ridge: float = 0.0):
        """Same basis on a finer (or coarser) grid; grids keep their per-input shape."""
        key = _resize_key(self.basis)
        new_basis = make_basis({**self.basis.config(), key: grid_size})
        st = self._export_state()
        trainable, buffers = basis_param_values(new_basis, self.groups, self.in_features)
        for name in new_basis.grid_names():
            if name in st["buffers"]:
                buffers[name] = gridlib.resample_grid(st["buffers"][name], buffers[name].shape[-1])
        xs = self._samples(x, self.basis, st["buffers"], 8 * new_basis.size + 64)
        self._refit_to(new_basis, trainable, buffers, xs, ridge)

    def change_basis(self, basis, x=None, *, ridge: float = 0.0):
        """Convert to another basis (e.g. B-spline -> Chebyshev) by least squares.

        Without ``x``, the fit uses points spread over the current grid ranges
        (or the new basis' ranges); a new grid inherits each input's range.
        """
        new_basis = make_basis(basis)
        st = self._export_state()
        old_ranges = gridlib.grid_ranges(self.basis, st["buffers"])
        trainable, buffers = basis_param_values(new_basis, self.groups, self.in_features, old_ranges)
        src = self.basis if old_ranges is not None else new_basis
        src_buffers = st["buffers"] if old_ranges is not None else buffers
        xs = self._samples(x, src, src_buffers, 8 * max(new_basis.size, self.basis.size) + 64)
        self._refit_to(new_basis, trainable, buffers, xs, ridge)


class InterpretMixin:
    """Edge masks, symbolic edges and sub-layers, shared by the backends.

    Buffers (always present, so state dicts stay compatible):
      ``edge_mask``  ``(out, in)`` — 0 removes the spline and base part of an edge
      ``sym_code``   ``(out, in)`` — library code of a symbolic edge, -1 for none
      ``sym_params`` ``(out, in, 4)`` — (a, b, c, d) of c * f(a * x + b) + d

    Backends implement ``_get(name)`` (backend array), ``_set(name, np_array)``,
    ``_to_numpy(array)``, ``_export_full()`` / ``_import_full(dict)`` and keep
    ``_init_kwargs`` (constructor arguments besides the sizes).
    """

    _sym_codes: tuple = ()

    def _refresh_symbolic(self) -> None:
        codes = np.unique(self._to_numpy(self._get("sym_code")).astype(int))
        # object.__setattr__: MLX modules would store a tuple as a child entry
        object.__setattr__(self, "_sym_codes", tuple(int(c) for c in codes if c >= 0))

    def symbolic_edges(self) -> dict[tuple[int, int], tuple[str, tuple[float, float, float, float]]]:
        """{(o, i): (function name, (a, b, c, d))} of the fixed edges."""
        from .interpret.library import NAMES

        code = self._to_numpy(self._get("sym_code")).astype(int)
        params = self._to_numpy(self._get("sym_params"))
        return {
            (int(o), int(i)): (NAMES[code[o, i]], tuple(float(v) for v in params[o, i]))
            for o, i in zip(*np.nonzero(code >= 0))
        }

    def fix_symbolic(self, o: int, i: int, name: str, params=(1.0, 0.0, 1.0, 0.0)) -> None:
        """Replace edge (o, i) by c * f(a * x + b) + d; its spline and base parts are masked."""
        from .interpret.library import CODES, get

        get(name)
        mask = self._to_numpy(self._get("edge_mask"))
        code = self._to_numpy(self._get("sym_code")).astype(np.int32)
        sp = self._to_numpy(self._get("sym_params"))
        mask[o, i], code[o, i], sp[o, i] = 0.0, CODES[name], np.asarray(params, dtype=np.float64)
        self._set("edge_mask", mask)
        self._set("sym_code", code)
        self._set("sym_params", sp)
        self._refresh_symbolic()

    def unfix_symbolic(self, o: int, i: int) -> None:
        """Back to the learned edge (spline and base parts unmasked)."""
        mask = self._to_numpy(self._get("edge_mask"))
        code = self._to_numpy(self._get("sym_code")).astype(np.int32)
        mask[o, i], code[o, i] = 1.0, -1
        self._set("edge_mask", mask)
        self._set("sym_code", code)
        self._refresh_symbolic()

    def set_edge_mask(self, mask) -> None:
        """0 removes an edge (pruning); symbolic edges stay as they are."""
        self._set("edge_mask", np.asarray(mask, dtype=np.float64))

    def active_edges(self) -> np.ndarray:
        """Boolean ``(out, in)``: learned (unmasked) or symbolic edges."""
        return (self._to_numpy(self._get("edge_mask")) != 0) | (self._to_numpy(self._get("sym_code")) >= 0)

    def _symbolic_part(self, x):
        """Symbolic edges, ``(..., out, in)``, or None when there are none."""
        if not self._sym_codes:
            return None
        from .interpret.library import SAFE_INPUT, get

        xp = namespace(x)
        code, p = self._get("sym_code"), self._get("sym_params")
        a, b, c, d = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        u = a * x[..., None, :] + b
        zero = 0 * u
        out = zero
        for k in self._sym_codes:
            sel = code == k
            v = c * get(k).fn(xp.where(sel, u, zero + SAFE_INPUT)) + d
            out = out + xp.where(sel, v, zero)
        return out

    def subset(self, out_idx=None, in_idx=None):
        """A new layer with only the given outputs and inputs (pruning)."""
        out_idx = np.arange(self.out_features) if out_idx is None else np.asarray(out_idx, dtype=int)
        in_idx = np.arange(self.in_features) if in_idx is None else np.asarray(in_idx, dtype=int)
        if self.groups not in (1, self.in_features) and len(in_idx) != self.in_features:
            raise ValueError("inputs of a layer with grouped basis parameters cannot be removed")
        st = self._export_full()
        new_st = {}
        for name, value in st.items():
            if value is None:
                new_st[name] = None
            elif name in ("coef", "base_weight", "edge_scale", "edge_mask", "sym_code", "sym_params"):
                new_st[name] = value[out_idx][:, in_idx]
            elif name == "bias":
                new_st[name] = value[out_idx]
            elif name.startswith("basis_"):
                per_input = value.shape[0] == self.in_features and self.groups != 1
                new_st[name] = value[in_idx] if per_input or name[6:] in self._buffer_names else value
            else:
                new_st[name] = value
        kwargs = dict(self._init_kwargs)
        if kwargs.get("groups") not in (None, 1):
            kwargs["groups"] = None if len(in_idx) == self.in_features else kwargs["groups"]
        layer = type(self)(len(in_idx), len(out_idx), **kwargs)
        layer._import_full(new_st)
        return layer
