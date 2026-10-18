"""KAN feed-forward blocks for transformers and model surgery for Hugging Face.

* ``KANFFN`` — the MLP of a transformer block made of KAN layers: either two
  ``KANLinear`` with any basis, or KAT's two group-rational layers (the first
  rational starts as the identity, the second as the activation).
* ``KATBlock`` — pre-norm transformer block: attention + ``KANFFN``.
* ``kanify(model)`` — replace the FFNs of a pretrained model in place:
  ``mode="exact"`` keeps the model's outputs bit for bit at the start (the
  activation becomes the base branch of a ``KANLinear``, splines start at
  zero), ``mode="grkan"`` transfers the weights into GR-KAN layers like KAT
  does from ViT (rationals approximate the activation to ~1e-2).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .layers import GRKAN, KANLinear

ACT_TARGETS = {"gelu": "gelu", "silu": "silu", "swish": "silu", "tanh": "tanh"}


class KANFFN(nn.Module):
    """dim -> hidden -> dim with KAN layers (``kind="kan"``) or GR-KAN (``"grkan"``)."""

    def __init__(
        self,
        dim: int,
        hidden: int | None = None,
        kind: str = "grkan",
        basis="bspline",
        groups: int = 8,
        activation: str = "gelu",
        dropout: float = 0.0,
        **kan_kwargs,
    ):
        super().__init__()
        hidden = hidden or 4 * dim
        if kind == "grkan":
            self.fc1 = GRKAN(dim, hidden, groups=groups, init="identity")
            self.fc2 = GRKAN(hidden, dim, groups=groups, init=activation)
        elif kind == "kan":
            self.fc1 = KANLinear(dim, hidden, basis, **kan_kwargs)
            self.fc2 = KANLinear(hidden, dim, basis, **kan_kwargs)
        else:
            raise ValueError("kind must be 'grkan' or 'kan'")
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.fc1(x))))


class KATBlock(nn.Module):
    """x + attn(norm(x)); x + KANFFN(norm(x)) — the block of the Kolmogorov–Arnold Transformer."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, kind: str = "grkan", dropout: float = 0.0, **ffn_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = KANFFN(dim, int(dim * mlp_ratio), kind, dropout=dropout, **ffn_kwargs)

    def forward(self, x: torch.Tensor, attn_mask=None, key_padding_mask=None) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return x + self.ffn(self.norm2(x))


# --- Hugging Face (or any) model surgery -----------------------------------------


def _linear_from(module) -> nn.Linear:
    """nn.Linear equal to ``module`` (nn.Linear or GPT-2's Conv1D)."""
    if isinstance(module, nn.Linear):
        return module
    if type(module).__name__ == "Conv1D":  # transformers.pytorch_utils.Conv1D: y = x @ W + b
        lin = nn.Linear(module.weight.shape[0], module.weight.shape[1], device=module.weight.device, dtype=module.weight.dtype)
        with torch.no_grad():
            lin.weight.copy_(module.weight.T)
            lin.bias.copy_(module.bias)
        return lin
    raise TypeError(f"not a linear layer: {type(module).__name__}")


def _act_name(act) -> str:
    name = type(act).__name__.lower()
    for key, target in ACT_TARGETS.items():
        if key in name:
            return target
    if act is F.gelu:
        return "gelu"
    if act is F.silu:
        return "silu"
    raise ValueError(f"no rational initialisation for activation {type(act).__name__}")


def _ffn_sites(model: nn.Module):
    """(name, owner_in, attr_in, owner_act, attr_act, owner_out, attr_out) of every FFN."""
    sites = []
    for name, m in model.named_modules():
        # ViT / CLIP / most encoders: fc1 -> act -> fc2
        for act_attr in ("activation_fn", "act", "act_fn", "activation"):
            if hasattr(m, "fc1") and hasattr(m, "fc2") and hasattr(m, act_attr):
                sites.append((name, m, "fc1", m, act_attr, m, "fc2"))
                break
        else:
            # GPT-2: c_fc -> act -> c_proj
            if hasattr(m, "c_fc") and hasattr(m, "c_proj") and hasattr(m, "act"):
                sites.append((name, m, "c_fc", m, "act", m, "c_proj"))
            # BERT-like: intermediate.dense -> intermediate_act_fn ; output.dense
            elif hasattr(m, "intermediate") and hasattr(m, "output") and hasattr(m.intermediate, "intermediate_act_fn"):
                sites.append((name, m.intermediate, "dense", m.intermediate, "intermediate_act_fn", m.output, "dense"))
    return sites


def _gated_sites(model: nn.Module):
    """LLaMA / Mistral / Qwen / Gemma: down_proj(act(gate_proj(x)) * up_proj(x))."""
    return [
        (name, m)
        for name, m in model.named_modules()
        if all(hasattr(m, a) for a in ("gate_proj", "up_proj", "down_proj"))
    ]


def kanify(model: nn.Module, mode: str = "exact", basis="bspline", groups: int = 8, **kan_kwargs) -> list[str]:
    """Replace the feed-forward blocks of ``model`` in place; returns their names.

    ``mode="exact"``: the second projection becomes ``KANLinear.from_linear``
    with the activation as base branch and the activation slot an identity;
    gated FFNs get a ``KANLinear`` down projection with an identity base. The
    model computes exactly the same function until the splines are trained.

    ``mode="grkan"``: fc1 -> GR-KAN with identity rationals, fc2 -> GR-KAN with
    rationals fitted to the activation (KAT's weight transfer); approximate.
    Gated FFNs are left alone in this mode.
    """
    if mode not in ("exact", "grkan"):
        raise ValueError("mode must be 'exact' or 'grkan'")
    done = []
    for name, own_in, a_in, own_act, a_act, own_out, a_out in _ffn_sites(model):
        act = getattr(own_act, a_act)
        lin_in, lin_out = _linear_from(getattr(own_in, a_in)), _linear_from(getattr(own_out, a_out))
        if mode == "exact":
            setattr(own_in, a_in, lin_in)
            setattr(own_out, a_out, KANLinear.from_linear(lin_out, act, basis, **kan_kwargs))
        else:
            target = _act_name(act)
            fc1 = GRKAN(lin_in.in_features, lin_in.out_features, groups=groups, init="identity", bias=lin_in.bias is not None)
            fc2 = GRKAN(lin_out.in_features, lin_out.out_features, groups=groups, init=target, bias=lin_out.bias is not None)
            for g, lin in ((fc1, lin_in), (fc2, lin_out)):
                g.to(lin.weight.device, lin.weight.dtype)
                with torch.no_grad():
                    g.coef.copy_(lin.weight[..., None])
                    if lin.bias is not None:
                        g.bias.copy_(lin.bias)
            setattr(own_in, a_in, fc1)
            setattr(own_out, a_out, fc2)
        setattr(own_act, a_act, nn.Identity())
        done.append(name)
    if mode == "exact":
        for name, m in _gated_sites(model):
            if not isinstance(m.down_proj, KANLinear):
                m.down_proj = KANLinear.from_linear(m.down_proj, nn.Identity(), basis, **kan_kwargs)
                done.append(name)
    return done
