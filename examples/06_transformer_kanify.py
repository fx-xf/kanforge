"""KAN layers inside a pretrained transformer (Hugging Face).

``kanify`` turns every FFN projection into a ``KANLinear`` whose base branch is
the original activation and whose splines start at zero: the model computes
exactly the same function, and fine-tuning trains the spline corrections. A
tiny randomly initialised GPT-2 keeps the example offline; use
``from_pretrained("gpt2")`` for the real one.
"""

import os

import torch
import transformers

from kanforge.torch import KANLinear, kanify

QUICK = os.environ.get("KANFORGE_QUICK") == "1"
torch.manual_seed(0)
config = transformers.GPT2Config(n_layer=2, n_embd=64, n_head=4, vocab_size=200, n_positions=64, bos_token_id=0, eos_token_id=0)
model = transformers.GPT2LMHeadModel(config).eval()
ids = torch.randint(0, 200, (4, 16))
with torch.no_grad():
    before = model(ids).logits
print("replaced:", kanify(model, mode="exact", basis={"name": "bspline", "grid_size": 5, "order": 3, "grid_range": [-3, 3]}))
with torch.no_grad():
    print("max change of logits:", (model(ids).logits - before).abs().max().item())

model.train()
splines = [p for m in model.modules() if isinstance(m, KANLinear) for p in (m.coef,)]
opt = torch.optim.Adam(splines, lr=1e-3)  # fine-tune only the KAN corrections
for step in range(2 if QUICK else 50):
    opt.zero_grad()
    loss = model(ids, labels=ids).loss
    loss.backward()
    opt.step()
print(f"language-model loss after fine-tuning the splines: {loss.item():.3f}")
