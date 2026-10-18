# Architecture blocks

| block | PyTorch (`kanforge.torch`) | MLX |
|---|---|---|
| convolutions | `KANConv1d/2d/3d` (`from_conv` exact) | channels-last versions |
| transformers | `KANFFN` (KAN or GR-KAN), `KATBlock`, `kanify(model)` | `KANFFN`, `KATBlock` |
| graphs | `KAGINConv`, `KAGCNConv`, `KAGATConv` (`edge_index` like PyG) | same |
| sequences | `KANLSTMCell`, `KANGRUCell`, `KANRNN` | same |
| scientific ML | `derivative`, `laplacian`, `sample_box`, `DeepONet`, `FBKAN`, `KANODE` | same |

`kanify(model, mode="exact")` replaces the FFN projections of Hugging Face
models (GPT-2, BERT, ViT/CLIP, LLaMA/Mistral/Qwen gated FFNs) by
`KANLinear`s that compute the same function; `mode="grkan"` transfers the
weights into KAT's group-rational layers.
