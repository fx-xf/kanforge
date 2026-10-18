# Tutorials

One runnable script per task, in `examples/`.

| task | what it shows |
|---|---|
| [Regression](regression.md) | Fit a smooth function of two variables, adapt the grids to the data and refine them. |
| [Formula discovery](formula.md) | Train with the sparsity penalty, prune, make every edge symbolic and read the formula (pykan's workflow). |
| [PDEs with PINNs](pinn.md) | Solve a Poisson equation with the PDE residual in the loss; `derivative` keeps the graph for second derivatives. |
| [Operator learning](operator.md) | A DeepONet with KAN branch and trunk learns the antiderivative operator. |
| [Images](images.md) | Classify 8x8 digits with KAN convolutions. |
| [Transformers](transformers.md) | Put KAN layers into a Hugging Face GPT-2 without changing its outputs, then fine-tune only the splines. |
| [Graphs](graphs.md) | Node classification with KAN message passing on `edge_index` graphs. |
| [Time series](time_series.md) | One-step forecasting with an LSTM whose gates are KAN layers. |
| [Neural ODEs](ode.md) | Learn the dynamics of a damped oscillator with dy/dt = KAN(y) and a differentiable RK4. |
| [Reinforcement learning](rl.md) | REINFORCE with a KAN policy on a small continuous control task (no gym needed). |
| [Deployment](deploy.md) | Lookup tables, int8, ONNX in onnxruntime and Verilog for an FPGA from one trained model. |
| [Backends](backends.md) | Convert a model between PyTorch, MLX and JAX and through one file format. |
