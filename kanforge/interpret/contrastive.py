"""pykan's unsupervised mode: find dependent features with a contrastive task.

Real samples (label 1) are mixed with samples whose columns are shuffled
independently (label 0), which destroys every dependence between features
but keeps their marginals. A KAN trained to tell them apart can only use
relations between features; pruning and ``feature_importance`` then show
which features are related, and ``auto_symbolic`` how.
"""

from __future__ import annotations

import numpy as np


def contrastive_samples(x, seed: int = 0):
    """(inputs, labels) NumPy arrays: ``x`` with label 1 and a column-shuffled copy with label 0."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    fake = np.stack([rng.permutation(x[:, j]) for j in range(x.shape[1])], 1)
    inputs = np.concatenate([x, fake], 0)
    labels = np.concatenate([np.ones(len(x)), np.zeros(len(x))])
    order = rng.permutation(len(inputs))
    return inputs[order], labels[order]
