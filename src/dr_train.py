"""Uniform GCN trainer over a (possibly preprocessed) adjacency, using DeepRobust's GCN.

Every grid cell trains the SAME vanilla GCN; methods differ only in the adjacency they
hand in. This isolates the effect of the graph transform from the model.
"""
from __future__ import annotations

import numpy as np
import torch

from deeprobust.graph.defense import GCN


def train_gcn(features, adj, labels, idx_train, idx_val, idx_test,
              *, seed: int = 0, device: str = "cpu",
              nhid: int = 16, dropout: float = 0.5, lr: float = 0.01,
              weight_decay: float = 5e-4, train_iters: int = 200) -> float:
    """Train a vanilla GCN on (features, adj) and return test accuracy (float)."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = GCN(
        nfeat=features.shape[1], nhid=nhid, nclass=int(labels.max()) + 1,
        dropout=dropout, lr=lr, weight_decay=weight_decay, device=device,
    ).to(device)
    model.fit(features, adj, labels, idx_train, idx_val,
              train_iters=train_iters, verbose=False)
    acc = model.test(idx_test)
    return float(acc)
