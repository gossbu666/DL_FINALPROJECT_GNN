"""Defense reference: GCN-Jaccard edge pruning, used as an adjacency preprocessor.

We reuse DeepRobust's validated `drop_dissimilar_edges` (so the defense is faithful to
the reference implementation) but apply it as a preprocessing step, then train a vanilla
GCN on the pruned graph. This keeps every grid method in the same shape:
    adjacency --(method)--> adjacency --> train vanilla GCN.
"""
from __future__ import annotations

import numpy as np
import torch

from deeprobust.graph.defense import GCNJaccard, GCNSVD


def train_gcnsvd(features, adj, labels, idx_train, idx_val, idx_test, *,
                 k: int = 15, seed: int = 0, device: str = "cpu",
                 nhid: int = 16, dropout: float = 0.5, lr: float = 0.01,
                 weight_decay: float = 5e-4, train_iters: int = 200) -> float:
    """GCN-SVD defense (Entezari et al. 2020): train on a rank-k low-rank approximation
    of the (possibly poisoned) adjacency. A second, mechanism-distinct defense reference
    (low-rank filtering rather than similarity pruning). Returns test accuracy."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = GCNSVD(
        nfeat=features.shape[1], nhid=nhid, nclass=int(labels.max()) + 1,
        dropout=dropout, lr=lr, weight_decay=weight_decay, device=device,
    ).to(device)
    model.fit(features, adj, labels, idx_train, idx_val, k=k,
              train_iters=train_iters, verbose=False)
    return float(model.test(idx_test))


def jaccard_prune(features, adj, *, threshold: float, nfeat: int, nclass: int,
                  binary_feature: bool = True, device: str = "cpu"):
    """Drop edges whose endpoint features are too dissimilar (Jaccard for binary feats).

    Returns the pruned (csr) adjacency. threshold is the similarity cutoff (DeepRobust
    default 0.01 for Cora's binary bag-of-words features).
    """
    jac = GCNJaccard(
        nfeat=nfeat, nhid=16, nclass=nclass,
        binary_feature=binary_feature, device=device,
    )
    jac.threshold = threshold
    return jac.drop_dissimilar_edges(features, adj)
