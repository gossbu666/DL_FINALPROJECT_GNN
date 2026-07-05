"""Metattack poisoning + disk caching (Task 2).

The attack is the expensive step (full Metattack meta-gradient is slow on CPU),
so we generate the poisoned adjacency ONCE per (ptb_rate, attack_seed) and cache
it to disk. Every method in the grid then reuses the same poisoned graph.

DeepRobust 'Meta-Self' (lambda_=0) is the standard for robust-accuracy benchmarks;
'Meta-Approx' (MetaApprox) is a faster approximation for when CPU time is tight.
"""
from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp

from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import MetaApprox, Metattack


def _cache_path(cache_dir, name, attack, ptb_rate, seed):
    tag = "approx" if attack == "Meta-Approx" else "self"
    return os.path.join(cache_dir, f"{name}_meta-{tag}_ptb{ptb_rate}_seed{seed}.npz")


def poisoned_adj(
    features, adj, labels, idx_train, idx_unlabeled,
    ptb_rate: float,
    *,
    attack: str = "Meta-Self",
    seed: int = 15,
    device: str = "cpu",
    cache_dir: str = "results/cache",
    name: str = "cora",
    surrogate_iters: int = 200,
    inner_iters: int = 100,
):
    """Return a poisoned (csr) adjacency for the given ptb_rate, from cache if present.

    ptb_rate == 0 returns the clean adjacency unchanged (the 'clean' condition).
    """
    if ptb_rate == 0:
        return adj if sp.issparse(adj) else sp.csr_matrix(adj)

    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, name, attack, ptb_rate, seed)
    if os.path.exists(path):
        return sp.load_npz(path)

    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)

    n = adj.shape[0]
    n_perturbations = int(ptb_rate * (adj.sum() // 2))

    # linearized surrogate the meta-attack differentiates through
    surrogate = GCN(
        nfeat=features.shape[1], nhid=16, nclass=int(labels.max()) + 1,
        dropout=0, with_relu=False, with_bias=False, device=device,
    ).to(device)
    surrogate.fit(features, adj, labels, idx_train, train_iters=surrogate_iters, verbose=False)

    AttackCls = MetaApprox if attack == "Meta-Approx" else Metattack
    attacker = AttackCls(
        surrogate, nnodes=n, feature_shape=features.shape,
        attack_structure=True, attack_features=False, device=device,
        lambda_=0, train_iters=inner_iters,
    ).to(device)
    attacker.attack(features, adj, labels, idx_train, idx_unlabeled,
                    n_perturbations, ll_constraint=False)

    modified = sp.csr_matrix(attacker.modified_adj.detach().cpu().numpy())
    sp.save_npz(path, modified)
    return modified
