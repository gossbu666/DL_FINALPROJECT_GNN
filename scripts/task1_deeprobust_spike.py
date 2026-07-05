"""Task 1 - DeepRobust gating spike (CPU).

Runs the full poisoning path ONCE end-to-end on Cora to de-risk the library
before building the grid:
    clean GCN  ->  Metattack(ptb=0.05) poisons adj  ->  GCN on poisoned (undefended)
                                                     ->  GCN-Jaccard on poisoned (defended)

If this prints a sensible clean > robust, and Jaccard recovers some accuracy,
the DeepRobust path works and Task 2 (the grid) can proceed. Numbers are real;
this is a smoke test, not a tuned result.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import scipy.sparse as sp
import torch

from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN, GCNJaccard
from deeprobust.graph.global_attack import Metattack

DEVICE = "cpu"
SEED = 15
PTB_RATE = 0.05


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def new_gcn(features, labels, dropout=0.5, **kw):
    return GCN(
        nfeat=features.shape[1],
        nhid=16,
        nclass=int(labels.max()) + 1,
        dropout=dropout,
        device=DEVICE,
        **kw,
    ).to(DEVICE)


def main():
    t0 = time.time()
    set_seed(SEED)

    # DeepRobust's own Cora load + the standard 'nettack' split used in robustness papers
    data = Dataset(root="/tmp/dr_data", name="cora", setting="nettack", seed=SEED)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)
    n = adj.shape[0]
    n_perturbations = int(PTB_RATE * (adj.sum() // 2))
    print(f"loaded cora (nettack split): nodes={n}, edges={int(adj.sum()//2)}, "
          f"perturbations={n_perturbations}")

    # 1) clean GCN -> clean accuracy
    gcn = new_gcn(features, labels)
    gcn.fit(features, adj, labels, idx_train, idx_val, patience=30, verbose=False)
    clean_acc = float(gcn.test(idx_test))
    print(f"[clean]            test acc = {clean_acc:.4f}  ({time.time()-t0:.0f}s)")

    # 2) surrogate (linear GCN) for the meta-attack
    surrogate = new_gcn(features, labels, with_relu=False, with_bias=False, dropout=0)
    surrogate.fit(features, adj, labels, idx_train, verbose=False)

    # 3) Metattack -> poisoned adjacency
    print(f"running Metattack (ptb={PTB_RATE}) ...")
    attacker = Metattack(
        surrogate, nnodes=n, feature_shape=features.shape,
        attack_structure=True, attack_features=False, device=DEVICE, lambda_=0,
    ).to(DEVICE)
    attacker.attack(features, adj, labels, idx_train, idx_unlabeled,
                    n_perturbations, ll_constraint=False)
    modified_adj = attacker.modified_adj  # dense torch tensor
    modified_adj_sp = sp.csr_matrix(modified_adj.detach().cpu().numpy())
    print(f"  attack done ({time.time()-t0:.0f}s)")

    # 4) GCN on poisoned graph -> robust accuracy (undefended)
    gcn_p = new_gcn(features, labels)
    gcn_p.fit(features, modified_adj_sp, labels, idx_train, idx_val, patience=30, verbose=False)
    robust_acc = float(gcn_p.test(idx_test))
    print(f"[poisoned, no def] test acc = {robust_acc:.4f}")

    # 5) GCN-Jaccard on poisoned graph -> defended accuracy
    jac = GCNJaccard(
        nfeat=features.shape[1], nhid=16, nclass=int(labels.max()) + 1,
        dropout=0.5, device=DEVICE, binary_feature=True,
    ).to(DEVICE)
    jac.fit(features, modified_adj_sp, labels, idx_train, idx_val, threshold=0.01, verbose=False)
    jac_acc = float(jac.test(idx_test))
    print(f"[poisoned, Jaccard]test acc = {jac_acc:.4f}")

    print("\n=== Task 1 spike summary (Cora, ptb=0.05, CPU) ===")
    print(f"  clean GCN        : {clean_acc:.4f}")
    print(f"  poisoned GCN     : {robust_acc:.4f}   (gap = {clean_acc - robust_acc:.4f})")
    print(f"  poisoned Jaccard : {jac_acc:.4f}   (recovery = {jac_acc - robust_acc:+.4f})")
    print(f"  total time       : {time.time()-t0:.0f}s")

    rec = {
        "dataset": "cora", "split": "nettack", "ptb_rate": PTB_RATE, "device": DEVICE,
        "seed": SEED, "n_perturbations": int(n_perturbations),
        "clean_acc": clean_acc, "robust_acc_gcn": robust_acc, "robust_acc_jaccard": jac_acc,
        "gap": clean_acc - robust_acc, "jaccard_recovery": jac_acc - robust_acc,
        "seconds": time.time() - t0,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/task1_spike.json", "w") as f:
        json.dump(rec, f, indent=2)
    print("saved -> results/task1_spike.json")


if __name__ == "__main__":
    main()
