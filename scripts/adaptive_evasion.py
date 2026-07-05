"""(c) Adaptive / defense-aware attack: can an attacker who KNOWS GCN-Jaccard evade it?

GCN-Jaccard removes edges between feature-DISSIMILAR nodes (Jaccard similarity < threshold).
A defense-aware attacker therefore adds edges that Jaccard will NOT remove: between nodes with
HIGH feature similarity but DIFFERENT labels. These edges (i) survive the defense and (ii) still
inject wrong-class signal into message passing. This is a heuristic (not gradient-optimal) adaptive
attack; we report its effect honestly.

Two informative outcomes: (A) the evasion attack hurts and Jaccard cannot recover -> the defense is
evadable (limitation confirmed); (B) constraining edges to be similarity-high makes the attack weak
-> effective structural attacks inherently use removable (dissimilar) edges, so Jaccard's defense is
fundamental. Either is reported as-is.

Usage: python -m scripts.adaptive_evasion --budgets 253 503 --threshold 0.01
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import scipy.sparse as sp

from deeprobust.graph.data import Dataset

from src.defenses import jaccard_prune
from src.dr_train import train_gcn


def feature_jaccard(features):
    F = sp.csr_matrix(features).astype(bool).astype(float)
    inter = (F @ F.T).toarray()
    deg = np.asarray(F.sum(1)).ravel()
    union = deg[:, None] + deg[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(union > 0, inter / union, 0.0)
    return jac


def build_evasion(adj, features, labels, budget, threshold):
    """Add up to `budget` edges between different-class, feature-similar (jac>=threshold)
    node pairs, chosen highest-similarity first (most stealthy). Returns (adj', added_edges)."""
    n = adj.shape[0]
    jac = feature_jaccard(features)
    A = (sp.csr_matrix(adj).toarray() > 0)
    diff_class = labels[:, None] != labels[None, :]
    ok = diff_class & (~A) & (jac >= threshold)
    np.fill_diagonal(ok, False)
    score = np.where(ok, jac, -1.0)
    iu = np.triu_indices(n, 1)
    s = score[iu]
    order = np.argsort(-s)
    added = []
    for idx in order:
        if s[idx] < threshold:
            break
        added.append((int(iu[0][idx]), int(iu[1][idx])))
        if len(added) >= budget:
            break
    Ap = sp.lil_matrix(sp.csr_matrix(adj))
    for i, j in added:
        Ap[i, j] = 1
        Ap[j, i] = 1
    return Ap.tocsr(), added


def edge_set(adj):
    u = sp.triu(sp.csr_matrix(adj), k=1).tocoo()
    return set(zip(u.row.tolist(), u.col.tolist()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="cora")
    p.add_argument("--budgets", type=int, nargs="+", default=[253, 503])
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--data_seed", type=int, default=15)
    p.add_argument("--root", default="/tmp/dr_data")
    p.add_argument("--out", default="results/adaptive_evasion.json")
    args = p.parse_args()

    data = Dataset(root=args.root, name=args.name, setting="nettack", seed=args.data_seed)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    nclass = int(labels.max()) + 1

    def acc_mean(adjX, defense):
        vals = []
        for s in args.seeds:
            if defense:
                aj = jaccard_prune(features, adjX, threshold=args.threshold,
                                   nfeat=features.shape[1], nclass=nclass)
            else:
                aj = adjX
            vals.append(train_gcn(features, aj, labels, idx_train, idx_val, idx_test, seed=s))
        return float(np.mean(vals) * 100), float(np.std(vals) * 100)

    clean_gcn = acc_mean(adj, False)
    print(f"clean GCN: {clean_gcn[0]:.2f}")
    rec = {"clean_gcn": clean_gcn, "budgets": {}}

    for B in args.budgets:
        adjE, added = build_evasion(adj, features, labels, B, args.threshold)
        # how many evasion edges does Jaccard remove? (expect few = evasion works)
        aj = jaccard_prune(features, adjE, threshold=args.threshold,
                           nfeat=features.shape[1], nclass=nclass)
        removed = edge_set(adjE) - edge_set(aj)
        jac_recall = len(set(added) & removed) / len(added) if added else 0.0

        gcn = acc_mean(adjE, False)
        jac = acc_mean(adjE, True)
        print(f"\nbudget {B}: requested {B}, added {len(added)} evasion edges "
              f"(Jaccard removes {jac_recall:.2f} of them)")
        print(f"  GCN under evasion    : {gcn[0]:.2f} +/- {gcn[1]:.2f}  (drop {clean_gcn[0]-gcn[0]:+.2f})")
        print(f"  Jaccard under evasion: {jac[0]:.2f} +/- {jac[1]:.2f}  (recovery {jac[0]-gcn[0]:+.2f})")
        rec["budgets"][str(B)] = {"requested": B, "added": len(added), "jaccard_recall_on_evasion": jac_recall,
                                  "gcn": gcn, "jaccard": jac}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rec, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
