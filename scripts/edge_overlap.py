"""Mechanism analysis: does each method remove the edges the attacker added?

The RQ2 finding is that only GCN-Jaccard defends. This script tests WHY: for each
method we measure how many of Metattack's inserted edges it removes.
  - recall    = |method_removed AND attacker_added| / |attacker_added|
                (of the attack edges, what fraction did the method delete?)
  - precision = |method_removed AND attacker_added| / |method_removed|
                (of the method's deletions, what fraction were attack edges?)
A random remover of k edges has recall ~ k / |poisoned edges| (the "chance" line).
Expectation: Jaccard's recall >> chance (it targets dissimilar = attacker edges),
while spectral/random removal sit at chance and FoSR (which adds) removes ~none.

Usage: python -m scripts.edge_overlap --ptb 0.05 0.10 --budget 643
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import scipy.sparse as sp

from deeprobust.graph.data import Dataset

from src.attacks import poisoned_adj
from src.methods import apply_method


def edge_set(adj):
    """Undirected edge set {(i,j), i<j} from a symmetric adjacency."""
    u = sp.triu(sp.csr_matrix(adj), k=1).tocoo()
    return set(zip(u.row.tolist(), u.col.tolist()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="cora")
    p.add_argument("--ptb", type=float, nargs="+", default=[0.05, 0.10])
    p.add_argument("--methods", nargs="+", default=["jaccard", "proxydelete", "dropedge", "fosr"])
    p.add_argument("--budget", type=int, default=643, help="remove/add budget for spectral/random/add arms")
    p.add_argument("--jaccard_threshold", type=float, default=0.01)
    p.add_argument("--attack", default="Meta-Self")
    p.add_argument("--attack_seed", type=int, default=15)
    p.add_argument("--data_seed", type=int, default=15)
    p.add_argument("--device", default="cpu")
    p.add_argument("--root", default="/tmp/dr_data")
    p.add_argument("--out", default="results/edge_overlap.json")
    args = p.parse_args()

    data = Dataset(root=args.root, name=args.name, setting="nettack", seed=args.data_seed)
    adj0, features, labels = data.adj, data.features, data.labels
    idx_train = data.idx_train
    idx_unlabeled = np.union1d(data.idx_val, data.idx_test)
    nclass = int(labels.max()) + 1
    E0 = edge_set(adj0)

    record = {}
    for ptb in args.ptb:
        adjp = poisoned_adj(features, adj0, labels, idx_train, idx_unlabeled, ptb,
                            attack=args.attack, seed=args.attack_seed, name=args.name)
        Ep = edge_set(adjp)
        added = Ep - E0          # edges Metattack inserted
        removed_by_attack = E0 - Ep
        chance = lambda k: (k / len(Ep)) if len(Ep) else 0.0
        print(f"\n=== {args.name} ptb={ptb}: attacker added {len(added)}, removed {len(removed_by_attack)} "
              f"(poisoned edges {len(Ep)}) ===")
        print(f"  {'method':12s} {'#rm':>6s} {'#attack_rm':>11s} {'recall':>8s} {'precision':>10s} {'chance':>8s}")
        record[str(ptb)] = {"attacker_added": len(added), "poisoned_edges": len(Ep), "methods": {}}
        for m in args.methods:
            ctx = {"nfeat": features.shape[1], "nclass": nclass, "device": args.device,
                   "remove_budget": args.budget, "add_budget": args.budget,
                   "jaccard_threshold": args.jaccard_threshold, "seed": 0}
            Am = apply_method(m, features, adjp, ctx)
            Em = edge_set(Am)
            removed = Ep - Em            # edges the method deleted from the poisoned graph
            hit = removed & added        # attack edges the method deleted
            recall = len(hit) / len(added) if added else 0.0
            precision = len(hit) / len(removed) if removed else 0.0
            ch = chance(len(removed))
            print(f"  {m:12s} {len(removed):6d} {len(hit):11d} {recall:8.3f} {precision:10.3f} {ch:8.3f}")
            record[str(ptb)]["methods"][m] = {
                "removed": len(removed), "added_edges": len(Em - Ep),
                "attack_removed": len(hit), "recall": recall,
                "precision": precision, "chance_recall": ch}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
