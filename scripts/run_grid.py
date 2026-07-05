"""Task 2 - the headline grid: methods x conditions x seeds -> robustness table.

For each method and each graph condition (clean, ptb 0.05, ptb 0.10), apply the method's
adjacency transform and train a vanilla GCN over several seeds; report mean +/- std test
accuracy and the robustness gap (clean - robust).

Pipeline order: poison (cached) -> method transform -> train -> eval. Reads cached
poisoned graphs (run scripts.precompute_attacks first); for ptb=0 it uses the clean graph.

Usage:
    python -m scripts.run_grid --methods gcn dropedge jaccard --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from datetime import datetime, timezone

import numpy as np

from deeprobust.graph.data import Dataset

from src.attacks import poisoned_adj
from src.dr_train import train_gcn
from src.defenses import train_gcnsvd
from src.methods import apply_method

# defenses that are trained as a model (not a pure adjacency->adjacency preprocess)
MODEL_METHODS = {"svd"}


def mean_std(xs):
    m = statistics.mean(xs)
    s = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return m, s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="cora", choices=["cora", "citeseer"])
    p.add_argument("--methods", nargs="+", default=["gcn", "dropedge", "jaccard"])
    p.add_argument("--ptb", type=float, nargs="+", default=[0.05, 0.10])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--attack", default="Meta-Self", choices=["Meta-Self", "Meta-Approx"])
    p.add_argument("--attack_seed", type=int, default=15)
    p.add_argument("--data_seed", type=int, default=15)
    p.add_argument("--jaccard_threshold", type=float, default=0.01)
    p.add_argument("--svd_k", type=int, default=15, help="SVD rank for GCN-SVD defense")
    p.add_argument("--remove_rate", type=float, default=0.05,
                   help="edge budget for dropedge/proxydelete, as a fraction of clean edges")
    p.add_argument("--remove_budget_abs", type=int, default=None,
                   help="absolute edge budget (overrides remove_rate); for budget-matched checks")
    p.add_argument("--device", default="cpu")
    p.add_argument("--root", default="/tmp/dr_data")
    p.add_argument("--out", default="results/grid_cora.json")
    args = p.parse_args()

    data = Dataset(root=args.root, name=args.name, setting="nettack", seed=args.data_seed)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)
    n_edges = int(adj.sum() // 2)
    nclass = int(labels.max()) + 1
    remove_budget = args.remove_budget_abs if args.remove_budget_abs else int(args.remove_rate * n_edges)

    conditions = [0.0] + list(args.ptb)            # 0.0 == clean
    # base graph per condition (clean, or cached poisoned)
    base = {}
    for c in conditions:
        base[c] = poisoned_adj(
            features, adj, labels, idx_train, idx_unlabeled, c,
            attack=args.attack, seed=args.attack_seed, device=args.device, name=args.name,
        )

    results = {}  # method -> condition -> (mean, std, [accs])
    for m in args.methods:
        results[m] = {}
        for c in conditions:
            accs = []
            for s in args.seeds:
                if m in MODEL_METHODS:  # model-based defense (e.g. GCN-SVD)
                    acc = train_gcnsvd(features, base[c], labels, idx_train, idx_val, idx_test,
                                       k=args.svd_k, seed=s, device=args.device)
                else:
                    ctx = {"nfeat": features.shape[1], "nclass": nclass, "device": args.device,
                           "remove_budget": remove_budget, "add_budget": remove_budget,
                           "jaccard_threshold": args.jaccard_threshold, "seed": s}
                    adj_m = apply_method(m, features, base[c], ctx)
                    acc = train_gcn(features, adj_m, labels, idx_train, idx_val, idx_test,
                                    seed=s, device=args.device)
                accs.append(acc)
            mu, sd = mean_std(accs)
            results[m][c] = {"mean": mu, "std": sd, "accs": accs}
            tag = "clean" if c == 0.0 else f"ptb{c}"
            print(f"  {m:12s} {tag:8s}: {mu*100:.2f} +/- {sd*100:.2f}")

    # markdown headline table
    print(f"\n### {args.name} headline (test acc %, {len(args.seeds)} seeds, {args.attack})")
    header = ["method", "clean"] + [f"robust@{c}" for c in args.ptb] + [f"gap@{c}" for c in args.ptb]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for m in args.methods:
        clean = results[m][0.0]["mean"]
        row = [m, f"{clean*100:.2f}"]
        for c in args.ptb:
            row.append(f"{results[m][c]['mean']*100:.2f}")
        for c in args.ptb:
            row.append(f"{(clean - results[m][c]['mean'])*100:.2f}")
        print("| " + " | ".join(row) + " |")

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.name, "split": "nettack", "attack": args.attack,
        "attack_seed": args.attack_seed, "seeds": args.seeds, "ptb": args.ptb,
        "jaccard_threshold": args.jaccard_threshold, "remove_budget": remove_budget,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
