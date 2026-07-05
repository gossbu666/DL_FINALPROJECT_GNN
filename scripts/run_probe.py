"""Task 4 - run the Jacobian over-squashing probe (RQ1).

Trains an L-layer GCN on the clean and on the poisoned Cora graph, measures Jacobian
sensitivity ||d h_v / d x_u||_F vs graph distance d(u,v), and compares the decay
curves. If poisoning measurably changes long-range sensitivity, that is evidence for
RQ1 (perturbation interacts with the over-squashing measure).

Usage:
    python -m scripts.run_probe --layers 6 --targets 20 --ptb 0.10
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from deeprobust.graph.data import Dataset

from src.attacks import poisoned_adj
from src.probe import (adj_to_edge_index, jacobian_sensitivity,
                       sensitivity_by_distance, train_probe_gcn)


def to_dense_x(features):
    arr = features.todense() if hasattr(features, "todense") else np.asarray(features)
    return torch.tensor(np.asarray(arr), dtype=torch.float32)


def run_one(label, adj, x, y, idx_train, *, layers, targets, max_dist, seed):
    edge_index = adj_to_edge_index(adj)
    model = train_probe_gcn(x, edge_index, y, idx_train, num_layers=layers, seed=seed)
    rng = np.random.default_rng(seed)
    target_nodes = rng.choice(adj.shape[0], size=targets, replace=False).tolist()
    sens = jacobian_sensitivity(model, x, edge_index, target_nodes)
    buckets = sensitivity_by_distance(adj, sens, target_nodes, max_dist)
    curve = {d: (float(np.mean(v)) if v else float("nan"),
                 int(len(v))) for d, v in buckets.items()}
    print(f"[{label}] mean ||J_vu||_F by distance:")
    for d, (m, k) in curve.items():
        print(f"   d={d}: {m:.3e}  (n={k})")
    return curve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="cora")
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--targets", type=int, default=20)
    p.add_argument("--max_dist", type=int, default=6)
    p.add_argument("--ptb", type=float, default=0.10)
    p.add_argument("--attack", default="Meta-Self")
    p.add_argument("--attack_seed", type=int, default=15)
    p.add_argument("--data_seed", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default="/tmp/dr_data")
    p.add_argument("--out", default="results/probe_cora.json")
    p.add_argument("--fig", default="final_session/report/figures/probe.png")
    args = p.parse_args()

    data = Dataset(root=args.root, name=args.name, setting="nettack", seed=args.data_seed)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train = data.idx_train
    idx_unlabeled = np.union1d(data.idx_val, data.idx_test)
    x = to_dense_x(features)
    y = torch.tensor(labels, dtype=torch.long)

    adj_poison = poisoned_adj(features, adj, labels, idx_train, idx_unlabeled, args.ptb,
                              attack=args.attack, seed=args.attack_seed, name=args.name)

    clean_curve = run_one("clean", adj, x, y, idx_train,
                          layers=args.layers, targets=args.targets,
                          max_dist=args.max_dist, seed=args.seed)
    poison_curve = run_one(f"poisoned(ptb={args.ptb})", adj_poison, x, y, idx_train,
                           layers=args.layers, targets=args.targets,
                           max_dist=args.max_dist, seed=args.seed)

    rec = {"dataset": args.name, "layers": args.layers, "targets": args.targets,
           "ptb": args.ptb, "clean": clean_curve, "poisoned": poison_curve}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"saved -> {args.out}")

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ds = list(range(1, args.max_dist + 1))
        cm = [clean_curve[d][0] for d in ds]
        pm = [poison_curve[d][0] for d in ds]
        plt.figure(figsize=(5, 3.2))
        plt.semilogy(ds, cm, "o-", label="clean")
        plt.semilogy(ds, pm, "s--", label=f"poisoned (ptb={args.ptb})")
        plt.xlabel("graph distance $d(u,v)$")
        plt.ylabel(r"mean $\|\partial h_v/\partial x_u\|_F$")
        plt.title(f"Jacobian sensitivity vs. distance ({args.name})", fontsize=10)
        plt.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(args.fig), exist_ok=True)
        plt.savefig(args.fig, dpi=150)
        print(f"saved figure -> {args.fig}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
