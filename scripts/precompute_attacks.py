"""Precompute + cache Metattack poisoned graphs (run once, e.g. overnight).

Metattack is the expensive step on CPU; this generates the poisoned adjacency for
each (ptb_rate, attack_seed) and caches it under results/cache/. The grid then reuses
these instantly. Run this BEFORE run_grid.py so the grid never triggers a slow attack.

Usage:
    python -m scripts.precompute_attacks --ptb 0.05 0.10 --attack_seeds 15 \
        --attack Meta-Self --inner_iters 100
    # if CPU time is tight, use --attack Meta-Approx (faster, slightly weaker)
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from deeprobust.graph.data import Dataset

from src.attacks import poisoned_adj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="cora", choices=["cora", "citeseer"])
    p.add_argument("--ptb", type=float, nargs="+", default=[0.05, 0.10])
    p.add_argument("--attack_seeds", type=int, nargs="+", default=[15])
    p.add_argument("--attack", default="Meta-Self", choices=["Meta-Self", "Meta-Approx"])
    p.add_argument("--inner_iters", type=int, default=100)
    p.add_argument("--data_seed", type=int, default=15)
    p.add_argument("--device", default="cpu")
    p.add_argument("--root", default="/tmp/dr_data")
    args = p.parse_args()

    data = Dataset(root=args.root, name=args.name, setting="nettack", seed=args.data_seed)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)
    print(f"{args.name} nettack split: nodes={adj.shape[0]}, edges={int(adj.sum()//2)}")

    for seed in args.attack_seeds:
        for ptb in args.ptb:
            t0 = time.time()
            mod = poisoned_adj(
                features, adj, labels, idx_train, idx_unlabeled, ptb,
                attack=args.attack, seed=seed, device=args.device,
                name=args.name, inner_iters=args.inner_iters,
            )
            print(f"  [{args.attack}] ptb={ptb} seed={seed}: "
                  f"edges {int(adj.sum()//2)} -> {int(mod.sum()//2)} "
                  f"({time.time()-t0:.0f}s)")
    print("done. cached under results/cache/")


if __name__ == "__main__":
    main()
