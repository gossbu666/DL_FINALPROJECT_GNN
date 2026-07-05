"""Do the over-squashing methods actually raise the spectral gap? (sanity for RQ2)

FoSR and ProxyDelete are *designed* to increase the normalized-Laplacian spectral gap
(lambda_2). This script confirms they succeed at that stated goal, and pairs each gap
with the method's robust accuracy. The point: gap goes up (methods work as intended)
yet robustness does not follow -> the spectral gap is orthogonal to robustness here,
so the methods are not simply "broken/misconfigured".

Usage: python -m scripts.spectral_gap_analysis --ptb 0.05 --budget 253
"""
from __future__ import annotations

import argparse
import json
import os

import networkx as nx
import numpy as np
import scipy.sparse as sp

from deeprobust.graph.data import Dataset

from src.attacks import poisoned_adj
from src.methods import apply_method


def norm_laplacian_gap(adj):
    """Second-smallest eigenvalue (lambda_2) of the normalized Laplacian = spectral gap."""
    g = nx.from_scipy_sparse_array(sp.csr_matrix(adj))
    g.remove_edges_from(list(nx.selfloop_edges(g)))
    L = sp.csr_matrix(nx.normalized_laplacian_matrix(g), dtype=float)
    try:
        vals = sp.linalg.eigsh(L, k=2, sigma=0, which="LM", return_eigenvectors=False)
    except Exception:
        vals = np.linalg.eigvalsh(L.todense())
    return float(np.sort(np.real(vals))[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="cora")
    p.add_argument("--ptb", type=float, default=0.05)
    p.add_argument("--budget", type=int, default=253)
    p.add_argument("--methods", nargs="+", default=["fosr", "proxydelete", "dropedge", "jaccard"])
    p.add_argument("--grid", default="results/grid_cora.json", help="to pair gap with robust acc")
    p.add_argument("--jaccard_threshold", type=float, default=0.01)
    p.add_argument("--attack_seed", type=int, default=15)
    p.add_argument("--data_seed", type=int, default=15)
    p.add_argument("--root", default="/tmp/dr_data")
    p.add_argument("--out", default="results/spectral_gap.json")
    args = p.parse_args()

    data = Dataset(root=args.root, name=args.name, setting="nettack", seed=args.data_seed)
    adj0, features, labels = data.adj, data.features, data.labels
    idx_train = data.idx_train
    idx_unlabeled = np.union1d(data.idx_val, data.idx_test)
    nclass = int(labels.max()) + 1
    adjp = poisoned_adj(features, adj0, labels, idx_train, idx_unlabeled, args.ptb,
                        seed=args.attack_seed, name=args.name)

    robust = {}
    if os.path.exists(args.grid):
        gj = json.load(open(args.grid))["results"]
        robust = {m: gj.get(m, {}).get(str(args.ptb), {}).get("mean") for m in ["gcn"] + args.methods}

    rows = {"clean": norm_laplacian_gap(adj0), "poisoned": norm_laplacian_gap(adjp)}
    for m in args.methods:
        ctx = {"nfeat": features.shape[1], "nclass": nclass, "device": "cpu",
               "remove_budget": args.budget, "add_budget": args.budget,
               "jaccard_threshold": args.jaccard_threshold, "seed": 0}
        rows[m] = norm_laplacian_gap(apply_method(m, features, adjp, ctx))

    print(f"\n=== spectral gap (lambda_2) vs robustness ({args.name}, ptb={args.ptb}) ===")
    print(f"  {'graph/method':16s} {'spectral_gap':>13s} {'robust@'+str(args.ptb):>12s}")
    print(f"  {'clean':16s} {rows['clean']:13.5f} {'--':>12s}")
    base_gap = rows['poisoned']
    ra = robust.get('gcn')
    print(f"  {'poisoned (GCN)':16s} {base_gap:13.5f} {(ra*100 if ra else float('nan')):12.2f}")
    for m in args.methods:
        ra = robust.get(m)
        d = rows[m] - base_gap
        print(f"  {m:16s} {rows[m]:13.5f} {(ra*100 if ra else float('nan')):12.2f}   (dgap {d:+.5f})")

    json.dump({"ptb": args.ptb, "budget": args.budget, "gaps": rows, "robust": robust},
              open(args.out, "w"), indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
