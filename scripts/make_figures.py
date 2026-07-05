"""Generate the headline results figure from a grid json.

Grouped bar chart of robust accuracy per method (at each ptb), with the defense
(GCN-Jaccard) highlighted and the vanilla-GCN clean baseline drawn as a reference
line. Makes the central finding visible at a glance: only Jaccard separates from
the pack.

Usage: python -m scripts.make_figures --grid results/grid_cora.json --out final_session/report/figures/headline.png
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {"gcn": "GCN", "dropedge": "DropEdge\n(rand rm)", "jaccard": "Jaccard\n(sim rm)",
          "fosr": "FoSR\n(add)", "proxydelete": "ProxyDelete\n(spec rm)", "svd": "GCN-SVD\n(low-rank)"}
ORDER = ["gcn", "dropedge", "fosr", "proxydelete", "jaccard", "svd"]
DEFENSES = {"jaccard", "svd"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", default="results/grid_cora.json")
    p.add_argument("--out", default="final_session/report/figures/headline.png")
    p.add_argument("--title", default="Cora")
    args = p.parse_args()

    with open(args.grid) as f:
        data = json.load(f)
    res = data["results"]
    ptbs = data["ptb"]
    methods = [m for m in ORDER if m in res]

    x = np.arange(len(methods))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for k, ptb in enumerate(ptbs):
        vals = [res[m][str(ptb)]["mean"] * 100 for m in methods]
        errs = [res[m][str(ptb)]["std"] * 100 for m in methods]
        cmap = {"jaccard": "#d62728", "svd": "#ff7f0e"}
        colors = [cmap.get(m, "#7f7f7f") for m in methods]
        bars = ax.bar(x + (k - 0.5) * width, vals, width, yerr=errs, capsize=3,
                      color=colors, alpha=0.95 if k == 0 else 0.55,
                      label=f"robust @ ptb {ptb}")
    # vanilla GCN clean baseline reference
    gcn_clean = res["gcn"]["0.0"]["mean"] * 100
    ax.axhline(gcn_clean, ls=":", color="black", lw=1, label=f"GCN clean ({gcn_clean:.1f})")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(m, m) for m in methods], fontsize=8)
    ax.set_ylabel("robust test accuracy (%)")
    ax.set_ylim(min(60, gcn_clean - 18), gcn_clean + 2)
    ax.set_title(f"Robust accuracy under Metattack ({args.title})", fontsize=11)
    ax.legend(fontsize=7, loc="lower left")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
