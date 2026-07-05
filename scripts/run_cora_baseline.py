"""Task 0 - GCN baseline on Cora (and other Planetoid graphs).

Produces the clean-accuracy number that the Phase 2 protocol needs, and confirms
dataset stats at load time (Open Question #4). Reproducible: seeds fixed, runs
averaged over multiple seeds so we report mean +/- std (integrity: real numbers).

Usage:
    python -m scripts.run_cora_baseline --dataset Cora --runs 5 --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from datetime import datetime, timezone

import numpy as np
import torch

from src.data import load_planetoid, describe
from src.models import build_model
from src.train import train_node_classifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Cora", choices=["Cora", "Citeseer", "PubMed"])
    p.add_argument("--model", default="gcn")
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--runs", type=int, default=5, help="seeds averaged")
    p.add_argument("--seed", type=int, default=42, help="base seed; run i uses seed+i")
    p.add_argument("--device", default="cpu", help="cpu (default) avoids Blackwell sm_120 issues")
    p.add_argument("--out", default="results/cora_baseline.json")
    args = p.parse_args()

    dataset, data = load_planetoid(args.dataset)
    stats = describe(data, dataset)
    print(f"[{args.dataset}] stats: {json.dumps(stats)}")

    accs = []
    for i in range(args.runs):
        set_seed(args.seed + i)
        model = build_model(
            args.model,
            in_channels=dataset.num_features,
            hidden_channels=args.hidden,
            out_channels=dataset.num_classes,
            dropout=args.dropout,
        )
        res = train_node_classifier(
            model, data,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            device=args.device, verbose=(i == 0),
        )
        print(f"  run {i} (seed {args.seed + i}): test_acc={res.test_acc:.4f} "
              f"(val={res.val_acc:.4f} @ epoch {res.best_epoch})")
        accs.append(res.test_acc)

    mean = statistics.mean(accs)
    std = statistics.pstdev(accs) if len(accs) > 1 else 0.0
    print(f"\n[{args.dataset}] {args.model.upper()} clean test acc: "
          f"{mean*100:.2f} +/- {std*100:.2f}  (n={args.runs} seeds)")

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "model": args.model,
        "config": {k: getattr(args, k) for k in
                   ["hidden", "dropout", "epochs", "lr", "weight_decay", "runs", "seed", "device"]},
        "dataset_stats": stats,
        "test_acc_mean": mean,
        "test_acc_std": std,
        "test_acc_per_seed": accs,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
