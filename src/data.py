"""Dataset loaders for the over-squashing x robustness study.

Task 0 only needs Planetoid node-classification graphs (Cora, Citeseer).
Perturbation / attack hooks are added later (Task 1+) in attacks.py.
"""
from __future__ import annotations

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid


def load_planetoid(name: str = "Cora", root: str = "data", normalize: bool = True):
    """Load a Planetoid dataset (Cora/Citeseer/PubMed).

    Uses the public (Kipf) split that ships with the dataset, so numbers are
    comparable to published GCN baselines. Row-normalizes features by default,
    which is the standard preprocessing for the ~81% Cora GCN result.

    Returns (dataset, data). `data` holds train/val/test masks.
    """
    transform = T.NormalizeFeatures() if normalize else None
    dataset = Planetoid(root=f"{root}/{name}", name=name, transform=transform)
    data = dataset[0]
    return dataset, data


def describe(data, dataset) -> dict:
    """Confirm dataset stats at load time (Open Question #4 in the brief)."""
    stats = {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),            # directed count (PyG stores both directions)
        "num_undirected_edges": int(data.num_edges // 2),
        "num_features": int(dataset.num_features),
        "num_classes": int(dataset.num_classes),
        "train_nodes": int(data.train_mask.sum()),
        "val_nodes": int(data.val_mask.sum()),
        "test_nodes": int(data.test_mask.sum()),
    }
    return stats
