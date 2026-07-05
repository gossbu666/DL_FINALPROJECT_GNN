"""Training loop for transductive node classification.

Selects the model at the epoch with the best validation accuracy (standard for
Cora-style benchmarks) and reports the corresponding test accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .eval import accuracy


@dataclass
class TrainResult:
    test_acc: float
    val_acc: float
    best_epoch: int


def train_node_classifier(
    model: torch.nn.Module,
    data,
    *,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    device: str = "cpu",
    verbose: bool = False,
) -> TrainResult:
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = 0.0
    best_test = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            val_acc = accuracy(logits, data.y, data.val_mask)
            test_acc = accuracy(logits, data.y, data.test_mask)

        # model selection on validation accuracy
        if val_acc > best_val:
            best_val, best_test, best_epoch = val_acc, test_acc, epoch

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d} | loss {loss.item():.4f} | val {val_acc:.4f} | test {test_acc:.4f}")

    return TrainResult(test_acc=best_test, val_acc=best_val, best_epoch=best_epoch)
