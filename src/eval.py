"""Evaluation metrics. Task 0: clean accuracy on a mask.

Robust accuracy / robustness gap (clean - robust) get added in metrics.py once
the attack pipeline exists (Task 2). Keeping accuracy here so models.py and
train.py stay focused.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    """Fraction of correctly classified nodes within `mask`."""
    pred = logits[mask].argmax(dim=-1)
    correct = (pred == labels[mask]).sum().item()
    return correct / int(mask.sum())
