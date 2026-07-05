"""GNN backbones. GCN is primary; GAT/GraphSAGE are stretch (IV3 in the brief)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """2-layer GCN (Kipf & Welling 2017 configuration).

    hidden=16, dropout=0.5, with the standard symmetric-normalized propagation.
    `cached=True` is safe for transductive single-graph training and speeds up
    repeated forward passes on the same edge_index.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
        cached: bool = True,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=cached)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=cached)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # raw logits; use cross_entropy in the training loop


def build_model(name: str, in_channels: int, hidden_channels: int, out_channels: int, **kw) -> nn.Module:
    name = name.lower()
    if name == "gcn":
        return GCN(in_channels, hidden_channels, out_channels, **kw)
    raise ValueError(f"unknown model '{name}' (only 'gcn' wired for Task 0)")
