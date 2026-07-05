"""Jacobian over-squashing probe (RQ1/RQ3).

Over-squashing means a node v's output is insensitive to a distant node u's input.
Following Di Giovanni et al. (2023), we measure that sensitivity by the Jacobian norm
||d h_v / d x_u||_F and study how it decays with the graph distance d(u,v): a sharp
decay is the signature of over-squashing. We need an L-layer GCN (depth >= distance)
for distant nodes to be reachable at all, so this probe uses a deeper GCN than the
2-layer backbone used for classification.

Scoped version: a handful of target nodes v, sensitivity to all u via autograd,
bucketed by distance. RQ1 compares the curve on the clean vs the poisoned graph.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class DeepGCN(nn.Module):
    """L-layer GCN for the sensitivity probe (no dropout at probe time)."""

    def __init__(self, in_channels, hidden, out_channels, num_layers=6):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden, cached=False))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden, hidden, cached=False))
        self.convs.append(GCNConv(hidden, out_channels, cached=False))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x


def adj_to_edge_index(adj):
    coo = sp.csr_matrix(adj).tocoo()
    return torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)


def train_probe_gcn(x, edge_index, y, idx_train, *, num_layers=6, hidden=32,
                    epochs=150, lr=0.01, weight_decay=5e-4, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DeepGCN(x.shape[1], hidden, int(y.max()) + 1, num_layers=num_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[idx_train], y[idx_train])
        loss.backward()
        opt.step()
    model.eval()
    return model


def jacobian_sensitivity(model, x, edge_index, targets):
    """For each target node v, return an array s[v] of length n where
    s[v][u] = ||d logits_v / d x_u||_F (Frobenius over classes x input features)."""
    x = x.clone().detach().requires_grad_(True)
    logits = model(x, edge_index)            # n x C
    n, C = logits.shape
    out = {}
    for v in targets:
        sq = torch.zeros(n)
        for c in range(C):
            if x.grad is not None:
                x.grad.zero_()
            g, = torch.autograd.grad(logits[v, c], x, retain_graph=True)
            sq = sq + (g ** 2).sum(dim=1)    # per-node squared grad of class c
        out[int(v)] = torch.sqrt(sq).detach().numpy()
    return out


def sensitivity_by_distance(adj, sens, targets, max_dist):
    """Bucket Jacobian sensitivities by graph distance d(u,v); return {d: [values]}."""
    g = nx.from_scipy_sparse_array(sp.csr_matrix(adj))
    buckets = {d: [] for d in range(1, max_dist + 1)}
    for v in targets:
        dist = nx.single_source_shortest_path_length(g, v, cutoff=max_dist)
        sv = sens[int(v)]
        for u, d in dist.items():
            if 1 <= d <= max_dist and u != v:
                buckets[d].append(float(sv[u]))
    return buckets
