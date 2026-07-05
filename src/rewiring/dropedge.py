"""DropEdge as a one-shot random edge-removal preprocessor (the random-removal baseline).

We use one-shot removal of a fixed edge budget (rather than the canonical per-epoch
stochastic DropEdge) so it is a like-for-like RANDOM control against the principled
spectral pruning (ProxyDelete), which also removes a fixed budget once. This isolates
"which edges you remove" from "how many". Note this choice in the writeup.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def random_edge_removal(adj, n_remove: int = None, rate: float = None, seed: int = 0):
    """Remove `n_remove` undirected edges uniformly at random (symmetric).

    Provide either an absolute count `n_remove` or a `rate` (fraction of edges).
    Returns a symmetric csr adjacency.
    """
    adj = sp.csr_matrix(adj)
    upper = sp.triu(adj, k=1).tocoo()
    rows, cols = upper.row, upper.col
    n_edges = len(rows)

    if n_remove is None:
        assert rate is not None, "give n_remove or rate"
        n_remove = int(rate * n_edges)
    n_remove = min(n_remove, n_edges)

    rng = np.random.default_rng(seed)
    keep = np.ones(n_edges, dtype=bool)
    drop_idx = rng.choice(n_edges, size=n_remove, replace=False)
    keep[drop_idx] = False

    r, c = rows[keep], cols[keep]
    data = np.ones(len(r))
    n = adj.shape[0]
    half = sp.csr_matrix((data, (r, c)), shape=(n, n))
    sym = half + half.T
    sym.data[:] = 1.0  # binary adjacency
    return sym.tocsr()
