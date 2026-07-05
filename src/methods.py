"""Method registry: each method is an adjacency->adjacency transform.

method(features, adj, ctx) -> adj'   where ctx carries shared params
(nfeat, nclass, device, remove_budget, jaccard_threshold, seed).

This is the single place the add/remove axis is encoded. FoSR (add) and ProxyDelete
(remove) are wired in Tasks 3 and 5; until then they raise NotImplementedError so the
grid fails loudly rather than silently skipping an arm.
"""
from __future__ import annotations

import scipy.sparse as sp

from .defenses import jaccard_prune
from .rewiring.dropedge import random_edge_removal


def _vanilla(features, adj, ctx):
    return sp.csr_matrix(adj)


def _dropedge(features, adj, ctx):
    return random_edge_removal(adj, n_remove=ctx["remove_budget"], seed=ctx["seed"])


def _jaccard(features, adj, ctx):
    return jaccard_prune(
        features, adj, threshold=ctx["jaccard_threshold"],
        nfeat=ctx["nfeat"], nclass=ctx["nclass"], device=ctx["device"],
    )


def _fosr(features, adj, ctx):
    from .rewiring.fosr import fosr_add
    return fosr_add(adj, num_add=ctx["add_budget"], seed=ctx["seed"])


def _proxydelete(features, adj, ctx):
    from .rewiring.pruning import proxydelete_remove
    return proxydelete_remove(adj, num_delete=ctx["remove_budget"], seed=ctx["seed"])


REGISTRY = {
    "gcn": _vanilla,          # baseline (no transform)
    "dropedge": _dropedge,    # random removal
    "jaccard": _jaccard,      # similarity-based removal (defense reference)
    "fosr": _fosr,            # spectral add  (Task 3)
    "proxydelete": _proxydelete,  # spectral remove (Task 5)
}


def apply_method(name, features, adj, ctx):
    if name not in REGISTRY:
        raise KeyError(f"unknown method '{name}'. known: {list(REGISTRY)}")
    return REGISTRY[name](features, adj, ctx)
