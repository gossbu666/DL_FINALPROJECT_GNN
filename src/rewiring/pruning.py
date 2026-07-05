"""Spectral graph pruning -- ProxyDelete (the spectral REMOVE arm).

Ported from Jamadandi et al., github.com/RelationalML/SpectralPruningBraess
(NodeClassification/rewiring/{spectral_utils,MinGapKupdates}.py), inlined into one
file and with the `from rewiring.spectral_utils import *` dependency resolved. The
algorithm is unchanged: it iteratively DELETES the edge that a first-order proxy says
most increases the spectral gap (connecting bottleneck structure), maintaining
connectivity. Pure networkx/scipy/numpy. We add a thin `proxydelete_remove` wrapper
(csr adjacency <-> networkx graph) for our node-classification pipeline.
"""
from __future__ import annotations

import random
import warnings

import networkx as nx
import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")

eps = 1e-6

# ----------------------------------------------------------------------------- spectral utils
def add_self_loops(g):
    g.add_edges_from([(i, i) for i in range(len(g.nodes))])
    return g


def obtain_Lnorm(g):
    adj = nx.adjacency_matrix(g)
    deg = np.array(adj.sum(axis=1)).flatten()
    D_sqrt_inv = sp.diags(np.power(deg + eps, -0.5))
    L_norm = (sp.eye(adj.shape[0]) - D_sqrt_inv @ adj @ D_sqrt_inv)
    return deg, L_norm


def update_Lnorm_deletion(u, v, L_norm, deg):
    L_norm[u, :] *= np.sqrt(deg[u] / (deg[u] - 1))
    L_norm[:, u] = L_norm[u, :].T
    L_norm[v, :] *= np.sqrt(deg[v] / (deg[v] - 1))
    L_norm[:, v] = L_norm[v, :].T
    deg[u] -= 1
    deg[v] -= 1
    L_norm[u, u] = 1 - 1 / deg[u]
    L_norm[v, v] = 1 - 1 / deg[v]
    L_norm[u, v] = 0
    L_norm[v, u] = 0
    return deg, L_norm


def update_Lnorm_addition(u, v, L_norm, deg):
    L_norm[u, :] *= np.sqrt(deg[u] / (deg[u] + 1))
    L_norm[:, u] = L_norm[u, :].T
    L_norm[v, :] *= np.sqrt(deg[v] / (deg[v] + 1))
    L_norm[:, v] = L_norm[v, :].T
    L_norm[u, v] = -1 / np.sqrt((deg[u] + 1) * (deg[v] + 1))
    L_norm[v, u] = L_norm[u, v]
    deg[u] += 1
    deg[v] += 1
    L_norm[u, u] = 1 - 1 / deg[u]
    L_norm[v, v] = 1 - 1 / deg[v]
    return deg, L_norm


def spectral_gap(g, params=None):
    deg, L_norm = obtain_Lnorm(g)
    try:  # sparse shift-invert if the factorization is non-singular
        vals, vecs = sp.linalg.eigsh(L_norm, k=2, sigma=0.0, which="LM")
        vecs = np.divide(vecs, np.sqrt(deg[:, np.newaxis]))
        vecs = vecs / np.linalg.norm(vecs, axis=1)[:, np.newaxis]
    except Exception:  # dense fallback
        dense_Lnorm = nx.normalized_laplacian_matrix(g).todense()
        vals, vecs = np.linalg.eigh(dense_Lnorm)
        vecs = np.divide(vecs, np.sqrt(deg[:, np.newaxis]))
        vecs = vecs / np.linalg.norm(vecs, axis=1)[:, np.newaxis]
    return vals[1], vecs, deg, L_norm


# ----------------------------------------------------------------------------- proxy-delete ranking
def gap_from_proxy(edge, gap, vecs, delta_w):
    i, j = edge
    return delta_w * ((vecs[i, 1] - vecs[j, 1]) ** 2 - gap * (vecs[i, 1] ** 2 + vecs[j, 1] ** 2))


proxy_delete_score = lambda g, edge, gap, vecs: (gap_from_proxy(edge, gap, vecs, -1), -1)
rank_by_proxy_delete_min = lambda g, gap, vecs: rank_by(g, gap, vecs, proxy_delete_score, "delete")


def rank_by(g, gap, vecs, score_method, add_or_delete):
    if add_or_delete == "add":
        edges = list(nx.non_edges(g))
        edges = random.sample(edges, 1500)
    elif add_or_delete == "delete":
        edges = list(g.edges - nx.selfloop_edges(g))
    else:
        edges = list(nx.non_edges(g)) + list(g.edges - nx.selfloop_edges(g))
    edge_dgap_mapping = dict()
    for i, j in edges:
        edge_dgap_mapping[(i, j)] = score_method(g, (i, j), gap, vecs)
    return list(edge_dgap_mapping.items())


def modify_k_edges(g, ranking_method, gap, vecs, deg, L_norm, k=1):
    best_edges = ranking_method(g, gap, vecs)
    counter = 0
    for _ in range(len(best_edges)):
        (s, t), (dgap, pm) = min(best_edges, key=lambda x: x[1][0])
        best_edges.remove(((s, t), (dgap, pm)))
        if pm == 1:
            g.add_edge(s, t)
            deg, L_norm = update_Lnorm_addition(s, t, L_norm, deg)
        else:
            g.remove_edge(s, t)
            if not nx.is_connected(g):
                g.add_edge(s, t)
                continue
            deg, L_norm = update_Lnorm_deletion(s, t, L_norm, deg)
        counter += 1
        if counter == k:
            return True, g, deg, L_norm
    return False, g, deg, L_norm


def min_and_update_edges(g, ranking_method, ranking_name, updating_period=1, max_iter=np.inf):
    g = add_self_loops(g)
    deg, L_norm = obtain_Lnorm(g)
    gap, vecs, _, _ = spectral_gap(g)
    counter = 0
    modified = True
    while modified and counter < max_iter:
        modified, g, deg, L_norm = modify_k_edges(g, ranking_method, gap, vecs, deg, L_norm, updating_period)
        gap, vecs, _, _ = spectral_gap(g)
        counter += 1
        if len(g.edges) == 0 or not modified:
            break
    return g


# ----------------------------------------------------------------------------- wrapper for our pipeline
def proxydelete_remove(adj, num_delete: int, seed: int = 0):
    """Delete `num_delete` spectral-gap-maximizing edges; return symmetric binary csr.

    Deterministic given `adj` (the delete ranking scans all edges, no sampling); `seed`
    is set only for parity with the other arms.
    """
    random.seed(seed)
    np.random.seed(seed)
    adj = sp.csr_matrix(adj)
    n = adj.shape[0]
    g = nx.from_scipy_sparse_array(adj)
    g.remove_edges_from(list(nx.selfloop_edges(g)))

    g = min_and_update_edges(g, rank_by_proxy_delete_min, "proxydeletemin",
                             updating_period=1, max_iter=int(num_delete))

    g.remove_edges_from(list(nx.selfloop_edges(g)))  # strip the self-loops the routine added
    A = nx.to_scipy_sparse_array(g, nodelist=range(n), format="csr", dtype=float)
    A.data[:] = 1.0
    return A
