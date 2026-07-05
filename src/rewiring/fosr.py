"""FoSR: First-Order Spectral Rewiring (the ADD arm).

Core routine ported verbatim from Karhadkar et al., github.com/kedar2/FoSR
(preprocessing/fosr.py) -- only the unused torch_geometric/networkx imports are
dropped; the algorithm is unchanged. It greedily ADDS edges that maximize the
spectral gap, connecting nodes on opposite ends of an approximate Fiedler vector
(maintained by power iteration). The routine is task-agnostic, so applying it to a
single Cora graph for node classification needs only an adjacency<->edge_index wrapper
(`fosr_add`); FoSR's own run_node_classification.py calls edge_rewire the same way.
"""
from __future__ import annotations

from math import inf

import numpy as np
import scipy.sparse as sp
from numba import jit


@jit(nopython=True)
def choose_edge_to_add(x, edge_index, degrees):
    # chooses edge (u, v) to add which minimizes y[u]*y[v]
    n = x.size
    m = edge_index.shape[1]
    y = x / ((degrees + 1) ** 0.5)
    products = np.outer(y, y)
    for i in range(m):
        u = edge_index[0, i]
        v = edge_index[1, i]
        products[u, v] = inf
    for i in range(n):
        products[i, i] = inf
    smallest_product = np.argmin(products)
    return (smallest_product % n, smallest_product // n)


@jit(nopython=True)
def compute_degrees(edge_index, num_nodes=None):
    if num_nodes is None:
        num_nodes = np.max(edge_index) + 1
    degrees = np.zeros(num_nodes)
    m = edge_index.shape[1]
    for i in range(m):
        degrees[edge_index[0, i]] += 1
    return degrees


@jit(nopython=True)
def add_edge(edge_index, u, v):
    new_edge = np.array([[u, v], [v, u]])
    return np.concatenate((edge_index, new_edge), axis=1)


@jit(nopython=True)
def adj_matrix_multiply(edge_index, x):
    # given edge_index, computes Ax for the corresponding adjacency matrix A
    n = x.size
    y = np.zeros(n)
    m = edge_index.shape[1]
    for i in range(m):
        u = edge_index[0, i]
        v = edge_index[1, i]
        y[u] += x[v]
    return y


@jit(nopython=True)
def _edge_rewire(edge_index, edge_type, x=None, num_iterations=50, initial_power_iters=50):
    n = np.max(edge_index) + 1
    if x is None:
        x = 2 * np.random.random(n) - 1
    degrees = compute_degrees(edge_index, num_nodes=n)
    for i in range(initial_power_iters):
        x = x - x.dot(degrees ** 0.5) * (degrees ** 0.5) / sum(degrees)
        y = x + adj_matrix_multiply(edge_index, x / (degrees ** 0.5)) / (degrees ** 0.5)
        x = y / np.linalg.norm(y)
    for I in range(num_iterations):
        i, j = choose_edge_to_add(x, edge_index, degrees=degrees)
        edge_index = add_edge(edge_index, i, j)
        degrees[i] += 1
        degrees[j] += 1
        edge_type = np.append(edge_type, 1)
        edge_type = np.append(edge_type, 1)
        x = x - x.dot(degrees ** 0.5) * (degrees ** 0.5) / sum(degrees)
        y = x + adj_matrix_multiply(edge_index, x / (degrees ** 0.5)) / (degrees ** 0.5)
        x = y / np.linalg.norm(y)
    return edge_index, edge_type, x


def edge_rewire(edge_index, x=None, edge_type=None, num_iterations=50, initial_power_iters=5):
    m = edge_index.shape[1]
    if x is None:
        n = np.max(edge_index) + 1
        x = 2 * np.random.random(n) - 1
    if edge_type is None:
        edge_type = np.zeros(m, dtype=np.int64)
    return _edge_rewire(edge_index, edge_type=edge_type, x=x,
                        num_iterations=num_iterations, initial_power_iters=initial_power_iters)


def fosr_add(adj, num_add: int, seed: int = 0):
    """Add `num_add` spectral-gap-maximizing edges to `adj`; return symmetric binary csr.

    Wrapper around FoSR's edge_rewire for node-classification on a single graph:
    csr adjacency -> edge_index (both directions) -> rewire -> csr adjacency.
    """
    adj = sp.csr_matrix(adj)
    n = adj.shape[0]
    coo = adj.tocoo()
    edge_index = np.vstack((coo.row, coo.col)).astype(np.int64)  # symmetric => both dirs present

    np.random.seed(seed)  # controls the random initial Fiedler estimate (reproducibility)
    new_ei, _, _ = edge_rewire(edge_index, num_iterations=int(num_add))

    r, c = new_ei[0], new_ei[1]
    A = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    A = A + A.T
    A.data[:] = 1.0          # binary
    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocsr()
