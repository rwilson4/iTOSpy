"""
matching.py

Implements zeta, computep, makenetwork, and makematch using numpy and networkx.
"""

import numpy as np
import networkx as nx
from math import comb


def zeta(bigN, n, m, g):
    if m < 0 or n < 0:
        return 0.0
    low = max(0, m + n - bigN)
    high = min(n, m)
    a = np.arange(low, high + 1)
    term = [comb(m, ai) * comb(bigN - m, n - ai) * g**ai for ai in a]
    return sum(term)


def computep(bigN, n, m, g):
    denom = zeta(bigN, n, m, g)
    return {
        "p1": g * zeta(bigN - 1, n - 1, m - 1, g) / denom,
        "p0": zeta(bigN - 1, n - 1, m, g) / denom,
        "p11": g**2 * zeta(bigN - 2, n - 2, m - 2, g) / denom,
        "p10": g * zeta(bigN - 2, n - 2, m - 1, g) / denom,
        "p01": g * zeta(bigN - 2, n - 2, m - 1, g) / denom,
        "p00": zeta(bigN - 2, n - 2, m, g) / denom,
    }


def makenetwork(costL, costR, ncontrols=1, controlcosts=None):
    m, k = costL.shape
    if controlcosts is None:
        controlcosts = np.zeros(k)
    G = nx.DiGraph()
    for i in range(m):
        G.add_edge("source", f"T{i}", capacity=1, weight=0)
    for j in range(k):
        G.add_edge(f"C{j}", "sink", capacity=ncontrols, weight=controlcosts[j])
    for i in range(m):
        for j in range(k):
            cost = costL[i, j] + costR[i, j]
            G.add_edge(f"T{i}", f"C{j}", capacity=1, weight=int(round(cost * 1e6)))
    return G


def makematch(dat, costL, costR, ncontrols=1, controlcosts=None, solver="networkx"):
    if solver != "networkx":
        raise ValueError("Only 'networkx' solver is supported in Python version")
    G = makenetwork(costL, costR, ncontrols, controlcosts)
    demand = {}
    for node in G.nodes:
        if node == "source":
            demand[node] = -sum(1 for n in G.successors(node))
        elif node == "sink":
            demand[node] = -sum(1 for n in G.predecessors(node))
        elif node.startswith("T"):
            demand[node] = 1
        elif node.startswith("C"):
            demand[node] = 0
    flow = nx.network_simplex(G, demand=demand)
    return flow
