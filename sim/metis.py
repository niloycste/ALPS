"""Sender-local weighted METIS baseline.

This is a general graph-partitioning comparator, not part of ALPS.  Vertices
are active senders, vertex weights are submitted-call counts, and undirected
edge weights are observed interactions between active senders.  Mapping each
transaction through its sender's vertex keeps the allocation sender-local.

``pymetis`` is imported lazily so the core artifact remains usable when the
optional baseline dependency is not installed.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

from .chain import Transaction


def sender_interaction_graph(
    txs: Sequence[Transaction],
) -> tuple[list[int], list[list[int]], list[int], list[int]]:
    """Return deterministic METIS arrays for the active-sender graph.

    Receiver-only accounts are excluded, matching ALPS's graph and the
    conditional active-account interaction-cut metric.
    """
    counts: Dict[int, int] = {}
    for tx in txs:
        counts[tx.sender] = counts.get(tx.sender, 0) + 1
    nodes = sorted(counts)
    index = {account: i for i, account in enumerate(nodes)}

    edges: Dict[tuple[int, int], int] = defaultdict(int)
    for tx in txs:
        if tx.receiver not in index or tx.receiver == tx.sender:
            continue
        u, v = sorted((index[tx.sender], index[tx.receiver]))
        edges[(u, v)] += 1

    weighted_neighbors: list[list[tuple[int, int]]] = [
        [] for _ in nodes
    ]
    for (u, v), weight in edges.items():
        weighted_neighbors[u].append((v, weight))
        weighted_neighbors[v].append((u, weight))

    adjacency: list[list[int]] = []
    edge_weights: list[int] = []
    for neighbors in weighted_neighbors:
        ordered = sorted(neighbors)
        adjacency.append([neighbor for neighbor, _ in ordered])
        edge_weights.extend(weight for _, weight in ordered)
    vertex_weights = [counts[account] for account in nodes]
    return nodes, adjacency, vertex_weights, edge_weights


def metis_sender(
    txs: Sequence[Transaction], n_regions: int, seed: int = 0, **kw,
) -> List[int]:
    """Partition active senders with weighted k-way METIS.

    METIS minimises weighted edge cut while balancing transaction-weighted
    vertices.  It has no semantic-retrieval objective and no ALPS-style exact
    hard cap, so achieved load is measured rather than assumed.
    """
    if n_regions < 1:
        raise ValueError("n_regions must be positive")
    if not txs:
        return []

    nodes, adjacency, vertex_weights, edge_weights = sender_interaction_graph(
        txs)
    if len(nodes) < n_regions:
        raise ValueError(
            "METIS-SENDER requires at least one active sender per region")

    try:
        import pymetis
    except ImportError as exc:
        raise RuntimeError(
            "METIS-SENDER requires the optional dependency 'pymetis'; "
            "install requirements-metis.txt"
        ) from exc

    options = pymetis.Options(seed=int(seed))
    _, membership = pymetis.part_graph(
        n_regions,
        adjacency=adjacency,
        vweights=vertex_weights,
        eweights=edge_weights,
        options=options,
    )
    region_of = {account: int(region)
                 for account, region in zip(nodes, membership)}
    return [region_of[tx.sender] for tx in txs]
