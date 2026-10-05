"""
Reimplementation of the TxAllo account-allocation algorithm as a baseline.

Zhang, Pan and Yu, "TxAllo: Dynamic Transaction Allocation in Sharded
Blockchain Systems", ICDE 2023.

Scope, stated precisely because a partial reimplementation that is presented as
the original would be misleading:

  included   the allocation algorithm itself -- Louvain initialisation on the
             account interaction graph, the candidate-community restriction,
             and deterministic coordinate descent on a throughput objective
             that charges cross-shard transactions a factor ``eta`` and is
             limited by the most heavily loaded shard.

  omitted    everything that is protocol rather than allocation: the sharded
             consensus itself, cross-shard settlement, broker accounts, and
             A-TxAllo's incremental re-allocation across epochs. Those change
             execution semantics and cannot be transplanted into the
             independent-barrier model without altering what is being measured.

The comparison is therefore between *allocation objectives* under one fixed
execution model, not between deployed systems. TxAllo optimises cross-shard
traffic and workload balance; it carries no payload-retrieval term, which is
the axis on which the two are expected to differ.

Being keyed on the account, the resulting partition is sender-local, so
Theorem 1 already certifies it: TxAllo is admissible, and the interesting
question is what it costs on the other metrics.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .chain import Transaction


def _interaction_graph(txs: Sequence[Transaction]):
    """Undirected weighted account graph, plus per-account transaction counts."""
    import networkx as nx

    G = nx.Graph()
    weight: Dict[int, int] = {}
    for t in txs:
        weight[t.sender] = weight.get(t.sender, 0) + 1
        if t.sender == t.receiver:
            continue
        if G.has_edge(t.sender, t.receiver):
            G[t.sender][t.receiver]["weight"] += 1
        else:
            G.add_edge(t.sender, t.receiver, weight=1)
    for a in weight:
        if a not in G:
            G.add_node(a)
    return G, weight


def _shard_workload(assign: Dict[int, int], G, weight: Dict[int, int],
                    n_regions: int, eta: float) -> np.ndarray:
    """sigma_q: intra-shard transactions plus eta times cross-shard ones."""
    load = np.zeros(n_regions)
    for a, r in assign.items():
        load[r] += weight.get(a, 0)
    for u, v, d in G.edges(data=True):
        ru, rv = assign.get(u), assign.get(v)
        if ru is None or rv is None or ru == rv:
            continue
        # A cross-shard transfer is charged to both endpoints' shards.
        load[ru] += (eta - 1.0) * d["weight"]
        load[rv] += (eta - 1.0) * d["weight"]
    return load


def txallo(
    txs: Sequence[Transaction], n_regions: int, seed: int = 0,
    eta: float = 2.0, max_iter: int = 8, **kw,
) -> List[int]:
    """Account-to-shard allocation in the style of G-TxAllo.

    ``eta`` is the workload of a cross-shard transaction relative to an
    intra-shard one; the ICDE paper sweeps it and we use its mid-range value.
    """
    import networkx as nx

    G, weight = _interaction_graph(txs)
    if G.number_of_nodes() == 0:
        return [t.sender % n_regions for t in txs]

    # --- initialisation: Louvain, then fold into exactly n_regions shards ----
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
    comms = sorted(comms, key=lambda c: -sum(weight.get(a, 0) for a in c))

    assign: Dict[int, int] = {}
    load = np.zeros(n_regions)
    for i, c in enumerate(comms):
        if i < n_regions:
            r = i
        else:
            # Small communities join the least loaded shard, as in the
            # initialisation phase of Algorithm 1.
            r = int(np.argmin(load))
        for a in c:
            assign[a] = r
        load[r] += sum(weight.get(a, 0) for a in c)

    # --- optimisation: move a node to the candidate shard with the best gain -
    nodes = sorted(G.nodes())
    for _ in range(max_iter):
        moved = False
        sigma = _shard_workload(assign, G, weight, n_regions, eta)
        for v in nodes:
            p = assign[v]
            # Candidate shards are those v actually connects to (Eq. 9);
            # an isolated node may go anywhere.
            cand = {assign[u] for u in G.neighbors(v) if u in assign}
            cand.discard(p)
            if not cand:
                cand = set(range(n_regions)) - {p}
            if not cand:
                continue

            wv = weight.get(v, 0)
            # Edge mass from v into each shard.
            to_shard = np.zeros(n_regions)
            for u in G.neighbors(v):
                ru = assign.get(u)
                if ru is not None:
                    to_shard[ru] += G[v][u]["weight"]
            total_edge = to_shard.sum()

            best, best_obj = p, None
            for q in [p] + sorted(cand):
                trial = sigma.copy()
                # v leaves p
                trial[p] -= wv
                trial[p] -= (eta - 1.0) * (total_edge - to_shard[p])
                # v joins q
                trial[q] += wv
                trial[q] += (eta - 1.0) * (total_edge - to_shard[q])
                obj = trial.max()          # throughput is set by the worst shard
                if best_obj is None or obj < best_obj - 1e-9:
                    best, best_obj = q, obj
            if best != p:
                # Update the exact workload in O(deg(v)).  Besides the source
                # and destination, moving v changes the cross-shard charge on
                # every neighbor's shard.  Recomputing from all E graph edges
                # after every move is equivalent but needlessly expensive.
                sigma[p] -= wv
                sigma[best] += wv
                factor = eta - 1.0
                for u in G.neighbors(v):
                    ru = assign[u]
                    edge = factor * G[v][u]["weight"]
                    if p != ru:
                        sigma[p] -= edge
                        sigma[ru] -= edge
                    if best != ru:
                        sigma[best] += edge
                        sigma[ru] += edge
                assign[v] = best
                moved = True
        if not moved:
            break

    return [assign.get(t.sender, t.sender % n_regions) for t in txs]
