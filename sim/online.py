"""
Fixed-horizon chronological evaluation: assign accounts from history, measure
on a later suffix.

Workload-driven allocators are normally scored on the same transactions they
partitioned. That overstates them, because a deployed system must place an
account *before* seeing what it will do. We therefore split the trace in
on-chain order: a prefix builds the allocation, and the disjoint suffix is
scored.  The experiment knows the suffix length only to set a fixed placement
horizon; payloads, receiver placements, and loads are revealed sequentially.

In that setting the two signals differ in availability. An account absent from
the history has no graph position at all, but the asset it is transferring is
visible in the very transaction being routed. Assignment happens once, at an
account's first appearance, so the region remains a function of the sender and
sender locality is preserved.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .chain import Transaction
from .partition import (
    STRATEGIES,
    _balance_groups,
    load_balance,
    semantic_locality,
    sender_locality_violations,
)


def split(txs: Sequence[Transaction], frac: float = 0.7):
    k = int(len(txs) * frac)
    return list(txs[:k]), list(txs[k:])


def _history_regions(history: Sequence[Transaction], n_regions: int,
                     seed: int = 0) -> Tuple[Dict[int, int], Dict[int, np.ndarray]]:
    """Louvain communities of the history graph, packed into balanced regions."""
    import networkx as nx

    G = nx.Graph()
    weight: Dict[int, int] = {}
    for t in history:
        weight[t.sender] = weight.get(t.sender, 0) + 1
        if G.has_edge(t.sender, t.receiver):
            G[t.sender][t.receiver]["weight"] += 1
        else:
            G.add_edge(t.sender, t.receiver, weight=1)

    if not G.number_of_nodes():
        return {}, {}

    comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
    groups = {i: list(c) for i, c in enumerate(comms)}
    return _balance_groups(groups, weight, n_regions), {}


def _token_affinity(history: Sequence[Transaction], region_of: Dict[int, int],
                    n_regions: int, n_cat: int) -> np.ndarray:
    """How strongly each region is associated with each asset, from history."""
    aff = np.zeros((n_regions, n_cat))
    for t in history:
        r = region_of.get(t.sender)
        if r is not None:
            aff[r, t.category] += 1
    tot = aff.sum(axis=0, keepdims=True)
    tot[tot == 0] = 1
    return aff / tot


def run_online(txs: Sequence[Transaction], n_regions: int, mode: str,
               frac: float = 0.7, seed: int = 0,
               n_cat: int | None = None) -> Dict[str, float]:
    """Assign from history, score on the future.

    mode:
      hash    every account by address, the stateless baseline
      graph   history communities; accounts unseen in history fall back to hash
      token   history communities; unseen accounts routed by the asset they are
              transferring at first appearance (ours)
    """
    history, future = split(txs, frac)
    if n_cat is None:
        n_cat = max(t.category for t in txs) + 1

    region_of: Dict[int, int] = {}
    if mode in ("graph", "token"):
        region_of, _ = _history_regions(history, n_regions, seed)

    aff = (_token_affinity(history, region_of, n_regions, n_cat)
           if mode in ("token", "recv_token") else None)

    load = np.zeros(n_regions)
    for t in history:
        r = region_of.get(t.sender)
        if r is not None:
            load[r] += 1

    assigned = dict(region_of)
    cold_new = 0

    def place(t: Transaction) -> int:
        """Assign at first appearance only, keeping the key sender-derived."""
        r = assigned.get(t.sender)
        if r is not None:
            return r

        if mode in ("recv", "recv_token"):
            # The counterparty of the very first transfer is a strong hint: if
            # it is already placed, joining its region makes that transfer local.
            rr = assigned.get(t.receiver)
            if rr is not None:
                r = rr
        if r is None and mode in ("token", "recv_token") and aff is not None:
            # Otherwise prefer the region already handling this asset, with a
            # mild load penalty so no region absorbs a popular token.
            score = aff[:, t.category] - 0.25 * (load / (load.mean() + 1e-9))
            r = int(np.argmax(score))
        if r is None:
            r = t.sender % n_regions

        assigned[t.sender] = r
        load[r] += 1
        return r

    cross = 0
    eligible = 0
    unseen = 0
    for t in future:
        if t.sender not in region_of:
            unseen += 1
        rs = place(t)
        rr = assigned.get(t.receiver)
        if rr is not None:
            eligible += 1
            cross += (rs != rr)

    counts = np.bincount([assigned[t.sender] for t in future],
                         minlength=n_regions).astype(float)
    mean = counts.mean() if counts.size else 0.0

    return {
        "mode": mode,
        "n_regions": n_regions,
        "future_tx": len(future),
        "unseen_sender_frac": unseen / max(len(future), 1),
        "cross_region_frac": cross / max(eligible, 1),
        "interaction_coverage": eligible / max(len(future), 1),
        "load_max_over_mean": float(counts.max() / mean) if mean > 0 else 0.0,
    }


def run_temporal_allocator(
    txs: Sequence[Transaction], n_regions: int, strategy: str,
    frac: float = 0.7, seed: int = 0, capacity: float = 1.15,
    semantic_weight: float = 1.0, graph_weight: float = 0.15,
    load_weight: float = 0.20,
    horizon_size: int | None = None,
) -> Dict[str, float]:
    """Train on a prefix and replay a disjoint suffix chronologically.

    Known accounts keep the mapping learned from history.  A cold-start account
    is placed exactly once using its first measured payload, the already-known
    receiver when available, and current test load; all later transactions from
    that account reuse the pinned mapping.  No future payload contributes to the
    training profile or region prototypes.  ``horizon_size`` is a predefined
    load-planning horizon; reported experiments set it to the suffix length.
    """
    history, future = split(txs, frac)
    if not history or not future:
        raise ValueError("temporal split must leave non-empty history and future")

    n_users = max(max(t.sender, t.receiver) for t in txs) + 1
    dim = len(txs[0].features)
    acc = np.zeros((n_users, dim)); cnt = np.zeros(n_users)
    for t in history:
        acc[t.sender] += np.asarray(t.features); cnt[t.sender] += 1
    safe_cnt = cnt.copy(); safe_cnt[safe_cnt == 0] = 1
    history_features = acc / safe_cnt[:, None]

    key = {
        "HASH": "ADDR-MOD",
        "SENDER-KMEANS": "SENDER-KMEANS",
        "SENDER-KMEANS-BAL": "SENDER-KMEANS-BAL",
        "ALPS": "ALPS",
    }[strategy]
    h_regions = STRATEGIES[key](
        history, n_regions, sender_features=history_features, seed=seed,
        over=capacity, semantic_weight=semantic_weight,
        graph_weight=graph_weight, load_weight=load_weight)
    assigned: Dict[int, int] = {}
    for t, r in zip(history, h_regions):
        assigned[t.sender] = int(r)

    # Historical semantic prototypes, weighted by transaction count.
    prototypes = np.zeros((n_regions, dim))
    proto_w = np.zeros(n_regions)
    for a in np.flatnonzero(cnt):
        r = assigned.get(int(a))
        if r is None:
            continue
        prototypes[r] += acc[a]
        proto_w[r] += cnt[a]
    for r in range(n_regions):
        if proto_w[r] > 0:
            prototypes[r] /= proto_w[r]

    future_load = np.zeros(n_regions)
    cold_senders = set()
    regions: List[int] = []
    cross = 0
    eligible = 0
    if horizon_size is None:
        horizon_size = len(future)
    if horizon_size <= 0:
        raise ValueError("horizon_size must be positive")
    cap = capacity * horizon_size / n_regions

    for t in future:
        r = assigned.get(t.sender)
        if r is None:
            cold_senders.add(t.sender)
            if strategy == "HASH":
                r = t.sender % n_regions
            else:
                x = np.asarray(t.features)
                sem = np.linalg.norm(prototypes - x, axis=1) ** 2
                sem /= max(float(np.median(sem[sem > 1e-12]))
                           if np.any(sem > 1e-12) else 1.0, 1e-12)
                rr = assigned.get(t.receiver)
                edge = np.array([0.0 if rr is None or q == rr else 1.0
                                 for q in range(n_regions)])
                load_penalty = (future_load / max(cap, 1.0)) ** 2
                if strategy == "SENDER-KMEANS":
                    score = sem
                    candidates = np.arange(n_regions)
                elif strategy == "SENDER-KMEANS-BAL":
                    score = sem + load_weight * load_penalty
                    feasible = np.flatnonzero(future_load + 1 <= cap + 1e-9)
                    candidates = (feasible if feasible.size
                                  else np.arange(n_regions))
                else:  # ALPS: semantic, counterparty and capacity jointly
                    score = (semantic_weight * sem
                             + graph_weight * edge
                             + load_weight * load_penalty)
                    feasible = np.flatnonzero(future_load + 1 <= cap + 1e-9)
                    candidates = (feasible if feasible.size
                                  else np.arange(n_regions))
                r = int(candidates[np.argmin(score[candidates])])
            assigned[t.sender] = int(r)

        regions.append(int(r))
        future_load[r] += 1
        rr = assigned.get(t.receiver)
        if rr is not None:
            eligible += 1
            cross += rr != r

    sem = semantic_locality(future, regions, n_regions)
    lb = load_balance(regions, n_regions)
    viol = sender_locality_violations(future, regions)
    return {
        "mode": strategy,
        "n_regions": n_regions,
        "split": frac,
        "seed": seed,
        "history_tx": len(history),
        "future_tx": len(future),
        "cold_senders": len(cold_senders),
        "cold_sender_tx_frac": sum(t.sender in cold_senders for t in future)
                               / len(future),
        "cross_region_frac": cross / max(eligible, 1),
        "interaction_coverage": eligible / len(future),
        "unknown_receiver_frac": 1.0 - eligible / len(future),
        "r90": sem["r90"],
        "purity": sem["purity"],
        "load_max_over_mean": lb["max_over_mean"],
        "split_senders": viol["split_senders"],
        "semantic_weight": semantic_weight if strategy == "ALPS" else "",
        "graph_weight": graph_weight if strategy == "ALPS" else "",
        "load_weight": load_weight if strategy == "ALPS" else "",
    }
