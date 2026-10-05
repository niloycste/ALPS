"""
Region assignment strategies.

Balance safety in the independent-barrier model rests on *guarded-key
locality*: all conditional debits of one ``(account, asset)`` state key must be
assigned to one region.  Ethereum's account-wide nonce imposes the stronger
*sender locality* invariant:

    for every account a, all transactions that debit a are assigned to the
    same region.

Debits are guarded and non-commutative; credits are unconditional additions.
The sender-local strategies enforce both invariants by construction;
TX_KMEANS can violate either one.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Sequence

from sklearn.cluster import KMeans

from .chain import Transaction
from .metis import metis_sender
from .txallo import txallo


def addr_mod(txs: Sequence[Transaction], n_regions: int, **kw) -> List[int]:
    """Base paper: R(U_i) = Adr(U_i) mod n.  Sender-local by construction."""
    return [t.sender % n_regions for t in txs]


def tx_kmeans(txs: Sequence[Transaction], n_regions: int, seed: int = 0, **kw) -> List[int]:
    """Thesis under study: k-means over the payload of each transaction.

    Clusters are semantically coherent, but a sender that trades in several
    categories has its debits split across regions, breaking sender locality.
    """
    X = np.array([t.features for t in txs])
    km = KMeans(n_clusters=n_regions, n_init=10, random_state=seed).fit(X)
    return [int(c) for c in km.labels_]


def sender_kmeans(
    txs: Sequence[Transaction],
    n_regions: int,
    sender_features: np.ndarray | None = None,
    seed: int = 0,
    **kw,
) -> List[int]:
    """Proposed: k-means over each *sender's* mean payload profile.

    The region key remains a function of the sender alone, so sender locality
    holds; but senders are grouped by what they trade in, so blocks retain the
    semantic coherence that motivated the payload-clustering design.
    """
    if sender_features is None:
        n_users = max(t.sender for t in txs) + 1
        dim = len(txs[0].features)
        acc = np.zeros((n_users, dim))
        cnt = np.zeros(n_users)
        for t in txs:
            acc[t.sender] += np.asarray(t.features)
            cnt[t.sender] += 1
        cnt[cnt == 0] = 1
        sender_features = acc / cnt[:, None]

    active = np.array(sorted({t.sender for t in txs}), dtype=int)
    km = KMeans(n_clusters=n_regions, n_init=10, random_state=seed).fit(
        sender_features[active])
    region_of = {int(a): int(r) for a, r in zip(active, km.labels_)}
    return [region_of[t.sender] for t in txs]


def _balance_groups(groups: Dict[int, List[int]], weight: Dict[int, int],
                    n_regions: int) -> Dict[int, int]:
    """Greedily pack account groups into n regions of similar transaction load.

    Groups are kept intact, so the resulting map is still a function of the
    account and sender locality is preserved.
    """
    order = sorted(groups, key=lambda g: -sum(weight.get(a, 0) for a in groups[g]))
    load = [0] * n_regions
    region_of: Dict[int, int] = {}
    for g in order:
        r = int(np.argmin(load))
        w = sum(weight.get(a, 0) for a in groups[g])
        load[r] += w
        for a in groups[g]:
            region_of[a] = r
    return region_of


def sender_kmeans_balanced(
    txs: Sequence[Transaction], n_regions: int,
    sender_features: np.ndarray | None = None, seed: int = 0,
    over: float = 1.15, **kw,
) -> List[int]:
    """SENDER-KMEANS with a capacity cap, to recover load balance.

    Clusters of accounts are inherently uneven because real senders are
    heavy-tailed. Here accounts are assigned to their nearest centroid subject
    to a per-region capacity of ``over`` times the mean transaction load; when a
    region is full, the account goes to the nearest region with room. Assignment
    remains a function of the account, so sender locality is preserved.
    """
    counts: Dict[int, int] = {}
    for t in txs:
        counts[t.sender] = counts.get(t.sender, 0) + 1

    if sender_features is None:
        n_users = max(max(t.sender, t.receiver) for t in txs) + 1
        dim = len(txs[0].features)
        acc = np.zeros((n_users, dim)); cnt = np.zeros(n_users)
        for t in txs:
            acc[t.sender] += np.asarray(t.features); cnt[t.sender] += 1
        cnt[cnt == 0] = 1
        sender_features = acc / cnt[:, None]

    active = np.array(sorted(counts), dtype=int)
    km = KMeans(n_clusters=n_regions, n_init=10, random_state=seed).fit(
        sender_features[active])
    cent = km.cluster_centers_

    cap = over * sum(counts.values()) / n_regions
    load = np.zeros(n_regions)
    region_of: Dict[int, int] = {}

    # Heaviest accounts first: they are the ones that would overflow a region.
    for a in sorted(counts, key=lambda x: -counts[x]):
        f = sender_features[a] if a < len(sender_features) else np.zeros(cent.shape[1])
        for r in np.argsort(np.linalg.norm(cent - f, axis=1)):
            if load[r] + counts[a] <= cap or load[r] == 0:
                region_of[a] = int(r); load[r] += counts[a]
                break
        else:
            r = int(np.argmin(load))
            region_of[a] = r; load[r] += counts[a]

    return [region_of.get(t.sender, t.sender % n_regions) for t in txs]


def alps(
    txs: Sequence[Transaction], n_regions: int,
    sender_features: np.ndarray | None = None, seed: int = 0,
    over: float = 1.15, semantic_weight: float = 1.0,
    graph_weight: float = 0.15, load_weight: float = 0.20,
    max_iter: int = 6, return_diagnostics: bool = False, **kw,
):
    """Admissibility-preserving locality and semantic partitioning (ALPS).

    ALPS treats each sender and its complete transaction sequence as one
    indivisible allocation unit.  It jointly minimises three measured costs:

      * distance from the sender's real payload profile to a region prototype;
      * historical sender/receiver edges cut by the assignment; and
      * region load, under a hard ``over * mean`` capacity whenever feasible.

    The initial heaviest-first assignment is followed by deterministic local
    refinement.  Every move operates on a whole sender, so the safety invariant
    is hard-constrained rather than encouraged by a soft penalty.  Refinement
    uses the exact delta of one normalized global objective; a move is accepted
    only when that objective strictly falls.  Weighted-centroid updates cannot
    increase its semantic term, so the recorded objective trace is monotone.
    """
    counts: Dict[int, int] = {}
    for t in txs:
        counts[t.sender] = counts.get(t.sender, 0) + 1

    if sender_features is None:
        n_users = max(max(t.sender, t.receiver) for t in txs) + 1
        dim = len(txs[0].features)
        acc = np.zeros((n_users, dim)); cnt = np.zeros(n_users)
        for t in txs:
            acc[t.sender] += np.asarray(t.features); cnt[t.sender] += 1
        cnt[cnt == 0] = 1
        sender_features = acc / cnt[:, None]

    active = np.array(sorted(counts), dtype=int)
    X = sender_features[active]
    weights = np.array([counts[int(a)] for a in active], dtype=float)
    km = KMeans(n_clusters=n_regions, n_init=10, random_state=seed).fit(
        X, sample_weight=weights)
    centroids = km.cluster_centers_.copy()

    # Undirected historical interaction weights between active senders.
    active_set = set(int(a) for a in active)
    neighbors: Dict[int, Dict[int, float]] = {int(a): {} for a in active}
    for t in txs:
        if t.receiver not in active_set or t.receiver == t.sender:
            continue
        neighbors[t.sender][t.receiver] = (
            neighbors[t.sender].get(t.receiver, 0.0) + 1.0)
        neighbors[t.receiver][t.sender] = (
            neighbors[t.receiver].get(t.sender, 0.0) + 1.0)

    mean = sum(counts.values()) / n_regions
    target_cap = max(over * mean, max(counts.values()))
    cap = target_cap
    load = np.zeros(n_regions)
    region_of: Dict[int, int] = {}

    raw = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2) ** 2
    positive = raw[raw > 1e-12]
    semantic_scale = float(np.median(positive)) if positive.size else 1.0

    total_tx = float(sum(counts.values()))
    total_edge = max(
        sum(sum(edges.values()) for edges in neighbors.values()) / 2.0, 1.0)

    def semantic_cost(a: int, r: int) -> float:
        distance = float(np.sum((sender_features[a] - centroids[r]) ** 2))
        return counts[a] * distance / (total_tx * semantic_scale)

    def initial_cost(a: int, r: int, mapping: Dict[int, int],
                     current_load: np.ndarray) -> float:
        """Exact incremental cost against already placed senders."""
        cut = sum(w for b, w in neighbors.get(a, {}).items()
                  if b in mapping and mapping[b] != r)
        before = (current_load[r] / max(mean, 1.0)) ** 2
        after = ((current_load[r] + counts[a]) / max(mean, 1.0)) ** 2
        return (
            semantic_weight * semantic_cost(a, r)
            + graph_weight * cut / total_edge
            + load_weight * (after - before) / n_regions
        )

    def components() -> Dict[str, float]:
        semantic = sum(semantic_cost(a, region_of[a]) for a in counts)
        cut = 0.0
        for a, edges in neighbors.items():
            for b, w in edges.items():
                if a < b and region_of[a] != region_of[b]:
                    cut += w
        graph = cut / total_edge
        load_term = float(np.mean((load / max(mean, 1.0)) ** 2))
        return {
            "semantic": semantic,
            "graph": graph,
            "load": load_term,
            "total": (semantic_weight * semantic
                      + graph_weight * graph
                      + load_weight * load_term),
        }

    def move_delta(a: int, new: int) -> float:
        """Exact objective change for moving one sender to ``new``."""
        old = region_of[a]
        sem = semantic_cost(a, new) - semantic_cost(a, old)
        graph = 0.0
        for b, w in neighbors.get(a, {}).items():
            rb = region_of[b]
            graph += w * (int(new != rb) - int(old != rb))

        weight = counts[a]
        old_before = (load[old] / max(mean, 1.0)) ** 2
        new_before = (load[new] / max(mean, 1.0)) ** 2
        old_after = ((load[old] - weight) / max(mean, 1.0)) ** 2
        new_after = ((load[new] + weight) / max(mean, 1.0)) ** 2
        load_delta = (old_after + new_after - old_before - new_before)
        load_delta /= n_regions
        return (semantic_weight * sem
                + graph_weight * graph / total_edge
                + load_weight * load_delta)

    # Heaviest accounts first makes the hard capacity attainable when possible.
    for a in sorted(counts, key=lambda x: (-counts[x], x)):
        feasible = [r for r in range(n_regions)
                    if load[r] + counts[a] <= cap + 1e-9]
        candidates = feasible or [int(np.argmin(load))]
        r = min(candidates,
                key=lambda q: (initial_cost(a, q, region_of, load), q))
        region_of[a] = int(r)
        load[r] += counts[a]

    # A cost-aware greedy prefix can consume capacity needed by later senders.
    # If that happens, restart from a deterministic LPT packing, which is a
    # concrete feasible certificate for the effective cap used by refinement.
    # On the reported experiments LPT fits the requested 1.15x cap, so no cap
    # relaxation occurs; the fallback makes the invariant explicit for other
    # workloads as well.
    capacity_relaxed = False
    if load.max() > cap + 1e-9:
        load = np.zeros(n_regions)
        region_of = {}
        for a in sorted(counts, key=lambda x: (-counts[x], x)):
            r = int(np.argmin(load))
            region_of[a] = r
            load[r] += counts[a]
        if load.max() > cap + 1e-9:
            cap = float(load.max())
            capacity_relaxed = True

    objective_trace = [components()["total"]]
    accepted_moves = 0

    # Exact coordinate descent with weighted-centroid updates. A move is
    # accepted only when the full normalized objective falls and the hard
    # capacity constraint remains satisfied.
    for _ in range(max_iter):
        for r in range(n_regions):
            members = [a for a in active if region_of[int(a)] == r]
            if members:
                ww = np.array([counts[int(a)] for a in members], dtype=float)
                centroids[r] = np.average(sender_features[members], axis=0,
                                          weights=ww)

        after_centroids = components()["total"]
        if after_centroids > objective_trace[-1] + 1e-10:
            raise AssertionError("weighted centroid update increased ALPS objective")
        objective_trace.append(after_centroids)

        moved = False
        for a in sorted(counts, key=lambda x: (-counts[x], x)):
            old = region_of[a]
            best, best_delta = old, 0.0
            for r in range(n_regions):
                if r == old or load[r] + counts[a] > cap + 1e-9:
                    continue
                delta = move_delta(a, r)
                if delta < best_delta - 1e-12:
                    best, best_delta = r, delta
            if best != old:
                load[old] -= counts[a]
                load[best] += counts[a]
                region_of[a] = int(best)
                moved = True
                accepted_moves += 1

        after_moves = components()["total"]
        if after_moves > objective_trace[-1] + 1e-10:
            raise AssertionError("accepted ALPS moves increased global objective")
        objective_trace.append(after_moves)
        if not moved:
            break

    regions = [region_of[t.sender] for t in txs]
    if return_diagnostics:
        return regions, {
            "objective_trace": objective_trace,
            "components": components(),
            "accepted_moves": accepted_moves,
            "target_capacity": target_cap,
            "capacity": cap,
            "capacity_relaxed": capacity_relaxed,
            "max_over_mean": float(load.max() / max(mean, 1.0)),
        }
    return regions


def alps_no_graph(txs: Sequence[Transaction], n_regions: int, **kw) -> List[int]:
    """ALPS ablation retaining semantics, load and hard admissibility."""
    return alps(txs, n_regions, graph_weight=0.0, **kw)


def alps_no_semantic(txs: Sequence[Transaction], n_regions: int,
                     **kw) -> List[int]:
    """Graph/load-only ALPS ablation; not a TxAllo reimplementation."""
    return alps(txs, n_regions, semantic_weight=0.0, **kw)


def alps_no_capacity(txs: Sequence[Transaction], n_regions: int,
                     **kw) -> List[int]:
    """ALPS ablation with only the soft load term, not the hard cap."""
    return alps(txs, n_regions, over=1e9, **kw)


STRATEGIES = {
    "ADDR-MOD": addr_mod,
    "TX-KMEANS": tx_kmeans,
    "SENDER-KMEANS": sender_kmeans,
    "SENDER-KMEANS-BAL": sender_kmeans_balanced,
    "ALPS": alps,
    # Reimplemented allocation objective only; see sim/txallo.py for the
    # precise scope of what is and is not reproduced.
    "TXALLO": txallo,
    # General weighted graph partitioner over indivisible sender vertices.
    "METIS-SENDER": metis_sender,
    "ALPS-NOGRAPH": alps_no_graph,
    "ALPS-NOSEM": alps_no_semantic,
    "ALPS-NOCAP": alps_no_capacity,
}


def cross_region_ratio(txs: Sequence[Transaction],
                       regions: Sequence[int]) -> Dict[str, float]:
    """Interaction cut among transfers with two observable placements.

    Every strategy in this study returns a placement for *active senders*.
    Receiver-only addresses have no comparable placement in sender-local
    baselines, so counting each of them as automatically cross-region would add
    a trace-dependent floor unrelated to allocation quality.  We therefore
    score the cut conditional on the receiver also appearing as an active
    sender, and report that condition's coverage explicitly.

    The same definition is used by ALPS's active-account interaction graph, the
    static comparison, and the TxAllo allocation-objective comparison.
    """
    region_of: Dict[int, int] = {}
    counts: Dict[int, int] = {}
    for t, r in zip(txs, regions):
        region_of[t.sender] = r
        counts[t.sender] = counts.get(t.sender, 0) + 1

    cross = eligible = 0
    cold_cross = cold_eligible = cold_n = 0
    for t, r in zip(txs, regions):
        rr = region_of.get(t.receiver)
        is_cold = counts.get(t.sender, 0) == 1
        if is_cold:
            cold_n += 1
        if rr is None:
            continue
        is_cross = rr != r
        eligible += 1
        cross += is_cross
        if is_cold:
            cold_eligible += 1
            cold_cross += is_cross

    return {
        "cross_region_frac": cross / max(eligible, 1),
        "interaction_coverage": eligible / max(len(txs), 1),
        "unknown_receiver_frac": 1.0 - eligible / max(len(txs), 1),
        "cold_cross_region_frac": cold_cross / max(cold_eligible, 1),
        "cold_interaction_coverage": cold_eligible / max(cold_n, 1),
    }


# --------------------------------------------------------------------------
# Static analysis of a partition
# --------------------------------------------------------------------------

def sender_locality_violations(
    txs: Sequence[Transaction], regions: Sequence[int]
) -> Dict[str, float]:
    """Measure how far a partition departs from sender locality.

    ``split_senders`` counts accounts whose debits land in more than one
    region; ``exposed_tx_frac`` is the fraction of transactions initiated by
    such an account, i.e. the share of the workload that a concurrent miner
    could validate against a stale balance.
    """
    by_sender: Dict[int, set] = {}
    tx_count: Dict[int, int] = {}
    for t, r in zip(txs, regions):
        by_sender.setdefault(t.sender, set()).add(r)
        tx_count[t.sender] = tx_count.get(t.sender, 0) + 1

    split = [s for s, rs in by_sender.items() if len(rs) > 1]
    exposed = sum(tx_count[s] for s in split)

    return {
        "split_senders": len(split),
        "split_sender_frac": len(split) / max(len(by_sender), 1),
        "exposed_tx_frac": exposed / max(len(txs), 1),
    }


def resource_locality_violations(
    txs: Sequence[Transaction], regions: Sequence[int]
) -> Dict[str, float]:
    """Violations for the actual guarded balance key ``(sender, asset)``."""
    by_key: Dict[tuple, set] = {}
    tx_count: Dict[tuple, int] = {}
    for t, r in zip(txs, regions):
        by_key.setdefault(t.debit_key, set()).add(r)
        tx_count[t.debit_key] = tx_count.get(t.debit_key, 0) + 1
    split = [k for k, rs in by_key.items() if len(rs) > 1]
    exposed = sum(tx_count[k] for k in split)
    return {
        "split_resources": len(split),
        "split_resource_frac": len(split) / max(len(by_key), 1),
        "resource_exposed_tx_frac": exposed / max(len(txs), 1),
    }


def load_balance(regions: Sequence[int], n_regions: int) -> Dict[str, float]:
    counts = np.bincount(np.asarray(regions), minlength=n_regions).astype(float)
    mean = counts.mean() if counts.size else 0.0
    return {
        "max_over_mean": float(counts.max() / mean) if mean > 0 else 0.0,
        "cv": float(counts.std() / mean) if mean > 0 else 0.0,
    }


def semantic_locality(
    txs: Sequence[Transaction], regions: Sequence[int], n_regions: int
) -> Dict[str, float]:
    """Quality of the partition as a category index.

    ``purity``        share of transactions belonging to their region's modal
                      category (how coherent a block is).
    ``concentration`` share of a category's transactions that sit in its single
                      dominant region, averaged over categories.
    ``r90``           mean number of regions that must be scanned to retrieve
                      90% of a category's transactions -- the retrieval cost the
                      thesis set out to reduce.  A count is used rather than a
                      "touched at all" test, which saturates at n once a single
                      stray transaction lands in every region.
    """
    cats = sorted({t.category for t in txs})
    per_region: Dict[int, Dict[int, int]] = {r: {} for r in range(n_regions)}
    for t, r in zip(txs, regions):
        per_region[r][t.category] = per_region[r].get(t.category, 0) + 1

    total = len(txs)
    pure = sum(max(d.values()) for d in per_region.values() if d)

    conc, r90 = [], []
    for c in cats:
        counts = np.array(
            sorted((per_region[r].get(c, 0) for r in range(n_regions)), reverse=True),
            dtype=float,
        )
        s = counts.sum()
        if s == 0:
            continue
        conc.append(counts[0] / s)
        r90.append(int(np.searchsorted(np.cumsum(counts) / s, 0.90) + 1))

    return {
        "purity": pure / max(total, 1),
        "concentration": float(np.mean(conc)) if conc else 0.0,
        "r90": float(np.mean(r90)) if r90 else float(n_regions),
    }
