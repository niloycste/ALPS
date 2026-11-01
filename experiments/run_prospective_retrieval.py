"""History-only allocation, features and query plans with causal suffix routing.

Example: python -m experiments.run_prospective_retrieval
Split runs: --methods ALPS,TXALLO,SENDER-KMEANS-BAL --output prospective_core
Graph baselines use an explicitly adapted first-arrival rule, not A-TxAllo.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from sim.chain import Transaction
from sim.partition import STRATEGIES, sender_locality_violations

ROOT = Path(__file__).resolve().parents[1]
TRACES = {"eth_aug14": "native_all.csv", "eth_aug13": "native_day2.csv",
          "eth_aug07": "eth_week2.csv", "bsc": "bsc_trace.csv"}
TARGETS = (.5, .7, .8, .9, .95, .99, 1.)


def read_rows(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda r: (int(r["block"]), int(r["tx_index"])))
    seen, valid = set(), []
    for r in rows:
        key = (r["from"].lower(), int(r["nonce"]))
        if r["from"].lower() == r["to"].lower() or key in seen:
            continue
        seen.add(key)
        valid.append(r)
    return valid


class PrefixEncoder:
    """All fitted statistics and feature dimensions depend on history only.

    Addresses/categories receive IDs on first appearance; unseen token contracts
    have separate query IDs but share a fixed OOV feature coordinate. Their amount
    scale uses the history-wide median raw amount (no suffix fitting).
    """
    def __init__(self, history):
        amounts = {}
        for r in history:
            amounts.setdefault(r["token"].lower(), []).append(int(r["value"]))
        self.tokens = {key: i for i, key in enumerate(sorted(amounts))}
        self.medians = {key: float(np.median(values)) or 1.
                        for key, values in amounts.items()}
        self.fallback = float(np.median([int(r["value"]) for r in history])) or 1.
        self.dim = len(self.tokens) + 2  # known token one-hot + OOV + amount
        self.addresses, self.categories = {}, dict(self.tokens)

    def encode(self, row, tid):
        for address in (row["from"].lower(), row["to"].lower()):
            if address not in self.addresses:
                self.addresses[address] = len(self.addresses)
        token = row["token"].lower()
        category = self.categories.setdefault(token, len(self.categories))
        features = np.zeros(self.dim)
        features[self.tokens.get(token, len(self.tokens))] = 4.
        relative = np.clip(int(row["value"]) / self.medians.get(token, self.fallback), .1, 20.)
        features[-1] = np.log1p(relative)
        return Transaction(tid, self.addresses[row["from"].lower()],
                           self.addresses[row["to"].lower()], float(relative * 8),
                           int(row["nonce"]), category, tuple(features))


def fit_history(history, n, method, seed, alpha, graph_weight=.15):
    # No suffix addresses or features are used to size or fill this matrix.
    features = np.zeros((1 + max(max(t.sender, t.receiver) for t in history),
                         len(history[0].features)))
    weights = np.zeros(len(features))
    for t in history:
        features[t.sender] += t.features
        weights[t.sender] += 1
    features /= np.maximum(weights, 1)[:, None]
    extra = {"eta": 2., "max_iter": 8} if method == "TXALLO" else {}
    if method == "ALPS":
        extra["graph_weight"] = graph_weight
    start = time.perf_counter()
    regions = STRATEGIES[method](history, n, sender_features=features,
                                seed=seed, over=alpha, **extra)
    elapsed = time.perf_counter() - start
    assigned = {t.sender: int(r) for t, r in zip(history, regions)}
    prototypes = np.zeros((n, features.shape[1]))
    loads = np.bincount(regions, minlength=n)
    for t, r in zip(history, regions):
        prototypes[r] += t.features
    prototypes /= np.maximum(loads, 1)[:, None]
    counts = {}
    for t, r in zip(history, regions):
        counts.setdefault(t.category, np.zeros(n, dtype=int))[r] += 1
    return assigned, prototypes, counts, elapsed


def frozen_plans(counts, targets, n):
    plans = {}
    for target in targets:
        plans[target] = {}
        for category, values in counts.items():
            order = np.argsort(-values, kind="stable")
            k = n if target == 1 else int(np.searchsorted(
                np.cumsum(values[order]), target * values.sum()) + 1)
            plans[target][category] = order[:k]
    return plans


def route_future(rows, encoder, assigned, prototypes, n, method, cap, graph_weight=.15):
    future, regions, loads = [], [], np.zeros(n)
    cross = eligible = cold_tx = admission_fallbacks = 0
    cold_senders = set()
    start = time.perf_counter()
    for i, row in enumerate(rows):
        t = encoder.encode(row, i)
        if t.sender not in assigned:
            cold_senders.add(t.sender)
            feasible = np.flatnonzero(loads + 1 <= cap + 1e-9)
            if not feasible.size:
                admission_fallbacks += 1
                feasible = np.arange(n)
            load_score = .2 * (loads / max(cap, 1)) ** 2
            rr = assigned.get(t.receiver)
            edge = np.array([float(rr is not None and r != rr) for r in range(n)])
            if method in ("ALPS", "SENDER-KMEANS-BAL"):
                dist = np.sum((prototypes - t.features) ** 2, axis=1)
                positive = dist[dist > 1e-12]
                semantic = dist / (float(np.median(positive)) if positive.size else 1.)
                score = semantic + load_score
                if method == "ALPS":
                    score += graph_weight * edge
            else:
                # Our shared causal extension: receiver affinity + load;
                # neither METIS nor the TxAllo paper is claimed to specify it.
                score = .15 * edge + load_score
            assigned[t.sender] = int(feasible[np.argmin(score[feasible])])
        r = assigned[t.sender]
        loads[r] += 1
        rr = assigned.get(t.receiver)
        if rr is not None:
            eligible += 1
            cross += rr != r
        cold_tx += t.sender in cold_senders
        future.append(t)
        regions.append(r)
    return future, regions, dict(
        routing_seconds=time.perf_counter() - start,
        future_calls=len(future), cold_senders=len(cold_senders),
        cold_sender_tx_fraction=cold_tx / len(future),
        interaction_coverage=eligible / len(future),
        active_cut=cross / eligible if eligible else "",
        max_mean_load=float(loads.max() / loads.mean()),
        admission_fallbacks=admission_fallbacks,
        split_senders=sender_locality_violations(future, regions)["split_senders"])


def query_benchmark(future, regions, plans, n, repetitions):
    stores = [np.array([t.category for t, rr in zip(future, regions) if rr == r],
                       dtype=np.int32) for r in range(n)]
    categories = sorted({t.category for t in future})
    result = []
    for target, by_category in plans.items():
        for category in categories:
            chosen = by_category.get(category, np.arange(n))
            total = sum(np.count_nonzero(s == category) for s in stores)
            returned = sum(np.count_nonzero(stores[r] == category) for r in chosen)
            samples = []
            for _ in range(repetitions):
                start = time.perf_counter_ns()
                matches = sum(np.count_nonzero(stores[r] == category) for r in chosen)
                samples.append((time.perf_counter_ns() - start) / 1000.)
                assert matches == returned
            result.append(dict(target_coverage=target, category=category,
                cold_token=category not in by_category,
                contacted_regions=len(chosen), rows_scanned=sum(stores[r].size for r in chosen),
                result_rows=int(returned), total_matching_rows=int(total),
                result_recall=returned / total,
                median_latency_us=float(np.median(samples)),
                latency_iqr_us=float(np.percentile(samples, 75) - np.percentile(samples, 25))))
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default="SENDER-KMEANS-BAL,METIS-SENDER,TXALLO,ALPS")
    parser.add_argument("--traces", default=",".join(TRACES))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--history", type=float, default=.7)
    parser.add_argument("--horizon-multiplier", type=float, default=.5)
    parser.add_argument("--repetitions", type=int, default=250)
    parser.add_argument("--graph-weight", type=float, default=.15,
                        help="ALPS graph weight in both history fitting and causal placement")
    parser.add_argument("--output", default="prospective_retrieval")
    args = parser.parse_args()
    if (not 0 < args.history < 1 or args.horizon_multiplier <= 0 or args.repetitions < 1
            or not math.isfinite(args.graph_weight) or args.graph_weight < 0):
        parser.error("invalid history fraction, horizon multiplier or repetitions")
    queries, allocations, manifests = [], [], []
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    for tag in args.traces.split(","):
        path = ROOT / "data" / TRACES[tag]
        rows = read_rows(path)
        cut = int(len(rows) * args.history)
        history_rows = rows[:cut]
        horizon = math.ceil(len(history_rows) * args.horizon_multiplier)
        fitted = PrefixEncoder(history_rows)
        manifests.append(dict(trace=tag, input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            history_calls=cut, cutoff_block=history_rows[-1]["block"],
            cutoff_tx_index=history_rows[-1]["tx_index"], horizon_calls=horizon,
            token_medians=fitted.medians, oov_amount_median=fitted.fallback,
            token_feature_coordinates=fitted.tokens))
        for method in args.methods.split(","):
            for seed in map(int, args.seeds.split(",")):
                encoder = PrefixEncoder(history_rows)
                history = [encoder.encode(r, i) for i, r in enumerate(history_rows)]
                assigned, proto, counts, seconds = fit_history(history, 6, method, seed, 1.15,
                                                              args.graph_weight)
                plans = frozen_plans(counts, TARGETS, 6)
                frozen_medians = dict(encoder.medians)
                future, regions, metrics = route_future(rows[cut:], encoder,
                    assigned, proto, 6, method, 1.15 * horizon / 6, args.graph_weight)
                assert frozen_medians == encoder.medians
                assert len(future) + len(history) == len(rows)
                assert metrics["split_senders"] == 0
                label = method + ("+CAUSAL" if method in ("METIS-SENDER", "TXALLO") else "")
                meta = dict(trace=tag, method=label, seed=seed, n_regions=6,
                    history_fraction=args.history, history_calls=cut, horizon_calls=horizon,
                    alps_graph_weight=args.graph_weight if method == "ALPS" else "")
                allocations.append(dict(**meta, allocation_seconds=seconds, **metrics))
                for row in query_benchmark(future, regions, plans, 6, args.repetitions):
                    queries.append(dict(**meta, repetitions=args.repetitions, **row))
                # Checkpoint complete configurations; reruns never append duplicates.
                write_csv(out / (args.output + ".csv"), queries)
                write_csv(out / (args.output + "_allocation.csv"), allocations)
                print(f"{tag} {label} seed={seed}: load={metrics['max_mean_load']:.3f}", flush=True)
    versions = {}
    for name in ("numpy", "scikit-learn", "networkx", "pymetis"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    config = dict(arguments=vars(args), targets=TARGETS, traces=manifests,
        python=platform.python_version(), dependencies=versions,
        planning="ceil(history_calls * preset horizon_multiplier); no suffix count used",
        cold_token_policy="all-region queries; OOV one-hot; history-wide median amount scale",
        graph_baseline_extension="first-arrival receiver indicator .15 + quadratic load .20; pinned",
        limitations="chronological holdout, not a deployed system; no receipt filtering; timings are local scans")
    (out / (args.output + "_config.json")).write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
