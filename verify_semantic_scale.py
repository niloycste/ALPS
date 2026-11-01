"""Verify the static ALPS semantic normalisation described in Equation (1).

The check distinguishes the implemented definition from two plausible errors:
using only assigned sender/prototype distances, or recomputing the median during
refinement. It writes SEMANTIC_SCALE_VERIFICATION.json.
"""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.cluster import KMeans

import sim.partition as partition
from sim.chain import Transaction


ROOT = Path(__file__).resolve().parent


def transaction(tid, sender, receiver, nonce, category, feature):
    return Transaction(tid, sender, receiver, 1.0, nonce, category,
                       tuple(feature))


def fixture():
    txs = []
    for sender in range(12):
        for nonce in range(4):
            category = sender % 3
            feature = np.eye(3)[category].tolist() + [nonce / 4]
            txs.append(transaction(len(txs), sender, (sender + 1) % 12,
                                   nonce, category, feature))
    features = np.zeros((12, 4))
    counts = np.zeros(12)
    for tx in txs:
        features[tx.sender] += np.asarray(tx.features)
        counts[tx.sender] += 1
    features /= counts[:, None]
    return txs, features, counts


def main():
    txs, sender_features, counts = fixture()
    initial = KMeans(n_clusters=3, n_init=10, random_state=0).fit(
        sender_features, sample_weight=counts)
    all_distances = np.linalg.norm(
        sender_features[:, None, :] - initial.cluster_centers_[None, :, :],
        axis=2) ** 2
    all_positive = all_distances[all_distances > 1e-12]
    expected = float(np.median(all_positive)) if all_positive.size else 1.0

    assigned = all_distances[np.arange(len(sender_features)), initial.labels_]
    assigned_positive = assigned[assigned > 1e-12]
    assigned_only = (float(np.median(assigned_positive))
                     if assigned_positive.size else 1.0)

    calls = []
    real_median = partition.np.median

    def capture_median(values, *args, **kwargs):
        calls.append(np.asarray(values).copy())
        return real_median(values, *args, **kwargs)

    partition.np.median = capture_median
    try:
        partition.alps(txs, 3, sender_features=sender_features, seed=0,
                       max_iter=6, return_diagnostics=True)
    finally:
        partition.np.median = real_median

    source = inspect.getsource(partition.alps)
    tree = ast.parse(source)
    scale_assignments = sum(
        isinstance(node, ast.Name) and node.id == "semantic_scale"
        for parent in ast.walk(tree)
        if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        for node in ([*parent.targets] if isinstance(parent, ast.Assign)
                     else [parent.target]))

    assert len(calls) == 1, f"expected one median call, observed {len(calls)}"
    observed = calls[0]
    np.testing.assert_allclose(np.sort(observed), np.sort(all_positive),
                               rtol=0, atol=0)
    observed_scale = float(real_median(observed)) if observed.size else 1.0
    assert observed_scale == expected
    assert expected != assigned_only, "fixture does not distinguish definitions"
    assert scale_assignments == 1, scale_assignments

    report = {
        "status": "passed",
        "python": sys.version.split()[0],
        "senders": len(sender_features),
        "initial_prototypes": initial.n_clusters,
        "all_pair_distances": int(all_distances.size),
        "nonzero_all_pair_distances": int(all_positive.size),
        "observed_median_calls_during_alps": len(calls),
        "semantic_scale_assignments_in_alps_source": scale_assignments,
        "observed_scale": observed_scale,
        "expected_all_pair_scale": expected,
        "assigned_only_scale": assigned_only,
        "conclusion": (
            "ALPS takes the median of all nonzero sender-to-initial-prototype "
            "squared distances exactly once and reuses that fixed scalar."
        ),
    }
    output = ROOT / "SEMANTIC_SCALE_VERIFICATION.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
