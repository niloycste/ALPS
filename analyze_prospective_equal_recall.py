"""Matched-recall scan-work analysis of frozen prospective query plans.

The underlying plans use seven coverage targets fixed from history. This script
interpolates their held-out recall/scan curves for descriptive comparison; it
does not select or validate a deployable plan on the suffix.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "ALPS_conference_submission" / "figures"
RECALL_LEVELS = (0.80, 0.85, 0.90, 0.95)


def interpolate_curve(group, recall):
    curve = (group.groupby("recall", as_index=False)
             .agg(rows=("rows", "min"), regions=("regions", "min"))
             .sort_values("recall"))
    x = np.r_[0.0, curve.recall.to_numpy()]
    rows = np.r_[0.0, curve.rows.to_numpy()]
    regions = np.r_[0.0, curve.regions.to_numpy()]
    assert np.all(np.diff(x) >= 0)
    assert np.all(np.diff(rows) >= -1e-9)
    assert np.all(np.diff(regions) >= -1e-9)
    return float(np.interp(recall, x, rows)), float(np.interp(recall, x, regions))


def main():
    query = pd.read_csv(RESULTS / "prospective_retrieval.csv")
    allocation = pd.read_csv(RESULTS / "prospective_retrieval_allocation.csv")
    curves = query.groupby(
        ["trace", "method", "seed", "target_coverage"], as_index=False
    ).agg(recall=("result_recall", "mean"),
          rows=("rows_scanned", "mean"),
          regions=("contacted_regions", "mean"))

    records = []
    for (trace, method, seed), group in curves.groupby(["trace", "method", "seed"]):
        assert set(group.target_coverage.round(2)) == {0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0}
        for recall in RECALL_LEVELS:
            rows, regions = interpolate_curve(group, recall)
            records.append(dict(trace=trace, method=method, seed=seed,
                                matched_recall=recall,
                                interpolated_rows=rows,
                                interpolated_regions=regions))
    matched = pd.DataFrame(records)
    matched.to_csv(RESULTS / "prospective_equal_recall_config.csv", index=False)
    summary = matched.groupby(["method", "matched_recall"], as_index=False).agg(
        rows_mean=("interpolated_rows", "mean"),
        rows_sd=("interpolated_rows", "std"),
        regions_mean=("interpolated_regions", "mean"),
        regions_sd=("interpolated_regions", "std"),
        configurations=("seed", "size"))
    summary.to_csv(RESULTS / "prospective_equal_recall_summary.csv", index=False)
    per_trace = matched.groupby(["trace", "method", "matched_recall"], as_index=False).agg(
        rows_mean=("interpolated_rows", "mean"),
        rows_seed_sd=("interpolated_rows", "std"),
        regions_mean=("interpolated_regions", "mean"),
        regions_seed_sd=("interpolated_regions", "std"))
    per_trace.to_csv(RESULTS / "prospective_equal_recall_per_trace.csv", index=False)

    # Worked amortisation examples against METIS at matched 90% recall.
    at90 = summary[summary.matched_recall.eq(.90)].set_index("method")
    alloc = allocation.groupby("method", as_index=True).agg(
        allocation_seconds=("allocation_seconds", "mean"),
        active_cut=("active_cut", "mean"),
        interaction_coverage=("interaction_coverage", "mean"))
    a, b = "ALPS", "METIS-SENDER+CAUSAL"
    delta_regions = float(at90.loc[b, "regions_mean"] - at90.loc[a, "regions_mean"])
    delta_cut = float(alloc.loc[a, "active_cut"] - alloc.loc[b, "active_cut"])
    coverage = float(alloc.loc[a, "interaction_coverage"])
    delta_allocation = float(alloc.loc[a, "allocation_seconds"]
                             - alloc.loc[b, "allocation_seconds"])
    scenarios = [
        ("retrieval-heavy", .10, 1.0, .10),
        ("equal query/interaction unit cost", .10, 1.0, 1.0),
        ("low query rate", .01, 1.0, .10),
    ]
    rows = []
    for name, q, query_ms, interaction_ms in scenarios:
        query_saving = q * query_ms / 1000 * delta_regions
        interaction_penalty = coverage * interaction_ms / 1000 * delta_cut
        net = query_saving - interaction_penalty
        rows.append(dict(
            scenario=name, queries_per_transaction=q,
            query_cost_per_region_ms=query_ms,
            active_interaction_cost_ms=interaction_ms,
            matched_recall=.90, alps_regions=at90.loc[a, "regions_mean"],
            metis_regions=at90.loc[b, "regions_mean"],
            alps_cut=alloc.loc[a, "active_cut"], metis_cut=alloc.loc[b, "active_cut"],
            interaction_coverage=coverage,
            extra_allocation_seconds=delta_allocation,
            net_saving_seconds_per_transaction=net,
            break_even_transactions=(delta_allocation / net if net > 0 else np.inf)))
    pd.DataFrame(rows).to_csv(RESULTS / "prospective_deployment_scenarios.csv", index=False)

    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, ax = plt.subplots(figsize=(3.35, 2.05), layout="constrained")
    styles = {
        "SENDER-KMEANS-BAL": ("S-KM-BAL", "s"),
        "METIS-SENDER+CAUSAL": ("METIS+C", "^"),
        "TXALLO+CAUSAL": ("TxAllo+C", "D"),
        "ALPS": ("ALPS", "o"),
    }
    for method, (label, marker) in styles.items():
        data = summary[summary.method.eq(method)].sort_values("matched_recall")
        ax.plot(100 * data.matched_recall, data.rows_mean / 1000,
                marker=marker, markersize=3, linewidth=1, label=label)
    ax.set_xlabel("Matched achieved suffix recall (%)")
    ax.set_ylabel("Interpolated rows scanned (thousands)")
    ax.grid(alpha=.25)
    ax.legend(fontsize=6, ncol=2)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "fig_prospective_equal_recall.pdf")
    fig.savefig(RESULTS / "prospective_equal_recall.png", dpi=200)
    print(summary.round(4).to_string(index=False))
    print(pd.DataFrame(rows).round(6).to_string(index=False))


if __name__ == "__main__":
    main()
