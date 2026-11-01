"""Combine optional split runs and summarise measured prospective retrieval.

python summarize_prospective.py --parts prospective_core,prospective_metis
"""
import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", default="prospective_retrieval")
    args = parser.parse_args()
    out = ROOT / "results"
    parts = args.parts.split(",")
    query = pd.concat([pd.read_csv(out / (p + ".csv")) for p in parts], ignore_index=True)
    allocation = pd.concat([pd.read_csv(out / (p + "_allocation.csv")) for p in parts], ignore_index=True)
    keys = ["trace", "method", "seed", "target_coverage", "category"]
    assert not query.duplicated(keys).any()
    assert not allocation.duplicated(["trace", "method", "seed"]).any()
    assert allocation.split_senders.eq(0).all()
    assert query[query.target_coverage.eq(1)].result_recall.eq(1).all()
    assert query[query.cold_token].result_recall.eq(1).all()
    query.sort_values(keys).to_csv(out / "prospective_retrieval.csv", index=False)
    allocation.to_csv(out / "prospective_retrieval_allocation.csv", index=False)
    summary = query.groupby(["method", "target_coverage"], as_index=False).agg(
        recall=("result_recall", "mean"), regions=("contacted_regions", "mean"),
        rows=("rows_scanned", "mean"), latency_us=("median_latency_us", "mean"),
        mean_iqr_us=("latency_iqr_us", "mean"), observations=("category", "size"))
    summary.to_csv(out / "prospective_retrieval_summary.csv", index=False)
    per_trace = query.groupby(["trace", "method", "target_coverage"], as_index=False).agg(
        recall=("result_recall", "mean"), regions=("contacted_regions", "mean"),
        rows=("rows_scanned", "mean"), latency_us=("median_latency_us", "mean"))
    per_trace.to_csv(out / "prospective_retrieval_per_trace.csv", index=False)
    metrics = allocation.groupby("method", as_index=False).agg(
        active_cut=("active_cut", "mean"), coverage=("interaction_coverage", "mean"),
        max_mean_load=("max_mean_load", "mean"), configurations=("seed", "size"))
    headline = summary[summary.target_coverage.eq(.9)].merge(metrics, on="method")
    headline.to_csv(out / "prospective_retrieval_headline.csv", index=False)
    print(headline.round(4).to_string(index=False))
    configs = [json.loads((out / (p + "_config.json")).read_text()) for p in parts]
    (out / "prospective_retrieval_runs.json").write_text(json.dumps(configs, indent=2) + "\n")
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, ax = plt.subplots(figsize=(5.2, 3.2), layout="constrained")
    for method, data in summary.groupby("method", sort=False):
        data = data.sort_values("target_coverage")
        ax.plot(data.recall * 100, data.rows / 1000, marker="o", markersize=3, label=method)
    ax.set(xlabel="Achieved suffix recall (%)", ylabel="Rows scanned (thousands)")
    ax.grid(alpha=.25)
    ax.legend(fontsize=7)
    fig.savefig(out / "prospective_retrieval.png", dpi=180)
    # Kept separate from the manuscript's existing retrospective figure.
    fig.savefig(out / "prospective_retrieval.svg")
    print(f"{len(query)} query rows; {len(allocation)} allocations; all invariants passed")


if __name__ == "__main__":
    main()
