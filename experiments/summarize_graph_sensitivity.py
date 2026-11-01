"""Summarise the declared .15/.6/2.4 prospective ALPS graph-weight sweep."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    results = ROOT / "results"
    records, per_trace, configurations = [], [], []
    for weight, tag in ((.15, "0p15"), (.6, "0p6"), (2.4, "2p4")):
        queries = pd.read_csv(results / f"prospective_graph_{tag}.csv")
        alloc = pd.read_csv(results / f"prospective_graph_{tag}_allocation.csv")
        assert len(alloc) == 12 and len(queries) == 672
        assert (alloc.split_senders == 0).all()
        assert (queries.loc[queries.target_coverage == 1, "result_recall"] == 1).all()
        chosen = queries[queries.target_coverage == .9]
        by_configuration = chosen.groupby(["trace", "seed"], as_index=False).agg(
            recall=("result_recall", "mean"), rows_scanned=("rows_scanned", "mean"),
            regions=("contacted_regions", "mean"))
        by_configuration = by_configuration.merge(
            alloc[["trace", "seed", "active_cut", "max_mean_load"]],
            on=["trace", "seed"], validate="one_to_one")
        by_configuration.insert(0, "graph_weight", weight)
        configurations.extend(by_configuration.to_dict("records"))
        for trace in [None, *sorted(alloc.trace.unique())]:
            q = chosen if trace is None else chosen[chosen.trace == trace]
            a = alloc if trace is None else alloc[alloc.trace == trace]
            record = dict(graph_weight=weight, trace=trace or "all", configurations=len(a),
                          query_observations=len(q), recall=q.result_recall.mean(),
                          rows_scanned=q.rows_scanned.mean(), regions=q.contacted_regions.mean(),
                          active_cut=a.active_cut.mean(), max_mean_load=a.max_mean_load.mean(),
                          cut_seed_min=a.active_cut.min(), cut_seed_max=a.active_cut.max())
            (records if trace is None else per_trace).append(record)
        if weight == .15:
            baseline = pd.read_csv(results / "prospective_retrieval.csv")
            baseline = baseline[baseline.method == "ALPS"]
            keys = ["trace", "seed", "target_coverage", "category"]
            for column in ["result_recall", "rows_scanned", "contacted_regions"]:
                pd.testing.assert_series_equal(queries.set_index(keys)[column].sort_index(),
                                               baseline.set_index(keys)[column].sort_index())
    pd.DataFrame(records).to_csv(results / "prospective_graph_summary.csv", index=False)
    pd.DataFrame(per_trace).to_csv(results / "prospective_graph_per_trace.csv", index=False)
    config = pd.DataFrame(configurations)
    config.to_csv(results / "prospective_graph_configurations.csv", index=False)
    variability = config.groupby("graph_weight", as_index=False).agg(
        recall_mean=("recall", "mean"), recall_sd=("recall", "std"),
        rows_mean=("rows_scanned", "mean"), rows_sd=("rows_scanned", "std"),
        cut_mean=("active_cut", "mean"), cut_sd=("active_cut", "std"),
        load_mean=("max_mean_load", "mean"), load_sd=("max_mean_load", "std"),
        configurations=("seed", "size"))
    variability.to_csv(results / "prospective_graph_variability.csv", index=False)
    within_trace = config.groupby(["graph_weight", "trace"], as_index=False).agg(
        recall_mean=("recall", "mean"), recall_seed_sd=("recall", "std"),
        rows_mean=("rows_scanned", "mean"), rows_seed_sd=("rows_scanned", "std"),
        cut_mean=("active_cut", "mean"), cut_seed_sd=("active_cut", "std"),
        load_mean=("max_mean_load", "mean"), load_seed_sd=("max_mean_load", "std"),
        seeds=("seed", "size"))
    within_trace.to_csv(results / "prospective_graph_within_trace_variability.csv",
                        index=False)
    print(pd.DataFrame(records).to_string(index=False))
    print("\nConfiguration-level mean and sample SD:\n" + variability.to_string(index=False))


if __name__ == "__main__":
    main()
