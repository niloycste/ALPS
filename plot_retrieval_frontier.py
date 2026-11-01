"""Plot the prespecified history-plan sweep; never select targets by suffix recall."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
d = pd.read_csv(ROOT / "results/retrieval_frontier.csv")
keys = ["method", "target_coverage"]
# Match Table V: each eligible token/seed contributes equally.
summary = d.groupby(keys, as_index=False).agg(
    result_recall=("result_recall", "mean"),
    rows_scanned=("rows_scanned", "mean"),
    contacted_regions=("contacted_regions", "mean"),
    token_seed_observations=("category", "size"))
summary.to_csv(ROOT / "results/retrieval_frontier_summary.csv", index=False)
plt.rcParams.update({"font.family": "serif", "font.size": 8})
fig, ax = plt.subplots(figsize=(3.35, 1.95), layout="constrained")
styles = {"SENDER-KMEANS-BAL": ("S-KM-BAL", "s"),
          "METIS-SENDER": ("METIS", "^"), "TXALLO": ("TxAllo", "D"),
          "ALPS": ("ALPS", "o")}
for method, (label, marker) in styles.items():
    s = summary[summary.method == method].sort_values("target_coverage")
    # Lines join training coverage targets in their prespecified order.
    ax.plot(100 * s.result_recall, s.rows_scanned / 1000,
            marker=marker, markersize=3, linewidth=1, label=label)
ax.set_xlabel("Achieved suffix recall (%)")
ax.set_ylabel("Rows scanned (thousands)")
ax.grid(alpha=.25)
ax.legend(fontsize=6, loc="upper left")
out = ROOT / "ALPS_conference_submission/figures/fig_retrieval_frontier.pdf"
fig.savefig(out)
fig.savefig(ROOT / "results/retrieval_frontier.png", dpi=200)
print(summary.round(4).to_string(index=False))
print(out)
