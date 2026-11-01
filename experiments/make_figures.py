"""
Generate the paper figures as vector PDFs in
ALPS_conference_submission/figures/.

Every figure now draws on the complete battery run over all traces
(results/all_*.csv), so each dataset receives the same treatment rather than
the primary trace carrying the plots alone.

Styling targets IEEEtran: serif type, single-column width 3.5in, double-column
7.16in, recessive grid, thin marks. Every series carries a marker and dash
pattern as well as a hue, so the figures survive grayscale printing.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "ALPS_conference_submission", "figures")
os.makedirs(FIGS, exist_ok=True)

COL1, COL2 = 3.5, 7.16

# Categorical slots validated with the data-viz palette checker
# (lightness band, chroma floor, CVD separation, contrast: all pass).
STYLE = {
    "ADDR-MOD":      dict(color="#2a78d6", marker="o", ls="-",  label="ADDR-MOD (base)"),
    "TX-KMEANS":     dict(color="#eb6834", marker="s", ls="--", label="TX-KMEANS (payload)"),
    "SENDER-KMEANS": dict(color="#00806a", marker="^", ls="-.", label="SENDER-KMEANS"),
    "SENDER-KMEANS-BAL": dict(color="#4a3aa7", marker="D", ls=(0, (3, 1, 1, 1)),
                              label="SENDER-KMEANS-BAL"),
    "METIS-SENDER": dict(color="#7b5a2e", marker="X", ls=(0, (1, 1)),
                          label="METIS-SENDER"),
    "ALPS":          dict(color="#a64b87", marker="P", ls=(0, (5, 1)),
                          label="ALPS (ours)"),
}
# Same validated hues, reused when the series identity is the trace.
TRACE_STYLE = {
    "eth_aug14": dict(color="#2a78d6", marker="o", ls="-",  label="D1 ETH 08-14"),
    "eth_aug13": dict(color="#eb6834", marker="s", ls="--", label="D2 ETH 08-13"),
    "eth_aug07": dict(color="#00806a", marker="^", ls="-.", label="D3 ETH 08-07"),
    "bsc":       dict(color="#4a3aa7", marker="D", ls=(0, (3, 1, 1, 1)),
                      label="C1 BSC"),
}
NEUTRAL = "#52514e"
ORDER = ["ADDR-MOD", "TX-KMEANS", "SENDER-KMEANS",
         "SENDER-KMEANS-BAL", "ALPS"]
KEY_ORDER = ["ADDR-MOD", "TX-KMEANS", "SENDER-KMEANS-BAL", "ALPS"]
TRACES = ["eth_aug14", "eth_aug13", "eth_aug07", "bsc"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "figure.dpi": 200,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def tidy(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=4)
    ax.grid(True, color="#b8b7b2")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8984")


def save(fig, name):
    p = os.path.join(FIGS, name)
    fig.savefig(p)
    fig.savefig(p.replace(".pdf", ".png"), dpi=220)
    plt.close(fig)
    print(f"  {p}")


def _load(name):
    return pd.read_csv(os.path.join(RESULTS, name))


# --------------------------------------------------------------------------
# Fig. 2  invariant violation and its scheduling cost, every trace
# --------------------------------------------------------------------------

def fig_sweep():
    s = _load("all_static.csv")
    s = s[s.trace.isin(TRACES)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.65),
                              constrained_layout=True)

    for t in TRACES:
        st = TRACE_STYLE[t]
        d = s[(s.trace == t) & (s.strategy == "TX-KMEANS")].sort_values("n_regions")
        a.plot(d.n_regions, d.exposed_tx_frac * 100, color=st["color"],
               marker=st["marker"], ls=st["ls"], label=st["label"])
        b.plot(d.n_regions, d.global_rounds, color=st["color"],
               marker=st["marker"], ls=st["ls"], label=st["label"])

    # every sender-local scheme is identically zero / flat, drawn once
    z = s[(s.strategy == "ALPS")].sort_values("n_regions")
    a.plot(z.n_regions.unique(), [0] * z.n_regions.nunique(), color=NEUTRAL,
           marker="v", ls=":", label="all sender-local (0%)")
    for t in TRACES:
        d = s[(s.trace == t) & (s.strategy == "ALPS")].sort_values("n_regions")
        b.plot(d.n_regions, d.global_rounds, color=NEUTRAL, ls=":", lw=1.0,
               alpha=.8)
    b.plot([], [], color=NEUTRAL, ls=":", label="ALPS (all traces)")

    tidy(a, "Number of regions $n$", "Transactions of split senders (%)",
         "(a) Nonce-locality exposure, TX-KMEANS")
    a.set_ylim(-4, 100)
    a.set_xticks([2, 3, 4, 6, 8, 11])
    a.legend(loc="lower right", frameon=False, fontsize=6.4,
             labelspacing=0.25)

    tidy(b, "Number of regions $n$", "Rounds to commit the full trace",
         "(b) Retry-aware scheduling cost")
    b.set_yscale("log")
    b.set_xticks([2, 3, 4, 6, 8, 11])
    b.legend(loc="center right", frameon=False, fontsize=6.4)
    save(fig, "fig_sweep.pdf")


# --------------------------------------------------------------------------
# Fig. 3  balance-only stress: overdrafts on every trace
# --------------------------------------------------------------------------

def fig_headroom():
    h = _load("all_headroom.csv")
    fig, ax = plt.subplots(figsize=(COL1, 2.45), constrained_layout=True)
    for t in TRACES:
        st = TRACE_STYLE[t]
        d = h[(h.trace == t) & (h.strategy == "TX-KMEANS")].sort_values(
            "initial_balance")
        ax.plot(d.initial_balance, d.overdrawn_accounts, color=st["color"],
                marker=st["marker"], ls=st["ls"], label=st["label"])
    ax.axhline(0, color=NEUTRAL, ls=":", lw=1.2)
    tidy(ax, "Initial balance per (account, token)",
         "Accounts overdrawn\n(invalid state)")
    ax.invert_xaxis()
    ax.legend(loc="upper left", frameon=False, fontsize=6.4,
              title="TX-KMEANS", title_fontsize=6.4)
    save(fig, "fig_headroom.pdf")


# --------------------------------------------------------------------------
# Fig. 4  retrieval cost per trace (small multiples)
# --------------------------------------------------------------------------

def fig_locality():
    q = _load("all_static.csv")
    fig, axes = plt.subplots(1, 4, figsize=(COL2, 2.25), sharey=True,
                             constrained_layout=True)
    for ax, t in zip(axes, TRACES):
        for k in ORDER:
            d = q[(q.trace == t) & (q.strategy == k)].sort_values("n_regions")
            st = STYLE[k]
            ax.plot(d.n_regions, d.r90, color=st["color"], marker=st["marker"],
                    ls=st["ls"], label=st["label"], markersize=3)
        tidy(ax, "$n$", "Regions for 90%" if t == TRACES[0] else "")
        ax.set_title(TRACE_STYLE[t]["label"], pad=3, fontsize=7)
        ax.set_xticks([2, 4, 6, 8, 11])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=5,
               frameon=False, fontsize=6.2, columnspacing=.8,
               handlelength=2.2)
    save(fig, "fig_locality.pdf")


# --------------------------------------------------------------------------
# Fig. 5  ALPS operating curve, every trace
# --------------------------------------------------------------------------

def fig_frontier():
    q = _load("all_analysis.csv")
    q = q[q.variant.str.startswith("GW-")].copy()
    q["w"] = q.variant.str.replace("GW-", "").astype(float)
    g = q.groupby(["trace", "w"]).agg(r90=("r90", "mean"),
                                      cross=("cross_region", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(COL1, 2.45), constrained_layout=True)
    for t in TRACES:
        d = g[g.trace == t].sort_values("w")
        st = TRACE_STYLE[t]
        ax.plot(d.r90, d.cross * 100, color=st["color"], marker=st["marker"],
                ls=st["ls"], label=st["label"])
        offsets = {"eth_aug14": (3, 3), "eth_aug13": (3, -7),
                   "eth_aug07": (-14, 3), "bsc": (-14, -7)}
        for row in d.itertuples():
            ax.annotate(f"{row.w:g}", (row.r90, row.cross * 100),
                        xytext=offsets[t], textcoords="offset points",
                        fontsize=4.2, color=st["color"])
        knee = d[abs(d.w - 0.15) < 1e-9]
        if len(knee):
            ax.scatter(knee.r90, knee.cross * 100, s=52, facecolors="none",
                       edgecolors=st["color"], linewidths=1.2, zorder=5)
    tidy(ax, "Regions scanned for 90% of a token ($R_{90}$)",
         "Active-account interaction cut (%)")
    ax.legend(loc="upper right", frameon=False, fontsize=6.4,
              labelspacing=.25)
    save(fig, "fig_frontier.pdf")


# --------------------------------------------------------------------------
# Fig. 6  chronological prefix-to-suffix evaluation, every trace
# --------------------------------------------------------------------------

def fig_temporal():
    o = _load("all_temporal.csv")
    modes = ["HASH", "SENDER-KMEANS", "SENDER-KMEANS-BAL", "ALPS"]
    short = {"HASH": "HASH", "SENDER-KMEANS": "S-KM",
             "SENDER-KMEANS-BAL": "S-KM-B", "ALPS": "ALPS"}
    cols = {"HASH": "#52514e", "SENDER-KMEANS": "#00806a",
            "SENDER-KMEANS-BAL": "#4a3aa7", "ALPS": "#a64b87"}
    panels = [("r90", "Regions scanned for 90%", "(a) Semantic retrieval"),
              ("cross_region_frac", "Active-account cut (%)",
               "(b) Interaction cut"),
              ("load_max_over_mean", "Maximum / mean load", "(c) Load balance")]

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.65),
                             constrained_layout=True)
    width = 0.2
    for ax, (col, ylab, title) in zip(axes, panels):
        for i, m in enumerate(modes):
            xs, ys, es = [], [], []
            for j, t in enumerate(TRACES):
                d = o[(o.trace == t) & (o["mode"] == m)][col]
                scale = 100 if col == "cross_region_frac" else 1
                xs.append(j + (i - 1.5) * width)
                ys.append(d.mean() * scale)
                es.append(d.std() * scale)
            ax.bar(xs, ys, width, yerr=es, color=cols[m], label=short[m],
                   capsize=1.5, error_kw=dict(lw=0.6))
        tidy(ax, "", ylab, title)
        ax.set_xticks(range(len(TRACES)))
        ax.set_xticklabels([TRACE_STYLE[t]["label"].split()[0] for t in TRACES],
                           fontsize=6.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4,
               frameon=False, fontsize=6.4, columnspacing=1.0)
    save(fig, "fig_temporal.pdf")


# --------------------------------------------------------------------------
# Fig. 7  where ALPS wins: all 24 static trace/region settings
# --------------------------------------------------------------------------

def fig_dominance():
    s = _load("all_static.csv")
    s = s[s.trace.isin(TRACES)]
    a = s[s.strategy == "ALPS"].set_index(["trace", "n_regions"])
    b = s[s.strategy == "SENDER-KMEANS-BAL"].set_index(["trace", "n_regions"])
    fig, ax = plt.subplots(figsize=(COL1, 2.45), constrained_layout=True)
    ax.axhline(0, color=NEUTRAL, lw=0.8)
    ax.axvline(0, color=NEUTRAL, lw=0.8)
    wins = ties = tradeoffs = 0
    for t in TRACES:
        st = TRACE_STYLE[t]
        xs, ys = [], []
        for k in a.index:
            if k[0] != t:
                continue
            x = ((b.loc[k, "r90"] - a.loc[k, "r90"])
                 / b.loc[k, "r90"] * 100)
            y = ((b.loc[k, "cross_region"] - a.loc[k, "cross_region"])
                 / b.loc[k, "cross_region"] * 100)
            xs.append(x); ys.append(y)
            if x > 1e-9 and y > 1e-9:
                wins += 1
            elif abs(x) <= 1e-9 and y > 1e-9:
                ties += 1
            else:
                tradeoffs += 1
        ax.scatter(xs, ys, color=st["color"], marker=st["marker"], s=26,
                   label=st["label"], zorder=4)
    xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
    y0 = min(max((0 - ymin) / (ymax - ymin), 0), 1)
    ax.axvspan(0, xmax, ymin=y0, ymax=1, color="#00806a", alpha=.06,
               zorder=0)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    tidy(ax, "$R_{90}$ improvement vs S-KM-BAL (%)",
         "Interaction-cut reduction\nvs S-KM-BAL (%)")
    ax.legend(loc="lower left", frameon=False, fontsize=6.2)
    ax.text(0.03, 0.96,
            f"{wins} improve / {ties} tie / {tradeoffs} trade off",
            ha="left",
            va="top", transform=ax.transAxes, fontsize=6.4, color="#00806a")
    save(fig, "fig_dominance.pdf")


# --------------------------------------------------------------------------
# Robustness: five-seed sensitivity to the hard-cap multiplier alpha
# --------------------------------------------------------------------------

def fig_alpha_sensitivity():
    data = _load("alpha_sensitivity_summary.csv")
    panels = [
        ("r90_mean", "r90_std", "$R_{90}$", "(a) Retrieval locality", 1),
        ("cross_region_mean", "cross_region_std", "Interaction cut (%)",
         "(b) Active-account cut", 100),
        ("load_mean", "load_std", "Maximum / mean load",
         "(c) Achieved load", 1),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.45),
                             constrained_layout=True)
    for ax, (mean_col, std_col, ylabel, title, scale) in zip(axes, panels):
        for trace in TRACES:
            style = TRACE_STYLE[trace]
            group = data[data.trace == trace].sort_values("alpha")
            ax.errorbar(
                group.alpha, group[mean_col] * scale,
                yerr=group[std_col] * scale,
                color=style["color"], marker=style["marker"],
                linestyle=style["ls"], label=style["label"], capsize=2,
            )
        tidy(ax, r"Capacity factor $\alpha$", ylabel, title)
        ax.set_xticks(sorted(data.alpha.unique()))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4,
               frameon=False, fontsize=6.4, columnspacing=1.0)
    save(fig, "fig_alpha.pdf")


if __name__ == "__main__":
    print("figures:")
    fig_sweep()
    fig_locality()
    fig_frontier()
    fig_temporal()
    if os.path.exists(os.path.join(RESULTS, "alpha_sensitivity_summary.csv")):
        fig_alpha_sensitivity()
