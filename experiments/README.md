# Experiments

Run every command from the repository root as a Python module, for example:

```powershell
python -m experiments.run_prospective_retrieval --methods ALPS
```

| Group | Entry points |
|---|---|
| Core allocation studies | `run_all_traces`, `run_headtohead`, `run_allocator_comparison` |
| Sensitivity studies | `run_alpha_sensitivity`, `run_txallo_sensitivity`, `run_allocator_scaling` |
| Retrieval studies | `run_retrieval_benchmark`, `run_retrieval_frontier`, `run_prospective_retrieval` |
| Analysis and plots | `summarize_prospective`, `summarize_graph_sensitivity`, `analyze_prospective_equal_recall`, `plot_retrieval_frontier`, `make_figures` |
| Legacy/auxiliary studies | `run_crosstrace`, `run_final` |

Outputs are written to `../results/`; manuscript figures are written to
`../ALPS_conference_submission/figures/`.
