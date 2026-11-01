# ALPS

Artifact for **ALPS: Admissible Locality and Payload Semantics for
Region-Based Parallel Mining**. ALPS keeps each sender indivisible and balances
semantic retrieval locality, active-account interaction locality, and regional
load under a hard capacity constraint.

[Repository](https://github.com/niloycste/ALPS) |
[clean-reproduction details](REPRODUCE_CLEAN.md) |
[artifact verification](ARTIFACT_VERIFICATION.md)

## Repository layout

| Path | Contents |
|---|---|
| `ALPS_conference_submission/` | LaTeX source, bibliography, figures, and PDF |
| `sim/` | ALPS simulator, allocators, baselines, and metrics |
| `data/` | Workload traces and SHA-256 manifests |
| `results/` | Expected CSV results and audit outputs |
| `tests/` | Unit tests |
| `run_*.py` | Experiment drivers |
| `verify_clean_environment.py` | End-to-end scientific-output verifier |

## Quick paper build

Requirements: Python 3.13, the packages in `requirements-core-lock.txt`, and a
LaTeX installation providing `pdflatex` and `bibtex`.

```powershell
git clone https://github.com/niloycste/ALPS.git
Set-Location ALPS

py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-core-lock.txt

# Regenerate the manuscript's experimental figures from supplied CSV results.
.\.venv\Scripts\python.exe -B make_figures.py
.\.venv\Scripts\python.exe -B plot_retrieval_frontier.py
.\.venv\Scripts\python.exe -B analyze_prospective_equal_recall.py

Set-Location ALPS_conference_submission
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The rendered paper is `ALPS_conference_submission/main.pdf`. Figure 1
(`figures/pipeline-v3.png`) is a conceptual overview; the remaining manuscript
figures are regenerated from the result CSV files by the commands above.

## Full clean reproduction

The complete verification uses separate pinned environments because PyMetis is
isolated from the core ALPS environment. From the repository root:

```powershell
py -3.13 -m venv .venv-core
.\.venv-core\Scripts\python.exe -m pip install -r requirements-core-lock.txt
conda env create -p .\.conda-metis -f environment-metis-lock.yml

.\.venv-core\Scripts\python.exe -B verify_clean_environment.py `
  --metis-python .\.conda-metis\python.exe
.\.venv-core\Scripts\python.exe -B analyze_prospective_equal_recall.py
.\.venv-core\Scripts\python.exe -B verify_semantic_scale.py
```

On success, `CLEAN_REPRODUCTION.json` has `"status": "passed"`. The verifier
checks dataset evidence and regenerates the scientific fields used in Tables
II--VI, the retrieval results, prospective allocations, graph-weight
sensitivity, experimental figures, and the cached 100-transaction BSC receipt
audit. Allocation and query timings are intentionally excluded from exact
comparisons because they depend on the machine and concurrent load.

| Evidence | Main generated data |
|---|---|
| Tables II--III | `results/all_static.csv`, `results/all_horizon.csv` |
| Table IV | `results/headtohead.csv` |
| Table V and retrieval frontier | `results/retrieval_benchmark.csv`, `results/retrieval_frontier.csv` |
| Table VI | `results/prospective_retrieval*.csv` |
| Matched-recall figure | `results/prospective_equal_recall_*.csv` |
| Graph-weight sensitivity | `results/prospective_graph_*.csv` |

For a lighter correctness check, run:

```powershell
.\.venv-core\Scripts\python.exe -B -m unittest discover -s tests -v
```

## Scope

- The evaluation uses direct ERC-20 calls; router-mediated execution and full
  EVM state reconstruction are outside scope.
- The retrieval experiments measure scan work, not distributed storage-engine
  latency, and do not claim an equal-recall latency speedup.
- The theorem assumes independent same-snapshot regional execution without
  cross-region coordination.
- The METIS and TxAllo baselines are sender-local experimental adaptations, not
  implementations of their complete protocols.
- The cached receipt audit confirms 100 BNB Smart Chain samples. Historical
  Ethereum RPC responses remain unresolved and are reported as a limitation.

These limits and the prospective retrieval--interaction--load trade-off are
reported explicitly in the manuscript.
