# ALPS

**Admissible Locality and Payload Semantics for Region-Based Parallel Mining**

ALPS is a reproducible research artifact for sender-local region allocation in
parallel blockchain execution. It combines semantic retrieval locality,
active-account interaction locality, and capacity-aware load balancing while
keeping each sender indivisible.

[Paper source](ALPS_conference_submission/) |
[Reproduction guide](docs/REPRODUCE_CLEAN.md) |
[Verification notes](docs/ARTIFACT_VERIFICATION.md) |
[GitHub](https://github.com/niloycste/ALPS)

## Project map

```text
ALPS/
|- ALPS_conference_submission/  IEEE manuscript, figures, and rendered PDF
|- ALPS_DASFAA_2027/            DASFAA/LNCS conversion draft
|- sim/                         ALPS model, allocators, baselines, and metrics
|- experiments/                 Experiment runners, analyses, and plot builders
|- tools/                       Data collection, receipt audit, and packaging tools
|  `- verification/            Clean-environment and artifact verifiers
|- data/                        Input traces and SHA-256 manifests
|- results/                     Recorded CSV/JSON outputs and generated plots
|- tests/                       Unit and invariant tests
|- docs/                        Reproduction, validation, and venue-format notes
|- config/                      Pinned Python and Conda environments
`- artifacts/                   Archived submission bundle and legacy run notes
```

## Quick start: rebuild the paper

Prerequisites: Python 3.13, a LaTeX installation with `pdflatex` and `bibtex`,
and the core pinned dependencies.

```powershell
git clone https://github.com/niloycste/ALPS.git
Set-Location ALPS

py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r config\requirements-core-lock.txt

# Regenerate the data-driven manuscript figures from recorded results.
.\.venv\Scripts\python.exe -m experiments.make_figures
.\.venv\Scripts\python.exe -m experiments.plot_retrieval_frontier
.\.venv\Scripts\python.exe -m experiments.analyze_prospective_equal_recall

Set-Location ALPS_conference_submission
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The paper is written to `ALPS_conference_submission/main.pdf`. The conceptual
pipeline image is versioned with the manuscript; experiment figures are rebuilt
from `results/` by the commands above.

## Full clean reproduction

ALPS uses isolated environments for the core methods and native METIS:

```powershell
py -3.13 -m venv .venv-core
.\.venv-core\Scripts\python.exe -m pip install -r config\requirements-core-lock.txt
conda env create -p .\.conda-metis -f config\environment-metis-lock.yml

.\.venv-core\Scripts\python.exe -m tools.verification.verify_clean_environment `
  --metis-python .\.conda-metis\python.exe
```

Success writes `docs/CLEAN_REPRODUCTION.json` with `"status": "passed"`. It
checks the scientific fields used by Tables II--VI, regenerates the experimental
figures, runs unit tests, and replays the cached BSC receipt audit. Timing values
are intentionally excluded from exact comparisons because they are
machine-dependent.

For a quick invariant check:

```powershell
.\.venv-core\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-core\Scripts\python.exe -m tools.verification.verify_semantic_scale
```

## Evidence index

| Manuscript evidence | Recorded output |
|---|---|
| Tables II--III | `results/all_static.csv`, `results/all_horizon.csv` |
| Table IV | `results/headtohead.csv` |
| Table V and retrieval frontier | `results/retrieval_benchmark.csv`, `results/retrieval_frontier.csv` |
| Table VI | `results/prospective_retrieval*.csv` |
| Matched-recall figure | `results/prospective_equal_recall_*.csv` |
| Graph-weight sensitivity | `results/prospective_graph_*.csv` |

## Scope and interpretation

- Workloads contain direct ERC-20 calls; router-mediated execution and complete
  EVM state reconstruction are not evaluated.
- Retrieval results measure scan work, not distributed storage-engine latency.
- The theorem assumes independent same-snapshot regional execution without
  cross-region coordination.
- METIS and TxAllo are sender-local experimental adaptations, not complete
  protocol implementations.
- The BSC receipt audit validates 100 cached samples; historical Ethereum
  receipt responses remain explicitly unresolved.

See [docs/](docs/) for full methodology, limitations, and validation records.
