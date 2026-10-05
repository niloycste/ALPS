# ALPS conference-paper artifact

This directory contains the data, simulator, result tables, figures, and IEEE
manuscript for **ALPS: Admissible Locality and Payload Semantics for
Region-Based Parallel Mining**.

ALPS assigns each sender as one indivisible unit. It combines payload
similarity, active-account interaction locality, and regional load under a hard
capacity constraint. The structural sender-local rule protects per-token debit
keys and account-wide nonce order in the paper's independent-barrier execution
model.

## Evidence and scope

The manuscript uses four disjoint direct-call traces containing 145,372 real
EVM calls:

| ID | Chain and date | Calls | represented/scanned blocks |
|---|---|---:|---:|
| D1 | Ethereum, 2026-08-14 | 34,326 | 513/714 |
| D2 | Ethereum, 2026-08-13 | 39,425 | 713/714 |
| D3 | Ethereum, 2026-08-07 | 45,893 | 499/500 |
| C1 | BNB Smart Chain | 25,728 | 2,489/2,500 |

The secondary `eth_all.csv` trace contains 17,445 retained ERC-20 `Transfer`
events. It broadens balance-locality coverage but is not used for real-nonce
claims because an event log does not identify the outer transaction signer.
Each `*.manifest.json` file records bounds, filters, provenance, and a SHA-256
hash.

The reported **active-account interaction cut** is conditional on the receiver
also appearing as an active sender. This makes the compared placement population
identical for all sender-local methods. Receiver-only addresses are reported as
uncovered rather than automatically labelled cross-region. Static coverage is
53.8% (D1), 56.6% (D2), 60.6% (D3), and 45.7% (C1).

## Main files

```text
data/*.csv                 measured traces
data/*.manifest.json       collection provenance and hashes
sim/partition.py           baselines, ALPS, TxAllo hook, and metrics
sim/metis.py               optional sender-local weighted METIS baseline
sim/experiment.py          sequential and retry-aware barrier execution
sim/online.py              chronological sender-pinning allocation
sim/native.py              direct-call adapter with measured nonce order
sim/ethereum.py            Transfer-event adapter (no real signer nonce)
tests/test_validity.py      invariant, retry, temporal, and metric tests
run_all_traces.py           complete cross-trace evidence battery
run_headtohead.py           three-seed TxAllo allocation comparison
run_txallo_sensitivity.py   six-region eta/sweep-cap audit
run_allocator_scaling.py    real-prefix ALPS time/memory scaling
run_alpha_sensitivity.py    five-seed hard-cap sensitivity audit
run_allocator_comparison.py fresh-process baseline runtime comparison
make_figures.py             publication figures (vector PDF + inspection PNG)
ALPS_conference_submission/main.tex
                            IEEE conference manuscript
ALPS_conference_submission/references.bib
                            focused 21-entry bibliography
ALPS_conference_submission/figures/*.pdf
                            four publication figures
ALPS_conference_submission/alps_conference_paper_clean.pdf
                            verified final manuscript
```

## Reproduce the manuscript evidence

Run from this directory:

```bash
pip install -r requirements-metis.txt
python -B -m unittest discover -s tests -v
python -B run_all_traces.py --workers 4
python -B run_headtohead.py
python -B run_txallo_sensitivity.py
python -B run_allocator_scaling.py
python -B run_alpha_sensitivity.py
python -B run_allocator_comparison.py
python -B make_figures.py
cd ALPS_conference_submission
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
Copy-Item main.pdf alps_conference_paper_clean.pdf  # PowerShell
# cp main.pdf alps_conference_paper_clean.pdf       # macOS/Linux
```

The complete battery is intentionally expensive. A targeted rerun accepts a
comma-separated subset, for example:

```bash
python -B run_all_traces.py --workers 4 --only static,analysis
```

Memory-constrained machines can rerun selected traces serially and merge only
those rows into the existing aggregate files:

```bash
python -B run_all_traces.py --workers 1 --only analysis \
  --traces eth_aug14,eth_aug13 --merge-existing
```

`pymetis` is optional for the original ALPS experiments and required only for
`METIS-SENDER`, `run_headtohead.py`, and `run_allocator_comparison.py`. The
METIS baseline uses active senders as transaction-weighted vertices and maps
every transaction through its sender, preserving sender and guarded-key
locality. Receiver-only accounts are excluded to match ALPS's graph and the
reported conditional interaction-cut population.

The alpha summary reports 95% t-intervals over five allocator initializations.
These intervals quantify initialization variability, not uncertainty over the
underlying blockchain workload.

`run_final.py` is retained for focused/legacy outputs. Its optional proof-of-work
benchmark now uses the direct-call trace with genuine signer nonce order, but it
is not evidence quoted by the paper and should be run only on an idle machine:

```bash
python -B run_final.py --throughput-only --reps 6
```

## Six-region headline (D1)

| Method | Split guarded keys | Rounds | Deferred checks | R90 | Load | Active cut |
|---|---:|---:|---:|---:|---:|---:|
| ADDR-MOD | 0 | 33 | 0 | 5.88 | 1.43 | .837 |
| TX-KMEANS | 1,021 | 1,068 | 1,562,573 | **1.38** | 2.27 | .425 |
| SENDER-KMEANS-BAL | 0 | **27** | **0** | 2.25 | **1.15** | .646 |
| ALPS | **0** | **27** | **0** | 3.00 | **1.15** | **.260** |

Across all 24 direct-trace/region settings, ALPS lowers the active-account cut
versus address hashing by 35.2--75.9% and versus balanced sender k-means by
12.8--82.5%. Against the balanced baseline it strictly improves R90 in 19
settings, ties in two, and trades retrieval for cut reduction in three. This is
a deployment trade-off, not universal metric dominance.

At the six-region headline, the TxAllo one-factor audit varies
`eta={1.25,1.5,2,3,4}` and sweep caps `{4,8,16,32}` over four traces and three
seeds. ALPS retains lower R90 in every trace/seed setting for every eta; TxAllo
retains lower cut through eta=3 and in 10/12 settings at eta=4. All sweep caps
produce identical allocation metrics at eta=2. The raw and summary outputs are
`results/txallo_sensitivity*.csv`.

On real D1 prefixes from 5,000 to 34,326 calls, three fresh-process runs take
median total ALPS allocation times from 0.89 to 9.94 seconds. Full-trace peak
working set is 226.2 MiB on the recorded Intel i9-14900 host, with no capacity
relaxation. Outputs are `results/allocator_scaling*.csv`.

## Important limits

- The theorem applies to independent same-snapshot regional execution without
  locks, atomic cross-shard settlement, optimistic re-execution, or equivalent
  coordination.
- The balance sweep is a controlled safety witness, not reconstructed historical
  balances or an estimate of real theft.
- Direct-call records are submitted workloads: receipts were not queried, so
  reverted calls are not distinguished. Router- and contract-mediated calls
  are excluded, and the measured spans are hours rather than a longitudinal study.
- Signer nonces provide observed-stream order only, and temporal feature scaling
  uses full-trace per-token medians; the chronological test is therefore not a
  strictly online preprocessing evaluation.
- Chronological placement is a fixed-horizon replay whose target uses the known
  evaluation-suffix length; it is not a fully online deployment experiment.
- The TxAllo result reimplements its allocation objective with `eta=2` and at
  most eight coordinate sweeps, not its consensus or settlement protocol.
- Barrier rounds and committed transactions per round measure dependency-limited
  scheduling throughput. The paper makes no production-TPS claim.
- Chronological evaluation is a robustness check, not a novelty claim;
  A-TxAllo already evaluates a temporal split with new-node placement.
