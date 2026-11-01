# Additional validation scripts and measured results

Run on 2026-10-16. The manuscript now includes the history-only protocol,
prospective allocation/query table, and receipt-audit outcomes. Its retrospective
tables are retained as separate experiments. The additional prospective curve
and detailed per-trace outputs remain in the artifact.

## Prospective allocation and queries

`experiments.run_prospective_retrieval` was run on all four frozen traces with allocation
seeds 0, 1, 2, six regions, seven preset query coverage targets and 250 timing
repetitions. It produced 48 allocation configurations and 2,688 token-level
query measurements. The original datasets and existing result files are unchanged.

The filtered records are ordered by block number and transaction index, then
split 70/30. Sender profiles, per-token amount medians, token feature vocabulary,
static assignments, regional prototypes, and query plans use only the history.
Address IDs are assigned on first appearance, without suffix-dependent numbering.
New token contracts share a fixed OOV feature coordinate, use a frozen
history-wide median raw amount as their fallback scale, and receive distinct
query category IDs. This fallback is a declared heuristic, not calibrated
cross-token economic value. All unseen-token queries scan every region.

The declared placement horizon is `ceil(0.5 * history_call_count)`, selected
before the run and computed without the suffix count. It is not a learned
prediction of stream volume. The suffix is a chronological holdout, not a live
deployment. New senders are placed once and pinned. ALPS uses its semantic,
receiver and load score; balanced sender k-means uses semantic and load terms.
Our METIS and TxAllo causal extensions use receiver affinity plus quadratic
load, with the same first-placement admission rule. They are explicitly labelled
`+CAUSAL`, and do not implement either paper's original streaming protocol.

At the preset 90% historical coverage target, means are:

| Method | Achieved recall | Regions | Rows scanned | Causal active cut | Max/mean load |
|---|---:|---:|---:|---:|---:|
| ALPS | 90.78% | 2.823 | 4,404 | .592 | 1.534 |
| Balanced sender k-means | 79.10% | 2.948 | 5,338 | .773 | 1.671 |
| METIS + causal placement | 88.46% | 4.583 | 8,312 | .097 | 1.191 |
| TxAllo + causal placement | 84.67% | 4.625 | 8,344 | .092 | 1.179 |

Query means weight each suffix token/seed equally, including the one cold token
on C1. Allocation metrics average the 12 trace/seed configurations per method.
Causal interaction coverage averages 34.79%; its denominator includes only
receivers already placed at arrival. The graph extensions have much lower
interaction cuts and lower imbalance than ALPS. No lifetime load cap is claimed.
All allocations preserve sender locality; all target-1.00 and cold-token query
controls return complete suffix results. Tests check frozen scaling, fixed
feature dimensions, historical plans and sender pinning.

`results/prospective_retrieval.png` plots recall against scan work. Timings are
local array scans and are not equal-recall or distributed latency evidence.
Core and METIS measurements used different Python environments, and split runs
overlapped in wall-clock time; latency comparisons are diagnostic only. Each
run's dependency versions and configuration are recorded in `*_config.json`
and combined in `prospective_retrieval_runs.json`.

## Prospective graph-weight sensitivity

The additional declared sweep uses graph weights 0.15, 0.60 and 2.40 in both
history fitting and chronological placement. Other parameters and all seven
query coverage targets stay fixed. All four traces and seeds 0, 1, 2 are used
(36 allocations, 2,016 query observations). No weight is selected using suffix
quality. The 0.15 rerun exactly reproduces the existing ALPS recall, region
and row-scan results. Only scan metrics are interpreted; the sweep uses one
timing repetition, which is not latency evidence.

| Graph weight | Recall at target .90 | Rows scanned | Cut | Max/mean load |
|---|---:|---:|---:|---:|
| .15 | 90.78% | 4,404 | .592 | 1.534 |
| .60 | 91.45% | 5,262 | .434 | 1.614 |
| 2.40 | 89.26% | 6,564 | .179 | 1.551 |

Configuration-level sample standard deviations (12 trace/seed configurations)
are:

| Graph weight | Recall SD (points) | Rows SD | Cut SD | Load SD |
|---|---:|---:|---:|---:|
| .15 | 4.94 | 1,226 | .131 | .046 |
| .60 | 5.29 | 1,592 | .059 | .143 |
| 2.40 | 6.36 | 1,409 | .061 | .209 |

The .67-percentage-point difference between the first two aggregate recalls is
reported descriptively and is not claimed to be statistically significant.
`prospective_graph_per_trace.csv`, `prospective_graph_configurations.csv`, and
`prospective_graph_variability.csv` provide per-trace, per-configuration, and
aggregate variability records respectively.

Because the four traces are heterogeneous workloads rather than four random
draws from one population, the supplement also separates seed variation within
each trace in `prospective_graph_within_trace_variability.csv`. It contains one
row per graph-weight/trace pair, with the mean and sample SD across the three
allocation seeds for recall, rows, cut and load. The across-configuration SD
above remains a descriptive spread over twelve configurations and is not used
as a sampling-error estimate.

Increasing graph weight recovers interaction locality at greater scan cost.
Load does not improve consistently: D3's mean load rises from 1.495 to 1.846
between the endpoints, while C1 improves from 1.514 to 1.357. The result does
not establish universal dominance or an equal-recall latency gain. Raw seed
results, configurations and per-trace aggregates are in `prospective_graph_*`.

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
foreach ($weight in @('0.15','0.6','2.4')) {
  python -m experiments.run_prospective_retrieval --methods ALPS --graph-weight $weight --repetitions 1 --output ('prospective_graph_' + $weight.Replace('.','p'))
}
python -m experiments.summarize_graph_sensitivity
```

## Receipt audit: original attempt

`tools.audit_receipts` samples 100 records uniformly without replacement per trace,
using seed 0. It verifies receipt transaction hash, block and index, status,
and exact Transfer contract/sender/recipient/raw-amount matches. For D1, which
lacks transaction hashes, it first attempts recovery by block and transaction
index and requires the recovered transaction's calldata, nonce and account
fields to match. Successful RPC responses are cached verbatim in JSONL files.

- BSC: all 100 sampled records have status 1 and an exact matching Transfer log.
  The responses were also replayed from the offline cache successfully. This is
  sample evidence, not validation of every BSC call or historical balances.
- D1, D2, D3: the Ethereum RPC probes returned null for the requested records.
  For each trace, one selected record is marked unavailable; the circuit breaker
  leaves the other 99 selected records explicitly **not attempted**. No claim
  about their execution status or dataset correctness follows from this result.
- The original PublicNode endpoints also returned HTTP 403 during probes.
  An alternate official BSC endpoint, `https://bsc-dataseed.bnbchain.org`, worked.
  Ethereum probes through `https://rpc.flashbots.net` remained unresolved.

`results/receipt_audit_summary.json` records the incomplete four-trace attempt;
`results/receipt_bsc_confirmed_summary.json` records the completed BSC sample.
The latter's confirmed subset is in `receipt_bsc_confirmed_confirmed_calls.csv`.
Do not replace the submitted-call nonce workload with this subset. No allocation
analysis on the sampled receipt-confirmed subset was performed: it would not be
comparable to the full-trace results.

Use `--limit 0` for a full-population audit and `--continue-on-error` to continue
after individual missing responses, preferably against a provider with access to
the requested historical data. This may require many RPC requests. Missing
responses are never classified as reverted calls.

## Reproduction

### Completed Ethereum retry

`python -m tools.retry_ethereum_audit --endpoint https://eth.llamarpc.com --timeout 5`
attempted all 100 original seed-0 records on each Ethereum trace, with two
attempts per RPC and no circuit breaker. Each record requests the reported
block/index transaction and checks its hash, sender, token, nonce and calldata
when available. D2/D3 also request receipts using their stored hashes even
when block lookup fails; D1 first needs hash recovery.

The alternate endpoint returned HTTP 525 for all requests: 200 network attempts
on D1 and 400 each on D2/D3. All 300 records remain `unavailable`, with none
classified as reverted or validated. D1's 100 receipt lookups cannot proceed
without recovered hashes. These transport failures establish no fact about
the transactions' execution or data accuracy. Detailed unsuccessful attempts
are supplementary; the paper retains a short limitation statement.

See `results/receipt_ethereum_retry.csv` and its summary JSON for per-record
block and receipt outcomes and errors. This completes the bounded attempt of
the intended sample, **not** Ethereum receipt validation. An accessible
historical provider is still required to resolve it.

From the repository root, in an environment with `config/requirements-metis.txt` installed:

```bash
python -m experiments.run_prospective_retrieval
python -m experiments.summarize_prospective
python -m tools.audit_receipts --limit 100 --ethereum-rpc https://rpc.flashbots.net --bsc-rpc https://bsc-dataseed.bnbchain.org
python -m tools.audit_receipts --offline --traces bsc --limit 100 --bsc-rpc https://bsc-dataseed.bnbchain.org --output receipt_bsc_confirmed
python -B -m unittest discover -s tests -v
```

For the recorded Windows split runs, core methods used the system Python and
METIS used the existing Python 3.12/PyMetis environment. Both were launched with
`OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`:

```bash
python -m experiments.run_prospective_retrieval --methods SENDER-KMEANS-BAL,TXALLO,ALPS --output prospective_core
python -m experiments.run_prospective_retrieval --methods METIS-SENDER --output prospective_metis
python -m experiments.summarize_prospective --parts prospective_core,prospective_metis
```

All 15 existing and new unit tests passed, including graph-weight sensitivity
of first placement. This remains a local run, not an
independent clean-room reproduction or full EVM validation.

A subsequent isolated reproduction used a fresh pinned Python 3.13 virtual
environment for the core methods and a fresh pinned Python 3.12 Conda
environment for native METIS. From an extracted anonymous archive, all 48
prospective allocations, 2,688 query rows, 36 graph-weight allocations and
2,016 sensitivity query rows matched the recorded scientific outputs exactly.
Timing fields were excluded. This is a clean local rerun, not a third-party
audit; its machine-readable report is packaged beside the anonymous ZIP.
