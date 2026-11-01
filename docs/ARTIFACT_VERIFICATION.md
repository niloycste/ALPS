# Local artifact verification

> This is a dated verification record. Use
> [REPRODUCE_CLEAN.md](REPRODUCE_CLEAN.md) and `CLEAN_REPRODUCTION.json` for the
> current reproducible workflow and result.

Checked on 2026-10-16. This is a local consistency check by the manuscript's
editing assistant, not independent replication or external certification.

Subsequent script runs on the same date are documented in
[NEW_VALIDATION_RESULTS.md](NEW_VALIDATION_RESULTS.md): prospective retrieval and
a partly resolved receipt sample. They supplement the original checks below;
the manuscript now reports their protocol and measured outcomes separately
from the retrospective benchmarks, including the unresolved Ethereum audit.

## Completed checks

- `tools.verification.verify_semantic_scale` confirms that static ALPS computes the semantic
  scale once from every nonzero sender-to-initial-prototype squared distance,
  rather than from assigned pairs, and does not reassign it during refinement.
  `SEMANTIC_SCALE_VERIFICATION.json` records the executable check.
- `experiments.analyze_prospective_equal_recall` regenerates per-configuration,
  per-trace, and aggregate matched-recall interpolation from the seven plans
  frozen before suffix evaluation. It also records the explicit assumptions
  and break-even transaction counts for three deployment scenarios.

- All 15 tests pass using `python -B -m unittest discover -s tests -v`.
  They cover balance-key isolation, the split-debit counterexample, nonce retry,
  sender locality, constrained refinement, temporal pinning, and graph population.
  The METIS assignment unit test substitutes its graph partitioner; it does not
  independently validate the native METIS algorithm. Real METIS was used for
  the recorded retrieval coverage sweep.
- All four direct-call data files match their manifest SHA-256 hashes and row
  counts: D1 34,326; D2 39,425; D3 45,893; C1 25,728, totalling 145,372.
  Represented block counts are respectively 513, 713, 499, and 2,489.
- Retrieval and head-to-head scripts explicitly use allocation seeds 0, 1, 2.
  Static headline sweeps use seed 0; runtime and capacity sensitivity scripts
  have separately documented repetitions. These are not independent chain samples.
- The 2,604-row coverage sweep has seven fixed training targets and 93 eligible
  token/seed observations per method/target. All 372 observations at target 0.90
  reproduce the existing History-90 region counts, scanned rows and recall.
  The target-1.00 controls contact all six regions and scan every suffix row.
- The chronological implementation squares Euclidean distances before taking
  their median and normalising. Equation (4) explicitly uses squared distances.
- The delivered manuscript PDF has eleven pages, including the full pipeline
  overview, an anonymous title block and
  `/Author=Anonymous`. The latest LaTeX log has no unresolved references or
  overfull-box warnings; rendered equations, tables and figures were inspected.

## Corrections made during this check

- Updated stale README scaling numbers to the recorded 2.50--12.60 seconds and
  248.4 MiB, consistent with the manuscript and scaling summary CSV.
- Clarified that the default retrieval scripts also require PyMetis.
- The working TeX retains an inactive camera-ready identity branch. The curated
  review archive removes that branch entirely from its source copy. It excludes
  Git history, previous archives, logs, caches, and the experimental LNCS draft.
  The working TeX and rendered scientific content are unchanged by packaging.

## Archive

Run `python -m tools.prepare_anonymous_artifact` from the repository root to check the
local files and write `../output/submission/ALPS_anonymous_artifact.zip`.
The archive contains code, data, results, active manuscript figures, anonymous
TeX and the current PDF. `SHA256SUMS.json` records content hashes for its files;
`LOCAL_CHECKS.json` records the automated checks. The archive is read back and
its hashes verified after creation. The checker additionally requires `pypdf`.

## Limits of verification

### Additional isolated package checks

The full clean-environment rerun passed on 2026-10-17. A fresh Python 3.13
virtual environment installed `config/requirements-core-lock.txt`; a fresh Python 3.12
Conda environment installed the native METIS stack in
`config/environment-metis-lock.yml`. From a newly extracted anonymous ZIP, the checker
regenerated all 150 static rows, 48 horizon rows, 360 head-to-head rows, 1,140
retrospective benchmark rows, 2,604 retrieval-frontier rows, 48 prospective
allocations and 2,688 prospective query rows, and 36 graph-weight allocations
with 2,016 query rows. It also regenerated the three experimental figures, ran
all 15 tests, and replayed 100 cached BSC receipts. Every scientific field
matched exactly; timing fields were intentionally excluded.
The machine-readable report is `../output/submission/CLEAN_REPRODUCTION.json`.

The first unpinned clean install selected NumPy 2.5.4 and scikit-learn 1.9.1,
which changed k-means partitions relative to the recorded NumPy 2.4.2 and
scikit-learn 1.8.0 run. The new lock files fix that reproducibility gap. This
finding concerns dependency resolution rather than the supplied recorded
results; the locked rerun reproduced them exactly.

`tools.verification.verify_packaged_artifact` extracts the curated ZIP into a new temporary
directory, verifies every inventory hash, runs all 15 tests, reconstructs the
prospective headline table and graph-weight summaries, replays all 100 cached
C1 receipts, and reruns D1/seed-0 prospective allocation for all four methods.
The METIS reproduction uses the native partitioner, rather than the unit-test
substitute. It compares recall, scanned rows and contacted regions to the
packaged results; latency is excluded. The report is saved separately at
`../output/submission/artifact_recheck.json` and identifies the checked ZIP hash.

This remains a local clean-environment reproduction performed by the manuscript
team, not an independent third-party audit or external certification.

The prospective graph-weight sweep and complete bounded Ethereum retry are
documented in `NEW_VALIDATION_RESULTS.md`. All 300 Ethereum sample records
were attempted via the alternate endpoint, but HTTP 525 errors left them
unresolved. The manuscript retains a short limitation; the artifact records
the unsuccessful requests separately from reverted transactions.

### Limits of the original packaging check

- No fresh chain download, receipt reconstruction, full allocator sweep, or
  timing experiment was run during this final packaging check. Hash matching
  verifies the supplied bytes, not the accuracy of the original RPC response.
- D1's acquisition lacks transaction hashes and a complete scanned-block ledger,
  as its manifest states. The secondary `eth_all.csv` log trace has no original
  acquisition manifest; its archive hash establishes only the packaged bytes.
- Historical query plans still use retrospective full-trace allocations. The
  new figure measures scan work, not matched-recall latency. C1's one suffix-only
  BUSD call is excluded from the historical-plan population.
- Migration, index rebuilds, mediated transactions, receipt outcomes and long-term
  workload drift remain outside the experiments. Numerical amortisation examples
  are assumptions, not deployment measurements.
- Anonymity checks cover known identity fields, included text and PDF metadata;
  they do not certify that research content cannot be linked to public work.
- The current manuscript uses IEEE conference formatting. The exact selected
  venue/year's submission rules must be checked before upload. This archive
  does not claim LNCS or DASFAA formatting compliance.
