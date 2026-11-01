# Tools

Utilities are separate from experiments so data handling, packaging, and
verification do not obscure the research runners.

| Tool | Purpose |
|---|---|
| `audit_receipts` | Audits sampled ERC-20 receipts and caches confirmed responses. |
| `retry_ethereum_audit` | Runs bounded retries against an alternate historical Ethereum RPC. |
| `fetch_ethereum`, `fetch_native` | Acquire supported input traces. |
| `merge_parts` | Merges split experimental outputs. |
| `prepare_anonymous_artifact` | Checks and builds the anonymous-review archive. |
| `verification/` | Clean-environment, package, and semantic-scale verifiers. |

Run from the repository root, e.g. `python -m tools.audit_receipts --help`.
