"""Resumable RPC receipt audit of a reproducible sample or the full call traces.

Default: 100 uniformly sampled records per trace, seed 0. --limit 0 audits all.
Writes row classifications, raw successful RPC responses, summaries and hashes.
No receipt-confirmed claim is made for unavailable or inconsistent responses.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
TRACES = {"eth_aug14": ("native_all.csv", "https://ethereum-rpc.publicnode.com"),
          "eth_aug13": ("native_day2.csv", "https://ethereum-rpc.publicnode.com"),
          "eth_aug07": ("eth_week2.csv", "https://ethereum-rpc.publicnode.com"),
          "bsc": ("bsc_trace.csv", "https://bsc-rpc.publicnode.com")}
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"


class NotAttempted(RuntimeError):
    """The endpoint circuit breaker prevented a network query for this row."""


class RPC:
    def __init__(self, endpoint, cache, timeout, offline=False, continue_on_error=False):
        self.endpoint, self.cache_path, self.timeout = endpoint, cache, timeout
        self.offline = offline
        self.continue_on_error = continue_on_error
        self.cache = {}
        self.failed = None
        self.network_requests = 0
        if cache.exists():
            for line in cache.read_text().splitlines():
                row = json.loads(line)
                self.cache[row["key"]] = row["result"]

    def get(self, method, params):
        key = json.dumps([self.endpoint, method, params], separators=(",", ":"))
        if key in self.cache:
            return self.cache[key]
        if self.offline or self.failed:
            raise NotAttempted(self.failed or "response not present in offline cache")
        payload = json.dumps(dict(jsonrpc="2.0", id=1, method=method, params=params)).encode()
        req = urllib.request.Request(self.endpoint, data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "ALPS-receipt-audit/1.0"})
        for attempt in range(2):
            try:
                self.network_requests += 1
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    body = json.load(response)
                if body.get("error"):
                    raise RuntimeError(json.dumps(body["error"]))
                result = body.get("result")
                if result is None:
                    raise RuntimeError("RPC returned null: historical record unavailable")
                self.cache[key] = result
                with self.cache_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(dict(key=key, result=result)) + "\n")
                return result
            except Exception as exc:
                if attempt == 0:
                    time.sleep(.5)
                else:
                    # Fail fast on this endpoint for this run; a later run retries.
                    message = f"{type(exc).__name__}: {exc}"
                    if not self.continue_on_error:
                        self.failed = message
                    raise RuntimeError(message) from exc


def validate_transaction(tx, row):
    data = tx.get("input", "").lower()
    return (tx.get("from", "").lower() == row["from"].lower()
        and tx.get("to", "").lower() == row["token"].lower()
        and int(tx["nonce"], 16) == int(row["nonce"])
        and int(tx["blockNumber"], 16) == int(row["block"])
        and int(tx["transactionIndex"], 16) == int(row["tx_index"])
        and data.startswith("0xa9059cbb") and len(data) >= 138
        and "0x" + data[34:74] == row["to"].lower()
        and int(data[74:138], 16) == int(row["value"]))


def classify(receipt, row, tx_hash):
    if (receipt.get("transactionHash", "").lower() != tx_hash.lower()
        or int(receipt["blockNumber"], 16) != int(row["block"])
        or int(receipt["transactionIndex"], 16) != int(row["tx_index"])):
        return "record_mismatch", 0
    status = int(receipt["status"], 16)
    if status == 0:
        return "reverted", 0
    if status != 1:
        return "invalid_receipt", 0
    matching = 0
    for log in receipt.get("logs", []):
        topics = log.get("topics", [])
        if (log.get("address", "").lower() != row["token"].lower()
            or len(topics) != 3 or topics[0].lower() != TRANSFER_TOPIC):
            continue
        if ("0x" + topics[1][-40:].lower() == row["from"].lower()
            and "0x" + topics[2][-40:].lower() == row["to"].lower()
            and int(log.get("data", "0x0"), 16) == int(row["value"])):
            matching += 1
    return ("success_matching_transfer" if matching else "success_no_exact_transfer"), matching


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", default=",".join(TRACES))
    p.add_argument("--limit", type=int, default=100, help="sample per trace; zero means all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=15)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--continue-on-error", action="store_true",
                   help="try subsequent records even after failed/missing RPC responses")
    p.add_argument("--ethereum-rpc", default=TRACES["eth_aug14"][1])
    p.add_argument("--bsc-rpc", default=TRACES["bsc"][1])
    p.add_argument("--output", default="receipt_audit")
    args = p.parse_args()
    if args.limit < 0:
        p.error("limit must be zero or positive")
    out = ROOT / "results"
    cache = out / "receipt_cache"
    cache.mkdir(parents=True, exist_ok=True)
    results, summaries, confirmed = [], [], []
    for trace in args.traces.split(","):
        filename, _ = TRACES[trace]
        path = ROOT / "data" / filename
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        selected = (list(range(len(rows))) if not args.limit else
                    sorted(random.Random(args.seed).sample(range(len(rows)), min(args.limit, len(rows)))))
        endpoint = args.bsc_rpc if trace == "bsc" else args.ethereum_rpc
        rpc = RPC(endpoint, cache / (trace + ".jsonl"), args.timeout, args.offline,
                  args.continue_on_error)
        counts = Counter()
        for index in selected:
            row = rows[index]
            tx_hash = row.get("tx_hash", "")
            result = dict(trace=trace, csv_row=index, block=row["block"],
                tx_index=row["tx_index"], token=row["token"], symbol=row["symbol"],
                tx_hash=tx_hash, classification="unavailable", matching_logs=0,
                gas_used="", error="")
            try:
                if not tx_hash:
                    tx = rpc.get("eth_getTransactionByBlockNumberAndIndex",
                                 [hex(int(row["block"])), hex(int(row["tx_index"]))])
                    if not validate_transaction(tx, row):
                        result["classification"] = "record_mismatch"
                        raise ValueError("recovered transaction does not match stored calldata/account fields")
                    tx_hash = tx["hash"]
                    result["tx_hash"] = tx_hash
                receipt = rpc.get("eth_getTransactionReceipt", [tx_hash])
                status, matches = classify(receipt, row, tx_hash)
                result.update(classification=status, matching_logs=matches,
                              gas_used=int(receipt["gasUsed"], 16))
            except NotAttempted as exc:
                result["classification"] = "not_attempted"
                result["error"] = str(exc)
            except Exception as exc:
                result["error"] = str(exc)
            counts[result["classification"]] += 1
            results.append(result)
            if result["classification"] == "success_matching_transfer":
                confirmed.append(dict(trace=trace, csv_row=index, block=row["block"],
                    tx_index=row["tx_index"], tx_hash=result["tx_hash"],
                    token=row["token"], symbol=row["symbol"], sender=row["from"],
                    receiver=row["to"], value=row["value"], nonce=row["nonce"]))
        summaries.append(dict(trace=trace, population=len(rows), selected=len(selected),
            rpc_endpoint=endpoint,
            selection="all" if not args.limit else "uniform_without_replacement",
            seed=args.seed, input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            classifications=dict(counts), network_requests=rpc.network_requests,
            rpc_error=rpc.failed))
        print(json.dumps(summaries[-1]), flush=True)
        with (out / (args.output + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        (out / (args.output + "_summary.json")).write_text(json.dumps(
            dict(limit_per_trace=args.limit, sample_seed=args.seed, traces=summaries,
                 full_population_audit=args.limit == 0,
                 sample_fully_resolved=all(not any(s["classifications"].get(k, 0) for k in
                     ("unavailable", "not_attempted", "invalid_receipt", "record_mismatch")) for s in summaries),
                 interpretation="exact event matching is a diagnostic; nonstandard token behaviour is not inferred"), indent=2) + "\n")
    if confirmed:
        with (out / (args.output + "_confirmed_calls.csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(confirmed[0]))
            writer.writeheader()
            writer.writerows(confirmed)


if __name__ == "__main__":
    main()
