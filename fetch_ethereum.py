"""
Fetch real ERC-20 Transfer events from Ethereum mainnet.

Uses a public JSON-RPC endpoint (no API key). Each Transfer log supplies a real
sender, receiver, value and token contract. The token contract plays the role of
the payload *category*: an account that mostly moves USDC and one that mostly
moves WETH are specialists in different goods, which is exactly the structure
the partitioning schemes are asked to exploit.

    python fetch_ethereum.py --blocks 60 --out data/eth_transfers.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.request

RPC = "https://ethereum-rpc.publicnode.com"

# keccak256("Transfer(address,address,uint256)")
TRANSFER_TOPIC = (
    "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
)

# Public endpoints refuse an unfiltered log scan, so we name the contracts.
# These eight high-volume ERC-20 tokens are the payload *categories*: each is a
# distinct class of transferred good, and an account's mix over them is a real,
# measured trading profile rather than one we invented.
TOKENS = {
    "0xdac17f958d2ee523a2206206994597c13d831ec7": "USDT",
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "USDC",
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "WETH",
    "0x6b175474e89094c44da98b954eedeac495271d0f": "DAI",
    "0x514910771af9ca656af840dff83e8264ecf986ca": "LINK",
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": "WBTC",
    "0x95ad61b0a150d79219dcf64e1e6cc01f0b64c4ce": "SHIB",
    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984": "UNI",
}


def rpc(method: str, params: list, retries: int = 4):
    body = json.dumps({"jsonrpc": "2.0", "method": method,
                       "params": params, "id": 1}).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                RPC, data=body,
                headers={"Content-Type": "application/json",
                         "User-Agent": "research-fetch/1.0"},
            )
            with urllib.request.urlopen(req, timeout=45) as r:
                out = json.loads(r.read())
            if "error" in out:
                raise RuntimeError(out["error"])
            return out["result"]
        except Exception as e:                      # transient RPC failures
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{method} failed after {retries} tries: {last}")


def topic_to_addr(t: str) -> str:
    """Topics are 32-byte left-padded; an address is the low 20 bytes."""
    return "0x" + t[-40:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=60,
                    help="how many consecutive blocks to scan")
    ap.add_argument("--chunk", type=int, default=4,
                    help="blocks per eth_getLogs call")
    ap.add_argument("--out", default="data/eth_transfers.csv")
    ap.add_argument("--back", type=int, default=64,
                    help="start this many blocks behind the head (reorg safety)")
    a = ap.parse_args()

    head = int(rpc("eth_blockNumber", []), 16)
    end = head - a.back
    start = end - a.blocks + 1
    print(f"chain head {head}; scanning blocks {start}..{end}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rows = []
    seen_blocks = 0
    FIELDS = ["block", "log_index", "token", "symbol", "from", "to", "value"]

    # Written after every chunk so an interrupted fetch still leaves usable data.
    out_f = open(a.out, "w", newline="")
    writer = csv.DictWriter(out_f, fieldnames=FIELDS)
    writer.writeheader()
    out_f.flush()

    for lo in range(start, end + 1, a.chunk):
        hi = min(lo + a.chunk - 1, end)
        try:
            logs = rpc("eth_getLogs", [{
                "fromBlock": hex(lo),
                "toBlock": hex(hi),
                "address": list(TOKENS),
                "topics": [TRANSFER_TOPIC],
            }])
        except Exception as e:
            print(f"  blocks {lo}-{hi}: skipped ({e})", flush=True)
            continue

        chunk_rows = []
        kept = 0
        for lg in logs:
            # Exactly three topics distinguishes an ERC-20 Transfer from an
            # ERC-721 Transfer (which indexes tokenId as a fourth topic).
            if len(lg.get("topics", [])) != 3:
                continue
            data = lg.get("data", "0x")
            if len(data) < 3:
                continue
            try:
                value = int(data, 16)
            except ValueError:
                continue
            if value == 0:
                continue
            addr = lg["address"].lower()
            chunk_rows.append({
                "block": int(lg["blockNumber"], 16),
                "log_index": int(lg["logIndex"], 16),
                "token": addr,
                "symbol": TOKENS.get(addr, "?"),
                "from": topic_to_addr(lg["topics"][1]).lower(),
                "to": topic_to_addr(lg["topics"][2]).lower(),
                "value": value,
            })
            kept += 1

        writer.writerows(chunk_rows)
        out_f.flush()
        rows.extend(chunk_rows)
        seen_blocks += (hi - lo + 1)
        print(f"  blocks {lo}-{hi}: {kept} ERC-20 transfers "
              f"(running total {len(rows)})", flush=True)
        time.sleep(0.4)                      # stay under the public rate limit

    out_f.close()
    if not rows:
        raise SystemExit("no transfers fetched; check RPC availability")

    # Rewrite once in canonical order (chunks arrive in order, but be certain).
    rows.sort(key=lambda r: (r["block"], r["log_index"]))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    tokens = {}
    accts = set()
    for r in rows:
        tokens[r["token"]] = tokens.get(r["token"], 0) + 1
        accts.add(r["from"])
        accts.add(r["to"])

    print(f"\nwrote {a.out}")
    print(f"  {len(rows)} transfers over {seen_blocks} blocks")
    print(f"  {len(accts)} distinct accounts, {len(tokens)} distinct tokens")
    print("  top tokens by transfer count:")
    for t, c in sorted(tokens.items(), key=lambda kv: -kv[1])[:10]:
        print(f"    {t}  {c}")


if __name__ == "__main__":
    main()
