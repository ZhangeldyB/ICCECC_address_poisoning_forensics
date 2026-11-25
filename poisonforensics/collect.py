"""Step 1: download ERC-20 Transfer logs for the configured tokens and window."""
from __future__ import annotations

import gzip
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

from .config import TRANSFER_TOPIC, Config
from .provenance import code_version, hash_logs, seal_manifest
from .rpc import Rpc, RpcError

SCHEMA_COLUMNS = ["block_number", "ts", "ts_exact", "tx_hash", "tx_index", "log_index",
                  "token", "from_addr", "to_addr", "value_raw", "value_usd"]


def decode_logs(logs: list[dict], tokens: dict, anchor_block: int, anchor_ts: int) -> pd.DataFrame:
    rows = []
    for l in logs:
        if l.get("removed"):
            continue
        topics = l.get("topics") or []
        tok = tokens.get(l["address"].lower())
        if tok is None or len(topics) != 3 or topics[0].lower() != TRANSFER_TOPIC:
            continue
        name, decimals = tok
        data = l.get("data") or "0x"
        raw = int(data, 16) if len(data) > 2 else 0
        block = int(l["blockNumber"], 16)
        if l.get("blockTimestamp"):
            ts, exact = int(l["blockTimestamp"], 16), True
        else:  # 12-second slots; exact enough for ordering, flagged as approximate
            ts, exact = anchor_ts + 12 * (block - anchor_block), False
        rows.append((block, ts, exact, l["transactionHash"].lower(), int(l["transactionIndex"], 16),
                     int(l["logIndex"], 16), name, "0x" + topics[1][-40:].lower(),
                     "0x" + topics[2][-40:].lower(), str(raw), raw / 10 ** decimals))
    return pd.DataFrame(rows, columns=SCHEMA_COLUMNS)


def _get_logs(rpc: Rpc, addresses: list[str], a: int, b: int) -> list[dict]:
    """eth_getLogs for [a, b]; splits the range when the provider refuses it."""
    try:
        return rpc.call("eth_getLogs", [{"fromBlock": hex(a), "toBlock": hex(b),
                                         "address": addresses, "topics": [TRANSFER_TOPIC]}]) or []
    except RpcError as e:
        if b > a and e.is_range_problem:
            mid = (a + b) // 2
            return _get_logs(rpc, addresses, a, mid) + _get_logs(rpc, addresses, mid + 1, b)
        raise


def run(cfg: Config) -> None:
    rpc = Rpc(os.environ.get("CHAINSTACK_URL", ""), max_rps=cfg["collect"]["max_rps"])
    tokens = cfg.token_by_address
    addresses = list(tokens)
    data_dir = cfg.data_dir
    parts, raw_dir = cfg.sub("transfers"), cfg.sub("raw")
    index_path = data_dir / "chunk_index.jsonl"

    start_ts, end_ts = cfg.window_ts()
    print("Resolving block range for the window ...")
    start_block = rpc.block_at_or_after(start_ts)
    end_block = rpc.block_at_or_after(end_ts) - 1
    print(f"Blocks {start_block} .. {end_block} ({end_block - start_block + 1:,} blocks)")

    done = set()
    if index_path.exists():
        for line in index_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["from_block"], r["to_block"]))

    step = int(cfg["collect"]["chunk_blocks"])
    chunks = [(a, min(a + step - 1, end_block)) for a in range(start_block, end_block + 1, step)]
    todo = [c for c in chunks if c not in done]
    print(f"{len(chunks):,} chunks, {len(todo):,} to download")

    lock = threading.Lock()
    keep_raw = bool(cfg["collect"].get("keep_raw", True))

    def work(chunk):
        a, b = chunk
        logs = _get_logs(rpc, addresses, a, b)
        anchor_ts = 0
        if logs and not logs[0].get("blockTimestamp"):
            anchor_ts = rpc.block_timestamp(a)
        df = decode_logs(logs, tokens, a, anchor_ts)
        df.to_parquet(parts / f"part_{a}_{b}.parquet", index=False)
        if keep_raw:
            with gzip.open(raw_dir / f"logs_{a}_{b}.jsonl.gz", "wt") as f:
                for l in logs:
                    f.write(json.dumps(l, separators=(",", ":")) + "\n")
        rec = {"from_block": a, "to_block": b, "n_logs": len(logs), "sha256": hash_logs(logs),
               "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        with lock:
            with open(index_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return len(logs)

    total = 0
    with ThreadPoolExecutor(max_workers=int(cfg["collect"]["workers"])) as ex:
        futures = [ex.submit(work, c) for c in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), unit="chunk"):
            total += fut.result()

    manifest = seal_manifest(data_dir, {
        "synthetic": False,
        "chain": cfg["chain"], "tokens": cfg["tokens"], "window": cfg["window"],
        "start_block": start_block, "end_block": end_block,
        "node_client": rpc.client_version(),
        "rpc_host": rpc.url.split("/")[2],   # host only; the key in the path is never stored
        "code_version": code_version(), "config_sha256": cfg.digest(),
        "sealed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    print(f"Downloaded {total:,} new logs. Manifest sealed: {manifest['n_logs']:,} logs, "
          f"root hash {manifest['root_hash'][:16]}…")
