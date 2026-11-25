"""Synthetic transfers for testing the pipeline offline. NOT for results in a paper."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .collect import SCHEMA_COLUMNS
from .config import Config
from .provenance import seal_manifest, sha256_hex

HEX = np.array(list("0123456789abcdef"))


def _addr(rng) -> str:
    return "0x" + "".join(rng.choice(HEX, 40))


def _lookalike(rng, target: str, prefix: int, suffix: int) -> str:
    body = target[2:]
    mid = "".join(rng.choice(HEX, 40 - prefix - suffix))
    return "0x" + body[:prefix] + mid + body[40 - suffix:]


def run(cfg: Config, n_blocks: int = 30000, n_users: int = 3000, n_payments: int = 60000, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    start_block, start_ts = 24_000_000, 1_790_000_000
    users = [_addr(rng) for _ in range(n_users)]
    contacts = [[users[j] for j in rng.choice(n_users, rng.integers(3, 9), replace=False)] for _ in range(n_users)]
    events = []  # (block, tx, from, to, value)
    tx = 0

    def new_tx() -> str:
        nonlocal tx
        tx += 1
        return "0x" + sha256_hex(f"tx{tx}".encode())

    campaigns = [{"batch": [], "block": None} for _ in range(5)]
    pay_blocks = np.sort(rng.integers(0, n_blocks - 600, n_payments))
    for b in pay_blocks:
        u = int(rng.integers(n_users))
        v, r = users[u], contacts[u][int(rng.integers(len(contacts[u])))]
        if v == r:
            continue
        if rng.random() < 0.08:  # benign test transfer before the real one
            events.append((b, new_tx(), v, r, round(float(rng.uniform(0.01, 0.99)), 6)))
        events.append((b + 1, new_tx(), v, r, round(float(np.exp(rng.normal(5, 2))) + 1, 6)))
        if rng.random() < 0.25:  # poisoned
            c = campaigns[int(rng.integers(len(campaigns)))]
            weak = rng.random() < 0.10  # below the similarity rule: label 0, behaviour of an attack
            look = _lookalike(rng, r, 2 if weak else int(rng.integers(3, 6)), int(rng.integers(4, 7)))
            pb = b + 1 + int(rng.integers(1, 40))
            if c["block"] is None or pb - c["block"] > 3 or len(c["batch"]) >= 60:
                c["block"], c["tx"], c["batch"] = pb, new_tx(), []
            c["batch"].append(look)
            if rng.random() < 0.6:
                events.append((c["block"], c["tx"], v, look, 0.0))
            else:
                events.append((c["block"], c["tx"], look, v, round(float(rng.uniform(0.001, 0.2)), 6)))
            if not weak and rng.random() < 0.02:  # victim copies the wrong address
                events.append((pb + int(rng.integers(50, 500)), new_tx(), v, look,
                               round(float(np.exp(rng.normal(7, 1.5))) + 100, 6)))
    for _ in range(n_payments // 15):  # unrelated benign dust and zero-value noise
        a, b_ = users[int(rng.integers(n_users))], users[int(rng.integers(n_users))]
        if a != b_:
            events.append((int(rng.integers(0, n_blocks)), new_tx(), a, b_,
                           0.0 if rng.random() < 0.3 else round(float(rng.uniform(0.01, 0.99)), 6)))

    df = pd.DataFrame(events, columns=["rel", "tx_hash", "from_addr", "to_addr", "value_usd"]).sort_values(
        ["rel", "tx_hash"], kind="stable").reset_index(drop=True)
    df["block_number"] = start_block + df["rel"]
    df["log_index"] = df.groupby("block_number").cumcount()
    df["tx_index"] = 0
    df["ts"] = start_ts + 12 * df["rel"]
    df["ts_exact"] = True
    df["token"] = np.where(rng.random(len(df)) < 0.7, "USDT", "USDC")
    df["value_raw"] = (df["value_usd"] * 1e6).round().astype("int64").astype(str)

    data_dir, parts = cfg.data_dir, cfg.sub("transfers")
    for old in parts.glob("*.parquet"):
        old.unlink()
    df[SCHEMA_COLUMNS].to_parquet(parts / "part_synth.parquet", index=False)
    with open(data_dir / "chunk_index.jsonl", "w") as f:
        for a in range(start_block, start_block + n_blocks, 1000):
            part = df[(df["block_number"] >= a) & (df["block_number"] < a + 1000)]
            f.write(json.dumps({"from_block": a, "to_block": a + 999, "n_logs": len(part),
                                "sha256": sha256_hex(part[SCHEMA_COLUMNS].to_csv(index=False).encode()),
                                "fetched_at": "synthetic"}) + "\n")
    seal_manifest(data_dir, {"synthetic": True, "chain": cfg["chain"], "start_block": start_block,
                             "end_block": start_block + n_blocks - 1, "node_client": "synthetic",
                             "sealed_at": "synthetic", "config_sha256": cfg.digest()})
    print(f"Wrote {len(df):,} synthetic transfers to {parts}/ (for testing only)")
