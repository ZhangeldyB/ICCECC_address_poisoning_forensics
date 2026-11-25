"""Hashing and manifest utilities that make the collected evidence verifiable.

A chunk hash is computed only over consensus fields of each log, so anyone can
recompute it from an independent Ethereum node and obtain the same value.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from . import __version__

CONSENSUS_FIELDS = ("address", "topics", "data", "blockNumber", "blockHash",
                    "transactionHash", "transactionIndex", "logIndex")


def canonical_log(log: dict) -> dict:
    out = {}
    for k in CONSENSUS_FIELDS:
        v = log.get(k)
        if isinstance(v, str):
            v = v.lower()
        elif isinstance(v, list):
            v = [str(x).lower() for x in v]
        out[k] = v
    return out


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_logs(logs: list[dict]) -> str:
    rows = sorted((canonical_log(l) for l in logs),
                  key=lambda l: (int(l["blockNumber"], 16), int(l["logIndex"], 16)))
    blob = "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in rows)
    return sha256_hex(blob.encode())


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def seal_manifest(data_dir: Path, meta: dict) -> dict:
    """Order chunk records by block and link them into a hash chain."""
    index = data_dir / "chunk_index.jsonl"
    records = {}
    for line in index.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            records[(r["from_block"], r["to_block"])] = r
    chain = "0" * 64
    chunks = []
    for key in sorted(records):
        r = records[key]
        chain = sha256_hex((chain + r["sha256"]).encode())
        chunks.append({**r, "chain_hash": chain})
    manifest = {**meta, "n_chunks": len(chunks), "n_logs": sum(c["n_logs"] for c in chunks),
                "root_hash": chain, "chunks": chunks}
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def load_manifest(data_dir: Path) -> dict:
    return json.loads((data_dir / "manifest.json").read_text())


def chunk_for_block(manifest: dict, block: int) -> dict | None:
    for c in manifest["chunks"]:
        if c["from_block"] <= block <= c["to_block"]:
            return {k: c[k] for k in ("from_block", "to_block", "n_logs", "sha256", "chain_hash")}
    return None


def code_version() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, timeout=5, cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        commit = ""
    return f"{__version__}+{commit}" if commit else __version__


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
