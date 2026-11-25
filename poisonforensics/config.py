"""Configuration and environment loading."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
ZERO_ADDRESS = "0x" + "0" * 40


def load_env(path: str = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines); existing variables win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Config:
    raw: dict
    path: Path

    @classmethod
    def load(cls, path: str = "config.yaml") -> "Config":
        p = Path(path)
        return cls(raw=yaml.safe_load(p.read_text()), path=p)

    def __getitem__(self, key: str):
        return self.raw[key]

    @property
    def data_dir(self) -> Path:
        d = Path(self.raw["paths"]["data_dir"])
        d.mkdir(parents=True, exist_ok=True)
        return d

    def sub(self, name: str) -> Path:
        d = self.data_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def token_by_address(self) -> dict:
        return {v["address"].lower(): (k, int(v["decimals"])) for k, v in self.raw["tokens"].items()}

    def window_ts(self) -> tuple[int, int]:
        def parse(s: str) -> int:
            return int(datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp())
        return parse(self.raw["window"]["start"]), parse(self.raw["window"]["end"])

    def digest(self) -> str:
        """SHA-256 of the canonical configuration; recorded in every evidence report."""
        blob = json.dumps(self.raw, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()
