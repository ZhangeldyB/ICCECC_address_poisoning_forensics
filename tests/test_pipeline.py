"""Offline tests: run with  python -m pytest -q"""
import pandas as pd
import yaml

from poisonforensics import dataset, synth
from poisonforensics.collect import decode_logs
from poisonforensics.config import TRANSFER_TOPIC, Config
from poisonforensics.provenance import hash_logs

USDT = "0xdac17f958d2ee523a2206206994597c13d831ec7"


def _log(block, index, value, frm="11" * 20, to="22" * 20):
    return {"address": USDT, "topics": [TRANSFER_TOPIC, "0x" + "0" * 24 + frm, "0x" + "0" * 24 + to],
            "data": hex(value), "blockNumber": hex(block), "blockHash": "0x" + "ab" * 32,
            "transactionHash": "0x" + "cd" * 32, "transactionIndex": "0x1", "logIndex": hex(index)}


def test_decode_logs():
    df = decode_logs([_log(100, 3, 2_500_000), _log(101, 0, 0)], {USDT: ("USDT", 6)}, 100, 1_000)
    assert df["value_usd"].tolist() == [2.5, 0.0]
    assert df["from_addr"][0] == "0x" + "11" * 20 and df["ts"].tolist() == [1_000, 1_012]


def test_chunk_hash_is_order_independent_and_ignores_provider_fields():
    a, b = _log(100, 3, 1), _log(100, 4, 2)
    assert hash_logs([a, b]) == hash_logs([dict(b, blockTimestamp="0x1"), a])
    assert hash_logs([a, b]) != hash_logs([a, _log(100, 4, 3)])


def test_lookalike_labelling_end_to_end(tmp_path):
    raw = yaml.safe_load(open("config.yaml"))
    raw["paths"]["data_dir"] = str(tmp_path / "d")
    cfg = Config(raw=raw, path=tmp_path / "c.yaml")
    synth.run(cfg, n_blocks=4000, n_users=300, n_payments=4000)
    dataset.run(cfg)
    ds = pd.read_parquet(cfg.data_dir / "dataset.parquet")
    pos = ds[ds["label"] == 1]
    assert len(pos) > 100
    assert (pos["sim_prefix_len"] >= 3).all() and (pos["sim_suffix_len"] >= 4).all()
    assert (pos["actor"] != pos["mimicked"]).all()
    assert (pos["actor"].str[-4:] == pos["mimicked"].str[-4:]).all()
