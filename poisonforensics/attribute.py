"""Step 4: group lookalike addresses into campaigns by shared on-chain infrastructure."""
from __future__ import annotations

import os
import time

import pandas as pd
import requests

from .config import Config
from .rpc import Rpc


class UnionFind:
    def __init__(self):
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _first_funder(address: str, key: str, chain_id: int) -> str | None:
    """Sender of the first transaction that touched `address` (Etherscan V2, free tier)."""
    for _ in range(4):
        r = requests.get("https://api.etherscan.io/v2/api", timeout=30, params={
            "chainid": chain_id, "module": "account", "action": "txlist", "address": address,
            "startblock": 0, "endblock": 99999999, "page": 1, "offset": 1, "sort": "asc", "apikey": key})
        time.sleep(0.4)
        res = r.json().get("result")
        if isinstance(res, list):
            return res[0]["from"].lower() if res else None
    return None


def run(cfg: Config, enrich: bool = False, funders: bool = False) -> None:
    import duckdb
    data_dir, out = cfg.data_dir, cfg.sub("results")
    con = duckdb.connect()
    ds = f"read_parquet('{(data_dir / 'dataset.parquet').as_posix()}')"
    # Addresses and hashes are replaced by 64-bit ids so that millions of rows fit in memory.
    pos = con.execute(f"""SELECT uid, block_number, hash(tx_hash) AS tx, hash(actor) AS a, hash(target) AS t,
                                 n_payoffs, payoff_usd FROM {ds} WHERE label = 1""").df()
    if pos.empty:
        print("No poisoning transfers; nothing to attribute.")
        return
    txs = con.execute(f"""SELECT tx_hash, hash(tx_hash) AS tx, count(*) AS n FROM {ds} WHERE label = 1
                          GROUP BY 1 ORDER BY n DESC, tx_hash""").df()
    print(f"{len(pos):,} poisoning transfers in {len(txs):,} transactions")

    uf = UnionFind()
    # Level 1: lookalikes used in the same (batch) transaction.
    pairs = pos[["tx", "a"]].drop_duplicates()
    for tx, a in zip(pairs["tx"].tolist(), pairs["a"].tolist()):
        uf.union(("a", a), ("tx", tx))

    tx_info = pd.DataFrame(columns=["tx_hash", "tx_from", "tx_to"])
    cache = out / "tx_info.parquet"
    if enrich:
        # Level 2: same submitting account; level 3: same (non-shared) contract.
        att = cfg["attribution"]
        hashes = txs["tx_hash"].head(int(att["max_tx_enrich"])).tolist()   # largest batches first
        if cache.exists():
            tx_info = pd.read_parquet(cache)
        missing = sorted(set(hashes) - set(tx_info["tx_hash"]))
        if missing:
            print(f"Fetching {len(missing):,} transactions (progress is saved; safe to interrupt and re-run) ...")
            rpc = Rpc(os.environ.get("CHAINSTACK_URL", ""), max_rps=att.get("max_rps", 20))
            from tqdm import tqdm
            for i in tqdm(range(0, len(missing), 500), unit="x500"):
                part = missing[i:i + 500]
                res = rpc.batch("eth_getTransactionByHash", [[h] for h in part])
                rows = [(h, r["from"].lower(), (r.get("to") or "").lower()) for h, r in zip(part, res) if r]
                tx_info = pd.concat([tx_info, pd.DataFrame(rows, columns=tx_info.columns)], ignore_index=True)
                tx_info.to_parquet(cache, index=False)
        ids = dict(zip(txs["tx_hash"], txs["tx"].tolist()))
        skip = {a.lower() for a in cfg.token_by_address} | {c.lower() for c in att["shared_contracts"]}
        for h, f, t in tx_info.itertuples(index=False):
            if h not in ids:
                continue
            uf.union(("tx", ids[h]), ("eoa", f))
            if t and t not in skip:
                uf.union(("tx", ids[h]), ("contract", t))
        if funders:
            key = os.environ.get("ETHERSCAN_API_KEY", "")
            if not key:
                print("ETHERSCAN_API_KEY not set; skipping funder lookup.")
            else:
                eoas = tx_info["tx_from"].value_counts().index[:2000]
                print(f"Looking up first funders of {len(eoas):,} submitting accounts ...")
                for e in eoas:
                    f = _first_funder(e, key, int(cfg["chain"]["chain_id"]))
                    if f and f != e:
                        uf.union(("eoa", e), ("funder", f))

    comp_id: dict = {}
    pos["component"] = [comp_id.setdefault(uf.find(("a", a)), len(comp_id)) for a in pos["a"].tolist()]
    camp = pos.groupby("component").agg(
        n_transfers=("uid", "size"), n_lookalikes=("a", "nunique"), n_victims=("t", "nunique"),
        n_txs=("tx", "nunique"), first_block=("block_number", "min"), last_block=("block_number", "max"),
    ).reset_index()
    if len(tx_info):
        ids = dict(zip(txs["tx_hash"], txs["tx"].tolist()))
        ti = tx_info[tx_info["tx_hash"].isin(ids)].copy()
        ti["component"] = [comp_id.get(uf.find(("tx", ids[h]))) for h in ti["tx_hash"]]
        camp = camp.merge(ti.groupby("component")["tx_from"].nunique().rename("n_submitters").reset_index(),
                          on="component", how="left")
    else:
        camp["n_submitters"] = float("nan")
    # Loss per campaign: count each (victim, lookalike) pair once.
    lp = pos[pos["n_payoffs"] > 0].groupby(["component", "t", "a"])["payoff_usd"].max().reset_index()
    loss = lp.groupby("component").agg(n_incidents=("a", "size"), loss_usd=("payoff_usd", "sum")).reset_index()
    camp = camp.merge(loss, on="component", how="left").fillna({"n_incidents": 0, "loss_usd": 0.0})
    camp["n_incidents"] = camp["n_incidents"].astype(int)
    camp = camp.sort_values("n_transfers", ascending=False).reset_index(drop=True)
    camp["campaign_id"] = [f"C{i + 1:05d}" for i in range(len(camp))]
    cid = dict(zip(camp["component"], camp["campaign_id"]))
    actors = con.execute(f"SELECT DISTINCT actor, hash(actor) AS a FROM {ds} WHERE label = 1").df()
    a2c = dict(zip(pos["a"].tolist(), pos["component"].tolist()))
    actors["campaign_id"] = [cid[a2c[a]] for a in actors["a"].tolist()]
    actors[["actor", "campaign_id"]].to_parquet(out / "lookalike_campaign.parquet", index=False)
    camp = camp.drop(columns="component")
    camp.to_parquet(out / "campaigns.parquet", index=False)
    level = ("transaction + submitter + contract" + (" + funder" if funders else "")) if enrich else "transaction only"
    top = camp.head(10)
    print(f"{len(camp):,} campaigns (linkage: {level}). "
          f"Top 10 cover {top['n_transfers'].sum() / camp['n_transfers'].sum():.1%} of poisoning transfers.")
    print(top.to_string(index=False))