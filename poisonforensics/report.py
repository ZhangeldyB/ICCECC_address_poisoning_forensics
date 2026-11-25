"""Step 5: per-incident evidence reports and their independent verification."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import duckdb
import pandas as pd

from .config import TRANSFER_TOPIC, Config
from .model import apply_calibration, load_primary, shap_values
from .provenance import canonical_json, chunk_for_block, code_version, load_manifest, sha256_hex
from .rpc import Rpc

CAVEATS = [
    "The poisoning label is derived from address similarity and timing; it shows that the lookalike "
    "address was placed in the victim's history, not who controls it.",
    "A payment to the lookalike address is consistent with a copy-paste error but does not by itself "
    "exclude an intentional transfer; intent must be established by the investigator.",
    "Campaign membership links addresses through shared transactions, submitting accounts and "
    "contracts; it does not identify a natural person.",
    "The classifier score is a calibrated probability estimated on historical data and should be "
    "read together with the listed feature contributions, never as sole evidence.",
]


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _align(a: str, b: str) -> str:
    return "".join("^" if x == y else " " for x, y in zip(a, b))


def build_report(cfg: Config, victim: str, lookalike: str, ctx: dict) -> dict:
    con, manifest, booster, meta = ctx["con"], ctx["manifest"], ctx["booster"], ctx["meta"]
    tokens = {name: v["address"].lower() for name, v in cfg["tokens"].items()}
    rows = con.execute(f"SELECT * FROM read_parquet('{ctx['ds']}') WHERE target = ? AND actor = ? AND label = 1 "
                       "ORDER BY uid", [victim, lookalike]).df()
    if rows.empty:
        raise ValueError(f"No poisoning transfers for victim {victim} / lookalike {lookalike}")
    mimicked = rows["mimicked"].iloc[0]
    first_uid = int(rows["uid"].iloc[0])
    glob = (cfg.data_dir / "transfers" / "*.parquet").as_posix()
    ref_uids = ",".join(str(int(u)) for u in rows["ref_uid"].dropna().unique())
    tr = con.execute(f"""
        SELECT block_number * 100000 + log_index AS uid, * FROM read_parquet('{glob}')
        WHERE block_number * 100000 + log_index IN ({ref_uids})
           OR (from_addr = ? AND to_addr = ? AND value_usd >= ?
               AND block_number * 100000 + log_index > ?)
        ORDER BY uid""", [victim, lookalike, float(cfg["detection"]["payoff_min_usd"]), first_uid]).df()

    def item(role, r):
        return {"role": role, "block_number": int(r["block_number"]), "time_utc": _iso(r["ts"]),
                "tx_hash": r["tx_hash"], "log_index": int(r["log_index"]), "token": r["token"],
                "token_address": tokens[r["token"]], "from": r["from_addr"], "to": r["to_addr"],
                "value": float(r["value_usd"]),
                "source_chunk": chunk_for_block(manifest, int(r["block_number"]))}

    timeline = [item("genuine_payment", r) for _, r in tr[tr["to_addr"] == mimicked].iterrows()]
    timeline += [item("poisoning_transfer", r) for _, r in rows.iterrows()]
    payoffs = tr[tr["to_addr"] == lookalike]
    timeline += [item("payment_to_lookalike", r) for _, r in payoffs.iterrows()]
    timeline.sort(key=lambda e: (e["block_number"], e["log_index"]))

    feats = meta["features"]
    x = rows.iloc[[0]][feats]
    raw = float(booster.predict(x)[0])
    contrib = shap_values(booster, x)[0]
    order = sorted(range(len(feats)), key=lambda i: -abs(contrib[i]))[:6]
    campaign = ctx["campaign"].get(lookalike)

    p, s = int(rows["sim_prefix_len"].iloc[0]), int(rows["sim_suffix_len"].iloc[0])
    report = {
        "schema": "address-poisoning-evidence/1",
        "case_id": sha256_hex(f"{cfg['chain']['chain_id']}|{victim}|{lookalike}".encode())[:16],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chain": cfg["chain"],
        "addresses": {"victim": victim, "genuine_counterparty": mimicked, "lookalike": lookalike,
                      "matched_prefix_hex": p, "matched_suffix_hex": s,
                      "alignment": [mimicked, lookalike, _align(mimicked, lookalike)]},
        "summary": {"n_poisoning_transfers": int(len(rows)), "n_payments_to_lookalike": int(len(payoffs)),
                    "loss_usd": float(payoffs["value_usd"].sum()),
                    "delay_blocks_payment_to_poisoning": int(rows["sim_delay_blocks"].iloc[0])},
        "timeline": timeline,
        "classifier": {"feature_set": meta["feature_set"], "model_sha256": meta["model_sha256"],
                       "explained_transfer_uid": first_uid, "score_raw": raw,
                       "score_calibrated": float(apply_calibration(meta["calibration"], [raw])[0]),
                       "decision_threshold": meta["threshold"],
                       "top_contributions": [{"feature": feats[i], "value": float(x.iloc[0, i]),
                                              "shap": float(contrib[i])} for i in order]},
        "campaign": campaign,
        "provenance": {"manifest_root_hash": manifest["root_hash"], "synthetic_data": bool(manifest.get("synthetic")),
                       "node_client": manifest.get("node_client"), "data_sealed_at": manifest.get("sealed_at"),
                       "block_range": [manifest.get("start_block"), manifest.get("end_block")],
                       "config_sha256": cfg.digest(), "code_version": code_version()},
        "caveats": CAVEATS,
    }
    report["report_sha256"] = sha256_hex(canonical_json(report))
    return report


def to_markdown(r: dict) -> str:
    a, s, c = r["addresses"], r["summary"], r["classifier"]
    L = [f"# Evidence report {r['case_id']}", "",
         f"Generated {r['generated_at']} · chain {r['chain']['name']} (id {r['chain']['chain_id']})"
         + (" · **SYNTHETIC TEST DATA**" if r["provenance"]["synthetic_data"] else ""), "",
         "## Addresses", "", "```",
         f"genuine   {a['alignment'][0]}", f"lookalike {a['alignment'][1]}", f"          {a['alignment'][2]}", "```",
         f"Victim: `{a['victim']}`. Matched {a['matched_prefix_hex']} leading and "
         f"{a['matched_suffix_hex']} trailing hex characters.", "",
         "## Summary", "",
         f"- Poisoning transfers: {s['n_poisoning_transfers']}",
         f"- Payments to the lookalike: {s['n_payments_to_lookalike']} (USD {s['loss_usd']:,.2f})",
         f"- Delay between genuine payment and first poisoning transfer: {s['delay_blocks_payment_to_poisoning']} blocks",
         "", "## Timeline", "", "| Time (UTC) | Block | Role | Token | Value | Tx hash | Log |", "|---|---|---|---|---|---|---|"]
    for e in r["timeline"]:
        L.append(f"| {e['time_utc']} | {e['block_number']} | {e['role']} | {e['token']} | {e['value']:,.6f} | "
                 f"`{e['tx_hash']}` | {e['log_index']} |")
    L += ["", "## Classifier", "",
          f"Calibrated probability **{c['score_calibrated']:.3f}** (raw {c['score_raw']:.3f}, threshold "
          f"{c['decision_threshold']:.3f}), feature set *{c['feature_set']}*, model `{c['model_sha256'][:16]}…`.", "",
          "| Feature | Value | SHAP contribution |", "|---|---|---|"]
    L += [f"| {t['feature']} | {t['value']:.4g} | {t['shap']:+.3f} |" for t in c["top_contributions"]]
    if r["campaign"]:
        k = r["campaign"]
        L += ["", "## Campaign", "", f"{k['campaign_id']}: {k['n_lookalikes']} lookalike addresses, "
              f"{k['n_victims']} victims, {k['n_transfers']} poisoning transfers."]
    pv = r["provenance"]
    L += ["", "## Provenance", "", f"- Manifest root hash: `{pv['manifest_root_hash']}`",
          f"- Node client: {pv['node_client']} · data sealed {pv['data_sealed_at']}",
          f"- Config `{pv['config_sha256'][:16]}…` · code {pv['code_version']}",
          f"- Report hash: `{r['report_sha256']}`", "", "## Caveats", ""]
    L += [f"- {x}" for x in r["caveats"]]
    return "\n".join(L) + "\n"


def run(cfg: Config, top: int = 10, victim: str | None = None, lookalike: str | None = None) -> None:
    data_dir, res, out = cfg.data_dir, cfg.sub("results"), cfg.sub("reports")
    booster, meta = load_primary(cfg)
    campaign = {}
    if (res / "campaigns.parquet").exists():
        camp = pd.read_parquet(res / "campaigns.parquet").set_index("campaign_id")
        link = pd.read_parquet(res / "lookalike_campaign.parquet")
        for a, cid in link.itertuples(index=False):
            k = camp.loc[cid]
            campaign[a] = {"campaign_id": cid, "n_lookalikes": int(k["n_lookalikes"]),
                           "n_victims": int(k["n_victims"]), "n_transfers": int(k["n_transfers"])}
    ctx = {"con": duckdb.connect(), "ds": (data_dir / "dataset.parquet").as_posix(),
           "manifest": load_manifest(data_dir), "booster": booster, "meta": meta, "campaign": campaign}
    if victim and lookalike:
        pairs = [(victim.lower(), lookalike.lower())]
    else:
        inc = pd.read_parquet(data_dir / "incidents.parquet").head(top)
        pairs = list(zip(inc["victim"], inc["lookalike"]))
    for v, l in pairs:
        r = build_report(cfg, v, l, ctx)
        (out / f"{r['case_id']}.json").write_text(json.dumps(r, indent=1))
        (out / f"{r['case_id']}.md").write_text(to_markdown(r))
    print(f"Wrote {len(pairs)} evidence report(s) to {out}/")


def verify(cfg: Config, path: str, rpc_url: str | None = None, chunks: bool = False) -> bool:
    """Re-check a report against an Ethereum node that was not used for collection."""
    from .provenance import hash_logs
    r = json.loads(open(path).read())
    body = {k: v for k, v in r.items() if k != "report_sha256"}
    ok = sha256_hex(canonical_json(body)) == r["report_sha256"]
    print(f"[{'PASS' if ok else 'FAIL'}] report hash")
    url = rpc_url or os.environ.get("VERIFY_RPC_URL") or os.environ.get("CHAINSTACK_URL", "")
    rpc = Rpc(url, max_rps=5)
    pad = lambda a: "0x" + "0" * 24 + a[2:]
    decimals = {v["address"].lower(): int(v["decimals"]) for v in cfg["tokens"].values()}
    for e in r["timeline"]:
        logs = rpc.call("eth_getLogs", [{"fromBlock": hex(e["block_number"]), "toBlock": hex(e["block_number"]),
                                         "address": e["token_address"],
                                         "topics": [TRANSFER_TOPIC, pad(e["from"]), pad(e["to"])]}]) or []
        hit = [l for l in logs if int(l["logIndex"], 16) == e["log_index"]
               and l["transactionHash"].lower() == e["tx_hash"]]
        good = bool(hit) and abs(int(hit[0]["data"], 16) / 10 ** decimals[e["token_address"]] - e["value"]) < 1e-9
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {e['role']:22s} block {e['block_number']} log {e['log_index']}")
    if chunks:
        seen = set()
        for e in r["timeline"]:
            c = e["source_chunk"]
            if not c or (c["from_block"], c["to_block"]) in seen:
                continue
            seen.add((c["from_block"], c["to_block"]))
            logs = rpc.call("eth_getLogs", [{"fromBlock": hex(c["from_block"]), "toBlock": hex(c["to_block"]),
                                             "address": list(decimals), "topics": [TRANSFER_TOPIC]}]) or []
            good = hash_logs(logs) == c["sha256"]
            ok &= good
            print(f"[{'PASS' if good else 'FAIL'}] chunk {c['from_block']}-{c['to_block']} sha256")
    print("VERIFIED" if ok else "VERIFICATION FAILED")
    return ok