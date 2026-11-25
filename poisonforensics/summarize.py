"""Step 6: collect every number the paper needs into results/paper_numbers.md."""
from __future__ import annotations

import json

import duckdb
import pandas as pd

from .config import Config
from .provenance import load_manifest


def run(cfg: Config) -> None:
    data_dir, res = cfg.data_dir, cfg.sub("results")
    man = load_manifest(data_dir)
    con = duckdb.connect()
    glob = (data_dir / "transfers" / "*.parquet").as_posix()
    ds = (data_dir / "dataset.parquet").as_posix()
    tot = con.execute(f"SELECT count(*), count(*) FILTER (WHERE token='USDT'), count(*) FILTER (WHERE token='USDC'), "
                      f"min(block_number), max(block_number) FROM read_parquet('{glob}')").fetchone()
    d = con.execute(f"""
        SELECT count(*), sum(is_zero), sum(label), sum(label) FILTER (WHERE is_zero=1),
               count(DISTINCT target) FILTER (WHERE label=1), count(DISTINCT actor) FILTER (WHERE label=1),
               median(sim_delay_blocks) FILTER (WHERE label=1),
               median(sim_prefix_len) FILTER (WHERE label=1), median(sim_suffix_len) FILTER (WHERE label=1),
               quantile_cont(sim_delay_blocks, 0.9) FILTER (WHERE label=1)
        FROM read_parquet('{ds}')""").fetchone()
    inc = pd.read_parquet(data_dir / "incidents.parquet")
    L = ["# Numbers for the paper", ""]
    if man.get("synthetic"):
        L += ["> **SYNTHETIC TEST DATA — do not put these numbers in the paper.**", ""]
    L += ["## Table 2 – Dataset", "", "| Placeholder | Quantity | Value |", "|---|---|---|",
          f"| D1 | Block range | {tot[3]:,} – {tot[4]:,} |",
          f"| D2 | Transfer events (total) | {tot[0]:,} |",
          f"| D3 | USDT / USDC | {tot[1]:,} / {tot[2]:,} |",
          f"| D4 | Low-value transfers | {d[0]:,} |",
          f"| D5 | of which zero-value / dust | {int(d[1]):,} / {d[0] - int(d[1]):,} |",
          f"| D6 | Poisoning transfers (weak label) | {int(d[2] or 0):,} ({(d[2] or 0) / max(d[0], 1):.1%} of low-value) |",
          f"| D7 | of which zero-value / dust | {int(d[3] or 0):,} / {int((d[2] or 0) - (d[3] or 0)):,} |",
          f"| D8 | Distinct victims | {d[4]:,} |",
          f"| D9 | Distinct lookalike addresses | {d[5]:,} |",
          f"| D10 | Median / 90th pct. reaction delay (blocks) | {d[6]} / {d[9]} |",
          f"| D11 | Median matched prefix / suffix (hex) | {d[7]} / {d[8]} |",
          f"| D12 | Loss incidents | {len(inc):,} |",
          f"| D13 | Loss (USD) | {inc['payoff_usd'].sum():,.0f} |",
          f"| D14 | Manifest root hash | `{man['root_hash']}` |",
          f"| D15 | Chunks / node client | {man['n_chunks']:,} / {man.get('node_client')} |", ""]
    mp = res / "metrics.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        L += ["## Table 3 – Detection performance (test period)", "",
              "| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |", "|---|---|---|---|---|---|"]
        L += [f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['pr_auc']:.3f} | {v['roc_auc']:.3f} |"
              for k, v in m.items() if not k.startswith("_")]
        x = m["_dataset"]
        L += ["", "| Placeholder | Quantity | Value |", "|---|---|---|",
              f"| M1 | Train / validation / test rows | {x['n_train']:,} / {x['n_valid']:,} / {x['n_test']:,} |",
              f"| M2 | Positive rate train / test | {x['pos_rate_train']:.3f} / {x['pos_rate_test']:.3f} |",
              f"| M3 | Brier score raw / calibrated | {x['brier_raw']:.4f} / {x['brier_calibrated']:.4f} |",
              f"| M4 | Confirmed incidents in test / recall on them | {m['_outcome_validation']['n_confirmed_test']:,} / "
              f"{m['_outcome_validation']['recall_on_confirmed']:.3f} |",
              f"| M5 | Rule-negative transfers flagged by the model | {m['_rule_missed_candidates']['n']:,} |", ""]
    sp = res / "shap_importance.csv"
    if sp.exists():
        imp = pd.read_csv(sp).head(8)
        L += ["## Fig. 3 and Section 6.2 – Global feature attribution (mean |SHAP|)", "", "| Rank | Feature | Mean abs. SHAP |", "|---|---|---|"]
        L += [f"| {i + 1} | {r.feature} | {r.mean_abs_shap:.3f} |" for i, r in enumerate(imp.itertuples())]
        L.append("")
    cp = res / "campaigns.parquet"
    if cp.exists():
        c = pd.read_parquet(cp)
        t = c.head(10)
        L += ["## Section 6.3 – Campaigns", "", "| Placeholder | Quantity | Value |", "|---|---|---|",
              f"| C1 | Campaigns | {len(c):,} |",
              f"| C2 | Share of poisoning transfers in top 10 | {t['n_transfers'].sum() / c['n_transfers'].sum():.1%} |",
              f"| C3 | Largest campaign: lookalikes / victims / transfers | {int(c.iloc[0]['n_lookalikes']):,} / "
              f"{int(c.iloc[0]['n_victims']):,} / {int(c.iloc[0]['n_transfers']):,} |",
              f"| C4 | Campaigns with at least one loss incident | {int((c['n_incidents'] > 0).sum()):,} |", "",
              t[["campaign_id", "n_transfers", "n_lookalikes", "n_victims", "n_txs", "n_incidents", "loss_usd"]].to_markdown(index=False)
              if hasattr(t, "to_markdown") else "", ""]
    (res / "paper_numbers.md").write_text("\n".join(L))
    print(f"Wrote {res / 'paper_numbers.md'}")
