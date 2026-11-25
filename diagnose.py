"""Sanity checks on the labels:  python diagnose.py [data_dir]   (prints to screen and data/diagnose.txt)"""
import sys

import duckdb

d = sys.argv[1] if len(sys.argv) > 1 else "data"
con = duckdb.connect()
ds, inc = f"read_parquet('{d}/dataset.parquet')", f"read_parquet('{d}/incidents.parquet')"
out = []

def show(title, sql):
    out.append(f"\n=== {title}\n" + con.execute(sql).df().to_string(index=False))

show("Top 20 loss incidents", f"""
    SELECT i.victim, i.lookalike, i.mimicked, round(i.payoff_usd) AS loss_usd, i.n_payoffs, i.n_poison_transfers AS n_poison,
           x.sim_prefix_len AS pre, x.sim_suffix_len AS suf, x.is_zero, x.target_n_recipients AS victim_payees,
           x.actor_n_sent, x.actor_n_recv, round(pow(10, x.actor_log_sum_recv)) AS actor_recv_usd, x.tx_n_transfers
    FROM {inc} i JOIN {ds} x ON x.uid = i.first_poison_uid ORDER BY i.payoff_usd DESC LIMIT 20""")
show("Poisoning labels by number of victim payees", f"""
    SELECT CASE WHEN target_n_recipients <= 10 THEN '1: <=10' WHEN target_n_recipients <= 100 THEN '2: 11-100'
                WHEN target_n_recipients <= 1000 THEN '3: 101-1000' WHEN target_n_recipients <= 10000 THEN '4: 1001-10000'
                ELSE '5: >10000' END AS victim_payees,
           count(*) AS transfers, count(DISTINCT target) AS victims, sum(CAST(n_payoffs > 0 AS INT)) AS with_loss
    FROM {ds} WHERE label = 1 GROUP BY 1 ORDER BY 1""")
show("Poisoning labels by matched characters (prefix + suffix)", f"""
    SELECT least(sim_prefix_len + sim_suffix_len, 12) AS matched, count(*) AS transfers,
           sum(is_zero) AS zero_value, sum(CAST(n_payoffs > 0 AS INT)) AS with_loss
    FROM {ds} WHERE label = 1 GROUP BY 1 ORDER BY 1""")
show("Poisoning labels by transfers in the same transaction", f"""
    SELECT CASE WHEN tx_n_transfers = 1 THEN '1' WHEN tx_n_transfers <= 5 THEN '2-5'
                WHEN tx_n_transfers <= 50 THEN '6-50' ELSE '>50' END AS tx_transfers,
           count(*) AS transfers, sum(is_zero) AS zero_value FROM {ds} WHERE label = 1 GROUP BY 1 ORDER BY 1""")
show("Lookalike addresses with real activity of their own", f"""
    SELECT CASE WHEN actor_log_sum_recv < 2 THEN '1: received < $100' WHEN actor_log_sum_recv < 4 THEN '2: $100-$10k'
                ELSE '3: > $10k' END AS lookalike_received, count(DISTINCT actor) AS lookalikes, count(*) AS transfers
    FROM {ds} WHERE label = 1 GROUP BY 1 ORDER BY 1""")
show("Most frequent 'victims'", f"""
    SELECT target, count(*) AS transfers, count(DISTINCT actor) AS lookalikes, max(target_n_recipients) AS payees
    FROM {ds} WHERE label = 1 GROUP BY 1 ORDER BY 2 DESC LIMIT 10""")
show("Busiest (victim, lookalike) pairs", f"""
    SELECT target, actor, count(*) AS transfers, sum(is_zero) AS zero_value, max(tx_n_transfers) AS max_tx_transfers,
           max(sim_prefix_len) AS pre, max(sim_suffix_len) AS suf, max(n_payoffs) AS payoffs
    FROM {ds} WHERE label = 1 GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 10""")
show("Loss incidents: totals", f"SELECT count(*) AS incidents, round(sum(payoff_usd)) AS loss_usd, round(median(payoff_usd)) AS median_usd, round(max(payoff_usd)) AS max_usd FROM {inc}")
text = "\n".join(out)
print(text)
open(f"{d}/diagnose.txt", "w").write(text)