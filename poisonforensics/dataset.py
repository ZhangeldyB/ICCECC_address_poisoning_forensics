"""Step 2: lookalike matching, weak labels, features and loss incidents (DuckDB)."""
from __future__ import annotations

import duckdb

from .config import ZERO_ADDRESS, Config

# Feature groups. The label is defined by address similarity, so the primary
# model deliberately excludes the SIMILARITY group (see README, "Label leakage").
BEHAVIOURAL = ["is_zero", "log_value", "token_is_usdt", "tx_n_transfers", "tx_frac_low",
               "tx_n_distinct_to", "tx_n_distinct_from", "actor_n_sent", "actor_n_recv",
               "actor_frac_low", "actor_n_cp", "actor_lifetime_blocks", "actor_age_blocks",
               "actor_log_sum_sent", "actor_log_sum_recv", "target_n_normal_out",
               "target_n_recipients", "target_log_sum_out"]
CONTEXT = ["target_blocks_since_out", "target_log_last_out"]
SIMILARITY = ["sim_prefix_len", "sim_suffix_len", "sim_delay_blocks"]
FEATURE_SETS = {
    "behavioural": BEHAVIOURAL,
    "behavioural+context": BEHAVIOURAL + CONTEXT,
    "all (incl. similarity)": BEHAVIOURAL + CONTEXT + SIMILARITY,
}
PRIMARY = "behavioural+context"


def _common_prefix(a: str, b: str, n: int = 20) -> str:
    whens = " ".join(f"WHEN substr({a},3,{k})=substr({b},3,{k}) THEN {k}" for k in range(n, 0, -1))
    return f"(CASE {whens} ELSE 0 END)"


def _common_suffix(a: str, b: str, n: int = 20) -> str:
    whens = " ".join(f"WHEN right({a},{k})=right({b},{k}) THEN {k}" for k in range(n, 0, -1))
    return f"(CASE {whens} ELSE 0 END)"


def run(cfg: Config, reuse: bool = False) -> None:
    d = cfg["detection"]
    low, minpay = float(d["low_value_usd"]), float(d["min_payment_usd"])
    W, P, S = int(d["lookback_blocks"]), int(d["min_prefix"]), int(d["min_suffix"])
    data_dir = cfg.data_dir
    glob = (data_dir / "transfers" / "*.parquet").as_posix()
    K = int(d.get("max_actor_payments", 20))
    db = data_dir / "work.duckdb"
    reuse = reuse and db.exists()
    if db.exists() and not reuse:
        db.unlink()
    con = duckdb.connect(db.as_posix())
    if reuse:
        print("Reusing matches and aggregates from the previous build ...")
    else:
        _heavy_stages(con, glob, low, minpay, W, S)

    # An address with many payment-sized transfers of its own is an established address
    # (exchange, bot, vanity-address sibling), not a single-purpose lookalike: drop it as actor.
    con.execute("""
        CREATE OR REPLACE TABLE paycount AS
        SELECT addr, count(*) AS n FROM (
            SELECT from_addr AS addr FROM normal UNION ALL SELECT to_addr FROM normal) GROUP BY addr
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE best_f AS
        SELECT b.* FROM best b JOIN low l ON l.uid = b.uid
        LEFT JOIN paycount c ON c.addr = l.actor
        WHERE coalesce(c.n, 0) <= {K}
    """)
    ex = con.execute(f"""
        SELECT count(*), count(DISTINCT l.actor) FROM best b JOIN low l ON l.uid = b.uid
        JOIN paycount c ON c.addr = l.actor WHERE c.n > {K} AND b.prefix_len >= {P}""").fetchone()
    print(f"Excluded as established addresses (> {K} payments of their own): "
          f"{ex[1]:,} addresses, {ex[0]:,} transfers")

    print("Finding loss incidents ...")
    con.execute(f"""
        CREATE OR REPLACE TABLE payoff AS
        SELECT l.uid, min(n.uid) AS first_payoff_uid, count(*) AS n_payoffs, sum(n.value_usd) AS payoff_usd
        FROM low l JOIN best_f b ON b.uid = l.uid AND b.prefix_len >= {P}
        JOIN normal n ON n.from_addr = l.target AND n.to_addr = l.actor
        WHERE n.uid > l.uid AND n.value_usd >= {float(d['payoff_min_usd'])}
          AND n.block_number - l.block_number <= {int(d['payoff_horizon_blocks'])}
        GROUP BY l.uid
    """)
    _export(con, cfg, P, K, ex)


def _heavy_stages(con, glob, low, minpay, W, S) -> None:
    print("Loading transfers ...")
    con.execute(f"""
        CREATE TABLE transfers AS
        SELECT block_number * 100000 + log_index AS uid, *
        FROM read_parquet('{glob}')
        QUALIFY row_number() OVER (PARTITION BY block_number, log_index ORDER BY tx_hash) = 1
    """)
    n = con.execute("SELECT count(*) FROM transfers").fetchone()[0]
    print(f"  {n:,} transfers")

    con.execute(f"""
        CREATE TABLE low AS
        SELECT uid, block_number, ts, tx_hash, log_index, token, from_addr, to_addr, value_usd,
               value_usd = 0 AS is_zero,
               CASE WHEN value_usd = 0 THEN to_addr ELSE from_addr END AS actor,
               CASE WHEN value_usd = 0 THEN from_addr ELSE to_addr END AS target
        FROM transfers
        WHERE value_usd < {low} AND from_addr <> to_addr
          AND from_addr <> '{ZERO_ADDRESS}' AND to_addr <> '{ZERO_ADDRESS}'
    """)
    con.execute(f"""
        CREATE TABLE normal AS
        SELECT uid, block_number, ts, tx_hash, log_index, token, from_addr, to_addr, value_usd,
               right(to_addr, {S}) AS suf
        FROM transfers WHERE value_usd >= {minpay}
    """)

    print("Matching lookalike addresses ...")
    con.execute(f"""
        CREATE TABLE cand AS
        SELECT l.uid, p.uid AS ref_uid, p.to_addr AS mimicked,
               l.block_number - p.block_number AS delay_blocks,
               {_common_prefix('l.actor', 'p.to_addr')} AS prefix_len,
               {_common_suffix('l.actor', 'p.to_addr')} AS suffix_len
        FROM low l JOIN normal p
          ON p.from_addr = l.target AND p.suf = right(l.actor, {S})
        WHERE p.to_addr <> l.actor AND p.uid < l.uid
          AND l.block_number - p.block_number <= {W}
    """)
    con.execute("""
        CREATE TABLE best AS
        SELECT * FROM cand
        QUALIFY row_number() OVER (PARTITION BY uid
                                   ORDER BY prefix_len + suffix_len DESC, delay_blocks ASC) = 1
    """)

    print("Aggregating behaviour ...")
    con.execute(f"""
        CREATE TABLE txagg AS
        SELECT tx_hash, count(*) AS n, count(*) FILTER (WHERE value_usd < {low}) AS n_low,
               count(DISTINCT to_addr) AS n_to, count(DISTINCT from_addr) AS n_from
        FROM transfers GROUP BY tx_hash
    """)
    con.execute(f"""
        CREATE TABLE addragg AS
        WITH edges AS (
            SELECT from_addr AS addr, to_addr AS cp, value_usd, block_number, TRUE AS is_out FROM transfers
            UNION ALL
            SELECT to_addr, from_addr, value_usd, block_number, FALSE FROM transfers)
        SELECT addr,
               count(*) FILTER (WHERE is_out) AS n_sent,
               count(*) FILTER (WHERE NOT is_out) AS n_recv,
               count(*) FILTER (WHERE value_usd < {low}) AS n_low,
               approx_count_distinct(cp) AS n_cp,
               min(block_number) AS first_block, max(block_number) AS last_block,
               coalesce(sum(value_usd) FILTER (WHERE is_out), 0) AS sum_sent,
               coalesce(sum(value_usd) FILTER (WHERE NOT is_out), 0) AS sum_recv,
               count(*) FILTER (WHERE is_out AND value_usd >= {minpay}) AS n_normal_out,
               approx_count_distinct(cp) FILTER (WHERE is_out AND value_usd >= {minpay}) AS n_recipients
        FROM edges GROUP BY addr
    """)
    con.execute("""
        CREATE TABLE lastout AS
        SELECT l.uid, l.block_number - n.block_number AS blocks_since_out, n.value_usd AS last_out
        FROM low l ASOF LEFT JOIN normal n ON l.target = n.from_addr AND l.uid > n.uid
    """)



def _export(con, cfg: Config, P: int, K: int, ex) -> None:
    data_dir = cfg.data_dir
    out = (data_dir / "dataset.parquet").as_posix()
    con.execute(f"""
        COPY (
        SELECT l.uid, l.block_number, l.ts, l.tx_hash, l.log_index, l.token,
               l.from_addr, l.to_addr, l.actor, l.target, l.value_usd,
               CAST(coalesce(b.prefix_len, 0) >= {P} AS INTEGER) AS label,
               b.mimicked, b.ref_uid,
               coalesce(p.n_payoffs, 0) AS n_payoffs, coalesce(p.payoff_usd, 0) AS payoff_usd,
               -- behavioural
               CAST(l.is_zero AS INTEGER) AS is_zero,
               log10(l.value_usd * 1e6 + 1) AS log_value,
               CAST(l.token = 'USDT' AS INTEGER) AS token_is_usdt,
               t.n AS tx_n_transfers, t.n_low / t.n AS tx_frac_low,
               t.n_to AS tx_n_distinct_to, t.n_from AS tx_n_distinct_from,
               a.n_sent AS actor_n_sent, a.n_recv AS actor_n_recv,
               a.n_low / (a.n_sent + a.n_recv) AS actor_frac_low, a.n_cp AS actor_n_cp,
               a.last_block - a.first_block AS actor_lifetime_blocks,
               l.block_number - a.first_block AS actor_age_blocks,
               log10(a.sum_sent + 1) AS actor_log_sum_sent, log10(a.sum_recv + 1) AS actor_log_sum_recv,
               g.n_normal_out AS target_n_normal_out, g.n_recipients AS target_n_recipients,
               log10(g.sum_sent + 1) AS target_log_sum_out,
               -- context
               coalesce(o.blocks_since_out, -1) AS target_blocks_since_out,
               log10(coalesce(o.last_out, 0) + 1) AS target_log_last_out,
               -- similarity (defines the label; used only in the upper-bound model)
               coalesce(b.prefix_len, 0) AS sim_prefix_len, coalesce(b.suffix_len, 0) AS sim_suffix_len,
               coalesce(b.delay_blocks, -1) AS sim_delay_blocks
        FROM low l
        JOIN txagg t ON t.tx_hash = l.tx_hash
        JOIN addragg a ON a.addr = l.actor
        JOIN addragg g ON g.addr = l.target
        LEFT JOIN best_f b ON b.uid = l.uid
        LEFT JOIN lastout o ON o.uid = l.uid
        LEFT JOIN payoff p ON p.uid = l.uid
        ORDER BY l.uid
        ) TO '{out}' (FORMAT PARQUET)
    """)

    # One row per (victim, lookalike) pair that ended in at least one payment to the lookalike.
    con.execute(f"""
        COPY (
        SELECT target AS victim, actor AS lookalike, any_value(mimicked) AS mimicked,
               count(*) AS n_poison_transfers, min(uid) AS first_poison_uid,
               max(payoff_usd) AS payoff_usd, max(n_payoffs) AS n_payoffs
        FROM read_parquet('{out}') WHERE label = 1 AND n_payoffs > 0
        GROUP BY target, actor ORDER BY payoff_usd DESC
        ) TO '{(data_dir / 'incidents.parquet').as_posix()}' (FORMAT PARQUET)
    """)

    s = con.execute(f"""
        SELECT count(*), sum(label), sum(is_zero), count(DISTINCT CASE WHEN label=1 THEN target END),
               count(DISTINCT CASE WHEN label=1 THEN actor END)
        FROM read_parquet('{out}')""").fetchone()
    inc = con.execute(f"SELECT count(*), coalesce(sum(payoff_usd),0) FROM read_parquet('{(data_dir / 'incidents.parquet').as_posix()}')").fetchone()
    print(f"Low-value transfers: {s[0]:,} | poisoning (weak label): {int(s[1] or 0):,} | zero-value: {int(s[2] or 0):,}")
    (data_dir / "label_filter.txt").write_text(
        f"max_actor_payments={K}; excluded addresses={ex[1]}; excluded transfers={ex[0]}\n")
    print(f"Victims: {s[3]:,} | lookalike addresses: {s[4]:,} | loss incidents: {inc[0]:,} (USD {inc[1]:,.0f})")
    con.close()