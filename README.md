# poison-forensics

Explainable forensic pipeline for ERC-20 **address-poisoning** attacks on Ethereum
(USDT and USDC). It collects Transfer logs from an RPC node, finds lookalike
addresses, trains an explainable detector, groups attacker addresses into campaigns,
and writes evidence reports that anyone can re-verify against the public chain.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then put your Chainstack endpoint in CHAINSTACK_URL
```

Python 3.10+ is required. Edit `config.yaml` to set the 30-day window.

## Run

```bash
python -m poisonforensics collect             # 1. download logs (resumable; re-run after a failure)
python -m poisonforensics build               # 2. lookalike matching, labels, features
python -m poisonforensics train               # 3. models, metrics, calibration, SHAP, figures
python -m poisonforensics attribute --enrich  # 4. campaigns (add --funders with an Etherscan key)
python -m poisonforensics report --top 10     # 5. evidence reports for the largest loss incidents
python -m poisonforensics verify data/reports/<case>.json --rpc <another node URL>
python -m poisonforensics summarize           # 6. data/results/paper_numbers.md
```

Offline test without any API key (synthetic data, **never use its numbers in the paper**):

```bash
python -m pytest -q
python -m poisonforensics --config config.synth.yaml synth   # then build, train, ... with the same --config
```

## What each step produces

| Step | Output (under `data/`) |
|---|---|
| collect | `transfers/*.parquet`, `raw/*.jsonl.gz`, `manifest.json` (SHA-256 per chunk + hash chain) |
| build | `dataset.parquet` (one row per low-value transfer), `incidents.parquet` |
| train | `results/metrics.json`, `model.txt`, `model_meta.json`, `fig_pr_curves.png`, `fig_shap_summary.png`, `shap_importance.csv`, `audit_candidates.csv` |
| attribute | `results/campaigns.parquet`, `results/lookalike_campaign.parquet` |
| report | `reports/<case_id>.json` and `.md` |
| summarize | `results/paper_numbers.md` – every number the paper's tables need |

## Method in brief

- **Low-value transfer**: value below `low_value_usd` (zero-value or dust).
  For a zero-value transfer the *actor* is the recipient and the *target* is the sender;
  for a dust transfer the actor is the sender and the target is the recipient.
- **Weak label**: a low-value transfer is poisoning when its actor shares at least
  `min_prefix` leading and `min_suffix` trailing hex characters with an address the target
  genuinely paid during the previous `lookback_blocks`, and is not that address.
- **Label leakage**: the label is defined by address similarity, so the primary model
  (`behavioural+context`) is *not* given the similarity features. The model with all
  features is reported only as an upper bound.
- **Independent check**: a poisoning transfer followed by a real payment from the
  victim to the lookalike is a confirmed loss incident; recall on these is reported separately.
- **Campaigns**: lookalikes are linked when they appear in the same transaction, are
  submitted by the same account, or go through the same non-shared contract
  (optionally: same first funder).
- **Provenance**: each chunk hash covers only consensus fields of the logs, so it can
  be recomputed from any Ethereum node. Reports carry the manifest root hash, model hash,
  config hash and their own SHA-256.

## Data sources and cost

- **Chainstack RPC** (`eth_getLogs`, `eth_getBlockByNumber`, `eth_getTransactionByHash`).
  30 days is about 216,000 blocks, i.e. roughly 2,200 `eth_getLogs` calls at 100 blocks per
  call; the code halves a range automatically when the provider rejects it.
  `attribute --enrich` adds one call per poisoning transaction (capped by `max_tx_enrich`).
- **Etherscan API V2** (free key, optional): first funder of submitting accounts.
- Disk: raw logs for 30 days can take several GB; set `keep_raw: false` to store hashes only.

## Scope and limits

- Counterfeit-token poisoning (fake "USDT" contracts) is out of scope: only transfers
  emitted by the genuine token contracts are collected.
- Address aggregates are past-only by default (`causal_features: true`): each transfer sees
  only earlier activity of its actor and target. Set `causal_features: false` in
  `config.yaml` to reproduce the whole-window variant (Table 3, last column).
- Addresses are pseudonymous public data. The pipeline does not link them to people.