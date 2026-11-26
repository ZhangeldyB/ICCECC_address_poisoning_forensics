"""Step 3: train, evaluate, calibrate and explain the detectors."""
from __future__ import annotations

import json
import warnings

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")
import matplotlib.pyplot as plt
import shap
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config
from .dataset import FEATURE_SETS, PRIMARY
from .provenance import file_sha256


def split_by_time(df: pd.DataFrame, test_fraction: float, valid_fraction: float):
    """Chronological split: no transfer in validation/test precedes a training transfer."""
    df = df.sort_values("uid")
    test_cut = df["block_number"].quantile(1 - test_fraction)
    train_all = df[df["block_number"] <= test_cut]
    test = df[df["block_number"] > test_cut]
    valid_cut = train_all["block_number"].quantile(1 - valid_fraction)
    return (train_all[train_all["block_number"] <= valid_cut],
            train_all[train_all["block_number"] > valid_cut], test)


def best_f1_threshold(y, score) -> float:
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.argmax(f1))]) if len(t) else 0.5


def metrics(y, score, thr) -> dict:
    pred = (score >= thr).astype(int)
    both = len(np.unique(y)) > 1
    return {"precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "pr_auc": float(average_precision_score(y, score)) if both else float("nan"),
            "roc_auc": float(roc_auc_score(y, score)) if both else float("nan"),
            "threshold": float(thr)}


def apply_calibration(cal: dict, score: np.ndarray) -> np.ndarray:
    return np.interp(score, cal["x"], cal["y"])


def run(cfg: Config) -> None:
    m = cfg["model"]
    data_dir, out = cfg.data_dir, cfg.sub("results")
    all_feats = FEATURE_SETS["all (incl. similarity)"]
    df = pd.read_parquet(data_dir / "dataset.parquet", columns=["uid", "block_number", "label", "n_payoffs"] + all_feats)
    df[all_feats] = df[all_feats].astype("float32")   # 11M+ rows: keep memory low
    train, valid, test = split_by_time(df, m["test_fraction"], m["valid_fraction"])
    if len(train) > m["max_train_rows"]:
        train = train.sample(m["max_train_rows"], random_state=m["seed"])
    print(f"train {len(train):,} | valid {len(valid):,} | test {len(test):,} | "
          f"positive rate (test) {test['label'].mean():.4f}")
    yt, yv, yte = train["label"].values, valid["label"].values, test["label"].values

    results, curves = {}, {}
    pos = max(int(yt.sum()), 1)
    spw = float((len(yt) - pos) / pos)

    # Trivial baseline: every zero-value transfer is poisoning.
    results["baseline: zero-value rule"] = metrics(yte, test["is_zero"].values.astype(float), 0.5)

    # Linear baseline on the primary feature set.
    feats = FEATURE_SETS[PRIMARY]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced"))
    lr_train = train.sample(min(len(train), 500_000), random_state=m["seed"])
    lr.fit(lr_train[feats].astype(float).fillna(-1), lr_train["label"].values)
    thr = best_f1_threshold(yv, lr.predict_proba(valid[feats].astype(float).fillna(-1))[:, 1])
    s = lr.predict_proba(test[feats].astype(float).fillna(-1))[:, 1]
    results[f"logistic regression ({PRIMARY})"] = metrics(yte, s, thr)
    curves["Logistic regression"] = precision_recall_curve(yte, s)

    primary_model, primary_scores, primary_thr = None, None, 0.5
    for name, feats in FEATURE_SETS.items():
        clf = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=63,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                 scale_pos_weight=spw, random_state=m["seed"], verbose=-1)
        clf.fit(train[feats], yt, eval_set=[(valid[feats], yv)], eval_metric="average_precision",
                callbacks=[lgb.early_stopping(50, verbose=False)])
        thr = best_f1_threshold(yv, clf.predict_proba(valid[feats])[:, 1])
        s = clf.predict_proba(test[feats])[:, 1]
        results[f"LightGBM ({name})"] = metrics(yte, s, thr)
        curves[f"LightGBM ({name})"] = precision_recall_curve(yte, s)
        if name == PRIMARY:
            primary_model, primary_scores, primary_thr = clf, s, thr
            sv = clf.predict_proba(valid[feats])[:, 1]

    # ---- primary model: persist, calibrate, explain -------------------------------
    feats = FEATURE_SETS[PRIMARY]
    model_path = out / "model.txt"
    primary_model.booster_.save_model(model_path.as_posix())
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(sv, yv)
    cal = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
    calibrated = apply_calibration(cal, primary_scores)
    meta = {"feature_set": PRIMARY, "features": feats, "threshold": primary_thr,
            "calibration": cal, "model_sha256": file_sha256(model_path),
            "train_blocks": [int(train["block_number"].min()), int(train["block_number"].max())],
            "test_blocks": [int(test["block_number"].min()), int(test["block_number"].max())],
            "brier_raw": float(brier_score_loss(yte, primary_scores)),
            "brier_calibrated": float(brier_score_loss(yte, calibrated))}
    (out / "model_meta.json").write_text(json.dumps(meta, indent=1))

    pred = test[["uid", "label", "n_payoffs"]].copy()
    pred["score"], pred["score_calibrated"] = primary_scores, calibrated
    pred.to_parquet(out / "test_predictions.parquet", index=False)

    # Outcome-based validation: poisoning transfers that were followed by a real payment
    # to the lookalike are confirmed independently of the classifier.
    confirmed = pred[(pred["label"] == 1) & (pred["n_payoffs"] > 0)]
    results["_outcome_validation"] = {
        "n_confirmed_test": int(len(confirmed)),
        "recall_on_confirmed": float((confirmed["score"] >= primary_thr).mean()) if len(confirmed) else float("nan")}
    # Transfers the model flags although the similarity rule did not: audit candidates.
    flagged_neg = pred[(pred["label"] == 0) & (pred["score"] >= primary_thr)]
    results["_rule_missed_candidates"] = {"n": int(len(flagged_neg))}
    top = flagged_neg.sort_values("score", ascending=False).head(500)[["uid", "score", "score_calibrated"]]
    if len(top):
        import duckdb
        con = duckdb.connect(); con.register("top", top)
        con.execute(f"""SELECT t.score, t.score_calibrated, d.* FROM top t
                        JOIN read_parquet('{(data_dir / 'dataset.parquet').as_posix()}') d ON d.uid = t.uid
                        ORDER BY t.score DESC""").df().to_csv(out / "audit_candidates.csv", index=False)

    del df
    results["_dataset"] = {"n_train": int(len(train)), "n_valid": int(len(valid)), "n_test": int(len(test)),
                           "pos_rate_train": float(yt.mean()), "pos_rate_test": float(yte.mean()),
                           "brier_raw": meta["brier_raw"], "brier_calibrated": meta["brier_calibrated"]}
    (out / "metrics.json").write_text(json.dumps(results, indent=1))

    # Figures (300 dpi, sized for the 16 cm text width of the template).
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for name, (p, r, _) in curves.items():
        ax.plot(r, p, label=name, linewidth=1.2)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(out / "fig_pr_curves.png", dpi=300); plt.close(fig)

    sample = test.sample(min(int(m["shap_sample"]), len(test)), random_state=m["seed"])
    sv_ = shap_values(primary_model.booster_, sample[feats])
    imp = pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(sv_).mean(axis=0)}).sort_values(
        "mean_abs_shap", ascending=False)
    imp.to_csv(out / "shap_importance.csv", index=False)
    shap.summary_plot(sv_, sample[feats], show=False, max_display=12, plot_size=(4.8, 3.6))
    plt.tight_layout(); plt.savefig(out / "fig_shap_summary.png", dpi=300, bbox_inches="tight"); plt.close()

    for k, v in results.items():
        if not k.startswith("_"):
            print(f"  {k:42s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} PR-AUC={v['pr_auc']:.3f}")
    print(f"Saved model, metrics and figures to {out}/")


def shap_values(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    sv = shap.TreeExplainer(booster).shap_values(X)
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)
    return sv[:, :, -1] if sv.ndim == 3 else sv


def load_primary(cfg: Config):
    out = cfg.sub("results")
    meta = json.loads((out / "model_meta.json").read_text())
    return lgb.Booster(model_file=(out / "model.txt").as_posix()), meta