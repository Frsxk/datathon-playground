#!/usr/bin/env python3
"""Production v4 — ordinal model + DISCOVERED interaction signal. CV-safe.

New signal (target-free, train/test-stable): the planted pairwise interaction
motiv_x_disc = skor_motivasi * skor_kedisiplinan (individually ~noise, product
rho=0.354, orthogonal to prior features) + daily-activity temporal patterns.
The explosive disc/motiv ratio is intentionally DROPPED (skor_motivasi crosses 0).

Model: ordinal decomposition S = sum_k P(y>k), seed-bagged, quartile-binned.
Writes outputs/submission_v4_interaction.csv + metrics_v4_interaction.json.
NO Kaggle submission is made.
"""
from __future__ import annotations
import sys, json, warnings, numpy as np, pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from cv_harness import build_features, quartile_bin, SEEDS

NC = 4; BAG = 3
Dc = [f"aktivitas_hari_{i:02d}" for i in range(1, 17)]
ORD = dict(n_estimators=550, max_depth=4, learning_rate=0.04, subsample=0.9,
           colsample_bytree=0.9, min_child_weight=3, reg_lambda=1.5)

def longest_run(mask):
    out = np.zeros(len(mask))
    for i, row in enumerate(mask):
        best = cur = 0
        for v in row: cur = cur + 1 if v else 0; best = max(best, cur)
        out[i] = best
    return out

def feats(df):
    f = build_features(df, "full")
    X = f.drop(columns=[c for c in f.columns if c.startswith("kelas")]).copy()
    Dm = df[Dc].to_numpy(float)
    # discovered interaction (the big lever) + daily temporal patterns
    X["motiv_x_disc"] = df["skor_motivasi"].values * df["skor_kedisiplinan"].values
    X["day_sign_changes"] = (np.diff(np.sign(Dm - Dm.mean(1, keepdims=True)), axis=1) != 0).sum(1)
    X["day_longest_high_run"] = longest_run(Dm > np.median(Dm, 1, keepdims=True))
    p = np.abs(Dm) / (np.abs(Dm).sum(1, keepdims=True) + 1e-9)
    X["day_entropy"] = -np.sum(p * np.log(p + 1e-9), 1)
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def xbin(seed): return XGBClassifier(**ORD, random_state=seed, n_jobs=-1, eval_metric="logloss", verbosity=0)

def ord_score(Xtr, ytr, Xpred, seed, bag=BAG):
    s = np.zeros(len(Xpred))
    for k in range(NC - 1):
        pk = np.zeros(len(Xpred))
        for bi in range(bag):
            b = xbin(seed + 100 * bi); b.fit(Xtr, (ytr > k).astype(int)); pk += b.predict_proba(Xpred)[:, 1]
        s += pk / bag
    return s

def main():
    tr = pd.read_csv("kaggle/train.csv"); te = pd.read_csv("kaggle/test.csv")
    sample = pd.read_csv("kaggle/sample_submission.csv")
    y = tr["target"]; X = feats(tr); Xte = feats(te)
    assert list(X.columns) == list(Xte.columns)
    print(f"nfeat={X.shape[1]} bag={BAG}")

    per_seed = []
    for s in SEEDS:
        cv = StratifiedKFold(5, shuffle=True, random_state=s); oof = np.zeros(len(y))
        for tr_i, va_i in cv.split(X, y):
            oof[va_i] = ord_score(X.iloc[tr_i], y.iloc[tr_i], X.iloc[va_i], s)
        per_seed.append(accuracy_score(y, quartile_bin(oof)))
        print(f"  seed {s}: {per_seed[-1]:.4f}")
    per_seed = np.array(per_seed)
    print(f"CV {per_seed.mean():.4f} +/- {per_seed.std():.4f}")

    test_scores = [pd.Series(ord_score(X, y, Xte, s)).rank().values for s in SEEDS]
    preds = quartile_bin(np.mean(test_scores, 0)).astype(int)
    sub = pd.DataFrame({"id": te["id"], "target": preds})
    assert list(sub.columns) == ["id", "target"] and sub.shape == sample.shape
    assert sub["id"].equals(sample["id"]) and set(sub["target"]).issubset({0, 1, 2, 3})
    sub.to_csv("outputs/submission_v4_interaction.csv", index=False)

    Path("outputs/metrics_v4_interaction.json").write_text(json.dumps({
        "experiment": "v4_ordinal_plus_discovered_interaction",
        "generated_at": datetime.now(timezone.utc).isoformat(), "seeds": list(SEEDS), "bag": BAG,
        "feature_shape": list(X.shape), "new_features": ["motiv_x_disc", "day_sign_changes", "day_longest_high_run", "day_entropy"],
        "cv_mean_accuracy": float(per_seed.mean()), "cv_std": float(per_seed.std()),
        "per_seed": per_seed.round(4).tolist(),
        "prediction_counts": {str(k): int(v) for k, v in sub["target"].value_counts().sort_index().items()},
        "no_kaggle_submission_made": True}, indent=2))
    print("counts:", sub["target"].value_counts().sort_index().to_dict())
    print("wrote outputs/submission_v4_interaction.csv + metrics_v4_interaction.json")

if __name__ == "__main__":
    main()
