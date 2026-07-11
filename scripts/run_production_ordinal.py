#!/usr/bin/env python3
"""Production ordinal model + submission generator (Datathon 2026 Playground).

Winning framing (see reports/ordinal_findings via notebook): the ordered target
is best estimated by ORDINAL DECOMPOSITION -- fit binary XGB models for P(y>0),
P(y>1), P(y>2); the cumulative sum S = sum_k P(y>k) is a latent rank in [0,3];
quartile-bin S into 200/200/200/200 balanced predictions.

Tuned binary params: n=550 d=4 lr=0.04. Seed-bagged (BAG internal seeds) for a
lower-variance rank. CV-safe (fit in-fold); quartile-bin on OOF only.

Writes outputs/submission_ordinal.csv + outputs/metrics_ordinal.json.
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
ORD_PARAMS = dict(n_estimators=550, max_depth=4, learning_rate=0.04, subsample=0.9,
                  colsample_bytree=0.9, min_child_weight=3, reg_lambda=1.5)

def feats(df):
    f = build_features(df, "full")
    return f.drop(columns=[c for c in f.columns if c.startswith("kelas")])

def xbin(seed): return XGBClassifier(**ORD_PARAMS, random_state=seed, n_jobs=-1,
    eval_metric="logloss", verbosity=0)

def ordinal_score(Xtr, ytr, Xpred, seed, bag=BAG):
    """Cumulative sum_k P(y>k), seed-bagged. Returns latent rank for Xpred rows."""
    s = np.zeros(len(Xpred))
    for kthr in range(NC - 1):
        pk = np.zeros(len(Xpred))
        for bi in range(bag):
            b = xbin(seed + 100 * bi); b.fit(Xtr, (ytr > kthr).astype(int))
            pk += b.predict_proba(Xpred)[:, 1]
        s += pk / bag
    return s

def main():
    tr = pd.read_csv("kaggle/train.csv"); te = pd.read_csv("kaggle/test.csv")
    sample = pd.read_csv("kaggle/sample_submission.csv")
    y = tr["target"]; X = feats(tr); Xte = feats(te)
    assert list(X.columns) == list(Xte.columns)
    print(f"nfeat={X.shape[1]} bag={BAG} params={ORD_PARAMS}")

    # OOF CV over seeds
    per_seed = []
    for s in SEEDS:
        cv = StratifiedKFold(5, shuffle=True, random_state=s); oof = np.zeros(len(y))
        for tr_i, va_i in cv.split(X, y):
            oof[va_i] = ordinal_score(X.iloc[tr_i], y.iloc[tr_i], X.iloc[va_i], s)
        per_seed.append(accuracy_score(y, quartile_bin(oof)))
        print(f"  seed {s}: {per_seed[-1]:.4f}")
    per_seed = np.array(per_seed)
    print(f"CV {per_seed.mean():.4f} +/- {per_seed.std():.4f}")

    # Refit on full train across seeds, average test rank, quartile-bin
    test_scores = []
    for s in SEEDS:
        test_scores.append(pd.Series(ordinal_score(X, y, Xte, s)).rank().values)
    preds = quartile_bin(np.mean(test_scores, 0)).astype(int)

    sub = pd.DataFrame({"id": te["id"], "target": preds})
    assert list(sub.columns) == ["id", "target"]
    assert sub.shape == sample.shape
    assert sub["id"].equals(sample["id"])
    assert set(sub["target"].unique()).issubset({0, 1, 2, 3})
    Path("outputs").mkdir(exist_ok=True)
    sub.to_csv("outputs/submission_ordinal.csv", index=False)

    payload = {
        "experiment": "ordinal_decomposition_seedbagged",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seeds": list(SEEDS), "bag": BAG, "ord_params": ORD_PARAMS,
        "feature_shape": list(X.shape),
        "cv_mean_accuracy": float(per_seed.mean()), "cv_std": float(per_seed.std()),
        "per_seed": per_seed.round(4).tolist(),
        "prediction_counts": {str(k): int(v) for k, v in sub["target"].value_counts().sort_index().items()},
        "no_kaggle_submission_made": True,
    }
    Path("outputs/metrics_ordinal.json").write_text(json.dumps(payload, indent=2))
    print("counts:", payload["prediction_counts"])
    print("wrote outputs/submission_ordinal.csv + metrics_ordinal.json")

if __name__ == "__main__":
    main()
