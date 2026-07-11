#!/usr/bin/env python3
"""Production v5 — ordinal model + temporal autocorrelation signal (daily
oscillation + weekly persistence) on top of the v4 interaction. CV-safe, seeds 42-46.

Discovered signals stacked here:
  * motiv_x_disc interaction (v4)
  * daily lag autocorrelations (lag1 -0.32, lag3 +0.34 oscillation) + FFT power
  * weekly lag autocorrelations (lag1 +0.32 -> lag6 -0.30 persistence) + FFT power
Model: ordinal decomposition S=sum_k P(y>k), seed-bagged, quartile-binned.
Writes outputs/submission_v6_temporal.csv + metrics_v6_temporal.json. NO Kaggle submission."""
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
D = [f"aktivitas_hari_{i:02d}" for i in range(1, 17)]
W = [f"nilai_minggu_{i:02d}" for i in range(1, 13)]
ORD = dict(n_estimators=550, max_depth=4, learning_rate=0.04, subsample=0.9,
           colsample_bytree=0.9, min_child_weight=3, reg_lambda=1.5)

def lag_autocorr(M, maxlag):
    Mw = M - M.mean(1, keepdims=True); den = (Mw ** 2).sum(1) + 1e-9; o = {}
    for lag in range(1, maxlag + 1):
        o[lag] = (Mw[:, :M.shape[1] - lag] * Mw[:, lag:]).sum(1) / den
    return o, Mw

def feats(df):
    f = build_features(df, "full")
    # drop kelas (junk) AND generic daily 'hari_*' summaries (replaced by lag feats)
    X = f.drop(columns=[c for c in f.columns if c.startswith("kelas") or c.startswith("hari")]).copy()
    Dm = df[D].to_numpy(float); Wm = df[W].to_numpy(float)
    X["motiv_x_disc"] = df["skor_motivasi"].values * df["skor_kedisiplinan"].values
    # daily temporal (period-~3 oscillation) — pure lag autocorrelations (FFT bins
    # were redundant and only added variance; see exp_v6_consolidate.json)
    dac, Dw = lag_autocorr(Dm, 6)
    for lag, v in dac.items(): X[f"day_ac_l{lag}"] = v
    # weekly temporal (persistence: lag1 +0.32 monotone in class)
    wac, Ww = lag_autocorr(Wm, 6)
    for lag, v in wac.items(): X[f"wk_ac_l{lag}"] = v
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
    sub.to_csv("outputs/submission_v6_temporal.csv", index=False)
    Path("outputs/metrics_v6_temporal.json").write_text(json.dumps({
        "experiment": "v6_ordinal_temporal_autocorr", "generated_at": datetime.now(timezone.utc).isoformat(),
        "seeds": list(SEEDS), "bag": BAG, "feature_shape": list(X.shape),
        "cv_mean_accuracy": float(per_seed.mean()), "cv_std": float(per_seed.std()),
        "per_seed": per_seed.round(4).tolist(),
        "prediction_counts": {str(k): int(v) for k, v in sub["target"].value_counts().sort_index().items()},
        "no_kaggle_submission_made": True}, indent=2))
    print("counts:", sub["target"].value_counts().sort_index().to_dict())
    print("wrote outputs/submission_v6_temporal.csv + metrics_v6_temporal.json")

if __name__ == "__main__":
    main()
