#!/usr/bin/env python3
"""Daily-block temporal signal: per-student lag autocorrelations (osc. pattern:
lag1 -0.32, lag3 +0.34) + lagged-product aggregates. Confirm CV lift on ordinal
harness, seeds 42-46. Writes exp_daily_lag.json. No Kaggle submission."""
from __future__ import annotations
import sys, json, warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from cv_harness import build_features, quartile_bin, SEEDS

tr = pd.read_csv("kaggle/train.csv"); y = tr["target"]
D = [f"aktivitas_hari_{i:02d}" for i in range(1, 17)]

def longest_run(mask):
    out = np.zeros(len(mask))
    for i, row in enumerate(mask):
        best = cur = 0
        for v in row: cur = cur + 1 if v else 0; best = max(best, cur)
        out[i] = best
    return out

def v4_base(df):
    f = build_features(df, "full"); X = f.drop(columns=[c for c in f.columns if c.startswith("kelas")]).copy()
    Dm = df[D].to_numpy(float)
    X["motiv_x_disc"] = df["skor_motivasi"].values * df["skor_kedisiplinan"].values
    X["day_sign_changes"] = (np.diff(np.sign(Dm - Dm.mean(1, keepdims=True)), axis=1) != 0).sum(1)
    X["day_longest_high_run"] = longest_run(Dm > np.median(Dm, 1, keepdims=True))
    p = np.abs(Dm) / (np.abs(Dm).sum(1, keepdims=True) + 1e-9)
    X["day_entropy"] = -np.sum(p * np.log(p + 1e-9), 1)
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def daily_lag_feats(df):
    Dm = df[D].to_numpy(float)
    Dw = Dm - Dm.mean(1, keepdims=True)  # within-row centered
    o = {}
    # per-student lag autocorrelations
    for lag in range(1, 7):
        num = (Dw[:, :16 - lag] * Dw[:, lag:]).sum(1)
        den = (Dw ** 2).sum(1) + 1e-9
        o[f"day_autocorr_lag{lag}"] = num / den
    o["day_ac_l3_minus_l1"] = o["day_autocorr_lag3"] - o["day_autocorr_lag1"]
    # FFT power at low freqs (periodicity)
    F = np.abs(np.fft.rfft(Dw, axis=1))
    for k in range(1, 6): o[f"day_fft_p{k}"] = F[:, k]
    o["day_fft_argmax"] = F[:, 1:].argmax(1).astype(float)
    return pd.DataFrame(o, index=df.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)

def ordb(s): return XGBClassifier(n_estimators=550, max_depth=4, learning_rate=0.04, subsample=0.9,
    colsample_bytree=0.9, min_child_weight=3, reg_lambda=1.5, random_state=s, n_jobs=-1,
    eval_metric="logloss", verbosity=0)
def cv_ord(X):
    accs = []
    for s in SEEDS:
        cv = StratifiedKFold(5, shuffle=True, random_state=s); o = np.zeros(len(y))
        for tr_i, va_i in cv.split(X, y):
            Xtr, Xva, ytr = X.iloc[tr_i], X.iloc[va_i], y.iloc[tr_i]; sc = np.zeros(len(va_i))
            for k in range(3):
                b = ordb(s); b.fit(Xtr, (ytr > k).astype(int)); sc += b.predict_proba(Xva)[:, 1]
            o[va_i] = sc
        accs.append(accuracy_score(y, quartile_bin(o)))
    return np.array(accs)

base = v4_base(tr); lag = daily_lag_feats(tr)
variants = {"v4_base": base, "+daily_lag": pd.concat([base, lag], axis=1),
            "lag_only+core": pd.concat([base[[c for c in base.columns if not c.startswith("hari")]], lag], axis=1)}
res = {}
for name, X in variants.items():
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    a = cv_ord(X); res[name] = {"mean": float(a.mean()), "std": float(a.std()), "nfeat": X.shape[1], "per_seed": a.round(4).tolist()}
    print(f"  {name:16s} nf={X.shape[1]:3d}  {a.mean():.4f} +/- {a.std():.4f}  {a.round(4)}")
best = max(res, key=lambda k: res[k]["mean"])
print(f"\nbest: {best} ({res[best]['mean']:.4f})  vs v4_base ({res['v4_base']['mean']:.4f})  delta {res[best]['mean']-res['v4_base']['mean']:+.4f}")
Path("outputs/exp_daily_lag.json").write_text(json.dumps({"seeds": list(SEEDS), "results": res, "best": best, "no_kaggle_submission_made": True}, indent=2))
print("wrote outputs/exp_daily_lag.json")
