#!/usr/bin/env python3
"""Consolidate temporal signal (v6): clean FFT + autocorr representation, test
feature variants and a capacity bump on the ordinal harness. Seeds 42-46.
Writes exp_v6_consolidate.json. No Kaggle submission."""
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
W = [f"nilai_minggu_{i:02d}" for i in range(1, 13)]

def base_feats(df):
    f = build_features(df, "full")
    X = f.drop(columns=[c for c in f.columns if c.startswith("kelas") or c.startswith("hari")]).copy()
    X["motiv_x_disc"] = df["skor_motivasi"].values * df["skor_kedisiplinan"].values
    return X

def temporal(df, maxlag_d=6, maxlag_w=6, fft=True):
    Dm = df[D].to_numpy(float); Wm = df[W].to_numpy(float)
    Dw = Dm - Dm.mean(1, keepdims=True); Ww = Wm - Wm.mean(1, keepdims=True)
    o = {}
    dd = (Dw ** 2).sum(1) + 1e-9; wd = (Ww ** 2).sum(1) + 1e-9
    for lag in range(1, maxlag_d + 1): o[f"day_ac_l{lag}"] = (Dw[:, :16 - lag] * Dw[:, lag:]).sum(1) / dd
    for lag in range(1, maxlag_w + 1): o[f"wk_ac_l{lag}"] = (Ww[:, :12 - lag] * Ww[:, lag:]).sum(1) / wd
    if fft:
        Fd = np.abs(np.fft.rfft(Dw, axis=1)); Fw = np.abs(np.fft.rfft(Ww, axis=1))
        for k in range(1, 9): o[f"day_fft{k}"] = Fd[:, k]
        for k in range(1, 7): o[f"wk_fft{k}"] = Fw[:, k]
        o["day_fft_norm4"] = Fd[:, 4] / (Fd[:, 1:].sum(1) + 1e-9)
        o["wk_fft_norm1"] = Fw[:, 1] / (Fw[:, 1:].sum(1) + 1e-9)
    return pd.DataFrame(o, index=df.index)

def make(df, **kw):
    return pd.concat([base_feats(df), temporal(df, **kw)], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)

def ordb(s, d=4, n=550, lr=0.04): return XGBClassifier(n_estimators=n, max_depth=d, learning_rate=lr,
    subsample=0.9, colsample_bytree=0.9, min_child_weight=3, reg_lambda=1.5, random_state=s,
    n_jobs=-1, eval_metric="logloss", verbosity=0)
def cv_ord(X, **mk):
    accs = []
    for s in SEEDS:
        cv = StratifiedKFold(5, shuffle=True, random_state=s); o = np.zeros(len(y))
        for tr_i, va_i in cv.split(X, y):
            Xtr, Xva, ytr = X.iloc[tr_i], X.iloc[va_i], y.iloc[tr_i]; sc = np.zeros(len(va_i))
            for k in range(3):
                b = ordb(s, **mk); b.fit(Xtr, (ytr > k).astype(int)); sc += b.predict_proba(Xva)[:, 1]
            o[va_i] = sc
        accs.append(accuracy_score(y, quartile_bin(o)))
    return np.array(accs)

Xfull = make(tr, fft=True)
variants = {
    "v5_repro(fft+ac6)": (Xfull, {}),
    "ac4+fft":           (make(tr, maxlag_d=4, maxlag_w=4, fft=True), {}),
    "no_fft(ac6)":       (make(tr, fft=False), {}),
    "full depth5":       (Xfull, {"d": 5}),
    "full depth6":       (Xfull, {"d": 6}),
    "full d5 n800":      (Xfull, {"d": 5, "n": 800, "lr": 0.03}),
}
res = {}
for name, (X, mk) in variants.items():
    a = cv_ord(X, **mk); res[name] = {"mean": float(a.mean()), "std": float(a.std()), "nfeat": X.shape[1], "per_seed": a.round(4).tolist()}
    print(f"  {name:20s} nf={X.shape[1]:3d}  {a.mean():.4f} +/- {a.std():.4f}  {a.round(4)}")
best = max(res, key=lambda k: res[k]["mean"])
print(f"\nbest: {best} ({res[best]['mean']:.4f})")
Path("outputs/exp_v6_consolidate.json").write_text(json.dumps({"seeds": list(SEEDS), "results": res, "best": best, "no_kaggle_submission_made": True}, indent=2))
print("wrote outputs/exp_v6_consolidate.json")
