#!/usr/bin/env python3
"""Bulk interaction expansion: hand the ordinal model explicit pairwise products
(weekly, behavioural, cross) since trees miss flat-marginal interactions.
5-seed ordinal CV. Writes exp_expand.json. No Kaggle submission."""
from __future__ import annotations
import sys, json, warnings, itertools, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from cv_harness import build_features, quartile_bin, SEEDS

tr = pd.read_csv("kaggle/train.csv"); y = tr["target"]
W = [f"nilai_minggu_{i:02d}" for i in range(1, 13)]
Dc = [f"aktivitas_hari_{i:02d}" for i in range(1, 17)]
SCAL = ["skor_motivasi","skor_kedisiplinan","skor_literasi","skor_minat_belajar",
        "indeks_kehadiran","skor_ekstrakurikuler","jarak_rumah_km","jumlah_saudara",
        "skor_tryout","urutan_ujian"]

def longest_run(mask):
    out = np.zeros(len(mask))
    for i, row in enumerate(mask):
        best = cur = 0
        for v in row: cur = cur + 1 if v else 0; best = max(best, cur)
        out[i] = best
    return out

def v4_base(df):
    f = build_features(df, "full"); X = f.drop(columns=[c for c in f.columns if c.startswith("kelas")]).copy()
    Dm = df[Dc].to_numpy(float)
    X["motiv_x_disc"] = df["skor_motivasi"].values * df["skor_kedisiplinan"].values
    X["day_sign_changes"] = (np.diff(np.sign(Dm - Dm.mean(1, keepdims=True)), axis=1) != 0).sum(1)
    X["day_longest_high_run"] = longest_run(Dm > np.median(Dm, 1, keepdims=True))
    p = np.abs(Dm) / (np.abs(Dm).sum(1, keepdims=True) + 1e-9)
    X["day_entropy"] = -np.sum(p * np.log(p + 1e-9), 1)
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def weekly_prod(df):
    o = {}
    for a, b in itertools.combinations(W, 2): o[f"wp_{a[-2:]}_{b[-2:]}"] = df[a].values * df[b].values
    return pd.DataFrame(o, index=df.index)
def behav_prod(df):
    o = {}
    for a, b in itertools.combinations(SCAL, 2): o[f"bp_{a[:6]}_{b[:6]}"] = df[a].values * df[b].values
    return pd.DataFrame(o, index=df.index)
def daily_prod(df):
    o = {}
    for a, b in itertools.combinations(Dc, 2): o[f"dp_{a[-2:]}_{b[-2:]}"] = df[a].values * df[b].values
    return pd.DataFrame(o, index=df.index)

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

base = v4_base(tr)
variants = {
    "v4_base": base,
    "+weekly_prod": pd.concat([base, weekly_prod(tr)], axis=1),
    "+behav_prod": pd.concat([base, behav_prod(tr)], axis=1),
    "+weekly+behav": pd.concat([base, weekly_prod(tr), behav_prod(tr)], axis=1),
    "+all_prod": pd.concat([base, weekly_prod(tr), behav_prod(tr), daily_prod(tr)], axis=1),
}
res = {}
for name, X in variants.items():
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    a = cv_ord(X); res[name] = {"mean": float(a.mean()), "std": float(a.std()), "nfeat": X.shape[1], "per_seed": a.round(4).tolist()}
    print(f"  {name:16s} nf={X.shape[1]:4d}  {a.mean():.4f} +/- {a.std():.4f}  {a.round(4)}")

best = max(res, key=lambda k: res[k]["mean"])
print(f"\nbest: {best} ({res[best]['mean']:.4f})  vs v4_base ({res['v4_base']['mean']:.4f})")
Path("outputs/exp_expand.json").write_text(json.dumps({"seeds": list(SEEDS), "results": res, "best": best, "no_kaggle_submission_made": True}, indent=2))
print("wrote outputs/exp_expand.json")
