#!/usr/bin/env python3
"""Residual screen vs v6 rank: what signal remains after temporal + interaction?
Screens refined temporal, cross-block, and composite candidates by partial corr.
Cheap (one v6 ordinal rank + correlations). Writes exp_residual_v6.json."""
from __future__ import annotations
import sys, json, warnings, itertools, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, rankdata
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from run_production_v6_temporal import feats
from cv_harness import quartile_bin

tr = pd.read_csv("kaggle/train.csv"); y = tr["target"]; yv = y.values
X = feats(tr)
D = [f"aktivitas_hari_{i:02d}" for i in range(1, 17)]; Dm = tr[D].to_numpy(float)
W = [f"nilai_minggu_{i:02d}" for i in range(1, 13)]; Wm = tr[W].to_numpy(float)

def ord_oof(seed=42):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed); s = np.zeros(len(y))
    for tr_i, va_i in cv.split(X, y):
        Xtr, Xva, ytr = X.iloc[tr_i], X.iloc[va_i], y.iloc[tr_i]
        for k in range(3):
            b = XGBClassifier(n_estimators=550, max_depth=4, learning_rate=0.04, subsample=0.9,
                colsample_bytree=0.9, min_child_weight=3, reg_lambda=1.5, random_state=seed,
                n_jobs=-1, eval_metric="logloss", verbosity=0)
            b.fit(Xtr, (ytr > k).astype(int)); s[va_i] += b.predict_proba(Xva)[:, 1]
    return s
cur = rankdata(ord_oof()); ry = rankdata(yv)
ty = ry - np.polyval(np.polyfit(cur, ry, 1), cur); tvar = np.sqrt((ty**2).mean())
print(f"v6 OOF quartile acc (seed42): {(quartile_bin(cur) == yv).mean():.4f}")
def partial(v):
    rv = rankdata(v); fr = rv - np.polyval(np.polyfit(cur, rv, 1), cur); fv = np.sqrt((fr**2).mean())
    return float((fr*ty).mean()/(fv*tvar+1e-12)) if fv > 1e-9 else 0.0

Dw = Dm - Dm.mean(1, keepdims=True); Ww = Wm - Wm.mean(1, keepdims=True)
dd = (Dw**2).sum(1)+1e-9; wd = (Ww**2).sum(1)+1e-9
dac = {l: (Dw[:, :16-l]*Dw[:, l:]).sum(1)/dd for l in range(1, 7)}
wac = {l: (Ww[:, :12-l]*Ww[:, l:]).sum(1)/wd for l in range(1, 7)}
mk = tr["skor_motivasi"].values * tr["skor_kedisiplinan"].values
cr = (tr["tugas_selesai"]/tr["tugas_diberikan"].replace(0, np.nan)).fillna(0).values

cand = {}
# cross-block temporal
cand["wac1_x_dac3"] = wac[1] * dac[3]
cand["wac1_x_mk"] = wac[1] * mk
cand["dac3_x_mk"] = dac[3] * mk
cand["wac1_x_compl"] = wac[1] * cr
cand["dac2_x_dac4"] = dac[2] * dac[4]
# higher lags / refined temporal
for l in range(7, 11):
    if 16 - l > 0: cand[f"day_ac_l{l}"] = (Dw[:, :16-l]*Dw[:, l:]).sum(1)/dd
for l in range(7, 10):
    if 12 - l > 0: cand[f"wk_ac_l{l}"] = (Ww[:, :12-l]*Ww[:, l:]).sum(1)/wd
# partial autocorr-ish: ac ratios
cand["dac3_over_dac1"] = dac[3] / (np.abs(dac[1]) + 0.1)
cand["wac1_sq"] = wac[1] ** 2
# second differences / curvature
cand["wk_2diff_std"] = np.diff(Wm, 2, axis=1).std(1)
cand["day_2diff_std"] = np.diff(Dm, 2, axis=1).std(1)
# cross-lag between blocks (weekly change t vs daily activity)
cand["wstd_x_dac3"] = Wm.std(1) * dac[3]

rows = [(n, round(spearmanr(v, yv).correlation, 3), round(partial(v), 3)) for n, v in cand.items()]
rows.sort(key=lambda t: -abs(t[2]))
print("\ncandidate            spearman  partial_vs_v6")
for n, sp, pa in rows: print(f"  {n:20s} {sp:+.3f}    {pa:+.3f}")
Path("outputs/exp_residual_v6.json").write_text(json.dumps({
    "v6_oof_acc_seed42": float((quartile_bin(cur) == yv).mean()),
    "candidates": [{"name": n, "spearman": sp, "partial": pa} for n, sp, pa in rows],
    "no_kaggle_submission_made": True}, indent=2))
print("\nwrote outputs/exp_residual_v6.json")
