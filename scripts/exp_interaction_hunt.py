#!/usr/bin/env python3
"""Exhaustive interaction discovery. Screen products/ratios/diffs of ALL raw
feature pairs by PARTIAL spearman vs the current v4 model rank (signal it misses).
Cheap: one ordinal OOF rank + vectorized correlations. Writes exp_interaction_hunt.json."""
from __future__ import annotations
import sys, json, warnings, itertools, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, rankdata
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from cv_harness import build_features

tr = pd.read_csv("kaggle/train.csv"); y = tr["target"]; yv = y.values
raw = [c for c in tr.columns if c not in ("id", "target")]
print(f"raw features: {len(raw)}  pairs: {len(raw)*(len(raw)-1)//2}")

# ---- current v4 model rank (base+interaction+day patterns), seed 42 ----
Dc = [f"aktivitas_hari_{i:02d}" for i in range(1, 17)]
def longest_run(mask):
    out = np.zeros(len(mask))
    for i, row in enumerate(mask):
        best = cur = 0
        for v in row: cur = cur + 1 if v else 0; best = max(best, cur)
        out[i] = best
    return out
def v4_feats(df):
    f = build_features(df, "full"); X = f.drop(columns=[c for c in f.columns if c.startswith("kelas")]).copy()
    Dm = df[Dc].to_numpy(float)
    X["motiv_x_disc"] = df["skor_motivasi"].values * df["skor_kedisiplinan"].values
    X["day_sign_changes"] = (np.diff(np.sign(Dm - Dm.mean(1, keepdims=True)), axis=1) != 0).sum(1)
    X["day_longest_high_run"] = longest_run(Dm > np.median(Dm, 1, keepdims=True))
    p = np.abs(Dm) / (np.abs(Dm).sum(1, keepdims=True) + 1e-9)
    X["day_entropy"] = -np.sum(p * np.log(p + 1e-9), 1)
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X = v4_feats(tr)
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
ty = ry - np.polyval(np.polyfit(cur, ry, 1), cur)          # target residual vs current model
tvar = np.sqrt((ty**2).mean())
def partial(v):
    rv = rankdata(v); fr = rv - np.polyval(np.polyfit(cur, rv, 1), cur)
    fvar = np.sqrt((fr**2).mean())
    return float((fr*ty).mean()/(fvar*tvar+1e-12)) if fvar > 1e-9 else 0.0
print("built current v4 rank\n")

R = tr[raw].to_numpy(float)
def scan(op, name):
    rows = []
    for i, j in itertools.combinations(range(len(raw)), 2):
        a, b = R[:, i], R[:, j]
        if op == "prod": v = a * b
        elif op == "ratio": v = a / (b + 1e-9)
        elif op == "diff": v = a - b
        if np.std(v) < 1e-9: continue
        sp = spearmanr(v, yv).correlation
        if abs(sp) < 0.12: continue
        rows.append((f"{raw[i]} {name} {raw[j]}", round(sp, 3), round(partial(v), 3)))
    return rows

allrows = []
for op, nm in [("prod", "*"), ("ratio", "/"), ("diff", "-")]:
    rows = scan(op, nm); allrows += rows
    print(f"== {op}: {len(rows)} pairs with |spearman|>0.12 ==")
res = sorted(allrows, key=lambda t: -abs(t[2]))
print("\nTOP by PARTIAL corr vs current v4 (new orthogonal signal):")
for n, sp, pa in res[:25]:
    print(f"  {n:44s} sp={sp:+.3f}  partial={pa:+.3f}")

Path("outputs/exp_interaction_hunt.json").write_text(json.dumps({
    "n_raw": len(raw), "top": [{"pair": n, "spearman": sp, "partial": pa} for n, sp, pa in res[:60]],
    "no_kaggle_submission_made": True}, indent=2))
print("\nwrote outputs/exp_interaction_hunt.json")
