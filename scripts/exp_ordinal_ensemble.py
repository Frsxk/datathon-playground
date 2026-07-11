#!/usr/bin/env python3
"""Diverse ordinal ensemble + fine tune. Rank-average two independent ordinal
score estimators (XGB, HistGB) to denoise the latent rank. CV-safe, seeds 42..46."""
from __future__ import annotations
import sys, json, warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from cv_harness import build_features, quartile_bin, z, SEEDS

tr = pd.read_csv("kaggle/train.csv"); y = tr["target"]
_f = build_features(tr, "full")
X = _f.drop(columns=[c for c in _f.columns if c.startswith("kelas")])
NC = 4
print(f"nfeat={X.shape[1]}")

def xbin(s, n, d, lr): return XGBClassifier(n_estimators=n, max_depth=d, learning_rate=lr,
    subsample=0.9, colsample_bytree=0.9, min_child_weight=3, reg_lambda=1.5, random_state=s,
    n_jobs=-1, eval_metric="logloss", verbosity=0)
def hbin(s): return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.04,
    max_depth=4, l2_regularization=1.0, random_state=s)

def rankz(v): return z(pd.Series(v).rank().values)

def ord_scores(seed, xp):
    """Return z-ranked ordinal scores from XGB and HGB, one seed."""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    ox = np.zeros(len(y)); oh = np.zeros(len(y))
    for tr_i, va_i in cv.split(X, y):
        Xtr, Xva, ytr = X.iloc[tr_i], X.iloc[va_i], y.iloc[tr_i]
        sx = np.zeros(len(va_i)); sh = np.zeros(len(va_i))
        for kthr in range(NC - 1):
            b = xbin(seed, *xp); b.fit(Xtr, (ytr > kthr).astype(int))
            sx += b.predict_proba(Xva)[:, 1]
            h = hbin(seed); h.fit(Xtr, (ytr > kthr).astype(int))
            sh += h.predict_proba(Xva)[:, 1]
        ox[va_i] = sx; oh[va_i] = sh
    return rankz(ox), rankz(oh)

print("\n=== fine tune XGB ordinal ===")
best = None
for xp in [(500,4,0.04),(450,4,0.045),(550,4,0.04),(500,4,0.05),(600,4,0.035),(500,4,0.04)]:
    accs = []
    for s in SEEDS:
        cv = StratifiedKFold(5, shuffle=True, random_state=s); o = np.zeros(len(y))
        for tr_i, va_i in cv.split(X, y):
            Xtr, Xva, ytr = X.iloc[tr_i], X.iloc[va_i], y.iloc[tr_i]; sc = np.zeros(len(va_i))
            for kthr in range(NC - 1):
                b = xbin(s, *xp); b.fit(Xtr, (ytr > kthr).astype(int)); sc += b.predict_proba(Xva)[:, 1]
            o[va_i] = sc
        accs.append(accuracy_score(y, quartile_bin(o)))
    a = np.array(accs); print(f"  xgb {str(xp):18s} {a.mean():.4f} +/- {a.std():.4f}")
    if best is None or a.mean() > best[0]: best = (a.mean(), xp)
xp = best[1]; print(f"best xgb ordinal params {xp} ({best[0]:.4f})")

print("\n=== diverse ordinal ensemble (XGB + HGB rank-avg) ===")
store = {s: ord_scores(s, xp) for s in SEEDS}
res = {}
for name, fn in [("xgb_ord", lambda ox, oh: ox), ("hgb_ord", lambda ox, oh: oh),
                 ("ens_50_50", lambda ox, oh: ox + oh), ("ens_70_30", lambda ox, oh: 0.7*ox + 0.3*oh)]:
    a = np.array([accuracy_score(y, quartile_bin(fn(*store[s]))) for s in SEEDS])
    res[name] = {"mean": float(a.mean()), "std": float(a.std()), "per_seed": a.round(4).tolist()}
    print(f"  {name:10s} {a.mean():.4f} +/- {a.std():.4f}  {a.round(4)}")

win = max(res, key=lambda k: res[k]["mean"])
print(f"\nWINNER: {win} ({res[win]['mean']:.4f} +/- {res[win]['std']:.4f})")
Path("outputs/exp_ordinal_ensemble.json").write_text(json.dumps({
    "feature_shape": list(X.shape), "seeds": list(SEEDS),
    "xgb_ordinal_params": {"n_estimators": xp[0], "max_depth": xp[1], "learning_rate": xp[2]},
    "results": res, "winner": win, "no_kaggle_submission_made": True}, indent=2))
print("wrote outputs/exp_ordinal_ensemble.json")
