#!/usr/bin/env python3
"""Ordinal-centric refinement: the P(y>k) cumulative score is the winner.
Tune the binary model + test ordinal-weighted blends. CV-safe, seeds 42..46."""
from __future__ import annotations
import sys, json, warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from cv_harness import build_features, quartile_bin, z, ev, SEEDS

tr = pd.read_csv("kaggle/train.csv"); y = tr["target"]
_f = build_features(tr, "full")
X = _f.drop(columns=[c for c in _f.columns if c.startswith("kelas")])
NC = 4
print(f"nfeat={X.shape[1]}")

def xbin(s, n, d, lr): return XGBClassifier(n_estimators=n, max_depth=d, learning_rate=lr,
    subsample=0.9, colsample_bytree=0.9, min_child_weight=3, reg_lambda=1.5, random_state=s,
    n_jobs=-1, eval_metric="logloss", verbosity=0)

def ord_oof(seed, n, d, lr):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    o = np.zeros(len(y))
    for tr_i, va_i in cv.split(X, y):
        Xtr, Xva, ytr = X.iloc[tr_i], X.iloc[va_i], y.iloc[tr_i]
        acc = np.zeros(len(va_i))
        for kthr in range(NC - 1):
            b = xbin(seed, n, d, lr); b.fit(Xtr, (ytr > kthr).astype(int))
            acc += b.predict_proba(Xva)[:, 1]
        o[va_i] = acc
    return o

print("\n=== tune ordinal binary model ===")
grid = [(400,3,0.04),(500,4,0.04),(700,3,0.03),(600,4,0.03),(800,4,0.02),(500,5,0.03),(1000,3,0.02)]
best = None
for (n, d, lr) in grid:
    accs = [accuracy_score(y, quartile_bin(ord_oof(s, n, d, lr))) for s in SEEDS]
    a = np.array(accs)
    tag = f"n={n} d={d} lr={lr}"
    print(f"  {tag:20s} {a.mean():.4f} +/- {a.std():.4f}")
    if best is None or a.mean() > best[0]: best = (a.mean(), a.std(), (n, d, lr), a.round(4).tolist())
print(f"\nbest ordinal: {best[2]}  CV {best[0]:.4f} +/- {best[1]:.4f}")

Path("outputs/exp_ordinal.json").write_text(json.dumps({
    "feature_shape": list(X.shape), "seeds": list(SEEDS),
    "best_params": {"n_estimators": best[2][0], "max_depth": best[2][1], "learning_rate": best[2][2]},
    "best_cv_mean": best[0], "best_cv_std": best[1], "best_per_seed": best[3],
    "no_kaggle_submission_made": True}, indent=2))
print("wrote outputs/exp_ordinal.json")
