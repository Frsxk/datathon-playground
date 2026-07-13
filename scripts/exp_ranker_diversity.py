#!/usr/bin/env python3
"""Ranker-diversity experiment. The v6 pipeline rides on ONE ordinal-XGB latent
rank; its ceiling is now per-student ESTIMATION noise, not missing signal. If a
different model family makes uncorrelated errors estimating the same latent rank,
a z-blend of ranks lowers the noise floor even with identical features.

Tests, on the SAME v6 feature matrix, latent-rank estimators:
  - ord_xgb   : v6 ordinal decomposition (reference)
  - ord_hgb   : ordinal decomposition with HistGradientBoosting binaries
  - reg_xgb   : direct XGBRegressor on y (expected-value rank)
  - reg_et    : ExtraTrees regressor
and z-blends of them. Reports per-seed CV and OOF rank correlations (diversity).

2 seeds by default (fast); --seeds 5 for full. NO Kaggle submission."""
from __future__ import annotations
import sys, json, argparse, warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesRegressor
from xgboost import XGBClassifier, XGBRegressor
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from cv_harness import quartile_bin, SEEDS
from run_production_v6_temporal import feats, ORD, NC, BAG

def z(v):
    v = np.asarray(v, float); return (v - v.mean()) / (v.std() + 1e-12)

def ord_xgb(Xtr, ytr, Xp, seed, bag=BAG):
    s = np.zeros(len(Xp))
    for k in range(NC - 1):
        pk = np.zeros(len(Xp))
        for bi in range(bag):
            m = XGBClassifier(**ORD, random_state=seed + 100 * bi, n_jobs=-1,
                              eval_metric="logloss", verbosity=0)
            m.fit(Xtr, (ytr > k).astype(int)); pk += m.predict_proba(Xp)[:, 1]
        s += pk / bag
    return s

def ord_hgb(Xtr, ytr, Xp, seed):
    s = np.zeros(len(Xp))
    for k in range(NC - 1):
        m = HistGradientBoostingClassifier(max_iter=400, max_depth=4, learning_rate=0.05,
                                           l2_regularization=1.0, random_state=seed)
        m.fit(Xtr, (ytr > k).astype(int)); s += m.predict_proba(Xp)[:, 1]
    return s

def reg_xgb(Xtr, ytr, Xp, seed):
    m = XGBRegressor(n_estimators=550, max_depth=4, learning_rate=0.04, subsample=0.9,
                     colsample_bytree=0.9, min_child_weight=3, reg_lambda=1.5,
                     random_state=seed, n_jobs=-1)
    m.fit(Xtr, ytr); return m.predict(Xp)

def reg_et(Xtr, ytr, Xp, seed):
    m = ExtraTreesRegressor(n_estimators=500, max_depth=None, min_samples_leaf=8,
                            max_features=0.5, random_state=seed, n_jobs=-1)
    m.fit(Xtr, ytr.astype(float)); return m.predict(Xp)

EST = {"ord_xgb": ord_xgb, "ord_hgb": ord_hgb, "reg_xgb": reg_xgb, "reg_et": reg_et}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=2); a = ap.parse_args()
    seeds = SEEDS[:a.seeds]
    tr = pd.read_csv("kaggle/train.csv"); y = tr["target"]; X = feats(tr)
    print(f"nfeat={X.shape[1]} seeds={seeds}")

    # OOF per estimator per seed
    oof = {k: {s: np.zeros(len(y)) for s in seeds} for k in EST}
    for s in seeds:
        cv = StratifiedKFold(5, shuffle=True, random_state=s)
        for tr_i, va_i in cv.split(X, y):
            Xtr, Xva, ytr = X.iloc[tr_i], X.iloc[va_i], y.iloc[tr_i]
            for name, fn in EST.items():
                oof[name][s][va_i] = fn(Xtr, ytr, Xva, s)

    # per-estimator accuracy
    print("\n-- solo estimators --")
    solo = {}
    for name in EST:
        accs = [accuracy_score(y, quartile_bin(oof[name][s])) for s in seeds]
        solo[name] = float(np.mean(accs)); print(f"  {name:9s} CV {np.mean(accs):.4f}  {np.round(accs,4)}")

    # rank correlations between estimators (diversity), avg over seeds
    print("\n-- OOF rank spearman (avg over seeds) --")
    names = list(EST)
    from scipy.stats import rankdata
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            cs = np.mean([np.corrcoef(rankdata(oof[names[i]][s]), rankdata(oof[names[j]][s]))[0, 1] for s in seeds])
            print(f"  {names[i]:9s} ~ {names[j]:9s}  {cs:.3f}")

    # blends (equal-weight z of ranks)
    print("\n-- blends (equal-weight z) --")
    blends = {
        "xgb+hgb": ["ord_xgb", "ord_hgb"],
        "xgb+reg": ["ord_xgb", "reg_xgb"],
        "xgb+et": ["ord_xgb", "reg_et"],
        "xgb+hgb+reg": ["ord_xgb", "ord_hgb", "reg_xgb"],
        "all4": ["ord_xgb", "ord_hgb", "reg_xgb", "reg_et"],
        "xgb+hgb+et": ["ord_xgb", "ord_hgb", "reg_et"],
    }
    results = {"solo": solo, "blends": {}}
    for bname, members in blends.items():
        accs = []
        for s in seeds:
            score = np.mean([z(oof[m][s]) for m in members], 0)
            accs.append(accuracy_score(y, quartile_bin(score)))
        results["blends"][bname] = float(np.mean(accs))
        d = np.mean(accs) - solo["ord_xgb"]
        print(f"  {bname:14s} CV {np.mean(accs):.4f}  (vs ord_xgb {d:+.4f})  {np.round(accs,4)}")

    results["seeds"] = list(seeds); results["no_kaggle_submission_made"] = True
    Path("outputs/exp_ranker_diversity.json").write_text(json.dumps(results, indent=2))
    print("\nwrote outputs/exp_ranker_diversity.json")

if __name__ == "__main__":
    main()
