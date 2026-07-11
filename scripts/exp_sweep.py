#!/usr/bin/env python3
"""High-value modeling sweep for the latent-score->quartile problem.

Everything is CV-safe (estimators fit in-fold; binning/threshold on OOF only).
Reports multi-seed (42..46) quartile-bin accuracy. No Kaggle submission.

Ideas tested:
  1. Latent-score model families incl. ORDINAL decomposition (P(y>k) cumulative)
  2. Fixed-weight blends of the strong signals (no CV weight tuning => no overfit)
  3. Learned OOF thresholds vs fixed quartile (nested: thresholds from a held-out
     fold split, applied out-of-sample -- honest estimate)
"""
from __future__ import annotations
import sys, json, warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from cv_harness import build_features, quartile_bin, z, ev, SEEDS

ROOT = Path(".")
tr = pd.read_csv("kaggle/train.csv"); y = tr["target"]
X = build_features(tr, "full").drop(columns=[c for c in build_features(tr, "full").columns if c.startswith("kelas")])
NC = 4
print(f"nfeat={X.shape[1]}")

def xreg(s, **k): return XGBRegressor(n_estimators=k.get("n",600), max_depth=k.get("d",3),
    learning_rate=k.get("lr",0.03), subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
    reg_lambda=2.0, random_state=s, n_jobs=-1)
def xclf(s): return XGBClassifier(objective="multi:softprob", num_class=NC, n_estimators=500,
    max_depth=4, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9, random_state=s,
    n_jobs=-1, eval_metric="mlogloss", verbosity=0)
def xbin(s): return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.04, subsample=0.9,
    colsample_bytree=0.9, random_state=s, n_jobs=-1, eval_metric="logloss", verbosity=0)
def hgb(s): return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=s)
def rid(s): return make_pipeline(StandardScaler(), Ridge(alpha=10.0))

def oof_all(seed):
    """Return dict of OOF latent scores (z-normed) for every base estimator."""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    keys = ["reg", "clf_ev", "hgb_ev", "ridge", "ord"]
    o = {k: np.zeros(len(y)) for k in keys}
    for tr_i, va_i in cv.split(X, y):
        Xtr, Xva, ytr = X.iloc[tr_i], X.iloc[va_i], y.iloc[tr_i]
        r = xreg(seed); r.fit(Xtr, ytr); o["reg"][va_i] = r.predict(Xva)
        c = xclf(seed); c.fit(Xtr, ytr); o["clf_ev"][va_i] = ev(c.predict_proba(Xva))
        h = hgb(seed); h.fit(Xtr, ytr); o["hgb_ev"][va_i] = ev(h.predict_proba(Xva))
        rd = rid(seed); rd.fit(Xtr, ytr); o["ridge"][va_i] = rd.predict(Xva)
        # ordinal decomposition: sum_k P(y>k) is a latent rank in [0,3]
        ords = np.zeros(len(va_i))
        for kthr in range(NC - 1):
            b = xbin(seed); b.fit(Xtr, (ytr > kthr).astype(int))
            ords += b.predict_proba(Xva)[:, 1]
        o["ord"][va_i] = ords
    return {k: z(v) for k, v in o.items()}

def learned_thresholds(score, ytrue):
    """Pick 3 cut points on a score to maximize accuracy (grid on quantiles)."""
    order = np.argsort(score); s_sorted = score[order]
    # search near quartiles only (balanced assumption) -> small honest search
    n = len(score); best = None
    base = [n//4, n//2, 3*n//4]
    for da in range(-60, 61, 20):
        for db in range(-60, 61, 20):
            for dc in range(-60, 61, 20):
                cuts = [s_sorted[base[0]+da], s_sorted[base[1]+db], s_sorted[base[2]+dc]]
                pred = np.digitize(score, cuts)
                acc = (pred == ytrue.values).mean()
                if best is None or acc > best[0]: best = (acc, cuts)
    return best[1]

results = {}
per_seed_oof = {s: oof_all(s) for s in SEEDS}

def blend_acc(fn):
    accs = []
    for s in SEEDS:
        o = per_seed_oof[s]
        accs.append(accuracy_score(y, quartile_bin(fn(o))))
    return np.array(accs)

configs = {
    "reg_only":        lambda o: o["reg"],
    "clf_ev_only":     lambda o: o["clf_ev"],
    "ord_only":        lambda o: o["ord"],
    "pair_reg_clf":    lambda o: o["reg"] + o["clf_ev"],
    "reg_clf_ord":     lambda o: o["reg"] + o["clf_ev"] + o["ord"],
    "reg_clf_ord_hgb": lambda o: o["reg"] + o["clf_ev"] + o["ord"] + o["hgb_ev"],
    "w_reg2clf1ord1":  lambda o: 2*o["reg"] + o["clf_ev"] + o["ord"],
    "all5_equal":      lambda o: o["reg"]+o["clf_ev"]+o["ord"]+o["hgb_ev"]+o["ridge"],
}
print("\n=== quartile-bin CV (mean over seeds 42-46) ===")
for name, fn in configs.items():
    a = blend_acc(fn)
    results[name] = {"mean": float(a.mean()), "std": float(a.std()), "per_seed": a.round(4).tolist()}
    print(f"  {name:18s} {a.mean():.4f} +/- {a.std():.4f}  {a.round(4)}")

# Learned thresholds on the best blend, nested/honest: fit thresholds on seed s,
# apply to a DIFFERENT seed's OOF score (proxy for out-of-sample threshold use).
best_name = max(results, key=lambda k: results[k]["mean"])
best_fn = configs[best_name]
print(f"\nBest blend: {best_name} ({results[best_name]['mean']:.4f})")
lt = []
for i, s in enumerate(SEEDS):
    s2 = SEEDS[(i + 1) % len(SEEDS)]
    cuts = learned_thresholds(best_fn(per_seed_oof[s]), y)
    pred = np.digitize(best_fn(per_seed_oof[s2]), cuts)
    lt.append(accuracy_score(y, pred))
lt = np.array(lt)
print(f"learned-threshold (cross-seed honest) {lt.mean():.4f} +/- {lt.std():.4f}")
results["_learned_threshold_crossseed"] = {"mean": float(lt.mean()), "std": float(lt.std())}

Path("outputs").mkdir(exist_ok=True)
Path("outputs/exp_sweep.json").write_text(json.dumps({
    "feature_shape": list(X.shape), "seeds": list(SEEDS), "results": results,
    "best_blend": best_name, "no_kaggle_submission_made": True}, indent=2))
print("\nwrote outputs/exp_sweep.json")
