#!/usr/bin/env python3
"""Final winner selection: seed-bagged tuned ordinal vs blends. CV-safe, seeds 42..46.

Ordinal binary params tuned in exp_ordinal.py: n=500 d=4 lr=0.04 (CV 0.5268).
Levers tested here:
  A. ordinal single (reference)
  B. ordinal seed-bagged (avg P(y>k) over BAG internal seeds per fold -> denoise rank)
  C. bagged-ordinal + reg + clf, ordinal-heavy weights
  D. learned near-quartile thresholds on the best, cross-seed honest
Writes outputs/exp_final.json.
"""
from __future__ import annotations
import sys, json, warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor, XGBClassifier
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from cv_harness import build_features, quartile_bin, z, ev, SEEDS

tr = pd.read_csv("kaggle/train.csv"); y = tr["target"]
_f = build_features(tr, "full")
X = _f.drop(columns=[c for c in _f.columns if c.startswith("kelas")])
NC = 4; BAG = 3
print(f"nfeat={X.shape[1]}  bag={BAG}")

def xbin(s): return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.04, subsample=0.9,
    colsample_bytree=0.9, min_child_weight=3, reg_lambda=1.5, random_state=s, n_jobs=-1,
    eval_metric="logloss", verbosity=0)
def xreg(s): return XGBRegressor(n_estimators=600, max_depth=3, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0, random_state=s, n_jobs=-1)
def xclf(s): return XGBClassifier(objective="multi:softprob", num_class=NC, n_estimators=500,
    max_depth=4, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9, random_state=s,
    n_jobs=-1, eval_metric="mlogloss", verbosity=0)

def oof(seed, bag=1):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    o = {k: np.zeros(len(y)) for k in ("ord", "reg", "clf")}
    for tr_i, va_i in cv.split(X, y):
        Xtr, Xva, ytr = X.iloc[tr_i], X.iloc[va_i], y.iloc[tr_i]
        oc = np.zeros(len(va_i))
        for kthr in range(NC - 1):
            pk = np.zeros(len(va_i))
            for bi in range(bag):
                b = xbin(seed + 100 * bi); b.fit(Xtr, (ytr > kthr).astype(int))
                pk += b.predict_proba(Xva)[:, 1]
            oc += pk / bag
        o["ord"][va_i] = oc
        r = xreg(seed); r.fit(Xtr, ytr); o["reg"][va_i] = r.predict(Xva)
        c = xclf(seed); c.fit(Xtr, ytr); o["clf"][va_i] = ev(c.predict_proba(Xva))
    return {k: z(v) for k, v in o.items()}

single = {s: oof(s, 1) for s in SEEDS}
bagged = {s: oof(s, BAG) for s in SEEDS}

def acc(store, fn): return np.array([accuracy_score(y, quartile_bin(fn(store[s]))) for s in SEEDS])

cfgs = {
    "A_ord_single":     (single, lambda o: o["ord"]),
    "B_ord_bagged":     (bagged, lambda o: o["ord"]),
    "C_bag_ord2reg1clf1": (bagged, lambda o: 2*o["ord"] + o["reg"] + o["clf"]),
    "C2_bag_ord3reg1clf1":(bagged, lambda o: 3*o["ord"] + o["reg"] + o["clf"]),
    "C3_bag_ordreg":    (bagged, lambda o: 2*o["ord"] + o["reg"]),
}
res = {}
print("\n=== final CV (seeds 42-46) ===")
for name, (store, fn) in cfgs.items():
    a = acc(store, fn)
    res[name] = {"mean": float(a.mean()), "std": float(a.std()), "per_seed": a.round(4).tolist()}
    print(f"  {name:22s} {a.mean():.4f} +/- {a.std():.4f}  {a.round(4)}")

best = max(res, key=lambda k: res[k]["mean"])
store_b, fn_b = cfgs[best]
print(f"\nbest: {best} ({res[best]['mean']:.4f})")

def learned_thr(score, ytrue):
    order = np.argsort(score); ss = score[order]; n = len(score)
    base = [n//4, n//2, 3*n//4]; bst = None
    for da in range(-50, 51, 25):
        for db in range(-50, 51, 25):
            for dc in range(-50, 51, 25):
                cuts = [ss[base[0]+da], ss[base[1]+db], ss[base[2]+dc]]
                p = np.digitize(score, cuts); a = (p == ytrue.values).mean()
                if bst is None or a > bst[0]: bst = (a, cuts)
    return bst[1]
lt = []
for i, s in enumerate(SEEDS):
    s2 = SEEDS[(i + 1) % len(SEEDS)]
    cuts = learned_thr(fn_b(store_b[s]), y)
    lt.append(accuracy_score(y, np.digitize(fn_b(store_b[s2]), cuts)))
lt = np.array(lt)
print(f"learned-threshold on best (cross-seed) {lt.mean():.4f} +/- {lt.std():.4f}")
res["_learned_thr"] = {"mean": float(lt.mean()), "std": float(lt.std())}

Path("outputs/exp_final.json").write_text(json.dumps({
    "feature_shape": list(X.shape), "seeds": list(SEEDS), "bag": BAG,
    "results": res, "best": best, "no_kaggle_submission_made": True}, indent=2))
print("wrote outputs/exp_final.json")
