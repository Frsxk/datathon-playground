#!/usr/bin/env python3
"""Confirm new-signal lift on the FULL harness (seeds 42-46).
New features discovered by screening:
  * motiv_x_disc = skor_motivasi * skor_kedisiplinan  (rho 0.354, ~orthogonal!)
  * disc_over_motiv ratio
  * daily-activity PATTERN: sign_changes, longest_high_run, entropy
Tested added to BOTH the ordinal model and the tuned v3-core (the current best).
CV-safe; no Kaggle submission. Writes outputs/exp_newsignal_confirm.json."""
from __future__ import annotations
import sys, json, warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, XGBRegressor
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from cv_harness import build_features, quartile_bin, z, ev, SEEDS

tr = pd.read_csv("kaggle/train.csv"); y = tr["target"]
_f = build_features(tr, "full"); base = _f.drop(columns=[c for c in _f.columns if c.startswith("kelas")])
Dc = [f"aktivitas_hari_{i:02d}" for i in range(1, 17)]; Dm = tr[Dc].to_numpy(float)
NC = 4
P = json.load(open("outputs/tune_blend_optuna.json"))["best_params"]
W_CORE = 0.7544216383299043

def longest_run(mask):
    out = np.zeros(len(mask))
    for i, row in enumerate(mask):
        best = cur = 0
        for v in row: cur = cur + 1 if v else 0; best = max(best, cur)
        out[i] = best
    return out

def add_new(X):
    X = X.copy()
    X["motiv_x_disc"] = tr["skor_motivasi"].values * tr["skor_kedisiplinan"].values
    X["disc_over_motiv"] = tr["skor_kedisiplinan"].values / (tr["skor_motivasi"].values + 1e-9)
    X["day_sign_changes"] = (np.diff(np.sign(Dm - Dm.mean(1, keepdims=True)), axis=1) != 0).sum(1)
    X["day_longest_high_run"] = longest_run(Dm > np.median(Dm, 1, keepdims=True))
    p = np.abs(Dm) / (np.abs(Dm).sum(1, keepdims=True) + 1e-9)
    X["day_entropy"] = -np.sum(p * np.log(p + 1e-9), 1)
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def add_interaction_only(X):
    X = X.copy(); X["motiv_x_disc"] = tr["skor_motivasi"].values * tr["skor_kedisiplinan"].values
    return X

def ordb(s): return XGBClassifier(n_estimators=550, max_depth=4, learning_rate=0.04, subsample=0.9,
    colsample_bytree=0.9, min_child_weight=3, reg_lambda=1.5, random_state=s, n_jobs=-1,
    eval_metric="logloss", verbosity=0)
def v3reg(s): return XGBRegressor(n_estimators=P["reg_n_estimators"], max_depth=P["reg_max_depth"],
    learning_rate=P["reg_learning_rate"], subsample=P["reg_subsample"], colsample_bytree=P["reg_colsample_bytree"],
    min_child_weight=P["reg_min_child_weight"], reg_lambda=P["reg_lambda"], reg_alpha=P["reg_alpha"],
    gamma=P["reg_gamma"], random_state=s, n_jobs=-1)
def v3clf(s): return XGBClassifier(objective="multi:softprob", num_class=NC, n_estimators=P["clf_n_estimators"],
    max_depth=P["clf_max_depth"], learning_rate=P["clf_learning_rate"], subsample=P["clf_subsample"],
    colsample_bytree=P["clf_colsample_bytree"], min_child_weight=P["clf_min_child_weight"], reg_lambda=P["clf_lambda"],
    reg_alpha=P["clf_alpha"], gamma=P["clf_gamma"], random_state=s, n_jobs=-1, eval_metric="mlogloss", verbosity=0)

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

def cv_core(X):
    accs = []
    for s in SEEDS:
        cv = StratifiedKFold(5, shuffle=True, random_state=s); o = np.zeros(len(y))
        for tr_i, va_i in cv.split(X, y):
            Xtr, Xva, ytr = X.iloc[tr_i], X.iloc[va_i], y.iloc[tr_i]
            r = v3reg(s); r.fit(Xtr, ytr); c = v3clf(s); c.fit(Xtr, ytr)
            o[va_i] = W_CORE * z(r.predict(Xva)) + (1 - W_CORE) * z(ev(c.predict_proba(Xva)))
        accs.append(accuracy_score(y, quartile_bin(o)))
    return np.array(accs)

variants = {"base": base, "base+interaction": add_interaction_only(base), "base+all_new": add_new(base)}
res = {}
print("=== ORDINAL model ===")
for name, X in variants.items():
    a = cv_ord(X); res[f"ord::{name}"] = {"mean": float(a.mean()), "std": float(a.std()), "nfeat": X.shape[1], "per_seed": a.round(4).tolist()}
    print(f"  {name:18s} nf={X.shape[1]:3d}  {a.mean():.4f} +/- {a.std():.4f}  {a.round(4)}")
print("=== V3 CORE model (current best) ===")
for name, X in variants.items():
    a = cv_core(X); res[f"core::{name}"] = {"mean": float(a.mean()), "std": float(a.std()), "nfeat": X.shape[1], "per_seed": a.round(4).tolist()}
    print(f"  {name:18s} nf={X.shape[1]:3d}  {a.mean():.4f} +/- {a.std():.4f}  {a.round(4)}")

Path("outputs/exp_newsignal_confirm.json").write_text(json.dumps({
    "seeds": list(SEEDS), "results": res, "no_kaggle_submission_made": True}, indent=2))
print("\nwrote outputs/exp_newsignal_confirm.json")
