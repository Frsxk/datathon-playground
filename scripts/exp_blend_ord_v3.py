#!/usr/bin/env python3
"""Rank-blend of the two decent rankers: ordinal Sigma P(y>k) + v3 core (tuned
reg + clf_ev). Question: does averaging their RANKS beat either alone on CV, as a
more robust (private-score) diversified candidate? CV-safe, seeds 42..46.
Writes outputs/exp_blend_ord_v3.json + candidate submission if competitive.
No Kaggle submission."""
from __future__ import annotations
import sys, json, warnings, numpy as np, pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, XGBRegressor
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from cv_harness import build_features, quartile_bin, z, ev, SEEDS

tr = pd.read_csv("kaggle/train.csv"); te = pd.read_csv("kaggle/test.csv")
sample = pd.read_csv("kaggle/sample_submission.csv")
y = tr["target"]
_f = build_features(tr, "full"); _ft = build_features(te, "full")
X  = _f.drop(columns=[c for c in _f.columns if c.startswith("kelas")])
Xte = _ft.drop(columns=[c for c in _ft.columns if c.startswith("kelas")])
NC = 4
P = json.load(open("outputs/tune_blend_optuna.json"))["best_params"]
W_CORE = 0.7544216383299043  # v3's selected blend weight
print(f"nfeat={X.shape[1]}  v3 blend_w={W_CORE:.3f}")

ORD = dict(n_estimators=550, max_depth=4, learning_rate=0.04, subsample=0.9,
           colsample_bytree=0.9, min_child_weight=3, reg_lambda=1.5)

def v3_reg(s): return XGBRegressor(n_estimators=P["reg_n_estimators"], max_depth=P["reg_max_depth"],
    learning_rate=P["reg_learning_rate"], subsample=P["reg_subsample"], colsample_bytree=P["reg_colsample_bytree"],
    min_child_weight=P["reg_min_child_weight"], reg_lambda=P["reg_lambda"], reg_alpha=P["reg_alpha"],
    gamma=P["reg_gamma"], random_state=s, n_jobs=-1)
def v3_clf(s): return XGBClassifier(objective="multi:softprob", num_class=NC, n_estimators=P["clf_n_estimators"],
    max_depth=P["clf_max_depth"], learning_rate=P["clf_learning_rate"], subsample=P["clf_subsample"],
    colsample_bytree=P["clf_colsample_bytree"], min_child_weight=P["clf_min_child_weight"], reg_lambda=P["clf_lambda"],
    reg_alpha=P["clf_alpha"], gamma=P["clf_gamma"], random_state=s, n_jobs=-1, eval_metric="mlogloss", verbosity=0)
def ordb(s): return XGBClassifier(**ORD, random_state=s, n_jobs=-1, eval_metric="logloss", verbosity=0)

def rankz(v): return z(pd.Series(v).rank().values)

def oof(seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    o = {k: np.zeros(len(y)) for k in ("ord", "core")}
    for tr_i, va_i in cv.split(X, y):
        Xtr, Xva, ytr = X.iloc[tr_i], X.iloc[va_i], y.iloc[tr_i]
        sc = np.zeros(len(va_i))
        for kthr in range(NC - 1):
            b = ordb(seed); b.fit(Xtr, (ytr > kthr).astype(int)); sc += b.predict_proba(Xva)[:, 1]
        o["ord"][va_i] = sc
        r = v3_reg(seed); r.fit(Xtr, ytr); rs = r.predict(Xva)
        c = v3_clf(seed); c.fit(Xtr, ytr); cs = ev(c.predict_proba(Xva))
        o["core"][va_i] = W_CORE * z(rs) + (1 - W_CORE) * z(cs)
    return {"ord": rankz(o["ord"]), "core": rankz(o["core"])}

store = {s: oof(s) for s in SEEDS}
cfgs = {"ord_only": lambda o: o["ord"], "core_only(v3)": lambda o: o["core"],
        "blend_50_50": lambda o: o["ord"] + o["core"],
        "blend_60ord": lambda o: 0.6*o["ord"] + 0.4*o["core"],
        "blend_40ord": lambda o: 0.4*o["ord"] + 0.6*o["core"]}
res = {}
print("\n=== CV (seeds 42-46) ===")
for name, fn in cfgs.items():
    a = np.array([accuracy_score(y, quartile_bin(fn(store[s]))) for s in SEEDS])
    res[name] = {"mean": float(a.mean()), "std": float(a.std()), "per_seed": a.round(4).tolist()}
    print(f"  {name:15s} {a.mean():.4f} +/- {a.std():.4f}  {a.round(4)}")

# Refit both on full train, build blended test rank, write candidate.
ord_t, core_t = [], []
for s in SEEDS:
    sc = np.zeros(len(Xte))
    for kthr in range(NC - 1):
        b = ordb(s); b.fit(X, (y > kthr).astype(int)); sc += b.predict_proba(Xte)[:, 1]
    ord_t.append(pd.Series(sc).rank().values)
    r = v3_reg(s); r.fit(X, y); c = v3_clf(s); c.fit(X, y)
    core_t.append(pd.Series(W_CORE*z(r.predict(Xte)) + (1-W_CORE)*z(ev(c.predict_proba(Xte)))).rank().values)
ord_r = rankz(np.mean(ord_t, 0)); core_r = rankz(np.mean(core_t, 0))
blend_test = 0.5*ord_r + 0.5*core_r
preds = quartile_bin(blend_test).astype(int)
sub = pd.DataFrame({"id": te["id"], "target": preds})
assert list(sub.columns) == ["id", "target"] and sub.shape == sample.shape
assert sub["id"].equals(sample["id"]) and set(sub["target"]).issubset({0,1,2,3})
sub.to_csv("outputs/submission_blend_ord_v3.csv", index=False)

Path("outputs/exp_blend_ord_v3.json").write_text(json.dumps({
    "generated_at": datetime.now(timezone.utc).isoformat(), "seeds": list(SEEDS),
    "results": res, "candidate": "outputs/submission_blend_ord_v3.csv",
    "candidate_counts": {str(k): int(v) for k,v in sub['target'].value_counts().sort_index().items()},
    "no_kaggle_submission_made": True}, indent=2))
print("counts:", sub['target'].value_counts().sort_index().to_dict())
print("wrote outputs/submission_blend_ord_v3.csv + exp_blend_ord_v3.json")
