#!/usr/bin/env python3
"""Production v3 — tuned + diverse latent-score blend, quartile-binned.

Combines the two validated accuracy levers on top of run_production_v2 (0.5127):
  1. Optuna-tuned XGB regressor + classifier   (scripts/tune_blend_optuna.py)
  2. Diverse score estimators (HistGB, Ridge)   (scripts/explore_score_estimators.py)

Reads the tuned params from outputs/tune_blend_optuna.json. RUN THAT FIRST:
    uv run python scripts/tune_blend_optuna.py --trials 150
    uv run python scripts/run_production_v3.py

The script evaluates several equal-/weighted blends by multi-seed CV, keeps the
best on ALL seeds (42-46), refits on full train, quartile-bins the test scores,
and writes a validated submission.

Assumption (same as v2): test set is balanced 25/25/25/25 -> quartile binning
forces 200/200/200/200 predictions. No Kaggle submission is made.

Writes: outputs/metrics_production_v3.json, outputs/submission_production_v3.csv
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_playground_loop import make_features  # noqa: E402
from exp_lean_tuned import augment  # noqa: E402

DATA = ROOT / "kaggle"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

N_CLASSES = 4
SEEDS = (42, 43, 44, 45, 46)
TUNE_JSON = OUT / "tune_blend_optuna.json"
BASELINE_V2 = 0.5127
warnings.filterwarnings("ignore")


def quartile_bin(s: np.ndarray) -> np.ndarray:
    r = pd.Series(s).rank(method="first")
    return np.clip((r / (len(s) + 1) * N_CLASSES).astype(int), 0, N_CLASSES - 1).values


def zscore(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / (v.std() + 1e-12)


def ev(p: np.ndarray) -> np.ndarray:
    return (p * np.arange(N_CLASSES)).sum(1)


def load_tuned() -> tuple[dict, float]:
    """Load tuned params if present, else fall back to v2 defaults."""
    if TUNE_JSON.exists():
        d = json.loads(TUNE_JSON.read_text())
        return d["best_params"], float(d.get("blend_w", d["best_params"].get("blend_w", 0.5)))
    print(f"[warn] {TUNE_JSON.name} not found — using v2 default params.")
    default = {
        "reg_n_estimators": 600, "reg_max_depth": 3, "reg_learning_rate": 0.03,
        "reg_subsample": 0.8, "reg_colsample_bytree": 0.7, "reg_min_child_weight": 5,
        "reg_lambda": 2.0, "reg_alpha": 0.0, "reg_gamma": 0.0,
        "clf_n_estimators": 500, "clf_max_depth": 4, "clf_learning_rate": 0.04,
        "clf_subsample": 0.9, "clf_colsample_bytree": 0.9, "clf_min_child_weight": 1,
        "clf_lambda": 1.0, "clf_alpha": 0.0, "clf_gamma": 0.0, "blend_w": 0.5,
    }
    return default, 0.5


def make_reg(p, seed):
    return XGBRegressor(
        n_estimators=p["reg_n_estimators"], max_depth=p["reg_max_depth"],
        learning_rate=p["reg_learning_rate"], subsample=p["reg_subsample"],
        colsample_bytree=p["reg_colsample_bytree"], min_child_weight=p["reg_min_child_weight"],
        reg_lambda=p["reg_lambda"], reg_alpha=p["reg_alpha"], gamma=p["reg_gamma"],
        random_state=seed, n_jobs=-1)


def make_clf(p, seed):
    return XGBClassifier(
        objective="multi:softprob", num_class=N_CLASSES,
        n_estimators=p["clf_n_estimators"], max_depth=p["clf_max_depth"],
        learning_rate=p["clf_learning_rate"], subsample=p["clf_subsample"],
        colsample_bytree=p["clf_colsample_bytree"], min_child_weight=p["clf_min_child_weight"],
        reg_lambda=p["clf_lambda"], reg_alpha=p["clf_alpha"], gamma=p["clf_gamma"],
        random_state=seed, n_jobs=-1, eval_metric="mlogloss", verbosity=0)


def make_hgb_ev(seed):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)


def make_ridge(seed):
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=seed))


def oof_scores(X, y, p, seed) -> dict:
    """OOF z-scored latent scores for every base estimator, one seed."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    out = {k: np.zeros(len(y)) for k in ("xgb_reg", "xgb_ev", "hgb_ev", "ridge")}
    for tr_idx, va_idx in cv.split(X, y):
        Xtr, Xva, ytr = X.iloc[tr_idx], X.iloc[va_idx], y.iloc[tr_idx]
        r = make_reg(p, seed); r.fit(Xtr, ytr); out["xgb_reg"][va_idx] = r.predict(Xva)
        c = make_clf(p, seed); c.fit(Xtr, ytr); out["xgb_ev"][va_idx] = ev(c.predict_proba(Xva))
        h = make_hgb_ev(seed); h.fit(Xtr, ytr); out["hgb_ev"][va_idx] = ev(h.predict_proba(Xva))
        rd = make_ridge(seed); rd.fit(Xtr, ytr); out["ridge"][va_idx] = rd.predict(Xva)
    return {k: zscore(v) for k, v in out.items()}


# Candidate blends. Each maps base z-scores -> a single latent score.
def blend_core(sc, w):       # tuned xgb pair (v2-style, tuned)
    return w * sc["xgb_reg"] + (1 - w) * sc["xgb_ev"]


def blend_diverse4(sc, w):   # tuned pair + hgb + ridge, equal weight
    return np.mean([sc["xgb_reg"], sc["xgb_ev"], sc["hgb_ev"], sc["ridge"]], axis=0)


def blend_explorer(sc, w):   # explorer winner shape: xgb_reg + hgb_ev + ridge
    return np.mean([sc["xgb_reg"], sc["hgb_ev"], sc["ridge"]], axis=0)


CANDIDATES = {
    "core_tuned": blend_core,
    "diverse4": blend_diverse4,
    "explorer3": blend_explorer,
}


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    y = train["target"]
    X = augment(train, make_features(train))
    X_test = augment(test, make_features(test))
    assert list(X.columns) == list(X_test.columns)

    p, w = load_tuned()
    print(f"Feature shape: {X.shape}   blend_w={w:.3f}")
    print(f"Tuned params source: {'tune_blend_optuna.json' if TUNE_JSON.exists() else 'v2 defaults'}\n")

    # Multi-seed CV of every candidate blend.
    per_seed_scores = [oof_scores(X, y, p, s) for s in SEEDS]
    cand_cv = {}
    for name, fn in CANDIDATES.items():
        accs = [accuracy_score(y.values, quartile_bin(fn(sc, w))) for sc in per_seed_scores]
        cand_cv[name] = {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "per_seed": accs}
        print(f"  {name:12s} {cand_cv[name]['mean']:.4f} +/- {cand_cv[name]['std']:.4f}")

    best_name = max(cand_cv, key=lambda k: cand_cv[k]["mean"])
    best_cv = cand_cv[best_name]["mean"]
    print(f"\nBest blend: {best_name}  CV {best_cv:.4f}  (v2 ref {BASELINE_V2}, "
          f"improvement {best_cv - BASELINE_V2:+.4f})")

    # Refit best blend on full train, score test, quartile-bin.
    reg_s, ev_s, hgb_s, rid_s = [], [], [], []
    for s in SEEDS:
        r = make_reg(p, s); r.fit(X, y); reg_s.append(zscore(r.predict(X_test)))
        c = make_clf(p, s); c.fit(X, y); ev_s.append(zscore(ev(c.predict_proba(X_test))))
        h = make_hgb_ev(s); h.fit(X, y); hgb_s.append(zscore(ev(h.predict_proba(X_test))))
        rd = make_ridge(s); rd.fit(X, y); rid_s.append(zscore(rd.predict(X_test)))
    sc_test = {"xgb_reg": np.mean(reg_s, 0), "xgb_ev": np.mean(ev_s, 0),
               "hgb_ev": np.mean(hgb_s, 0), "ridge": np.mean(rid_s, 0)}
    test_score = CANDIDATES[best_name](sc_test, w)
    preds = quartile_bin(test_score).astype(int)

    submission = pd.DataFrame({"id": test["id"], "target": preds})
    if list(submission.columns) != ["id", "target"]:
        raise RuntimeError("Bad submission columns")
    if submission.shape != sample.shape:
        raise RuntimeError(f"Bad submission shape {submission.shape} != {sample.shape}")
    if not submission["id"].equals(sample["id"]):
        raise RuntimeError("Submission IDs do not match sample_submission order")
    if not set(submission["target"].unique()).issubset({0, 1, 2, 3}):
        raise RuntimeError("Submission contains invalid target labels")

    sub_path = OUT / "submission_production_v3.csv"
    submission.to_csv(sub_path, index=False)

    payload = {
        "experiment": "production_v3_tuned_diverse_quartile",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seeds": list(SEEDS),
        "tuned_params_source": "tune_blend_optuna.json" if TUNE_JSON.exists() else "v2_defaults",
        "blend_w": w,
        "candidate_cv": cand_cv,
        "best_blend": best_name,
        "best_cv_mean_accuracy": best_cv,
        "baseline_v2_ref": BASELINE_V2,
        "improvement_over_v2": best_cv - BASELINE_V2,
        "assumption": "Test set balanced 25/25/25/25 (quartile binning).",
        "submission_path": str(sub_path),
        "submission_prediction_counts": {
            str(k): int(v) for k, v in submission["target"].value_counts().sort_index().items()},
        "no_kaggle_submission_made": True,
    }
    (OUT / "metrics_production_v3.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {sub_path}")
    print(f"Prediction counts: {payload['submission_prediction_counts']}")


if __name__ == "__main__":
    main()
