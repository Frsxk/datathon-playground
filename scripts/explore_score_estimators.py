#!/usr/bin/env python3
"""Explore diverse latent-score estimators — Datathon 2026 Playground.

The target is the quartile-rank of a hidden performance score (see
run_production_v2.py). Better we estimate that score's RANK, the better the
quartile-binned accuracy. This script tests whether adding DIVERSE, independent
score estimators to the blend improves the rank beyond the current
XGB-reg + XGB-clf-EV pair (baseline CV 0.5127 over seeds 42-46).

Each estimator emits a continuous latent score (z-normalized); a blend is the
mean of a chosen subset's z-scores, then quartile_bin -> 0..3.

We use EQUAL-WEIGHT blends only (no weight tuning) so nothing here can overfit
weights to the CV. Every estimator is fit inside each fold (CV-safe).

Estimators:
    xgb_reg      XGBRegressor  -> raw prediction
    xgb_ev       XGBClassifier -> E[class] = Σ c·p(c)
    hgb_reg      HistGradientBoostingRegressor -> raw prediction
    hgb_ev       HistGradientBoostingClassifier -> E[class]
    ridge        Ridge on standardized features -> raw prediction
    logit_ev     LogisticRegression -> E[class]

Run:  uv run python scripts/explore_score_estimators.py
Writes: outputs/explore_score_estimators.json
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_playground_loop import make_features  # noqa: E402
from exp_lean_tuned import augment, xgb_base  # noqa: E402

DATA = ROOT / "kaggle"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

N_CLASSES = 4
SEEDS = (42, 43, 44, 45, 46)
BASELINE_V2 = 0.5127
warnings.filterwarnings("ignore")


def quartile_bin(scores: np.ndarray) -> np.ndarray:
    r = pd.Series(scores).rank(method="first")
    return np.clip((r / (len(scores) + 1) * N_CLASSES).astype(int), 0, N_CLASSES - 1).values


def zscore(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / (v.std() + 1e-12)


def ev(proba: np.ndarray) -> np.ndarray:
    return (proba * np.arange(N_CLASSES)).sum(1)


def estimator_factories(seed: int) -> dict:
    """Return {name: (model, kind)} where kind in {'reg','clf'}."""
    return {
        "xgb_reg": (XGBRegressor(
            n_estimators=600, max_depth=3, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0,
            random_state=seed, n_jobs=-1), "reg"),
        "xgb_ev": (XGBClassifier(
            objective="multi:softprob", num_class=4, n_estimators=500, max_depth=4,
            learning_rate=0.04, subsample=0.9, colsample_bytree=0.9,
            random_state=seed, n_jobs=-1, eval_metric="mlogloss", verbosity=0), "clf"),
        "hgb_reg": (HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.04, max_depth=3, l2_regularization=1.0,
            random_state=seed), "reg"),
        "hgb_ev": (HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=seed), "clf"),
        "ridge": (make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=seed)), "reg"),
        "logit_ev": (make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, C=0.5, random_state=seed)), "clf"),
    }


def all_oof_scores(X, y, seed: int) -> dict:
    """OOF z-scored latent score for every estimator, one seed."""
    names = list(estimator_factories(seed).keys())
    oof = {n: np.zeros(len(y)) for n in names}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, va_idx in cv.split(X, y):
        facts = estimator_factories(seed)
        for n, (model, kind) in facts.items():
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            if kind == "reg":
                oof[n][va_idx] = model.predict(X.iloc[va_idx])
            else:
                oof[n][va_idx] = ev(model.predict_proba(X.iloc[va_idx]))
    return {n: zscore(v) for n, v in oof.items()}


def blend_acc(scores_per_seed: list[dict], subset: tuple, y) -> tuple[float, float]:
    accs = []
    for sc in scores_per_seed:
        blended = np.mean([sc[n] for n in subset], axis=0)
        accs.append(accuracy_score(y.values, quartile_bin(blended)))
    return float(np.mean(accs)), float(np.std(accs))


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    y = train["target"]
    X = augment(train, make_features(train))
    print(f"Feature shape: {X.shape}")
    print("Computing OOF scores for all estimators across seeds (this is the slow part)...")

    scores_per_seed = []
    for s in SEEDS:
        print(f"  seed {s} ...", flush=True)
        scores_per_seed.append(all_oof_scores(X, y, s))
    names = list(scores_per_seed[0].keys())

    # 1. single-estimator quartile accuracy
    print("\n--- single estimators (quartile-binned) ---")
    single = {}
    for n in names:
        m, sd = blend_acc(scores_per_seed, (n,), y)
        single[n] = m
        print(f"  {n:10s} {m:.4f} +/- {sd:.4f}")

    # 2. the current v2 pair as reference
    m2, sd2 = blend_acc(scores_per_seed, ("xgb_reg", "xgb_ev"), y)
    print(f"\ncurrent v2 pair xgb_reg+xgb_ev: {m2:.4f} +/- {sd2:.4f}  (ref {BASELINE_V2})")

    # 3. all subsets of size 2..N, report the best few
    print("\n--- searching equal-weight blends (size 2..6) ---")
    results = []
    for k in range(2, len(names) + 1):
        for sub in combinations(names, k):
            m, sd = blend_acc(scores_per_seed, sub, y)
            results.append({"subset": list(sub), "mean_accuracy": m, "std": sd})
    results.sort(key=lambda r: r["mean_accuracy"], reverse=True)

    print("Top 12 blends:")
    for r in results[:12]:
        print(f"  {r['mean_accuracy']:.4f} +/- {r['std']:.4f}  {'+'.join(r['subset'])}")

    best = results[0]
    print("\n" + "=" * 60)
    print(f"BEST BLEND: {'+'.join(best['subset'])}")
    print(f"  CV {best['mean_accuracy']:.4f} +/- {best['std']:.4f}  vs v2 pair {m2:.4f} / ref {BASELINE_V2}")
    print("=" * 60)

    payload = {
        "experiment": "explore_score_estimators",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seeds": list(SEEDS),
        "baseline_v2_ref": BASELINE_V2,
        "single_estimator_accuracy": single,
        "current_v2_pair_accuracy": m2,
        "best_blend": best,
        "top_blends": results[:20],
        "no_kaggle_submission_made": True,
    }
    (OUT / "explore_score_estimators.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT / 'explore_score_estimators.json'}")


if __name__ == "__main__":
    main()
