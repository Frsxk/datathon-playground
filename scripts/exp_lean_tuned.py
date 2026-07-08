#!/usr/bin/env python3
"""Rigorous multi-seed experiment: augmented trajectory features + tuned XGBoost.

Motivation (from fresh EDA):
- Only two feature families carry real signal vs `target`:
    * assignment completion ratio (tugas_selesai / tugas_diberikan)   ~0.41
    * weekly grade-change volatility (minggu_std / abs_change_sum)     ~0.40
  Both are cleanly monotonic across the 4 ordered classes.
- The 16 daily-activity columns are ~noise (|corr| < 0.05).
- Cumulative grade *trajectory* (running sum of weekly changes) was never
  engineered; cum_range / cum_std reach ~0.36 corr.

Baseline is re-estimated across 5 seeds to defeat single-seed CV noise
(the previously reported 0.502 was a lucky seed; robust baseline ~0.4939).

This script ONLY evaluates CV; it does not write a submission. A separate,
approved step promotes the winning config to a submission.

Creates: outputs/exp_lean_tuned.json
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_playground_loop import make_features, WEEK_COLS  # noqa: E402

DATA = ROOT / "kaggle"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

SEEDS = (42, 43, 44, 45, 46)
warnings.filterwarnings("ignore")


def augment(df: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative-trajectory features and strong-signal interactions."""
    W = df[WEEK_COLS].to_numpy(float)
    cum = np.cumsum(W, axis=1)
    out = base.copy()
    out["cum_range"] = cum.max(1) - cum.min(1)
    out["cum_std"] = cum.std(1)
    out["cum_max"] = cum.max(1)
    out["cum_min"] = cum.min(1)
    out["abs_change_sum"] = np.abs(W).sum(1)
    cr = (df["tugas_selesai"] / df["tugas_diberikan"].replace(0, np.nan)).fillna(0)
    out["vol_x_compl"] = base["minggu_std"] * cr
    out["tryout_x_compl"] = df["skor_tryout"] * cr
    out["range_x_compl"] = base["minggu_range"] * cr
    return out


def xgb_base(s: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softmax", num_class=4, n_estimators=500, max_depth=4,
        learning_rate=0.04, subsample=0.9, colsample_bytree=0.9,
        random_state=s, n_jobs=-1, eval_metric="mlogloss", verbosity=0,
    )


def xgb_reg(s: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softmax", num_class=4, n_estimators=700, max_depth=3,
        learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
        min_child_weight=5, reg_lambda=2.0, reg_alpha=0.5, gamma=0.1,
        random_state=s, n_jobs=-1, eval_metric="mlogloss", verbosity=0,
    )


def multiseed(X: pd.DataFrame, y: pd.Series, model_fn, seeds=SEEDS) -> dict:
    per_seed = []
    for s in seeds:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        sc = cross_val_score(model_fn(s), X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        per_seed.append(float(sc.mean()))
    return {
        "per_seed": per_seed,
        "mean_accuracy": float(np.mean(per_seed)),
        "std_across_seeds": float(np.std(per_seed)),
    }


def main() -> None:
    tr = pd.read_csv(DATA / "train.csv")
    y = tr["target"]
    X = make_features(tr)
    Xaug = augment(tr, X)

    configs = {
        "base_fe__base_xgb": (X, xgb_base),
        "aug_fe__base_xgb": (Xaug, xgb_base),
        "base_fe__reg_xgb": (X, xgb_reg),
        "aug_fe__reg_xgb": (Xaug, xgb_reg),
    }
    results = {}
    for name, (Xc, fn) in configs.items():
        r = multiseed(Xc, y, fn)
        results[name] = r
        print(f"{name:22s} {r['mean_accuracy']:.4f} +/- {r['std_across_seeds']:.4f}  "
              f"{[round(v,4) for v in r['per_seed']]}")

    best_name = max(results, key=lambda k: results[k]["mean_accuracy"])
    payload = {
        "experiment": "lean_tuned",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seeds": list(SEEDS),
        "robust_baseline_ref": 0.4939,
        "results": results,
        "best_config": best_name,
        "best_mean_accuracy": results[best_name]["mean_accuracy"],
        "feature_shape_base": list(X.shape),
        "feature_shape_aug": list(Xaug.shape),
        "no_kaggle_submission_made": True,
    }
    (OUT / "exp_lean_tuned.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nBest:", best_name, round(results[best_name]["mean_accuracy"], 4))


if __name__ == "__main__":
    main()
