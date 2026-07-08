#!/usr/bin/env python3
"""Optuna tuning of the latent-score blend — Datathon 2026 Playground.

Tunes, jointly:
  - an XGBRegressor  (predicts target 0..3 as a continuous latent score)
  - an XGBClassifier (its expected value E[class]=Σ c·p(c) is a second score)
  - the blend weight w:  score = w·z(reg) + (1-w)·z(clf_ev)
Final label = quartile_bin(score).  This is the winning framing from
run_production_v2.py (baseline to beat: CV 0.5127 ± 0.0056 over seeds 42-46).

ANTI-OVERFIT PROTOCOL (important):
  - Optuna optimizes the MEAN quartile-binned accuracy on TUNE_SEEDS only.
  - The best trial is then re-scored on held-out VAL_SEEDS it never saw.
  - We report BOTH. Only trust the tuned params if the val gain holds up;
    a big tune-vs-val gap means Optuna overfit the tuning seeds.

Everything is CV-safe: reg/clf are fit inside each fold; quartile_bin is applied
to OOF scores only. No Kaggle submission is made.

Run:
    uv run python scripts/tune_blend_optuna.py --trials 150
    # smaller first if you want a quick smoke test:
    uv run python scripts/tune_blend_optuna.py --trials 20

Writes: outputs/tune_blend_optuna.json  (best params + tune/val accuracies)
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor, XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_playground_loop import make_features  # noqa: E402
from exp_lean_tuned import augment  # noqa: E402

DATA = ROOT / "kaggle"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

N_CLASSES = 4
TUNE_SEEDS = (42, 43, 44)
VAL_SEEDS = (45, 46)
ALL_SEEDS = (42, 43, 44, 45, 46)
BASELINE_V2 = 0.5127
warnings.filterwarnings("ignore")

# Loaded once in main(), referenced by the objective.
_X: pd.DataFrame | None = None
_y: pd.Series | None = None


def quartile_bin(scores: np.ndarray) -> np.ndarray:
    r = pd.Series(scores).rank(method="first")
    return np.clip((r / (len(scores) + 1) * N_CLASSES).astype(int), 0, N_CLASSES - 1).values


def zscore(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / (v.std() + 1e-12)


def make_reg(p: dict, seed: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=p["reg_n_estimators"], max_depth=p["reg_max_depth"],
        learning_rate=p["reg_learning_rate"], subsample=p["reg_subsample"],
        colsample_bytree=p["reg_colsample_bytree"], min_child_weight=p["reg_min_child_weight"],
        reg_lambda=p["reg_lambda"], reg_alpha=p["reg_alpha"], gamma=p["reg_gamma"],
        random_state=seed, n_jobs=-1,
    )


def make_clf(p: dict, seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob", num_class=N_CLASSES,
        n_estimators=p["clf_n_estimators"], max_depth=p["clf_max_depth"],
        learning_rate=p["clf_learning_rate"], subsample=p["clf_subsample"],
        colsample_bytree=p["clf_colsample_bytree"], min_child_weight=p["clf_min_child_weight"],
        reg_lambda=p["clf_lambda"], reg_alpha=p["clf_alpha"], gamma=p["clf_gamma"],
        random_state=seed, n_jobs=-1, eval_metric="mlogloss", verbosity=0,
    )


def blended_oof_score(X, y, p: dict, seed: int) -> np.ndarray:
    """OOF blended latent score for one seed under params p."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    reg_oof = np.zeros(len(y))
    ev_oof = np.zeros(len(y))
    for tr_idx, va_idx in cv.split(X, y):
        r = make_reg(p, seed)
        r.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        reg_oof[va_idx] = r.predict(X.iloc[va_idx])
        c = make_clf(p, seed)
        c.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        ev_oof[va_idx] = (c.predict_proba(X.iloc[va_idx]) * np.arange(N_CLASSES)).sum(1)
    w = p["blend_w"]
    return w * zscore(reg_oof) + (1.0 - w) * zscore(ev_oof)


def mean_acc(X, y, p: dict, seeds) -> float:
    accs = [accuracy_score(y.values, quartile_bin(blended_oof_score(X, y, p, s))) for s in seeds]
    return float(np.mean(accs))


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        # regressor
        "reg_n_estimators": trial.suggest_int("reg_n_estimators", 300, 1200, step=100),
        "reg_max_depth": trial.suggest_int("reg_max_depth", 2, 5),
        "reg_learning_rate": trial.suggest_float("reg_learning_rate", 0.01, 0.08, log=True),
        "reg_subsample": trial.suggest_float("reg_subsample", 0.6, 1.0),
        "reg_colsample_bytree": trial.suggest_float("reg_colsample_bytree", 0.5, 1.0),
        "reg_min_child_weight": trial.suggest_int("reg_min_child_weight", 1, 12),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 6.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
        "reg_gamma": trial.suggest_float("reg_gamma", 0.0, 0.6),
        # classifier
        "clf_n_estimators": trial.suggest_int("clf_n_estimators", 300, 1000, step=100),
        "clf_max_depth": trial.suggest_int("clf_max_depth", 2, 5),
        "clf_learning_rate": trial.suggest_float("clf_learning_rate", 0.01, 0.08, log=True),
        "clf_subsample": trial.suggest_float("clf_subsample", 0.6, 1.0),
        "clf_colsample_bytree": trial.suggest_float("clf_colsample_bytree", 0.5, 1.0),
        "clf_min_child_weight": trial.suggest_int("clf_min_child_weight", 1, 12),
        "clf_lambda": trial.suggest_float("clf_lambda", 0.0, 6.0),
        "clf_alpha": trial.suggest_float("clf_alpha", 0.0, 3.0),
        "clf_gamma": trial.suggest_float("clf_gamma", 0.0, 0.6),
        # blend
        "blend_w": trial.suggest_float("blend_w", 0.2, 0.8),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--sampler-seed", type=int, default=2026)
    args = ap.parse_args()

    global _X, _y
    train = pd.read_csv(DATA / "train.csv")
    _y = train["target"]
    _X = augment(train, make_features(train))
    print(f"Feature shape: {_X.shape}")
    print(f"Tuning on seeds {TUNE_SEEDS}; validating on {VAL_SEEDS}.")
    print(f"Baseline to beat (run_production_v2, seeds 42-46): {BASELINE_V2}\n")

    def objective(trial: optuna.Trial) -> float:
        p = suggest_params(trial)
        return mean_acc(_X, _y, p, TUNE_SEEDS)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
    )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = study.best_params
    tune_acc = study.best_value
    val_acc = mean_acc(_X, _y, best, VAL_SEEDS)
    all_acc = mean_acc(_X, _y, best, ALL_SEEDS)

    print("\n" + "=" * 60)
    print("BEST PARAMS:")
    print(json.dumps(best, indent=2))
    print("=" * 60)
    print(f"Tune-seed acc {TUNE_SEEDS}: {tune_acc:.4f}")
    print(f"Val-seed  acc {VAL_SEEDS}: {val_acc:.4f}   <-- must hold up vs {BASELINE_V2}")
    print(f"All-seed  acc {ALL_SEEDS}: {all_acc:.4f}")
    print(f"Tune-vs-val gap: {tune_acc - val_acc:+.4f}  (large gap => overfit tuning seeds)")

    payload = {
        "experiment": "tune_blend_optuna",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_trials": args.trials,
        "sampler_seed": args.sampler_seed,
        "tune_seeds": list(TUNE_SEEDS),
        "val_seeds": list(VAL_SEEDS),
        "baseline_v2_ref": BASELINE_V2,
        "best_params": best,
        "tune_seed_accuracy": tune_acc,
        "val_seed_accuracy": val_acc,
        "all_seed_accuracy": all_acc,
        "tune_minus_val_gap": tune_acc - val_acc,
        "no_kaggle_submission_made": True,
    }
    (OUT / "tune_blend_optuna.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT / 'tune_blend_optuna.json'}")


if __name__ == "__main__":
    main()
