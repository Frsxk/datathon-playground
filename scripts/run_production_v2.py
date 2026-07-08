#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_playground_loop import make_features  # noqa: E402
from exp_lean_tuned import augment, xgb_base, xgb_reg  # noqa: E402

DATA = ROOT / "kaggle"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

SEEDS = (42, 43, 44, 45, 46)
N_CLASSES = 4
ROBUST_BASELINE = 0.4939
warnings.filterwarnings("ignore")


def xgb_regressor(seed: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=600, max_depth=3, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0,
        random_state=seed, n_jobs=-1,
    )


def clf_softprob(factory, seed: int):
    m = factory(seed)
    m.set_params(objective="multi:softprob")
    return m


def quartile_bin(scores: np.ndarray) -> np.ndarray:
    """Map continuous scores to 0..3 by rank quartiles (balanced design)."""
    r = pd.Series(scores).rank(method="first")
    return np.clip((r / (len(scores) + 1) * N_CLASSES).astype(int), 0, N_CLASSES - 1).values


def zscore(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / (v.std() + 1e-12)


def blended_score_oof(X: pd.DataFrame, y: pd.Series, seed: int) -> np.ndarray:
    """One-seed OOF blended latent score."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    reg_oof = np.zeros(len(y))
    ev_oof = np.zeros(len(y))
    for tr_idx, va_idx in cv.split(X, y):
        r = xgb_regressor(seed)
        r.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        reg_oof[va_idx] = r.predict(X.iloc[va_idx])
        ev = np.zeros(len(va_idx))
        for f in (xgb_base, xgb_reg):
            m = clf_softprob(f, seed)
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            ev += (m.predict_proba(X.iloc[va_idx]) * np.arange(N_CLASSES)).sum(1)
        ev_oof[va_idx] = ev / 2
    return zscore(reg_oof) + zscore(ev_oof)


def cv_eval(X: pd.DataFrame, y: pd.Series) -> dict:
    accs = []
    for s in SEEDS:
        score = blended_score_oof(X, y, s)
        accs.append(float(accuracy_score(y.values, quartile_bin(score))))
    return {"per_seed": accs, "mean_accuracy": float(np.mean(accs)),
            "std_across_seeds": float(np.std(accs))}


def full_train_test_score(X, y, X_test) -> np.ndarray:
    """Fit all (model, seed) on full train; average z-normalized test scores."""
    reg_scores = []
    ev_scores = []
    for s in SEEDS:
        r = xgb_regressor(s)
        r.fit(X, y)
        reg_scores.append(zscore(r.predict(X_test)))
        ev = np.zeros(len(X_test))
        for f in (xgb_base, xgb_reg):
            m = clf_softprob(f, s)
            m.fit(X, y)
            ev += (m.predict_proba(X_test) * np.arange(N_CLASSES)).sum(1)
        ev_scores.append(zscore(ev / 2))
    return np.mean(reg_scores, 0) + np.mean(ev_scores, 0)


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")

    y = train["target"]
    X = augment(train, make_features(train))
    X_test = augment(test, make_features(test))
    assert list(X.columns) == list(X_test.columns), "train/test feature mismatch"

    print(f"Feature shape: {X.shape}")
    cv = cv_eval(X, y)
    print(f"Blend+quartile CV: {cv['mean_accuracy']:.4f} +/- {cv['std_across_seeds']:.4f}")
    print(f"Robust baseline:   {ROBUST_BASELINE:.4f}  (improvement {cv['mean_accuracy']-ROBUST_BASELINE:+.4f})")

    test_score = full_train_test_score(X, y, X_test)
    preds = quartile_bin(test_score)
    submission = pd.DataFrame({"id": test["id"], "target": preds.astype(int)})

    if list(submission.columns) != ["id", "target"]:
        raise RuntimeError("Bad submission columns")
    if submission.shape != sample.shape:
        raise RuntimeError(f"Bad submission shape {submission.shape} != {sample.shape}")
    if not submission["id"].equals(sample["id"]):
        raise RuntimeError("Submission IDs do not match sample_submission order")
    if not set(submission["target"].unique()).issubset({0, 1, 2, 3}):
        raise RuntimeError("Submission contains invalid target labels")

    sub_path = OUT / "submission_production_v2.csv"
    submission.to_csv(sub_path, index=False)

    payload = {
        "experiment": "production_v2_latent_score_quartile",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Reconstruct latent performance score via blended XGB-regression + "
            "XGB-classifier expected value; quartile-bin to 0..3. target is the "
            "quartile split of a hidden continuous score."
        ),
        "seeds": list(SEEDS),
        "feature_shape": list(X.shape),
        "cv_per_seed": cv["per_seed"],
        "cv_mean_accuracy": cv["mean_accuracy"],
        "cv_std_across_seeds": cv["std_across_seeds"],
        "robust_baseline_ref": ROBUST_BASELINE,
        "improvement_over_baseline": cv["mean_accuracy"] - ROBUST_BASELINE,
        "assumption": "Test set has balanced 25/25/25/25 class design (quartile binning).",
        "submission_path": str(sub_path),
        "submission_prediction_counts": {
            str(k): int(v) for k, v in submission["target"].value_counts().sort_index().items()
        },
        "no_kaggle_submission_made": True,
    }
    (OUT / "metrics_production_v2.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {sub_path}")
    print(f"Prediction counts: {payload['submission_prediction_counts']}")


if __name__ == "__main__":
    main()
