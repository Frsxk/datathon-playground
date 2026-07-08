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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_playground_loop import make_features, WEEK_COLS  # noqa: E402
from exp_lean_tuned import augment, xgb_base, xgb_reg  # noqa: E402

DATA = ROOT / "kaggle"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

TRAIN_PATH = DATA / "train.csv"
TEST_PATH = DATA / "test.csv"
SAMPLE_PATH = DATA / "sample_submission.csv"

SEEDS = (42, 43, 44, 45, 46)
FACTORIES = [xgb_base, xgb_reg]  # softprob is forced below
N_CLASSES = 4
ROBUST_BASELINE = 0.4939

warnings.filterwarnings("ignore")


def make_softprob(factory, seed: int):
    m = factory(seed)
    m.set_params(objective="multi:softprob")
    return m


def cv_eval(X: pd.DataFrame, y: pd.Series) -> dict:
    """Per-seed 5-fold CV of the seed-bagged ensemble."""
    seed_accs = []
    for s in SEEDS:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        oof = np.zeros((len(y), N_CLASSES))
        for tr_idx, va_idx in cv.split(X, y):
            for f in FACTORIES:
                m = make_softprob(f, s)
                m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                oof[va_idx] += m.predict_proba(X.iloc[va_idx])
        seed_accs.append(float(accuracy_score(y, oof.argmax(1))))
    return {
        "per_seed": seed_accs,
        "mean_accuracy": float(np.mean(seed_accs)),
        "std_across_seeds": float(np.std(seed_accs)),
    }


def fit_predict_full(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    """Fit every (model, seed) on full train; average test soft probabilities."""
    test_p = np.zeros((len(X_test), N_CLASSES))
    n = 0
    for s in SEEDS:
        for f in FACTORIES:
            m = make_softprob(f, s)
            m.fit(X, y)
            test_p += m.predict_proba(X_test)
            n += 1
    return (test_p / n).argmax(1).astype(int)


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_PATH)

    y = train["target"]
    X = augment(train, make_features(train))
    X_test = augment(test, make_features(test))
    assert list(X.columns) == list(X_test.columns), "train/test feature mismatch"

    print(f"Feature shape: {X.shape}  (baseline FE was 95 cols)")
    cv = cv_eval(X, y)
    print(f"Seed-bagged ensemble CV: {cv['mean_accuracy']:.4f} +/- {cv['std_across_seeds']:.4f}")
    print(f"Robust baseline:         {ROBUST_BASELINE:.4f}")
    print(f"Improvement:             {cv['mean_accuracy'] - ROBUST_BASELINE:+.4f}")

    preds = fit_predict_full(X, y, X_test)
    submission = pd.DataFrame({"id": test["id"], "target": preds})

    # Validate exactly like the existing scripts.
    if list(submission.columns) != ["id", "target"]:
        raise RuntimeError("Bad submission columns")
    if submission.shape != sample.shape:
        raise RuntimeError(f"Bad submission shape {submission.shape} != {sample.shape}")
    if not submission["id"].equals(sample["id"]):
        raise RuntimeError("Submission IDs do not match sample_submission order")
    if not set(submission["target"].unique()).issubset({0, 1, 2, 3}):
        raise RuntimeError("Submission contains invalid target labels")

    sub_path = OUT / "submission_production.csv"
    submission.to_csv(sub_path, index=False)

    payload = {
        "experiment": "production_seed_bagged_xgb",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Augmented trajectory FE + soft-prob bag of {xgb_base, xgb_reg} over "
            "5 seeds. Argmax of averaged probabilities."
        ),
        "seeds": list(SEEDS),
        "models": ["xgb_base", "xgb_reg"],
        "feature_shape": list(X.shape),
        "cv_per_seed": cv["per_seed"],
        "cv_mean_accuracy": cv["mean_accuracy"],
        "cv_std_across_seeds": cv["std_across_seeds"],
        "robust_baseline_ref": ROBUST_BASELINE,
        "improvement_over_baseline": cv["mean_accuracy"] - ROBUST_BASELINE,
        "submission_path": str(sub_path),
        "submission_prediction_counts": {
            str(k): int(v) for k, v in submission["target"].value_counts().sort_index().items()
        },
        "no_kaggle_submission_made": True,
    }
    (OUT / "metrics_production.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {sub_path}")
    print(f"Prediction counts: {payload['submission_prediction_counts']}")


if __name__ == "__main__":
    main()
