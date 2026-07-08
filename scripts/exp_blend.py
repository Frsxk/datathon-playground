#!/usr/bin/env python3
"""Bounded ensemble/blend experiment for Datathon 2026 Playground.

Trains 3 existing-style models on engineered features (reusing make_features
from run_playground_loop), evaluates out-of-fold soft probability averaging
under 5-fold StratifiedKFold (random_state=44), and compares against the
baseline CV accuracy of 0.49031.

If the blend beats baseline, writes outputs/submission_blend.csv (validated
for shape/id/order) and always writes outputs/exp_blend.json with full results.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "kaggle"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

TRAIN_PATH = DATA / "train.csv"
TEST_PATH = DATA / "test.csv"
SAMPLE_PATH = DATA / "sample_submission.csv"

BASELINE_CV = 0.49031
RANDOM_STATE = 44
N_SPLITS = 5

# ── Import make_features from the existing loop script ──────────────────────
sys.path.insert(0, str(ROOT / "scripts"))
from run_playground_loop import make_features  # noqa: E402


def build_models() -> list[tuple[str, object]]:
    """Three existing-style models with modest configs."""
    models: list[tuple[str, object]] = [
        (
            "logreg_fe",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, C=1.0, random_state=RANDOM_STATE),
            ),
        ),
        (
            "histgb_fe",
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_depth=None,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
    if XGBClassifier is not None:
        models.append(
            (
                "xgboost_fe",
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=4,
                    n_estimators=500,
                    max_depth=4,
                    learning_rate=0.04,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=RANDOM_STATE,
                    n_jobs=2,
                    eval_metric="mlogloss",
                    verbosity=0,
                ),
            )
        )
    return models


def get_predict_proba(model, X) -> np.ndarray:
    """Return probability matrix; fallback to one-hot of predict."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    preds = model.predict(X)
    n_classes = 4
    probs = np.zeros((len(preds), n_classes))
    probs[np.arange(len(preds)), preds.astype(int)] = 1.0
    return probs


def run_oof_blend(
    models: list[tuple[str, object]],
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    cv: StratifiedKFold,
) -> dict:
    """Run OOF blend with soft probability averaging.

    Returns per-model OOF scores, blend OOF score, and test predictions.
    """
    n_classes = 4
    n_train = len(X)
    n_test = len(X_test)

    # Storage for OOF probabilities and test probabilities per model
    oof_probs: dict[str, np.ndarray] = {}
    test_probs: dict[str, np.ndarray] = {}

    per_model_results: list[dict] = []

    for name, model in models:
        oof_p = np.zeros((n_train, n_classes))
        test_p = np.zeros((n_test, n_classes))
        fold_accs = []

        for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            # Clone the model for each fold to avoid warm-start leakage
            from sklearn.base import clone

            m = clone(model)
            m.fit(X_tr, y_tr)

            va_probs = get_predict_proba(m, X_va)
            oof_p[va_idx] = va_probs

            fold_pred = va_probs.argmax(axis=1)
            fold_accs.append(float(accuracy_score(y_va, fold_pred)))

            # Accumulate test predictions (average across folds)
            test_p += get_predict_proba(m, X_test) / cv.get_n_splits()

        oof_pred = oof_p.argmax(axis=1)
        oof_acc = float(accuracy_score(y, oof_pred))

        oof_probs[name] = oof_p
        test_probs[name] = test_p

        per_model_results.append(
            {
                "name": name,
                "fold_scores": fold_accs,
                "mean_accuracy": float(np.mean(fold_accs)),
                "std_accuracy": float(np.std(fold_accs)),
                "oof_accuracy": oof_acc,
            }
        )
        print(
            f"  {name}: CV mean={np.mean(fold_accs):.5f}  OOF acc={oof_acc:.5f}"
        )

    # ── Soft vote blend: average OOF probabilities ───────────────────────────
    blend_oof = np.mean(list(oof_probs.values()), axis=0)
    blend_oof_pred = blend_oof.argmax(axis=1)
    blend_oof_acc = float(accuracy_score(y, blend_oof_pred))

    blend_test = np.mean(list(test_probs.values()), axis=0)
    blend_test_pred = blend_test.argmax(axis=1).astype(int)

    # ── Also evaluate weighted blend (weight by OOF accuracy) ────────────────
    weights = np.array([r["oof_accuracy"] for r in per_model_results])
    weights = weights / weights.sum()

    wblend_oof = np.zeros_like(blend_oof)
    wblend_test = np.zeros_like(blend_test)
    for w, (name, _) in zip(weights, models):
        wblend_oof += w * oof_probs[name]
        wblend_test += w * test_probs[name]

    wblend_oof_pred = wblend_oof.argmax(axis=1)
    wblend_oof_acc = float(accuracy_score(y, wblend_oof_pred))
    wblend_test_pred = wblend_test.argmax(axis=1).astype(int)

    print(f"  blend_soft (equal):   OOF acc={blend_oof_acc:.5f}")
    print(f"  blend_soft (weighted): OOF acc={wblend_oof_acc:.5f}")

    # Pick the better blend for test predictions
    if wblend_oof_acc >= blend_oof_acc:
        best_blend_name = "soft_weighted"
        best_blend_acc = wblend_oof_acc
        best_test_pred = wblend_test_pred
    else:
        best_blend_name = "soft_equal"
        best_blend_acc = blend_oof_acc
        best_test_pred = blend_test_pred

    return {
        "per_model": per_model_results,
        "blend_equal": {
            "oof_accuracy": blend_oof_acc,
            "weights": {name: 1.0 / len(models) for name, _ in models},
        },
        "blend_weighted": {
            "oof_accuracy": wblend_oof_acc,
            "weights": {name: float(w) for (name, _), w in zip(models, weights)},
        },
        "best_blend_name": best_blend_name,
        "best_blend_oof_accuracy": best_blend_acc,
        "test_predictions": best_test_pred,
    }


def main() -> None:
    print("=" * 70)
    print("Datathon 2026 Playground - Blend Experiment")
    print(f"Baseline CV to beat: {BASELINE_CV}")
    print(f"CV: StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────────
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_PATH)

    y = train["target"]
    X_fe = make_features(train)
    X_test = make_features(test)

    print(f"Train: {train.shape}  Test: {test.shape}  Features: {X_fe.shape[1]}")
    print(f"Target counts: {y.value_counts().sort_index().to_dict()}")
    print()

    # ── Build models ─────────────────────────────────────────────────────────
    models = build_models()
    print(f"Models ({len(models)}): {[name for name, _ in models]}")
    print()

    # ── Run OOF blend ────────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    print("Running 5-fold OOF blend...")
    results = run_oof_blend(models, X_fe, y, X_test, cv)
    print()

    # ── Determine if blend beats baseline ────────────────────────────────────
    best_blend_acc = results["best_blend_oof_accuracy"]
    beats_baseline = best_blend_acc > BASELINE_CV

    print(f"Best blend: {results['best_blend_name']}  OOF accuracy={best_blend_acc:.5f}")
    print(f"Baseline CV: {BASELINE_CV}")
    print(f"Beats baseline: {beats_baseline}")
    print()

    # ── Write submission if blend beats baseline ─────────────────────────────
    submission_path = None
    submission_validated = False
    submission_pred_counts = None

    if beats_baseline:
        test_pred = results["test_predictions"]
        submission = pd.DataFrame({"id": test["id"], "target": test_pred})

        # Validate shape, columns, IDs, labels
        assert list(submission.columns) == ["id", "target"], "Bad submission columns"
        assert submission.shape == sample.shape, (
            f"Bad submission shape {submission.shape} != {sample.shape}"
        )
        assert submission["id"].equals(sample["id"]), (
            "Submission IDs do not match sample_submission order"
        )
        assert set(submission["target"].unique()).issubset({0, 1, 2, 3}), (
            "Submission contains invalid target labels"
        )

        submission_path = OUT / "submission_blend.csv"
        submission.to_csv(submission_path, index=False)
        submission_validated = True
        submission_pred_counts = {
            str(k): int(v)
            for k, v in submission["target"].value_counts().sort_index().items()
        }
        print(f"✓ Wrote validated submission: {submission_path}")
    else:
        print("✗ Blend did not beat baseline — no submission written.")

    # ── Write experiment JSON ────────────────────────────────────────────────
    exp_output = {
        "experiment": "blend_exp",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": "OOF soft probability averaging blend of logreg/histgb/xgboost on engineered features",
        "baseline_cv": BASELINE_CV,
        "cv": f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})",
        "feature_shape": list(X_fe.shape),
        "models": [name for name, _ in models],
        "per_model_results": results["per_model"],
        "blend_equal": results["blend_equal"],
        "blend_weighted": results["blend_weighted"],
        "best_blend_name": results["best_blend_name"],
        "best_blend_oof_accuracy": best_blend_acc,
        "beats_baseline": beats_baseline,
        "submission_path": str(submission_path) if submission_path else None,
        "submission_validated": submission_validated,
        "submission_prediction_counts": submission_pred_counts,
        "no_kaggle_submission_made": True,
    }

    exp_path = OUT / "exp_blend.json"
    exp_path.write_text(json.dumps(exp_output, indent=2), encoding="utf-8")
    print(f"\n✓ Wrote experiment JSON: {exp_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Per-model OOF accuracies:")
    for m in results["per_model"]:
        print(f"    {m['name']}: {m['oof_accuracy']:.5f}")
    print(f"  Blend (equal):    {results['blend_equal']['oof_accuracy']:.5f}")
    print(f"  Blend (weighted): {results['blend_weighted']['oof_accuracy']:.5f}")
    print(f"  Best blend:       {best_blend_acc:.5f} ({results['best_blend_name']})")
    print(f"  Baseline:         {BASELINE_CV}")
    print(f"  Beats baseline:   {beats_baseline}")
    if submission_path:
        print(f"  Submission:       {submission_path} (validated)")
    print("=" * 70)


if __name__ == "__main__":
    main()
