#!/usr/bin/env python3
"""Bounded feature-variant experiment for Datathon 2026 Playground.

Focus: sequence/assignment features beyond baseline — add log transforms,
ratios, and CV-safe simple transforms. Evaluate 2 models (HistGB + XGBoost)
with 5-fold StratifiedKFold(random_state=43). Bounded runtime; no broad search.

Creates:
- outputs/exp_feature_variant.json
- outputs/submission_feature_variant.csv  (only if CV beats baseline 0.49031)
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "kaggle"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

TRAIN_PATH = DATA / "train.csv"
TEST_PATH = DATA / "test.csv"
SAMPLE_PATH = DATA / "sample_submission.csv"

WEEK_COLS = [f"nilai_minggu_{i:02d}" for i in range(1, 13)]
DAY_COLS = [f"aktivitas_hari_{i:02d}" for i in range(1, 17)]
BASE_DROP = {"id", "target"}
RANDOM_STATE = 43  # per task spec
BASELINE_CV = 0.49031  # baseline to beat


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def row_slope(values: np.ndarray) -> np.ndarray:
    x = np.arange(values.shape[1], dtype=float)
    x_centered = x - x.mean()
    denom = np.square(x_centered).sum()
    y_centered = values - values.mean(axis=1, keepdims=True)
    return (y_centered @ x_centered) / denom


def add_sequence_features(out: pd.DataFrame, source: pd.DataFrame, cols: list[str], prefix: str) -> None:
    """Baseline sequence features (same as run_playground_loop.py) PLUS new variant features."""
    arr = source[cols].to_numpy(dtype=float)
    first_half = cols[: len(cols) // 2]
    second_half = cols[len(cols) // 2:]

    # --- Baseline sequence features (copied to stay comparable) ---
    out[f"{prefix}_mean"] = source[cols].mean(axis=1)
    out[f"{prefix}_std"] = source[cols].std(axis=1)
    out[f"{prefix}_min"] = source[cols].min(axis=1)
    out[f"{prefix}_max"] = source[cols].max(axis=1)
    out[f"{prefix}_range"] = out[f"{prefix}_max"] - out[f"{prefix}_min"]
    out[f"{prefix}_median"] = source[cols].median(axis=1)
    out[f"{prefix}_q25"] = source[cols].quantile(0.25, axis=1)
    out[f"{prefix}_q75"] = source[cols].quantile(0.75, axis=1)
    out[f"{prefix}_iqr"] = out[f"{prefix}_q75"] - out[f"{prefix}_q25"]
    out[f"{prefix}_first"] = source[cols[0]]
    out[f"{prefix}_last"] = source[cols[-1]]
    out[f"{prefix}_last_minus_first"] = source[cols[-1]] - source[cols[0]]
    out[f"{prefix}_early_mean"] = source[first_half].mean(axis=1)
    out[f"{prefix}_late_mean"] = source[second_half].mean(axis=1)
    out[f"{prefix}_late_minus_early"] = out[f"{prefix}_late_mean"] - out[f"{prefix}_early_mean"]
    diffs = np.diff(arr, axis=1)
    out[f"{prefix}_diff_mean"] = diffs.mean(axis=1)
    out[f"{prefix}_diff_std"] = diffs.std(axis=1)
    out[f"{prefix}_diff_abs_mean"] = np.abs(diffs).mean(axis=1)
    out[f"{prefix}_positive_steps"] = (diffs > 0).sum(axis=1)
    out[f"{prefix}_negative_steps"] = (diffs < 0).sum(axis=1)
    out[f"{prefix}_slope"] = row_slope(arr)

    # --- NEW: Variant sequence features (log/ratio/CV-safe transforms) ---
    # CV of sequence (coefficient of variation — normalized volatility)
    out[f"{prefix}_cv"] = safe_div(out[f"{prefix}_std"], out[f"{prefix}_mean"].abs() + 1e-8)

    # Log-transform of mean (handle negatives via shift)
    mean_shifted = out[f"{prefix}_mean"] - out[f"{prefix}_mean"].min() + 1.0
    out[f"{prefix}_log_mean"] = np.log1p(mean_shifted)

    # Log-transform of range
    range_shifted = out[f"{prefix}_range"] - out[f"{prefix}_range"].min() + 1.0
    out[f"{prefix}_log_range"] = np.log1p(range_shifted)

    # Ratio of late to early (shifted to avoid div-by-zero and negatives)
    early_shifted = out[f"{prefix}_early_mean"] - out[f"{prefix}_early_mean"].min() + 1.0
    late_shifted = out[f"{prefix}_late_mean"] - out[f"{prefix}_late_mean"].min() + 1.0
    out[f"{prefix}_late_early_ratio"] = safe_div(late_shifted, early_shifted)

    # Log of absolute slope (trend strength)
    out[f"{prefix}_log_abs_slope"] = np.log1p(out[f"{prefix}_slope"].abs())

    # Max-to-mean ratio (peak relative to average)
    out[f"{prefix}_max_to_mean_ratio"] = safe_div(
        out[f"{prefix}_max"] - out[f"{prefix}_min"],
        out[f"{prefix}_mean"].abs() + 1e-8,
    )

    # Coefficient of variation of diffs (volatility of changes)
    out[f"{prefix}_diff_cv"] = safe_div(out[f"{prefix}_diff_std"], out[f"{prefix}_diff_mean"].abs() + 1e-8)

    # Positive step ratio (fraction of increasing steps)
    total_steps = (out[f"{prefix}_positive_steps"] + out[f"{prefix}_negative_steps"]).replace(0, np.nan)
    out[f"{prefix}_positive_step_ratio"] = safe_div(out[f"{prefix}_positive_steps"], total_steps)

    # Quartile ratio (dispersion asymmetry)
    out[f"{prefix}_q_ratio"] = safe_div(out[f"{prefix}_q75"], out[f"{prefix}_q25"].abs() + 1e-8)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline features + new sequence/assignment variant features."""
    out = df[[c for c in df.columns if c not in BASE_DROP]].copy()
    add_sequence_features(out, df, WEEK_COLS, "minggu")
    add_sequence_features(out, df, DAY_COLS, "hari")

    # --- Baseline assignment features ---
    out["tugas_completion_ratio"] = safe_div(df["tugas_selesai"], df["tugas_diberikan"])
    out["tugas_remaining"] = df["tugas_diberikan"] - df["tugas_selesai"]
    out["tugas_completion_gap"] = out["tugas_completion_ratio"] - out["tugas_completion_ratio"].median()

    out["tryout_x_kehadiran"] = df["skor_tryout"] * df["indeks_kehadiran"]
    out["motivasi_x_minat"] = df["skor_motivasi"] * df["skor_minat_belajar"]
    out["disiplin_x_tugas_ratio"] = df["skor_kedisiplinan"] * out["tugas_completion_ratio"]
    out["literasi_x_tryout"] = df["skor_literasi"] * df["skor_tryout"]
    out["weekly_slope_x_tryout"] = out["minggu_slope"] * df["skor_tryout"]
    out["activity_consistency"] = -out["hari_std"]

    # --- NEW: Variant assignment/ratio features ---
    # Log transform of tasks assigned (skewed count)
    out["tugas_diberikan_log"] = np.log1p(df["tugas_diberikan"].clip(lower=0))
    out["tugas_selesai_log"] = np.log1p(df["tugas_selesai"].clip(lower=0))

    # Remaining-to-given ratio (incomplete fraction)
    out["tugas_incomplete_ratio"] = safe_div(out["tugas_remaining"], df["tugas_diberikan"])

    # Log of completion ratio (shifted to handle negatives/zeros)
    comp_shifted = out["tugas_completion_ratio"] - out["tugas_completion_ratio"].min() + 1.0
    out["tugas_completion_ratio_log"] = np.log1p(comp_shifted)

    # Assignment efficiency: tasks done per unit of discipline
    out["tugas_per_disiplin"] = safe_div(df["tugas_selesai"], df["skor_kedisiplinan"].abs() + 1e-8)

    # Tryout per task (academic density)
    out["tryout_per_tugas"] = safe_div(df["skor_tryout"], df["tugas_diberikan"].replace(0, np.nan))

    # Score-to-distance ratio (proximity-adjusted score)
    out["tryout_per_km"] = safe_div(df["skor_tryout"], df["jarak_rumah_km"].abs() + 1e-8)

    # Attendance-weighted literacy
    out["literasi_x_kehadiran"] = df["skor_literasi"] * df["indeks_kehadiran"]

    # Motivation-to-discipline ratio (willpower balance)
    out["motivasi_per_disiplin"] = safe_div(df["skor_motivasi"], df["skor_kedisiplinan"].abs() + 1e-8)

    # Extracurricular per sibling (resource dilution proxy)
    out["ekskul_per_saudara"] = safe_div(df["skor_ekstrakurikuler"], df["jumlah_saudara"].replace(0, np.nan))

    # Class identifier features (CV-safe, target-free)
    class_counts = df["kelas"].value_counts()
    out["kelas_frequency"] = df["kelas"].map(class_counts).astype(float)
    out["kelas_mod_10"] = df["kelas"] % 10
    out["kelas_mod_100"] = df["kelas"] % 100

    # Log of class frequency
    out["kelas_frequency_log"] = np.log1p(out["kelas_frequency"])

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(out.median(numeric_only=True))
    return out


def cv_model(name: str, model, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> dict:
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    return {
        "name": name,
        "fold_scores": [float(x) for x in scores],
        "mean_accuracy": float(scores.mean()),
        "std_accuracy": float(scores.std()),
    }


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_PATH)

    y = train["target"]
    X_fe = make_features(train)
    X_test = make_features(test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 2 models with modest settings
    histgb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, random_state=RANDOM_STATE
    )
    xgb = XGBClassifier(
        objective="multi:softmax",
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
    )

    models = [
        ("histgb_variant", histgb),
        ("xgboost_variant", xgb),
    ]

    metrics: list[dict] = []
    for name, model in models:
        print(f"Evaluating {name} on shape {X_fe.shape}...")
        metrics.append(cv_model(name, model, X_fe, y, cv))

    best = max(metrics, key=lambda m: m["mean_accuracy"])
    best_name = best["name"]
    best_model = dict(models)[best_name]

    beats_baseline = best["mean_accuracy"] > BASELINE_CV

    submission_path = None
    submission_prediction_counts = None
    submission_validated = False

    if beats_baseline:
        best_model.fit(X_fe, y)
        preds = best_model.predict(X_test).astype(int)
        submission = pd.DataFrame({"id": test["id"], "target": preds})

        # Validate shape/id/order
        if list(submission.columns) != ["id", "target"]:
            raise RuntimeError("Bad submission columns")
        if submission.shape != sample.shape:
            raise RuntimeError(f"Bad submission shape {submission.shape} != {sample.shape}")
        if not submission["id"].equals(sample["id"]):
            raise RuntimeError("Submission IDs do not match sample_submission order")
        if not set(submission["target"].unique()).issubset({0, 1, 2, 3}):
            raise RuntimeError("Submission contains invalid target labels")
        submission_validated = True

        submission_path = OUT / "submission_feature_variant.csv"
        submission.to_csv(submission_path, index=False)
        submission_prediction_counts = {
            str(k): int(v) for k, v in submission["target"].value_counts().sort_index().items()
        }

    result = {
        "experiment": "feature_variant",
        "description": "Sequence/assignment features: log transforms, ratios, CV-safe transforms beyond baseline",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "cv": f"StratifiedKFold(n_splits=5, shuffle=True, random_state={RANDOM_STATE})",
        "baseline_cv": BASELINE_CV,
        "feature_shape": list(X_fe.shape),
        "n_new_features": X_fe.shape[1] - 95,  # baseline had 95 features
        "models_evaluated": [m["name"] for m in metrics],
        "metrics": metrics,
        "best_model": best,
        "beats_baseline": beats_baseline,
        "submission_path": str(submission_path) if submission_path else None,
        "submission_validated": submission_validated,
        "submission_prediction_counts": submission_prediction_counts,
        "no_kaggle_submission_made": True,
    }

    (OUT / "exp_feature_variant.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
