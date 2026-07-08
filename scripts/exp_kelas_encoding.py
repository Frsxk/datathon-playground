#!/usr/bin/env python3
"""CV-safe target encoding for `kelas` feature - Datathon 2026 Playground.

Evaluates whether CV-safe target encoding of the high-cardinality `kelas`
column improves over the baseline XGBoost (CV acc 0.49031 with rs=42).

Key leakage precautions:
- Target encoding is computed strictly inside each CV fold using only
  the training portion of that fold.
- Smoothing (Bayesian shrinkage) is applied to shrink per-kelas means
  toward the global mean, reducing overfitting on rare kelas values.
- Test/fold-validation rows use a fallback (global mean) for unseen kelas.

Creates:
- outputs/exp_kelas_encoding.json
- outputs/submission_kelas_encoding.csv (only if candidate beats baseline)
"""
from __future__ import annotations

import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

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
RANDOM_STATE = 45
N_SPLITS = 5
N_CLASSES = 4
SMOOTHING = 10.0  # Bayesian shrinkage prior strength

warnings.filterwarnings("ignore", category=FutureWarning)


def safe_div(num, den):
    return num / den.replace(0, np.nan)


def row_slope(values):
    x = np.arange(values.shape[1], dtype=float)
    xc = x - x.mean()
    denom = np.square(xc).sum()
    yc = values - values.mean(axis=1, keepdims=True)
    return (yc @ xc) / denom


def add_sequence_features(out, source, cols, prefix):
    arr = source[cols].to_numpy(dtype=float)
    fh = cols[: len(cols) // 2]
    sh = cols[len(cols) // 2:]
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
    out[f"{prefix}_early_mean"] = source[fh].mean(axis=1)
    out[f"{prefix}_late_mean"] = source[sh].mean(axis=1)
    out[f"{prefix}_late_minus_early"] = out[f"{prefix}_late_mean"] - out[f"{prefix}_early_mean"]
    diffs = np.diff(arr, axis=1)
    out[f"{prefix}_diff_mean"] = diffs.mean(axis=1)
    out[f"{prefix}_diff_std"] = diffs.std(axis=1)
    out[f"{prefix}_diff_abs_mean"] = np.abs(diffs).mean(axis=1)
    out[f"{prefix}_positive_steps"] = (diffs > 0).sum(axis=1)
    out[f"{prefix}_negative_steps"] = (diffs < 0).sum(axis=1)
    out[f"{prefix}_slope"] = row_slope(arr)


def make_features(df):
    """Same feature engineering as baseline, but WITHOUT kelas target encoding."""
    out = df[[c for c in df.columns if c not in BASE_DROP]].copy()
    add_sequence_features(out, df, WEEK_COLS, "minggu")
    add_sequence_features(out, df, DAY_COLS, "hari")
    out["tugas_completion_ratio"] = safe_div(df["tugas_selesai"], df["tugas_diberikan"])
    out["tugas_remaining"] = df["tugas_diberikan"] - df["tugas_selesai"]
    out["tugas_completion_gap"] = out["tugas_completion_ratio"] - out["tugas_completion_ratio"].median()
    out["tryout_x_kehadiran"] = df["skor_tryout"] * df["indeks_kehadiran"]
    out["motivasi_x_minat"] = df["skor_motivasi"] * df["skor_minat_belajar"]
    out["disiplin_x_tugas_ratio"] = df["skor_kedisiplinan"] * out["tugas_completion_ratio"]
    out["literasi_x_tryout"] = df["skor_literasi"] * df["skor_tryout"]
    out["weekly_slope_x_tryout"] = out["minggu_slope"] * df["skor_tryout"]
    out["activity_consistency"] = -out["hari_std"]
    class_counts = df["kelas"].value_counts()
    out["kelas_frequency"] = df["kelas"].map(class_counts).astype(float)
    out["kelas_mod_10"] = df["kelas"] % 10
    out["kelas_mod_100"] = df["kelas"] % 100
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(out.median(numeric_only=True))
    return out


def fit_target_encoding(kelas_train, y_train, smoothing=SMOOTHING):
    """Compute smoothed per-kelas target probabilities from training fold only.

    Returns (kelas_to_enc, global_probs) where:
    - kelas_to_enc: dict {kelas_val -> np.array of 4 class probs}
    - global_probs: np.array of 4 class probs (fallback + shrinkage prior)
    """
    global_counts = np.bincount(y_train, minlength=N_CLASSES).astype(float)
    global_probs = global_counts / global_counts.sum()

    df = pd.DataFrame({"kelas": kelas_train, "y": y_train})
    kelas_to_enc = {}
    for cls in range(N_CLASSES):
        df[f"is_{cls}"] = (df["y"] == cls).astype(float)

    grp = df.groupby("kelas")
    n = grp.size()
    kelas_to_enc = {}
    for kelas_val, row in n.items():
        enc = np.zeros(N_CLASSES)
        for cls in range(N_CLASSES):
            s = grp[f"is_{cls}"].sum().loc[kelas_val]
            enc[cls] = (s + smoothing * global_probs[cls]) / (row + smoothing)
        kelas_to_enc[kelas_val] = enc
    return kelas_to_enc, global_probs


def apply_target_encoding(kelas_series, kelas_to_enc, global_probs):
    """Apply encoding to a Series of kelas values; unseen -> global_probs."""
    n = len(kelas_series)
    result = np.tile(global_probs, (n, 1))
    for i, kv in enumerate(kelas_series.values):
        if kv in kelas_to_enc:
            result[i] = kelas_to_enc[kv]
    return result


def make_xgb():
    return XGBClassifier(
        objective="multi:softmax", num_class=4,
        n_estimators=500, max_depth=4, learning_rate=0.04,
        subsample=0.9, colsample_bytree=0.9,
        random_state=RANDOM_STATE, n_jobs=2,
        eval_metric="mlogloss", verbosity=0,
    )


def make_histgb():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, random_state=RANDOM_STATE)


def cv_fold_eval(X_base, y, kelas_series, use_encoding, model_factory):
    """Manual CV with optional in-fold target encoding.

    Returns list of fold accuracies.
    """
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []
    for tr_idx, va_idx in cv.split(X_base, y):
        X_tr = X_base.iloc[tr_idx].copy()
        X_va = X_base.iloc[va_idx].copy()
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        if use_encoding:
            enc_map, global_probs = fit_target_encoding(
                kelas_series.iloc[tr_idx], y_tr.values)
            enc_tr = apply_target_encoding(kelas_series.iloc[tr_idx], enc_map, global_probs)
            enc_va = apply_target_encoding(kelas_series.iloc[va_idx], enc_map, global_probs)
            for cls in range(N_CLASSES):
                X_tr[f"kelas_te_{cls}"] = enc_tr[:, cls]
                X_va[f"kelas_te_{cls}"] = enc_va[:, cls]

        model = model_factory()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        fold_scores.append(float(accuracy_score(y_va, preds)))
    return fold_scores


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_PATH)
    y = train["target"]
    kelas_train = train["kelas"]
    kelas_test = test["kelas"]

    X_base = make_features(train)
    X_test_base = make_features(test)
    print(f"Base feature shape: {X_base.shape}")

    results = {}
    configs = []
    if XGBClassifier is not None:
        configs.append(("xgboost", make_xgb))
    configs.append(("histgb", make_histgb))

    # Baseline (no target encoding) under rs=45
    for name, factory in configs:
        label = f"{name}_baseline"
        print(f"Evaluating {label}...")
        scores = cv_fold_eval(X_base, y, kelas_train, False, factory)
        mean_acc = float(np.mean(scores))
        results[label] = {
            "fold_scores": scores, "mean_accuracy": mean_acc,
            "std_accuracy": float(np.std(scores)),
        }
        print(f"  {label}: {mean_acc:.5f} +/- {np.std(scores):.5f}")

    # Candidate (with CV-safe target encoding)
    for name, factory in configs:
        label = f"{name}_te"
        print(f"Evaluating {label}...")
        scores = cv_fold_eval(X_base, y, kelas_train, True, factory)
        mean_acc = float(np.mean(scores))
        results[label] = {
            "fold_scores": scores, "mean_accuracy": mean_acc,
            "std_accuracy": float(np.std(scores)),
        }
        print(f"  {label}: {mean_acc:.5f} +/- {np.std(scores):.5f}")

    # Baseline reference from original loop (rs=42)
    baseline_ref = 0.49031
    baseline_items = [(k, v) for k, v in results.items() if "baseline" in k]
    candidate_items = [(k, v) for k, v in results.items() if "_te" in k]
    best_baseline_name, best_baseline = max(baseline_items, key=lambda kv: kv[1]["mean_accuracy"])
    best_candidate_name, best_candidate = max(candidate_items, key=lambda kv: kv[1]["mean_accuracy"])

    beats_baseline = best_candidate["mean_accuracy"] > best_baseline["mean_accuracy"]
    beats_ref = best_candidate["mean_accuracy"] > baseline_ref

    print(f"\nBest baseline (rs=45): {best_baseline_name} {best_baseline['mean_accuracy']:.5f}")
    print(f"Best candidate (TE):  {best_candidate_name} {best_candidate['mean_accuracy']:.5f}")
    print(f"Reference baseline (rs=42): {baseline_ref}")
    print(f"Beats baseline (rs=45): {beats_baseline}")
    print(f"Beats reference (rs=42): {beats_ref}")

    submission_path = None
    submission_validated = False
    submission_prediction_counts = None

    if beats_ref:
        # Fit full-train target encoding, add class-probability features to train/test,
        # then fit the best TE model on all training rows.
        factory_name = best_candidate_name.replace("_te", "")
        factory = dict(configs)[factory_name]
        enc_map, global_probs = fit_target_encoding(kelas_train, y.values)
        enc_train = apply_target_encoding(kelas_train, enc_map, global_probs)
        enc_test = apply_target_encoding(kelas_test, enc_map, global_probs)
        X_train_full = X_base.copy()
        X_test_full = X_test_base.copy()
        for cls in range(N_CLASSES):
            X_train_full[f"kelas_te_{cls}"] = enc_train[:, cls]
            X_test_full[f"kelas_te_{cls}"] = enc_test[:, cls]

        model = factory()
        model.fit(X_train_full, y)
        preds = model.predict(X_test_full).astype(int)
        submission = pd.DataFrame({"id": test["id"], "target": preds})
        if list(submission.columns) != ["id", "target"]:
            raise RuntimeError("Bad submission columns")
        if submission.shape != sample.shape:
            raise RuntimeError(f"Bad submission shape {submission.shape} != {sample.shape}")
        if not submission["id"].equals(sample["id"]):
            raise RuntimeError("Submission IDs do not match sample_submission order")
        if not set(submission["target"].unique()).issubset({0, 1, 2, 3}):
            raise RuntimeError("Submission contains invalid target labels")
        submission_path = OUT / "submission_kelas_encoding.csv"
        submission.to_csv(submission_path, index=False)
        submission_validated = True
        submission_prediction_counts = {str(k): int(v) for k, v in submission["target"].value_counts().sort_index().items()}
        print(f"Wrote validated submission: {submission_path}")

    result = {
        "experiment": "kelas_encoding",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": "CV-safe smoothed target encoding for kelas; encodings built inside each fold only.",
        "baseline_ref": baseline_ref,
        "random_state": RANDOM_STATE,
        "cv": f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})",
        "smoothing": SMOOTHING,
        "feature_shape_base": list(X_base.shape),
        "results": results,
        "best_baseline_name": best_baseline_name,
        "best_baseline": best_baseline,
        "best_candidate_name": best_candidate_name,
        "best_candidate": best_candidate,
        "beats_same_seed_baseline": beats_baseline,
        "beats_reference_baseline": beats_ref,
        "leakage_precautions": [
            "Target encodings are computed inside each CV training fold only.",
            "Validation rows use mappings learned from training-fold rows only.",
            "Unseen kelas values fall back to fold/global class probabilities.",
            "Full-train encoding is used only after CV for optional test submission generation.",
        ],
        "submission_path": str(submission_path) if submission_path else None,
        "submission_validated": submission_validated,
        "submission_prediction_counts": submission_prediction_counts,
        "no_kaggle_submission_made": True,
    }
    (OUT / "exp_kelas_encoding.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
