#!/usr/bin/env python3
"""Datathon 2026 Playground baseline + reports.

Creates:
- outputs/metrics.json
- outputs/submission_baseline.csv
- reports/eda_feature_recon.md
- reports/competition_brief.md

No Kaggle submission is performed here.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "kaggle"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
PAGES = DATA / "pages"
OUT.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

TRAIN_PATH = DATA / "train.csv"
TEST_PATH = DATA / "test.csv"
SAMPLE_PATH = DATA / "sample_submission.csv"

WEEK_COLS = [f"nilai_minggu_{i:02d}" for i in range(1, 13)]
DAY_COLS = [f"aktivitas_hari_{i:02d}" for i in range(1, 17)]
BASE_DROP = {"id", "target"}
RANDOM_STATE = 42


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def row_slope(values: np.ndarray) -> np.ndarray:
    """Fast row-wise linear slope against positions 0..n-1."""
    x = np.arange(values.shape[1], dtype=float)
    x_centered = x - x.mean()
    denom = np.square(x_centered).sum()
    y_centered = values - values.mean(axis=1, keepdims=True)
    return (y_centered @ x_centered) / denom


def add_sequence_features(out: pd.DataFrame, source: pd.DataFrame, cols: list[str], prefix: str) -> None:
    arr = source[cols].to_numpy(dtype=float)
    first_half = cols[: len(cols) // 2]
    second_half = cols[len(cols) // 2 :]
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


def make_features(df: pd.DataFrame) -> pd.DataFrame:
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

    # Treat class identifier as a high-cardinality numeric/categorical-ish value.
    # Use frequency based on the current frame only; CV-safe because it does not use target.
    class_counts = df["kelas"].value_counts()
    out["kelas_frequency"] = df["kelas"].map(class_counts).astype(float)
    out["kelas_mod_10"] = df["kelas"] % 10
    out["kelas_mod_100"] = df["kelas"] % 100

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


def load_page(name: str) -> str:
    path = PAGES / f"{name}.json"
    if not path.exists():
        return ""
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return str(obj[0].get("content", ""))
    return str(obj)


def write_competition_brief(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> None:
    abstract = load_page("abstract")
    desc = load_page("Description")
    eval_page = load_page("Evaluation")
    data_desc = load_page("data-description")
    brief = f"""# Datathon 2026 Playground — Competition Brief

Generated: {datetime.now(timezone.utc).isoformat()}

## Status

- Playground window from broadcast: **July 8–17, 2026**.
- Broadcast states this playground is **learning-only** and **not part of competition evaluation**.
- Kaggle slug: `datathon-playground-2026`.
- Kaggle CLI access verified via `~/.kaggle/access_token`.

## Objective

{abstract.strip() or desc.strip()}

## Labels

Target `target` is a 4-class ordered performance label:

- `0` — Very Low Performance
- `1` — Low Performance
- `2` — Good Performance
- `3` — Excellent Performance

## Evaluation

{eval_page.strip()}

## Files and Shapes

- `train.csv`: {train.shape[0]} rows × {train.shape[1]} columns
- `test.csv`: {test.shape[0]} rows × {test.shape[1]} columns
- `sample_submission.csv`: {sample.shape[0]} rows × {sample.shape[1]} columns

## Submission Format

Must contain exactly:

```csv
id,target
```

One row per test `id`; `target` must be one of 0, 1, 2, 3.

## Data Description Extract

{data_desc.strip()}

## Notes / Caveats

- FAQ sheet was public/exportable but currently mostly empty.
- Kaggle discussion topics currently returned no topics.
- Do not submit from Hermes without explicit approval from frisky.
"""
    (REPORTS / "competition_brief.md").write_text(brief, encoding="utf-8")


def write_eda_report(train: pd.DataFrame, test: pd.DataFrame, X: pd.DataFrame, metrics: list[dict]) -> None:
    y = train["target"]
    feature_cols = [c for c in train.columns if c not in BASE_DROP]
    raw_corr = (
        train[feature_cols]
        .apply(lambda col: abs(np.corrcoef(col, y)[0, 1]))
        .sort_values(ascending=False)
    )
    engineered_corr = (
        X.apply(lambda col: abs(np.corrcoef(col, y)[0, 1]))
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values(ascending=False)
    )
    target_counts = y.value_counts().sort_index().to_dict()
    best = max(metrics, key=lambda m: m["mean_accuracy"])
    md = [
        "# Datathon 2026 Playground — EDA & Feature Recon",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Dataset Snapshot",
        "",
        f"- Train shape: `{train.shape}`",
        f"- Test shape: `{test.shape}`",
        f"- Missing values train: `{int(train.isna().sum().sum())}`",
        f"- Missing values test: `{int(test.isna().sum().sum())}`",
        f"- Target counts: `{target_counts}`",
        "- Classes are balanced enough that accuracy is a reasonable metric.",
        "",
        "## Baseline Results",
        "",
    ]
    for m in sorted(metrics, key=lambda x: x["mean_accuracy"], reverse=True):
        md.append(f"- **{m['name']}**: {m['mean_accuracy']:.5f} ± {m['std_accuracy']:.5f} folds={m['fold_scores']}")
    md += [
        "",
        f"Best local CV model in this first loop: **{best['name']}** ({best['mean_accuracy']:.5f}).",
        "",
        "## Top Raw Feature Correlations with Target",
        "",
        raw_corr.head(15).round(4).to_markdown(),
        "",
        "## Top Engineered Feature Correlations with Target",
        "",
        engineered_corr.head(20).round(4).to_markdown(),
        "",
        "## Recommended Next Feature Pounces",
        "",
        "1. **Sequence trend features**: keep weekly/day slopes, late-vs-early deltas, diff volatility, positive/negative step counts.",
        "2. **Assignment behavior**: `tugas_selesai / tugas_diberikan`, remaining tasks, and interactions with discipline/motivation.",
        "3. **Activity consistency**: day std/IQR/diff_abs_mean may capture stable vs erratic learning patterns.",
        "4. **Academic trajectory**: interactions between `skor_tryout`, weekly slope, literacy, and attendance.",
        "5. **Class identifier caution**: `kelas` may encode groups. Frequency/mod features are target-safe; target encoding must be done inside CV folds only to avoid leakage.",
        "6. **Ordered labels**: metric is accuracy, but labels have order. Try regression/ordinal-inspired features or calibrated class boundaries later, while still submitting class labels.",
        "7. **Model stack**: compare ExtraTrees/RandomForest/HistGradientBoosting/XGBoost, then blend if CV is stable.",
        "",
        "## Submission Safety",
        "",
        "No submission was made. Generated submission must be approved by frisky before using Kaggle CLI submit.",
    ]
    (REPORTS / "eda_feature_recon.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_PATH)

    y = train["target"]
    X_raw = train[[c for c in train.columns if c not in BASE_DROP]].copy()
    X_fe = make_features(train)
    X_test = make_features(test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = [
        ("logreg_raw_scaled", make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))),
        ("logreg_fe_scaled", make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))),
        ("extratrees_fe", ExtraTreesClassifier(n_estimators=600, random_state=RANDOM_STATE, n_jobs=-1, max_features="sqrt")),
        ("randomforest_fe", RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1, max_features="sqrt")),
        ("histgb_fe", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=RANDOM_STATE)),
    ]
    if XGBClassifier is not None:
        models.append((
            "xgboost_fe",
            XGBClassifier(
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
            ),
        ))

    metrics: list[dict] = []
    # Raw only for the raw model; all others use FE.
    for name, model in models:
        X = X_raw if name.endswith("raw_scaled") else X_fe
        print(f"Evaluating {name} on shape {X.shape}...")
        metrics.append(cv_model(name, model, X, y, cv))

    best = max(metrics, key=lambda m: m["mean_accuracy"])
    best_name = best["name"]
    best_model = dict(models)[best_name]
    best_X = X_raw if best_name.endswith("raw_scaled") else X_fe
    best_X_test = test[[c for c in test.columns if c not in BASE_DROP]].copy() if best_name.endswith("raw_scaled") else X_test
    best_model.fit(best_X, y)
    preds = best_model.predict(best_X_test).astype(int)

    submission = pd.DataFrame({"id": test["id"], "target": preds})
    # Validate exact submission shape/order against sample IDs.
    if list(submission.columns) != ["id", "target"]:
        raise RuntimeError("Bad submission columns")
    if submission.shape != sample.shape:
        raise RuntimeError(f"Bad submission shape {submission.shape} != {sample.shape}")
    if not submission["id"].equals(sample["id"]):
        raise RuntimeError("Submission IDs do not match sample_submission order")
    if not set(submission["target"].unique()).issubset({0, 1, 2, 3}):
        raise RuntimeError("Submission contains invalid target labels")

    sub_path = OUT / "submission_baseline.csv"
    submission.to_csv(sub_path, index=False)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "competition_slug": "datathon-playground-2026",
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "sample_submission_shape": list(sample.shape),
        "target_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "feature_shape_raw": list(X_raw.shape),
        "feature_shape_fe": list(X_fe.shape),
        "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "metrics": metrics,
        "best_model": best,
        "submission_path": str(sub_path),
        "submission_prediction_counts": {str(k): int(v) for k, v in submission["target"].value_counts().sort_index().items()},
        "no_kaggle_submission_made": True,
    }
    (OUT / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    write_competition_brief(train, test, sample)
    write_eda_report(train, test, X_fe, metrics)
    print(json.dumps({"best_model": best, "submission_path": str(sub_path)}, indent=2))


if __name__ == "__main__":
    main()
