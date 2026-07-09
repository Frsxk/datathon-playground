#!/usr/bin/env python3
"""Feature recipe scaffold for future Datathon production v4 experiments.

This module is intentionally safe by default:
- It creates deterministic, target-free feature recipes.
- It does NOT train models.
- It does NOT run cross-validation.
- It does NOT tune hyperparameters.
- It does NOT create Kaggle submissions.

Use `--dry-run` to inspect feature shapes/names on a small row sample.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_playground_loop import DAY_COLS, WEEK_COLS, make_features, safe_div  # noqa: E402

DATA = ROOT / "kaggle"


def _longest_run(mask: np.ndarray) -> np.ndarray:
    """Return longest consecutive True run for each row in a boolean matrix."""
    out = np.zeros(mask.shape[0], dtype=int)
    cur = np.zeros(mask.shape[0], dtype=int)
    for j in range(mask.shape[1]):
        cur = np.where(mask[:, j], cur + 1, 0)
        out = np.maximum(out, cur)
    return out


def _mad(arr: np.ndarray) -> np.ndarray:
    med = np.median(arr, axis=1, keepdims=True)
    return np.median(np.abs(arr - med), axis=1)


def _trimmed_mean(arr: np.ndarray, trim_each_side: int = 1) -> np.ndarray:
    if arr.shape[1] <= trim_each_side * 2:
        return arr.mean(axis=1)
    sorted_arr = np.sort(arr, axis=1)
    return sorted_arr[:, trim_each_side:-trim_each_side].mean(axis=1)


def add_boundary_features(df: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Features aimed at adjacent class boundaries.

    These are target-free and can be computed for train/test the same way.
    """
    out = pd.DataFrame(index=df.index)
    weekly = df[WEEK_COLS].to_numpy(float)
    daily = df[DAY_COLS].to_numpy(float)
    completion = safe_div(df["tugas_selesai"], df["tugas_diberikan"]).fillna(0.0)

    early_week = df[WEEK_COLS[:3]].mean(axis=1)
    mid_week = df[WEEK_COLS[3:9]].mean(axis=1)
    late_week = df[WEEK_COLS[9:]].mean(axis=1)

    out["v4_tryout_relative_to_weekly_mean"] = df["skor_tryout"] - base["minggu_mean"]
    out["v4_late_vs_early_week_mean"] = late_week - early_week
    out["v4_late_vs_mid_week_mean"] = late_week - mid_week
    out["v4_weekly_recovery_from_min"] = late_week - np.min(weekly[:, :9], axis=1)
    out["v4_completion_shortfall"] = 1.0 - completion
    out["v4_completion_shortfall_x_weekly_vol"] = out["v4_completion_shortfall"] * base["minggu_std"]
    out["v4_tryout_x_completion_shortfall"] = df["skor_tryout"] * out["v4_completion_shortfall"]

    daily_median = np.median(daily, axis=1, keepdims=True)
    daily_iqr = np.subtract(*np.percentile(daily, [75, 25], axis=1))
    daily_spike_threshold = daily_median + daily_iqr.reshape(-1, 1)
    out["v4_activity_spike_count"] = (daily > daily_spike_threshold).sum(axis=1)
    out["v4_activity_spike_ratio"] = out["v4_activity_spike_count"] / len(DAY_COLS)
    return out


def add_robust_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Robust summaries to reduce sensitivity to noisy single columns."""
    out = pd.DataFrame(index=df.index)
    weekly = df[WEEK_COLS].to_numpy(float)
    daily = df[DAY_COLS].to_numpy(float)
    week_diff = np.diff(weekly, axis=1)
    day_diff = np.diff(daily, axis=1)

    out["v4_week_trimmed_mean"] = _trimmed_mean(weekly)
    out["v4_day_trimmed_mean"] = _trimmed_mean(daily)
    out["v4_week_mad"] = _mad(weekly)
    out["v4_day_mad"] = _mad(daily)
    out["v4_week_sign_changes"] = (np.diff(np.sign(week_diff), axis=1) != 0).sum(axis=1)
    out["v4_day_sign_changes"] = (np.diff(np.sign(day_diff), axis=1) != 0).sum(axis=1)
    out["v4_week_longest_increase_run"] = _longest_run(week_diff > 0)
    out["v4_week_longest_decrease_run"] = _longest_run(week_diff < 0)

    early_vol = df[WEEK_COLS[:6]].std(axis=1).replace(0, np.nan)
    late_vol = df[WEEK_COLS[6:]].std(axis=1)
    out["v4_late_to_early_week_vol_ratio"] = (late_vol / early_vol).fillna(0.0)
    return out


def add_rank_features(df: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Target-free rank/composite features.

    For train/test production use, call this on the concatenated feature frame or
    accept that ranks are frame-local. This scaffold uses frame-local ranks only
    for dry-run inspection and future experiment design.
    """
    out = pd.DataFrame(index=df.index)
    rank_cols = [
        "tugas_completion_ratio",
        "minggu_mean",
        "minggu_std",
        "minggu_iqr",
        "minggu_range",
        "skor_tryout",
        "indeks_kehadiran",
        "skor_literasi",
    ]
    for col in rank_cols:
        if col in base:
            out[f"v4_rank_{col}"] = base[col].rank(pct=True, method="average")
        elif col in df:
            out[f"v4_rank_{col}"] = df[col].rank(pct=True, method="average")
    rank_feature_cols = list(out.columns)
    if rank_feature_cols:
        out["v4_rank_composite_academic"] = out[rank_feature_cols].mean(axis=1)
    return out


def make_features_v4(df: pd.DataFrame, include_base: bool = True) -> pd.DataFrame:
    """Return target-free v4 feature matrix.

    Parameters
    ----------
    df:
        Raw train or test dataframe.
    include_base:
        If true, include existing `run_playground_loop.make_features` output.
    """
    base = make_features(df)
    parts = []
    if include_base:
        parts.append(base)
    parts.extend([
        add_boundary_features(df, base),
        add_robust_sequence_features(df),
        add_rank_features(df, base),
    ])
    out = pd.concat(parts, axis=1)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(out.median(numeric_only=True))
    return out


def dry_run(rows: int) -> None:
    train_path = DATA / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing data file: {train_path}")
    raw = pd.read_csv(train_path).head(rows)
    features = make_features_v4(raw)
    v4_cols = [c for c in features.columns if c.startswith("v4_")]
    print("DRY RUN ONLY — no training/CV/tuning/submission executed")
    print(f"raw_shape={raw.shape}")
    print(f"feature_shape={features.shape}")
    print(f"base_feature_count={features.shape[1] - len(v4_cols)}")
    print(f"v4_feature_count={len(v4_cols)}")
    print("v4_feature_names:")
    for name in v4_cols:
        print(f"  - {name}")
    print("sample_v4_values:")
    print(features[v4_cols].head(min(rows, 5)).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Datathon v4 feature recipe scaffold")
    parser.add_argument("--dry-run", action="store_true", help="Inspect feature output only; required for CLI use")
    parser.add_argument("--rows", type=int, default=5, help="Rows to inspect in dry-run mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        raise SystemExit("Refusing to run without --dry-run. This script is a non-training scaffold.")
    dry_run(max(1, args.rows))


if __name__ == "__main__":
    main()
