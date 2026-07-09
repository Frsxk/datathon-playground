#!/usr/bin/env python3
"""Train/test feature-shift audit for Datathon Playground.

Safe diagnostic only:
- no model training
- no CV
- no tuning
- no Kaggle calls/submission
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from feature_factory_v4 import make_features_v4  # noqa: E402

DATA = ROOT / "kaggle"
DEFAULT_OUT = ROOT / "reports" / "feature_shift_audit.md"


def psi(train_values: pd.Series, test_values: pd.Series, bins: int = 10) -> float:
    """Population Stability Index using train quantile bins."""
    train = pd.Series(train_values).replace([np.inf, -np.inf], np.nan).dropna()
    test = pd.Series(test_values).replace([np.inf, -np.inf], np.nan).dropna()
    if train.empty or test.empty or train.nunique() <= 1:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(train, quantiles))
    if len(edges) <= 2:
        edges = np.linspace(float(train.min()), float(train.max()), min(bins, train.nunique()) + 1)
    if len(edges) <= 1:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    train_bins = pd.cut(train, bins=edges, include_lowest=True)
    test_bins = pd.cut(test, bins=edges, include_lowest=True)
    train_pct = train_bins.value_counts(sort=False, normalize=True).replace(0, 1e-6)
    test_pct = test_bins.value_counts(sort=False, normalize=True).reindex(train_pct.index, fill_value=1e-6).replace(0, 1e-6)
    return float(((test_pct - train_pct) * np.log(test_pct / train_pct)).sum())


def standardized_mean_diff(train: pd.Series, test: pd.Series) -> float:
    tr = pd.Series(train).astype(float)
    te = pd.Series(test).astype(float)
    pooled = np.sqrt((tr.var(ddof=0) + te.var(ddof=0)) / 2.0)
    if not np.isfinite(pooled) or pooled == 0:
        return 0.0
    return float((te.mean() - tr.mean()) / pooled)


def quantile_delta(train: pd.Series, test: pd.Series, q: float) -> float:
    return float(test.quantile(q) - train.quantile(q))


def shift_table(train_features: pd.DataFrame, test_features: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in train_features.columns or col not in test_features.columns:
            continue
        tr = train_features[col].astype(float)
        te = test_features[col].astype(float)
        rows.append({
            "feature": col,
            "train_mean": float(tr.mean()),
            "test_mean": float(te.mean()),
            "smd": standardized_mean_diff(tr, te),
            "psi": psi(tr, te),
            "train_std": float(tr.std(ddof=0)),
            "test_std": float(te.std(ddof=0)),
            "median_delta": quantile_delta(tr, te, 0.5),
            "q10_delta": quantile_delta(tr, te, 0.1),
            "q90_delta": quantile_delta(tr, te, 0.9),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_smd"] = out["smd"].abs()
    return out.sort_values(["psi", "abs_smd"], ascending=False)


def risk_label(psi_value: float, abs_smd: float) -> str:
    if psi_value >= 0.25 or abs_smd >= 0.50:
        return "high"
    if psi_value >= 0.10 or abs_smd >= 0.25:
        return "medium"
    return "low"


def markdown_table(df: pd.DataFrame, limit: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    return df.head(limit).to_markdown(index=False)


def write_report(out_path: Path, top_n: int) -> None:
    train_path = DATA / "train.csv"
    test_path = DATA / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing train.csv or test.csv under kaggle/")

    raw_train = pd.read_csv(train_path)
    raw_test = pd.read_csv(test_path)
    train_features = make_features_v4(raw_train)
    test_features = make_features_v4(raw_test)

    shared_cols = [c for c in train_features.columns if c in test_features.columns]
    table = shift_table(train_features, test_features, shared_cols)
    if not table.empty:
        table["risk"] = [risk_label(float(p), float(s)) for p, s in zip(table["psi"], table["abs_smd"])]
    high = int((table["risk"] == "high").sum()) if not table.empty else 0
    medium = int((table["risk"] == "medium").sum()) if not table.empty else 0
    low = int((table["risk"] == "low").sum()) if not table.empty else 0
    v4 = table[table["feature"].str.startswith("v4_")] if not table.empty else table

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Feature Shift Audit",
        "",
        f"Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Safety note",
        "",
        "This diagnostic computes target-free feature statistics only. It does not train models, run CV, tune hyperparameters, or call Kaggle.",
        "",
        "## Inputs",
        "",
        f"- Train: `{train_path}` shape `{raw_train.shape}`",
        f"- Test: `{test_path}` shape `{raw_test.shape}`",
        f"- Feature matrix shapes: train `{train_features.shape}`, test `{test_features.shape}`",
        "",
        "## Shift summary",
        "",
        f"- Shared features audited: `{len(shared_cols)}`",
        f"- High-risk shift features: `{high}`",
        f"- Medium-risk shift features: `{medium}`",
        f"- Low-risk shift features: `{low}`",
        "",
        "Risk heuristic: high if PSI >= 0.25 or |SMD| >= 0.50; medium if PSI >= 0.10 or |SMD| >= 0.25.",
        "",
        f"## Top {top_n} shifted features",
        "",
        markdown_table(table.round(5), top_n),
        "",
        f"## Top {top_n} shifted v4 scaffold features",
        "",
        markdown_table(v4.round(5), top_n),
        "",
        "## Interpretation",
        "",
        "- Features with high train/test shift should be used cautiously in future smoke experiments, especially if they dominate model gain.",
        "- If a v4 feature is high-shift, prefer rank-normalized or threshold-stable variants before full CV.",
        "- If most v4 features are low/medium shift, the next safe modeling step is a one-seed smoke test rather than another broad Optuna run.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"features_audited={len(shared_cols)}")
    print(f"risk_counts={{'high': {high}, 'medium': {medium}, 'low': {low}}}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Target-free train/test feature-shift audit")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_report(args.out, max(1, args.top))


if __name__ == "__main__":
    main()
