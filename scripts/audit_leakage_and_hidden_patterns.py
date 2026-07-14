#!/usr/bin/env python3
"""Leakage and hidden-pattern audit for Datathon Playground.

Read-only diagnostics only: no submissions, no target leakage into test labels, and no
competition API writes. The goal is to decide whether the 0.82 public LB jump
looks reachable from local structural clues.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "kaggle"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"


def feature_columns(train: pd.DataFrame) -> list[str]:
    return [c for c in train.columns if c not in {"id", "target"}]


def exact_feature_overlap(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Count exact duplicate feature rows across train/test and within each split."""
    cols = feature_columns(train)
    train_feat = train[cols].copy()
    test_feat = test[cols].copy()
    train_hash = pd.util.hash_pandas_object(train_feat, index=False)
    test_hash = pd.util.hash_pandas_object(test_feat, index=False)
    common = set(train_hash).intersection(set(test_hash))
    return {
        "overlap_count": int(train_hash.isin(common).sum()),
        "test_overlap_count": int(test_hash.isin(common).sum()),
        "train_duplicate_feature_rows": int(train_feat.duplicated(keep=False).sum()),
        "test_duplicate_feature_rows": int(test_feat.duplicated(keep=False).sum()),
    }


def standardized_mean_difference(a, b) -> float:
    """Train/test standardized mean difference for numeric arrays."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return 0.0
    denom = np.sqrt((np.var(x) + np.var(y)) / 2.0)
    if denom == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / denom)


def id_pattern_features(frame: pd.DataFrame, modulus: int, divisor: int) -> pd.DataFrame:
    """Deterministic ID/row-order pattern features for a chosen modulus/divisor."""
    ids = frame["id"].to_numpy(dtype=int)
    row_idx = np.arange(len(frame), dtype=int)
    return pd.DataFrame({
        f"id_mod_{modulus}": ids % modulus,
        f"id_div_{divisor}": ids // divisor,
        f"row_idx_mod_{modulus}": row_idx % modulus,
        f"row_idx_div_{divisor}": row_idx // divisor,
    }, index=frame.index)


def make_model(kind: str, seed: int):
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=180, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=0.5, random_state=seed)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=8, max_features="sqrt", n_jobs=-1, random_state=seed)
    raise ValueError(f"unknown model kind: {kind}")


def cv_accuracy(X: pd.DataFrame, y: np.ndarray, kind: str = "hgb", seed: int = 42) -> float:
    X = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=int)
    for fold, (tr, va) in enumerate(cv.split(X, y)):
        model = make_model(kind, seed + fold)
        model.fit(X.iloc[tr], y[tr])
        pred[va] = model.predict(X.iloc[va])
    return float(accuracy_score(y, pred))


def train_test_shift(train: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    rows = []
    for col in feature_columns(train):
        if pd.api.types.is_numeric_dtype(train[col]):
            smd = standardized_mean_difference(train[col], test[col])
            rows.append({"feature": col, "smd": smd, "abs_smd": abs(smd)})
    return sorted(rows, key=lambda r: r["abs_smd"], reverse=True)


def nearest_neighbor_summary(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    cols = feature_columns(train)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train[cols])
    test_x = scaler.transform(test[cols])
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(train_x)
    dist, idx = nn.kneighbors(test_x)
    dist = dist[:, 0]
    idx = idx[:, 0]
    nearest_targets = train["target"].to_numpy()[idx]
    return {
        "min_distance": float(np.min(dist)),
        "p01_distance": float(np.quantile(dist, 0.01)),
        "p05_distance": float(np.quantile(dist, 0.05)),
        "median_distance": float(np.median(dist)),
        "nearest_target_counts": {str(k): int(v) for k, v in zip(*np.unique(nearest_targets, return_counts=True))},
        "near_duplicate_threshold_count_lt_1e_minus_6": int(np.sum(dist < 1e-6)),
    }


def evaluate_feature_sets(train: pd.DataFrame) -> list[dict]:
    y = train["target"].to_numpy(dtype=int)
    weekly = [c for c in train.columns if c.startswith("nilai_minggu_")]
    daily = [c for c in train.columns if c.startswith("aktivitas_hari_")]
    scalar = [c for c in feature_columns(train) if c not in weekly + daily]
    sets = {
        "id_only": ["id"],
        "admin_only": ["id", "kelas", "urutan_ujian"],
        "kelas_urutan": ["kelas", "urutan_ujian"],
        "weekly_only": weekly,
        "daily_only": daily,
        "scalar_nonseq": scalar,
        "full_raw": feature_columns(train),
    }
    rows = []
    for name, cols in sets.items():
        rows.append({"feature_set": name, "n_features": len(cols), "hgb_accuracy": cv_accuracy(train[cols], y, "hgb")})
    # ID/row pattern search with small feature frames.
    for modulus in list(range(2, 21)) + [24, 25, 32, 40, 50, 64, 80, 100]:
        divisor = max(1, modulus)
        X = id_pattern_features(train, modulus=modulus, divisor=divisor)
        rows.append({"feature_set": f"id_patterns_m{modulus}", "n_features": int(X.shape[1]), "hgb_accuracy": cv_accuracy(X, y, "hgb")})
        X2 = pd.concat([X, train[["kelas", "urutan_ujian"]]], axis=1)
        rows.append({"feature_set": f"id_patterns_m{modulus}+kelas_urutan", "n_features": int(X2.shape[1]), "hgb_accuracy": cv_accuracy(X2, y, "hgb")})
    return sorted(rows, key=lambda r: r["hgb_accuracy"], reverse=True)


def markdown_report(report: dict) -> str:
    lines = [
        "# Datathon Leakage and Hidden-Pattern Audit",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        "## Safety",
        "",
        "- Read-only audit. No Kaggle submission was made.",
        "- No test labels are inferred or used; this checks structural clues only.",
        "",
        "## Exact overlap / duplicates",
        "",
        "```json",
        json.dumps(report["exact_overlap"], indent=2),
        "```",
        "",
        "## Nearest-neighbor summary",
        "",
        "```json",
        json.dumps(report["nearest_neighbor"], indent=2),
        "```",
        "",
        "## Top train/test shifts",
        "",
        "| feature | SMD |",
        "|---|---:|",
    ]
    for row in report["top_train_test_shift"][:20]:
        lines.append(f"| {row['feature']} | {row['smd']:.4f} |")
    lines += [
        "",
        "## Top local hidden-pattern feature-set checks",
        "",
        "| feature set | n | HGB OOF accuracy |",
        "|---|---:|---:|",
    ]
    for row in report["feature_set_results"][:25]:
        lines.append(f"| {row['feature_set']} | {row['n_features']} | {row['hgb_accuracy']:.6f} |")
    lines += [
        "",
        "## Initial interpretation",
        "",
        report["interpretation"],
    ]
    return "\n".join(lines) + "\n"


def build_interpretation(report: dict) -> str:
    overlap = report["exact_overlap"]
    if overlap["overlap_count"] or overlap["test_duplicate_feature_rows"]:
        return "Exact duplicate structure exists and deserves direct exploitation review before modeling."
    structural_prefixes = ("id_patterns_", "admin_only", "kelas_urutan", "id_only")
    structural_rows = [
        row for row in report["feature_set_results"]
        if row["feature_set"].startswith(structural_prefixes)
    ]
    best_structural = max((row["hgb_accuracy"] for row in structural_rows), default=0.0)
    if best_structural < 0.40:
        return "No obvious row-order/ID/admin shortcut appears in local CV. The 0.82 public score likely requires a deeper feature-generation rule, external clue, or public-LB probing rather than a simple deterministic ID leak."
    return "A simple structural/admin feature set shows notable local signal and should be promoted to a focused candidate experiment."


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    report = {
        "experiment": "leakage_hidden_pattern_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "target_counts": {str(k): int(v) for k, v in train["target"].value_counts().sort_index().items()},
        "exact_overlap": exact_feature_overlap(train, test),
        "nearest_neighbor": nearest_neighbor_summary(train, test),
        "top_train_test_shift": train_test_shift(train, test)[:30],
        "feature_set_results": evaluate_feature_sets(train)[:40],
        "no_kaggle_submission_made": True,
    }
    report["interpretation"] = build_interpretation(report)
    OUT.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    (OUT / "leakage_hidden_pattern_audit.json").write_text(json.dumps(report, indent=2))
    (REPORTS / "leakage_hidden_pattern_audit.md").write_text(markdown_report(report))
    print(json.dumps({
        "exact_overlap": report["exact_overlap"],
        "nearest_neighbor": report["nearest_neighbor"],
        "top_feature_sets": report["feature_set_results"][:10],
        "interpretation": report["interpretation"],
        "json": "outputs/leakage_hidden_pattern_audit.json",
        "markdown": "reports/leakage_hidden_pattern_audit.md",
    }, indent=2))


if __name__ == "__main__":
    main()
