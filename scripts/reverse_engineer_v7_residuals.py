#!/usr/bin/env python3
"""Read-only v7 residual and hidden-generator signal audit.

Uses the cached v7 OOF latent scores from outputs/v8_v7_scores.npz. It does
not train models, generate synthetic rows, alter thresholds, or call Kaggle.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "kaggle"
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "scripts"))
from cv_harness import build_features, quartile_bin  # noqa: E402

NC = 4
BOUNDARIES = np.array([0.25, 0.50, 0.75])


def rank_percentile(values: np.ndarray) -> np.ndarray:
    """Convert values to deterministic mid-rank percentiles in (0, 1)."""
    ranks = pd.Series(np.asarray(values, dtype=float)).rank(method="average").to_numpy()
    return (ranks - 0.5) / len(ranks)


def target_midpoints(target: np.ndarray) -> np.ndarray:
    """Map ordered labels 0..3 to their class-bin midpoints."""
    return (np.asarray(target, dtype=float) + 0.5) / NC


def boundary_distance(score_pct: np.ndarray) -> np.ndarray:
    """Distance from each score percentile to its nearest quartile boundary."""
    values = np.asarray(score_pct, dtype=float)
    return np.min(np.abs(values[:, None] - BOUNDARIES[None, :]), axis=1)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Small dependency-light Spearman correlation with finite filtering."""
    a = pd.Series(np.asarray(x, dtype=float))
    b = pd.Series(np.asarray(y, dtype=float))
    mask = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or a[mask].nunique() < 2 or b[mask].nunique() < 2:
        return 0.0
    return float(a[mask].rank(method="average").corr(b[mask].rank(method="average")))


def feature_table(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build target-free raw + v7 feature tables with unique names."""
    raw_cols = [c for c in train.columns if c not in {"id", "target"}]
    raw_train = train[raw_cols].copy().add_prefix("raw__")
    raw_test = test[raw_cols].copy().add_prefix("raw__")
    v7_train = pd.DataFrame(__import__("run_production_v7", fromlist=["feats"]).feats(train), index=train.index)
    v7_test = pd.DataFrame(__import__("run_production_v7", fromlist=["feats"]).feats(test), index=test.index)
    v7_train.columns = [f"v7__{c}" for c in v7_train.columns]
    v7_test.columns = list(v7_train.columns)
    engineered_train = build_features(train, "full")
    engineered_test = build_features(test, "full")
    engineered_train.columns = [f"fe__{c}" for c in engineered_train.columns]
    engineered_test.columns = list(engineered_train.columns)
    return (
        pd.concat([raw_train, engineered_train, v7_train], axis=1),
        pd.concat([raw_test, engineered_test, v7_test], axis=1),
    )


def audit_feature(name: str, train_values: np.ndarray, test_values: np.ndarray,
                  residuals: np.ndarray, abs_residuals: np.ndarray,
                  errors: np.ndarray, per_seed_residuals: list[np.ndarray]) -> dict:
    values = np.asarray(train_values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return {"feature": name, "valid": False}
    vals = np.nan_to_num(values, nan=float(np.nanmedian(values[finite])))
    test_vals = np.asarray(test_values, dtype=float)
    test_vals = np.nan_to_num(test_vals, nan=float(np.nanmedian(vals)))
    std = float(np.std(vals))
    smd = float((np.mean(vals) - np.mean(test_vals)) / (np.sqrt((np.var(vals) + np.var(test_vals)) / 2) + 1e-12))
    per_corr = np.array([spearman(vals, r) for r in per_seed_residuals])
    signed = spearman(vals, residuals)
    absolute = spearman(vals, abs_residuals)
    error_corr = spearman(vals, errors.astype(float))
    return {
        "feature": name,
        "valid": True,
        "signed_residual_spearman": signed,
        "absolute_residual_spearman": absolute,
        "error_spearman": error_corr,
        "per_seed_signed_mean": float(per_corr.mean()),
        "per_seed_signed_std": float(per_corr.std()),
        "per_seed_signed_sign_agreement": float(np.mean(np.sign(per_corr) == np.sign(per_corr.mean()))) if per_corr.mean() else 0.0,
        "train_test_smd": smd,
        "train_std": std,
        "test_std": float(np.std(test_vals)),
        "shift_risk": "high" if abs(smd) >= 0.5 else ("medium" if abs(smd) >= 0.25 else "low"),
    }


def markdown_report(report: dict) -> str:
    lines = [
        "# v7 Residual Signal Audit",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        "## Safety",
        "",
        "- Read-only audit; no model fitting, synthetic generation, threshold changes, or Kaggle calls.",
        "- OOF scores came from `outputs/v8_v7_scores.npz`.",
        "- Features are target-free; labels are used only for residual/error analysis.",
        "",
        "## v7 OOF summary",
        "",
        f"- Seeds: `{report['seeds']}`",
        f"- Per-seed accuracy: `{report['per_seed_accuracy']}`",
        f"- Mean accuracy: `{report['mean_accuracy']:.6f}`",
        f"- Mean-score accuracy: `{report['mean_score_accuracy']:.6f}`",
        f"- Mean boundary distance: `{report['mean_boundary_distance']:.6f}`",
        "",
        "### Confusion matrix",
        "",
        "```text",
        json.dumps(report["confusion_matrix"]),
        "```",
        "",
        "## Top stable residual signals",
        "",
        "| feature | signed rho | abs rho | error rho | seed sign agreement | train/test SMD | risk |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["top_stable_signals"][:20]:
        lines.append(f"| {row['feature']} | {row['signed_residual_spearman']:.4f} | {row['absolute_residual_spearman']:.4f} | {row['error_spearman']:.4f} | {row['per_seed_signed_sign_agreement']:.2f} | {row['train_test_smd']:.3f} | {row['shift_risk']} |")
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- A residual correlation is a discovery lead, not proof of causal signal.",
        "- Promote only features with stable direction across seeds and low train/test shift.",
        "- Any next model experiment must preserve v7's exact 200/200/200/200 submission balancing.",
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="outputs/v8_v7_scores.npz")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    cache_path = ROOT / args.cache
    if not cache_path.exists():
        raise SystemExit(f"missing score cache: {cache_path}")
    cached = np.load(cache_path, allow_pickle=False)
    oof = np.asarray(cached["oof"], dtype=float)
    seeds = [int(v) for v in cached["seeds"]]
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["target"].to_numpy(dtype=int)
    if oof.shape != (len(seeds), len(train)):
        raise SystemExit(f"unexpected OOF shape {oof.shape}; expected {(len(seeds), len(train))}")

    score_pct = np.vstack([rank_percentile(row) for row in oof])
    per_seed_pred = np.vstack([quartile_bin(row) for row in oof])
    per_seed_accuracy = [float(np.mean(pred == y)) for pred in per_seed_pred]
    mean_pct = score_pct.mean(axis=0)
    mean_pred = quartile_bin(mean_pct)
    target_pct = target_midpoints(y)
    residual = target_pct - mean_pct
    abs_residual = np.abs(residual)
    errors = mean_pred != y
    distance = boundary_distance(mean_pct)

    X_train, X_test = feature_table(train, test)
    rows = []
    per_seed_residuals = [target_pct - score_pct[i] for i in range(len(seeds))]
    for col in X_train.columns:
        rows.append(audit_feature(col, X_train[col].to_numpy(), X_test[col].to_numpy(), residual, abs_residual, errors, per_seed_residuals))
    rows = [r for r in rows if r.get("valid")]

    def ranked(key, reverse=True):
        return sorted(rows, key=lambda r: abs(r[key]), reverse=reverse)[: args.top]

    stable = [r for r in rows if r["per_seed_signed_sign_agreement"] >= 0.8 and r["shift_risk"] == "low"]
    stable = sorted(stable, key=lambda r: abs(r["per_seed_signed_mean"]), reverse=True)[: args.top]
    conf = confusion_matrix(y, mean_pred, labels=list(range(NC))).tolist()
    report = {
        "experiment": "v7_residual_signal_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cache": str(cache_path.relative_to(ROOT)),
        "seeds": seeds,
        "n_rows": len(train),
        "n_features_audited": len(rows),
        "per_seed_accuracy": [round(v, 6) for v in per_seed_accuracy],
        "mean_accuracy": float(np.mean(per_seed_accuracy)),
        "mean_score_accuracy": float(np.mean(mean_pred == y)),
        "mean_boundary_distance": float(np.mean(distance)),
        "error_rate": float(np.mean(errors)),
        "confusion_matrix": conf,
        "error_counts_by_true_class": {str(k): int(np.sum(errors & (y == k))) for k in range(NC)},
        "error_counts_by_pred_class": {str(k): int(np.sum(errors & (mean_pred == k))) for k in range(NC)},
        "top_signed_residual": ranked("signed_residual_spearman"),
        "top_absolute_residual": ranked("absolute_residual_spearman"),
        "top_error_association": ranked("error_spearman"),
        "top_stable_signals": stable,
        "no_model_fitting": True,
        "no_kaggle_submission_made": True,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v7_residual_signal_audit.json").write_text(json.dumps(report, indent=2))
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "v7_residual_signal_audit.md").write_text(markdown_report(report))
    print(json.dumps({
        "mean_accuracy": report["mean_accuracy"],
        "error_rate": report["error_rate"],
        "top_stable_signals": [r["feature"] for r in stable[:10]],
        "json": "outputs/v7_residual_signal_audit.json",
        "markdown": "reports/v7_residual_signal_audit.md",
    }, indent=2))


if __name__ == "__main__":
    main()
