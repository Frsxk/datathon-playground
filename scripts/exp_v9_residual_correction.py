#!/usr/bin/env python3
"""Fold-safe v9 residual/latent-score correction experiment.

This script reuses cached v7 OOF/test latent scores, learns a small correction
model for ordered residuals in nested OOF folds, and writes exact-balanced
candidate submissions. No Kaggle submission is performed.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "kaggle"
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "scripts"))
from cv_harness import quartile_bin  # noqa: E402
from run_production_v7 import feats as v7_feats  # noqa: E402
from reverse_engineer_v7_residuals import rank_percentile, target_midpoints  # noqa: E402

NC = 4
DEFAULT_ALPHAS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
SELECTED_FEATURES = [
    "abschg_x_compl",
    "vol_x_compl",
    "range_x_compl",
    "minggu_std",
    "minggu_iqr",
    "minggu_range",
    "abs_change_mean",
    "abs_change_sum",
    "cum_std",
    "cum_range",
    "tryout_x_compl",
    "compl_ratio",
    "minggu_min",
    "day_ac_l1",
    "day_ac_l2",
    "day_ac_l4",
    "day_ac_l6",
    "wk_ac_l1",
    "wk_ac_l2",
    "wk_ac_l4",
    "wk_ac_l6",
    "wk_ar1_ols",
    "wk_pacf1",
    "urutan_ujian",
]


def residual_targets(y: np.ndarray, score_pct: np.ndarray) -> np.ndarray:
    """Residual target in percentile space."""
    return target_midpoints(np.asarray(y, dtype=int)) - np.asarray(score_pct, dtype=float)


def apply_residual_correction(base_pct: np.ndarray, correction: np.ndarray, alpha: float) -> np.ndarray:
    """Apply a shrunken correction and keep scores in [0, 1]."""
    return np.clip(np.asarray(base_pct, dtype=float) + float(alpha) * np.asarray(correction, dtype=float), 0.0, 1.0)


def select_best_config(rows: list[dict]) -> dict:
    """Select highest mean accuracy, then lower std, then smaller alpha, then ridge."""
    if not rows:
        raise ValueError("no rows to select")
    method_order = {"ridge": 0, "hgb": 1}
    return sorted(
        rows,
        key=lambda r: (-r["mean_accuracy"], r["std_accuracy"], abs(r["alpha"]), method_order.get(r["method"], 99)),
    )[0]


def correction_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Target-free correction feature matrix based on residual audit leads."""
    X = v7_feats(frame).copy()
    keep = [c for c in SELECTED_FEATURES if c in X.columns]
    if not keep:
        raise ValueError("no selected correction features were found")
    out = X[keep].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def make_model(method: str, seed: int):
    if method == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if method == "hgb":
        return HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=15, learning_rate=0.04,
                                             l2_regularization=1.0, random_state=seed)
    raise ValueError(f"unknown method: {method}")


def nested_correction_oof(X: pd.DataFrame, y: np.ndarray, residual: np.ndarray, seed: int, method: str) -> np.ndarray:
    """OOF correction predictions for one v7 seed."""
    pred = np.zeros(len(y), dtype=float)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed + 10_000)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        model = make_model(method, seed + fold * 101)
        model.fit(X.iloc[tr_idx], residual[tr_idx])
        pred[va_idx] = model.predict(X.iloc[va_idx])
    return pred


def evaluate_methods(oof_scores: np.ndarray, y: np.ndarray, X: pd.DataFrame, seeds: list[int], methods: list[str], alphas: list[float]) -> tuple[list[dict], dict]:
    per_seed_payload = {}
    rows = []
    for method in methods:
        per_seed_payload[method] = []
        for si, seed in enumerate(seeds):
            base_pct = rank_percentile(oof_scores[si])
            residual = residual_targets(y, base_pct)
            corr = nested_correction_oof(X, y, residual, seed, method)
            per_seed_payload[method].append({
                "seed": seed,
                "base_pct": base_pct,
                "correction": corr,
                "baseline_accuracy": float(accuracy_score(y, quartile_bin(base_pct))),
            })
        for alpha in alphas:
            accs = []
            for payload in per_seed_payload[method]:
                corrected = apply_residual_correction(payload["base_pct"], payload["correction"], alpha)
                accs.append(float(accuracy_score(y, quartile_bin(corrected))))
            rows.append({
                "method": method,
                "alpha": float(alpha),
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "per_seed_accuracy": [round(v, 6) for v in accs],
            })
    return rows, per_seed_payload


def fit_full_test_correction(X: pd.DataFrame, Xtest: pd.DataFrame, y: np.ndarray, base_pct: np.ndarray, method: str, seed: int) -> np.ndarray:
    residual = residual_targets(y, base_pct)
    model = make_model(method, seed + 50_000)
    model.fit(X, residual)
    return model.predict(Xtest)


def write_candidate(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame, cached: dict, best: dict, X: pd.DataFrame, Xtest: pd.DataFrame, seeds: list[int]) -> dict:
    test_scores = np.asarray(cached["test"], dtype=float)
    oof_scores = np.asarray(cached["oof"], dtype=float)
    corrected_test_rows = []
    for si, seed in enumerate(seeds):
        base_train_pct = rank_percentile(oof_scores[si])
        base_test_pct = rank_percentile(test_scores[si])
        corr_test = fit_full_test_correction(X, Xtest, train["target"].to_numpy(dtype=int), base_train_pct, best["method"], seed)
        corrected_test_rows.append(apply_residual_correction(base_test_pct, corr_test, best["alpha"]))
    final_score = np.mean(corrected_test_rows, axis=0)
    pred = quartile_bin(final_score).astype(int)
    sub = pd.DataFrame({"id": test["id"], "target": pred})
    if list(sub.columns) != ["id", "target"] or sub.shape != sample.shape or not sub["id"].equals(sample["id"]):
        raise AssertionError("candidate submission format mismatch")
    if not set(sub["target"]).issubset({0, 1, 2, 3}):
        raise AssertionError("candidate contains invalid labels")
    suffix = f"{best['method']}_a{str(best['alpha']).replace('.', 'p')}"
    path = OUT / f"submission_v9_residual_{suffix}.csv"
    sub.to_csv(path, index=False)
    v7_path = OUT / "submission_v7.csv"
    if v7_path.exists():
        v7 = pd.read_csv(v7_path)
        diff = (v7["target"].to_numpy(dtype=int) != pred).sum()
        adjacent = (np.abs(v7["target"].to_numpy(dtype=int) - pred) == 1).sum()
    else:
        diff = None
        adjacent = None
    return {
        "path": str(path.relative_to(ROOT)),
        "prediction_counts": {str(k): int(v) for k, v in sub["target"].value_counts().sort_index().items()},
        "changed_vs_v7": None if diff is None else int(diff),
        "adjacent_changes_vs_v7": None if adjacent is None else int(adjacent),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", nargs="+", default=["ridge", "hgb"])
    ap.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    ap.add_argument("--cache", default="outputs/v8_v7_scores.npz")
    args = ap.parse_args()
    cache_path = ROOT / args.cache
    cached_np = np.load(cache_path, allow_pickle=False)
    cached = {"oof": cached_np["oof"], "test": cached_np["test"], "seeds": cached_np["seeds"]}
    seeds = [int(v) for v in cached["seeds"]]
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    y = train["target"].to_numpy(dtype=int)
    X = correction_features(train)
    Xtest = correction_features(test)
    if list(X.columns) != list(Xtest.columns):
        raise SystemExit("train/test correction features differ")

    rows, payload = evaluate_methods(cached["oof"], y, X, seeds, args.methods, list(args.alphas))
    best = select_best_config(rows)
    baseline = [r for r in rows if r["alpha"] == 0.0][0]
    candidate = write_candidate(train, test, sample, cached, best, X, Xtest, seeds) if best["alpha"] > 0 else None
    report = {
        "experiment": "v9_fold_safe_residual_correction",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cache": str(cache_path.relative_to(ROOT)),
        "seeds": seeds,
        "methods": args.methods,
        "alphas": list(args.alphas),
        "features": list(X.columns),
        "baseline": baseline,
        "best": best,
        "improvement_over_baseline": float(best["mean_accuracy"] - baseline["mean_accuracy"]),
        "results": rows,
        "candidate": candidate,
        "selection_rule": "highest nested-OOF mean accuracy; lower std and smaller alpha break ties",
        "exact_quartile_balancing": True,
        "no_kaggle_submission_made": True,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "exp_v9_residual_correction.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "baseline": baseline,
        "best": best,
        "improvement": report["improvement_over_baseline"],
        "candidate": candidate,
        "report": "outputs/exp_v9_residual_correction.json",
    }, indent=2))


if __name__ == "__main__":
    main()
