#!/usr/bin/env python3
"""Search stable symbolic formulas that could add signal beyond v7.

This is a read-only, no-submission audit. It screens target-free candidate
formulae against cached five-seed v7 OOF/test latent scores. Candidate rank
blends are scored with leave-one-seed-out transfer rather than a single
in-sample alpha choice.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "kaggle"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
WEEKLY = [f"nilai_minggu_{i:02d}" for i in range(1, 13)]
DAILY = [f"aktivitas_hari_{i:02d}" for i in range(1, 17)]
ALPHAS = np.array([-0.80, -0.60, -0.40, -0.25, -0.15, -0.08, 0.0, 0.08, 0.15, 0.25, 0.40, 0.60, 0.80])


def zscore(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    sd = x.std()
    return np.zeros_like(x) if sd < 1e-12 else (x - x.mean()) / sd


def quartile_labels(scores) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    ranks = rankdata(scores, method="ordinal") - 1
    return np.minimum(3, (ranks * 4 // n).astype(int))


def quartile_accuracy(scores, y) -> float:
    return float(np.mean(quartile_labels(scores) == np.asarray(y, dtype=int)))


def rank_residual_correlation(feature, y, baseline) -> float:
    fr = rankdata(np.asarray(feature, dtype=float))
    yr = rankdata(np.asarray(y, dtype=float))
    br = rankdata(np.asarray(baseline, dtype=float))
    f_res = fr - np.polyval(np.polyfit(br, fr, 1), br)
    y_res = yr - np.polyval(np.polyfit(br, yr, 1), br)
    if f_res.std() < 1e-12 or y_res.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(f_res, y_res)[0, 1])


def _safe_corr_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    Ac = A - A.mean(axis=1, keepdims=True)
    Bc = B - B.mean(axis=1, keepdims=True)
    denom = np.sqrt((Ac * Ac).sum(axis=1) * (Bc * Bc).sum(axis=1))
    return np.divide((Ac * Bc).sum(axis=1), denom, out=np.zeros(len(A)), where=denom > 1e-12)


def _fft_energy(M: np.ndarray, period: float) -> np.ndarray:
    n = M.shape[1]
    t = np.arange(n, dtype=float)
    centered = M - M.mean(axis=1, keepdims=True)
    w = 2 * np.pi / period
    c = centered @ np.cos(w * t)
    s = centered @ np.sin(w * t)
    return (c * c + s * s) / n


def _slope(M: np.ndarray) -> np.ndarray:
    t = np.arange(M.shape[1], dtype=float)
    tc = t - t.mean()
    centered = M - M.mean(axis=1, keepdims=True)
    return (centered @ tc) / (tc @ tc)


def build_symbolic_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Target-free, row-local candidate hidden-generator formula bank."""
    W = frame[WEEKLY].to_numpy(float)
    D = frame[DAILY].to_numpy(float)
    completion = (frame["tugas_selesai"].to_numpy(float) / np.maximum(frame["tugas_diberikan"].to_numpy(float), 1.0))
    weekly_std = W.std(axis=1)
    weekly_range = W.max(axis=1) - W.min(axis=1)
    weekly_slope = _slope(W)
    daily_std = D.std(axis=1)
    daily_slope = _slope(D)
    Wc = W - W.mean(axis=1, keepdims=True)
    Dc = D - D.mean(axis=1, keepdims=True)
    # Align daily/weekly trajectories to twelve positions for a shape coupling measure.
    d_grid = np.linspace(0, 1, D.shape[1])
    w_grid = np.linspace(0, 1, W.shape[1])
    D12 = np.array([np.interp(w_grid, d_grid, row) for row in Dc])
    shape_correlation = _safe_corr_rows(Wc, D12)
    motivation = frame["skor_motivasi"].to_numpy(float)
    discipline = frame["skor_kedisiplinan"].to_numpy(float)
    tryout = frame["skor_tryout"].to_numpy(float)
    attendance = frame["indeks_kehadiran"].to_numpy(float)
    literacy = frame["skor_literasi"].to_numpy(float)
    interest = frame["skor_minat_belajar"].to_numpy(float)
    exam_order = frame["urutan_ujian"].to_numpy(float)
    kelas = frame["kelas"].to_numpy(float)

    candidates = {
        # Completion × temporal regime.
        "compl_x_wk_vol": completion * weekly_std,
        "compl_x_wk_range": completion * weekly_range,
        "compl_x_wk_slope": completion * weekly_slope,
        "compl_x_daily_vol": completion * daily_std,
        "compl_x_daily_slope": completion * daily_slope,
        "compl_x_shape_correlation": completion * shape_correlation,
        # Periodic daily and weekly energy, including the suspected short rhythms.
        "daily_period2_energy": _fft_energy(D, 2.0),
        "daily_period3_energy": _fft_energy(D, 3.0),
        "daily_period4_energy": _fft_energy(D, 4.0),
        "daily_period5_energy": _fft_energy(D, 5.0),
        "weekly_period2_energy": _fft_energy(W, 2.0),
        "weekly_period3_energy": _fft_energy(W, 3.0),
        "weekly_period4_energy": _fft_energy(W, 4.0),
        # Shape / regime changes.
        "weekly_late_minus_early": W[:, -4:].mean(axis=1) - W[:, :4].mean(axis=1),
        "daily_late_minus_early": D[:, -4:].mean(axis=1) - D[:, :4].mean(axis=1),
        "weekly_curvature": W[:, -4:].mean(axis=1) - 2 * W[:, 4:8].mean(axis=1) + W[:, :4].mean(axis=1),
        "daily_curvature": D[:, -4:].mean(axis=1) - 2 * D[:, 6:10].mean(axis=1) + D[:, :6].mean(axis=1),
        "daily_weekly_shape_corr": shape_correlation,
        "daily_weekly_slope_product": daily_slope * weekly_slope,
        "daily_weekly_vol_ratio": daily_std / (weekly_std + 1e-6),
        # Behavioral score formulae not present as exactly these forms in v7.
        "motivation_discipline_balance": (motivation - discipline) / (np.abs(motivation) + np.abs(discipline) + 1e-6),
        "motivation_discipline_harmonic": 2 * motivation * discipline / (motivation + discipline + 1e-6),
        "motivation_discipline_x_compl": motivation * discipline * completion,
        "interest_x_attendance_x_compl": interest * attendance * completion,
        "literacy_x_tryout": literacy * tryout,
        "tryout_x_wk_slope": tryout * weekly_slope,
        "tryout_x_wk_vol": tryout * weekly_std,
        "tryout_x_daily_period3": tryout * _fft_energy(D, 3.0),
        # Administrative variables only as interactions, never standalone shortcut claims.
        "exam_order_x_tryout": exam_order * tryout,
        "exam_order_x_completion": exam_order * completion,
        "kelas_x_tryout": kelas * tryout,
        "kelas_x_weekly_vol": kelas * weekly_std,
    }
    return pd.DataFrame(candidates, index=frame.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def smd(train_values, test_values) -> float:
    a = np.asarray(train_values, dtype=float)
    b = np.asarray(test_values, dtype=float)
    denom = np.sqrt((a.var() + b.var()) / 2)
    return 0.0 if denom < 1e-12 else float((a.mean() - b.mean()) / denom)


def cross_seed_transfer(oof: np.ndarray, y: np.ndarray, feature: np.ndarray) -> dict:
    """Choose feature blend alpha on 4 v7 seeds, evaluate it on the held seed."""
    per_seed = []
    base_acc = []
    feature_z = zscore(feature)
    for hold in range(oof.shape[0]):
        train_idx = [i for i in range(oof.shape[0]) if i != hold]
        selection_scores = np.mean([zscore(oof[i]) for i in train_idx], axis=0)
        candidates = [quartile_accuracy(selection_scores + a * feature_z, y) for a in ALPHAS]
        alpha = float(ALPHAS[int(np.argmax(candidates))])
        held_scores = zscore(oof[hold])
        base_acc.append(quartile_accuracy(held_scores, y))
        per_seed.append({"held_seed_index": hold, "alpha": alpha, "accuracy": quartile_accuracy(held_scores + alpha * feature_z, y)})
    return {
        "baseline_mean": float(np.mean(base_acc)),
        "transfer_mean": float(np.mean([r["accuracy"] for r in per_seed])),
        "gain": float(np.mean([r["accuracy"] for r in per_seed]) - np.mean(base_acc)),
        "per_seed": per_seed,
    }


def report_markdown(result: dict) -> str:
    lines = [
        "# Datathon Symbolic Hidden-Rule Search",
        "",
        f"Generated: `{result['generated_at']}`",
        "",
        "## Safety",
        "",
        "- Target-free formula screening only.",
        "- Uses cached v7 OOF/test scores; no model retraining and no Kaggle submission.",
        "- Cross-seed alpha transfer is diagnostic only and does not authorize a submission.",
        "",
        f"Baseline mean v7 OOF quartile accuracy: `{result['baseline_mean_accuracy']:.6f}`",
        "",
        "## Top stable symbolic candidates",
        "",
        "| feature | transfer gain | transfer accuracy | residual corr | train/test SMD | best in-sample alpha |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"][:25]:
        lines.append(
            f"| {row['feature']} | {row['transfer_gain']:+.6f} | {row['transfer_accuracy']:.6f} | "
            f"{row['residual_correlation']:+.4f} | {row['smd']:+.4f} | {row['best_in_sample_alpha']:+.2f} |"
        )
    lines += ["", "## Interpretation", "", result["interpretation"], ""]
    return "\n".join(lines)


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["target"].to_numpy(int)
    cache = np.load(OUT / "v8_v7_scores.npz")
    oof = cache["oof"]
    test_scores = cache["test"]
    train_features = build_symbolic_features(train)
    test_features = build_symbolic_features(test)
    baseline = np.mean([zscore(s) for s in oof], axis=0)
    baseline_acc = quartile_accuracy(baseline, y)
    rows = []
    for col in train_features.columns:
        feature = train_features[col].to_numpy(float)
        transfer = cross_seed_transfer(oof, y, feature)
        in_sample = [quartile_accuracy(baseline + alpha * zscore(feature), y) for alpha in ALPHAS]
        best_idx = int(np.argmax(in_sample))
        rows.append({
            "feature": col,
            "residual_correlation": rank_residual_correlation(feature, y, baseline),
            "smd": smd(feature, test_features[col].to_numpy(float)),
            "best_in_sample_accuracy": float(in_sample[best_idx]),
            "best_in_sample_alpha": float(ALPHAS[best_idx]),
            "transfer_accuracy": transfer["transfer_mean"],
            "transfer_gain": transfer["gain"],
            "transfer_per_seed": transfer["per_seed"],
        })
    rows.sort(key=lambda r: (r["transfer_gain"], -abs(r["smd"])), reverse=True)
    stable = [r for r in rows if abs(r["smd"]) < 0.10 and r["transfer_gain"] >= 0.001]
    if stable:
        interpretation = (
            "At least one low-shift symbolic formula has a positive cross-seed diagnostic gain. "
            "It merits a separate, full nested-CV model-addition experiment before any Kaggle candidate is considered."
        )
    else:
        interpretation = (
            "No low-shift symbolic formula produced a meaningful cross-seed transfer gain. "
            "Do not create or submit a candidate from this screen; the 0.82 leaderboard route remains unexplained."
        )
    result = {
        "experiment": "symbolic_hidden_rule_search",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cache": "outputs/v8_v7_scores.npz",
        "seeds": cache["seeds"].tolist(),
        "baseline_mean_accuracy": baseline_acc,
        "alpha_grid": ALPHAS.tolist(),
        "rows": rows,
        "stable_positive_candidates": [r["feature"] for r in stable],
        "interpretation": interpretation,
        "no_kaggle_submission_made": True,
    }
    OUT.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    (OUT / "symbolic_hidden_rule_search.json").write_text(json.dumps(result, indent=2))
    (REPORTS / "symbolic_hidden_rule_search.md").write_text(report_markdown(result))
    print(json.dumps({
        "baseline_mean_accuracy": baseline_acc,
        "top_10": rows[:10],
        "stable_positive_candidates": result["stable_positive_candidates"],
        "interpretation": interpretation,
    }, indent=2))


if __name__ == "__main__":
    main()
