#!/usr/bin/env python3
"""Advanced generator-fingerprint audit for Datathon Playground.

Read-only diagnostics for hidden data-generation clues beyond simple duplicates,
ID modulo shortcuts, threshold shifts, or one-off symbolic blends. No Kaggle
submission and no test labels are inferred.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "kaggle"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
WEEKLY = [f"nilai_minggu_{i:02d}" for i in range(1, 13)]
DAILY = [f"aktivitas_hari_{i:02d}" for i in range(1, 17)]


def decimal_places(series: pd.Series) -> int:
    """Maximum number of visible decimal places in a numeric series."""
    max_places = 0
    for value in pd.Series(series).dropna().head(5000):
        x = float(value)
        if abs(x - round(x)) < 1e-10:
            places = 0
        else:
            text = f"{x:.10f}".rstrip("0").rstrip(".")
            places = len(text.split(".")[1]) if "." in text else 0
        max_places = max(max_places, places)
    return int(max_places)


def quartile_labels(values: np.ndarray) -> np.ndarray:
    order = pd.Series(values).rank(method="first").to_numpy() - 1
    return np.minimum(3, (order * 4 // len(order)).astype(int))


def monotone_quartile_accuracy(values: np.ndarray, y: np.ndarray) -> float:
    labels = quartile_labels(np.asarray(values, dtype=float))
    return float(max(accuracy_score(y, labels), accuracy_score(y, 3 - labels)))


def cluster_label_purity(clusters: np.ndarray, y: np.ndarray) -> dict:
    clusters = np.asarray(clusters)
    y = np.asarray(y)
    total_majority = 0
    cluster_rows = []
    for cl in sorted(np.unique(clusters)):
        mask = clusters == cl
        counts = Counter(y[mask].tolist())
        majority_label, majority_count = counts.most_common(1)[0]
        total_majority += majority_count
        cluster_rows.append({
            "cluster": int(cl),
            "size": int(mask.sum()),
            "majority_label": int(majority_label),
            "purity": float(majority_count / mask.sum()),
            "counts": {str(k): int(v) for k, v in sorted(counts.items())},
        })
    return {
        "n_clusters": int(len(np.unique(clusters))),
        "weighted_purity": float(total_majority / len(y)),
        "clusters": cluster_rows,
    }


def _slope(M: np.ndarray) -> np.ndarray:
    t = np.arange(M.shape[1], dtype=float)
    t = t - t.mean()
    return ((M - M.mean(axis=1, keepdims=True)) @ t) / (t @ t)


def _fft_energy(M: np.ndarray, period: float) -> np.ndarray:
    t = np.arange(M.shape[1], dtype=float)
    centered = M - M.mean(axis=1, keepdims=True)
    w = 2 * np.pi / period
    return ((centered @ np.cos(w * t)) ** 2 + (centered @ np.sin(w * t)) ** 2) / M.shape[1]


def _longest_run(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape[0], dtype=float)
    for i, row in enumerate(mask):
        best = cur = 0
        for v in row:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        out[i] = best
    return out


def generator_parameter_features(frame: pd.DataFrame) -> pd.DataFrame:
    W = frame[WEEKLY].to_numpy(float)
    D = frame[DAILY].to_numpy(float)
    completion = frame["tugas_selesai"].to_numpy(float) / np.maximum(frame["tugas_diberikan"].to_numpy(float), 1.0)
    p_daily = np.abs(D) / (np.abs(D).sum(axis=1, keepdims=True) + 1e-9)
    p_weekly = np.abs(W) / (np.abs(W).sum(axis=1, keepdims=True) + 1e-9)
    feats = pd.DataFrame({
        "wk_mean": W.mean(axis=1),
        "wk_std": W.std(axis=1),
        "wk_range": W.max(axis=1) - W.min(axis=1),
        "wk_slope": _slope(W),
        "wk_curvature": W[:, -4:].mean(axis=1) - 2 * W[:, 4:8].mean(axis=1) + W[:, :4].mean(axis=1),
        "wk_longest_positive_run": _longest_run(W > 0),
        "wk_longest_negative_run": _longest_run(W < 0),
        "wk_entropy": -np.sum(p_weekly * np.log(p_weekly + 1e-9), axis=1),
        "wk_period2": _fft_energy(W, 2.0),
        "wk_period3": _fft_energy(W, 3.0),
        "wk_period4": _fft_energy(W, 4.0),
        "day_mean": D.mean(axis=1),
        "day_std": D.std(axis=1),
        "day_range": D.max(axis=1) - D.min(axis=1),
        "day_slope": _slope(D),
        "day_curvature": D[:, -4:].mean(axis=1) - 2 * D[:, 6:10].mean(axis=1) + D[:, :6].mean(axis=1),
        "day_longest_high_run": _longest_run(D > np.median(D, axis=1, keepdims=True)),
        "day_entropy": -np.sum(p_daily * np.log(p_daily + 1e-9), axis=1),
        "day_period2": _fft_energy(D, 2.0),
        "day_period3": _fft_energy(D, 3.0),
        "day_period4": _fft_energy(D, 4.0),
        "day_period5": _fft_energy(D, 5.0),
        "completion_ratio": completion,
        "completion_remaining": frame["tugas_diberikan"].to_numpy(float) - frame["tugas_selesai"].to_numpy(float),
        "motiv_x_disc": frame["skor_motivasi"].to_numpy(float) * frame["skor_kedisiplinan"].to_numpy(float),
        "tryout_x_completion": frame["skor_tryout"].to_numpy(float) * completion,
        "attendance_x_completion": frame["indeks_kehadiran"].to_numpy(float) * completion,
        "literacy_x_interest": frame["skor_literasi"].to_numpy(float) * frame["skor_minat_belajar"].to_numpy(float),
        "exam_order_x_completion": frame["urutan_ujian"].to_numpy(float) * completion,
    }, index=frame.index)
    return feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def quantization_report(train: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    rows = []
    for col in [c for c in train.columns if c != "target"]:
        combined = pd.concat([train[col], test[col]], ignore_index=True)
        values = np.sort(combined.dropna().unique().astype(float))
        diffs = np.diff(values)
        positive = diffs[diffs > 1e-12]
        rows.append({
            "feature": col,
            "unique_train": int(train[col].nunique()),
            "unique_test": int(test[col].nunique()),
            "unique_combined": int(len(values)),
            "decimal_places": decimal_places(combined),
            "min_positive_step": float(positive.min()) if len(positive) else 0.0,
            "is_integer_grid": bool(np.all(np.abs(values - np.round(values)) < 1e-10)),
        })
    return sorted(rows, key=lambda r: (r["decimal_places"], -r["unique_combined"]))


def single_feature_screen(features: pd.DataFrame, y: np.ndarray) -> list[dict]:
    rows = []
    for col in features.columns:
        values = features[col].to_numpy(float)
        rho = spearmanr(values, y).correlation
        if not np.isfinite(rho):
            rho = 0.0
        rows.append({
            "feature": col,
            "spearman": float(rho),
            "quartile_accuracy_best_orientation": monotone_quartile_accuracy(values, y),
        })
    return sorted(rows, key=lambda r: r["quartile_accuracy_best_orientation"], reverse=True)


def cluster_majority_cv(features: pd.DataFrame, y: np.ndarray, k: int, seed: int = 42) -> float:
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    preds = np.zeros(len(y), dtype=int)
    for tr_idx, va_idx in cv.split(features, y):
        scaler = StandardScaler().fit(features.iloc[tr_idx])
        xtr = scaler.transform(features.iloc[tr_idx])
        xva = scaler.transform(features.iloc[va_idx])
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        tr_clusters = km.fit_predict(xtr)
        va_clusters = km.predict(xva)
        global_majority = Counter(y[tr_idx].tolist()).most_common(1)[0][0]
        mapping = {}
        for cl in range(k):
            mask = tr_clusters == cl
            if mask.any():
                mapping[cl] = Counter(y[tr_idx][mask].tolist()).most_common(1)[0][0]
            else:
                mapping[cl] = global_majority
        preds[va_idx] = [mapping[int(cl)] for cl in va_clusters]
    return float(accuracy_score(y, preds))


def cluster_screen(raw: pd.DataFrame, params: pd.DataFrame, y: np.ndarray) -> list[dict]:
    rows = []
    reps = {"raw": raw, "generator_params": params}
    for rep_name, X in reps.items():
        Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
        for transform, Xt in [("none", Xs), ("pca8", pd.DataFrame(PCA(n_components=min(8, Xs.shape[1]), random_state=42).fit_transform(Xs)) )]:
            for k in [4, 8, 16, 32, 64]:
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                full_clusters = km.fit_predict(Xt)
                purity = cluster_label_purity(full_clusters, y)
                cv_acc = cluster_majority_cv(Xt, y, k)
                rows.append({
                    "representation": rep_name,
                    "transform": transform,
                    "k": k,
                    "full_train_weighted_purity": purity["weighted_purity"],
                    "cv_majority_accuracy": cv_acc,
                })
    return sorted(rows, key=lambda r: r["cv_majority_accuracy"], reverse=True)


def markdown(result: dict) -> str:
    lines = [
        "# Advanced Generator Fingerprint Audit",
        "",
        f"Generated: `{result['generated_at']}`",
        "",
        "## Safety",
        "",
        "Read-only local audit. No Kaggle submission was made, and no test labels were inferred.",
        "",
        "## Top monotone generator-parameter features",
        "",
        "| feature | quartile acc | spearman |",
        "|---|---:|---:|",
    ]
    for row in result["top_single_features"][:20]:
        lines.append(f"| {row['feature']} | {row['quartile_accuracy_best_orientation']:.6f} | {row['spearman']:+.4f} |")
    lines += ["", "## Top unsupervised cluster-majority checks", "", "| rep | transform | k | CV acc | full train purity |", "|---|---|---:|---:|---:|"]
    for row in result["cluster_results"][:20]:
        lines.append(f"| {row['representation']} | {row['transform']} | {row['k']} | {row['cv_majority_accuracy']:.6f} | {row['full_train_weighted_purity']:.6f} |")
    lines += ["", "## Quantization / grid notes", "", "| feature | unique combined | decimals | min step | integer grid |", "|---|---:|---:|---:|---|"]
    for row in result["quantization"][:25]:
        lines.append(f"| {row['feature']} | {row['unique_combined']} | {row['decimal_places']} | {row['min_positive_step']:.8g} | {row['is_integer_grid']} |")
    lines += ["", "## Interpretation", "", result["interpretation"], ""]
    return "\n".join(lines)


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["target"].to_numpy(int)
    raw_cols = [c for c in train.columns if c not in {"id", "target"}]
    raw = train[raw_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    params = generator_parameter_features(train)
    single = single_feature_screen(pd.concat([raw, params], axis=1), y)
    clusters = cluster_screen(raw, params, y)
    quant = quantization_report(train, test)
    best_single = single[0]["quartile_accuracy_best_orientation"]
    best_cluster = clusters[0]["cv_majority_accuracy"]
    if best_single > 0.55 or best_cluster > 0.55:
        interpretation = "A target-free generator fingerprint has unusually high local signal and should be promoted to a focused model experiment."
    else:
        interpretation = "No advanced local fingerprint explains a 0.82 public score. Single formulae and unsupervised clusters are far below v7, so further local climb is unlikely without an external clue or leaderboard probing."
    result = {
        "experiment": "advanced_generator_fingerprint_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "top_single_features": single[:40],
        "cluster_results": clusters,
        "quantization": quant,
        "interpretation": interpretation,
        "no_kaggle_submission_made": True,
    }
    OUT.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    (OUT / "advanced_generator_fingerprint_audit.json").write_text(json.dumps(result, indent=2))
    (REPORTS / "advanced_generator_fingerprint_audit.md").write_text(markdown(result))
    print(json.dumps({
        "best_single": single[:5],
        "best_clusters": clusters[:5],
        "interpretation": interpretation,
    }, indent=2))


if __name__ == "__main__":
    main()
