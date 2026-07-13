#!/usr/bin/env python3
"""Fold-safe, structure-preserving synthetic augmentation for Datathon v7.

Synthetic rows are same-label interpolations of real training students. They are
created separately inside each CV training fold, never from validation/test
rows, and preserve weekly/daily sequence blocks plus task constraints.
No Kaggle submission is performed.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "kaggle"
sys.path.insert(0, str(ROOT / "scripts"))
from run_production_v7 import SEEDS, blend_rank, feats, quartile_bin  # noqa: E402

WEEKLY_COLS = [f"nilai_minggu_{i:02d}" for i in range(1, 13)]
DAILY_COLS = [f"aktivitas_hari_{i:02d}" for i in range(1, 17)]
CONTINUOUS_COLS = WEEKLY_COLS + DAILY_COLS + [
    "skor_motivasi", "skor_kedisiplinan", "skor_tryout", "jarak_rumah_km",
    "skor_ekstrakurikuler", "indeks_kehadiran", "skor_literasi",
    "jumlah_saudara", "skor_minat_belajar", "urutan_ujian",
    "tugas_selesai", "tugas_diberikan",
]
CATEGORICAL_COLS = ["kelas"]


def synthesize_class_rows(
    frame: pd.DataFrame,
    n_rows: int,
    rng: np.random.Generator,
    id_start: int = -1,
) -> pd.DataFrame:
    """Create same-label interpolations from rows in ``frame``.

    The two donors always share a target label. One interpolation coefficient is
    shared by each sequence/scalar block, preserving within-student structure.
    Categorical values come from one of the donors and task counts are repaired.
    """
    if n_rows <= 0:
        return frame.iloc[:0].copy()
    if "target" not in frame or "id" not in frame:
        raise ValueError("frame must contain id and target columns")
    labels = frame["target"].to_numpy(dtype=int)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        raise ValueError("at least two target classes are required")
    probs = counts / counts.sum()
    chosen = rng.choice(classes, size=n_rows, p=probs)
    rows = []
    for j, target in enumerate(chosen):
        pool = np.flatnonzero(labels == target)
        a_idx, b_idx = rng.choice(pool, size=2, replace=True)
        a = frame.iloc[a_idx]
        b = frame.iloc[b_idx]
        lam = float(rng.uniform(0.25, 0.75))
        row = a.copy()
        for col in CONTINUOUS_COLS:
            if col in frame.columns:
                row[col] = (1.0 - lam) * float(a[col]) + lam * float(b[col])
        for col in CATEGORICAL_COLS:
            if col in frame.columns:
                row[col] = a[col] if rng.random() < 0.5 else b[col]
        if "tugas_diberikan" in frame.columns:
            row["tugas_diberikan"] = max(1, int(round(float(row["tugas_diberikan"]))))
        if "tugas_selesai" in frame.columns:
            row["tugas_selesai"] = int(round(float(row["tugas_selesai"])))
            row["tugas_selesai"] = min(max(row["tugas_selesai"], 0), row["tugas_diberikan"])
        row["id"] = id_start - j
        row["target"] = int(target)
        rows.append(row)
    return pd.DataFrame(rows, columns=frame.columns).reset_index(drop=True)


def augment_training_frame(
    frame: pd.DataFrame,
    factor: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Append ``factor * len(frame)`` synthetic rows to a training fold."""
    if factor < 0:
        raise ValueError("factor must be nonnegative")
    n_rows = int(round(len(frame) * factor))
    if n_rows == 0:
        return frame.copy()
    real_ids = set(frame["id"].tolist())
    id_start = min([-1, *[int(v) for v in real_ids]]) - 1
    synthetic = synthesize_class_rows(frame, n_rows, rng, id_start=id_start)
    if real_ids.intersection(set(synthetic["id"].tolist())):
        raise AssertionError("synthetic IDs overlap real IDs")
    return pd.concat([frame, synthetic], ignore_index=True)


def evaluate_factor(train: pd.DataFrame, seed: int, factor: float) -> dict:
    """Evaluate one augmentation factor using five OOF folds for one seed."""
    y = train["target"]
    X = feats(train)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(train), dtype=float)
    synthetic_rows = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        fold_frame = train.iloc[tr_idx].reset_index(drop=True)
        rng_seed = 2_026_071_300 + seed * 100 + fold * 10 + int(round(factor * 100))
        augmented = augment_training_frame(fold_frame, factor, np.random.default_rng(rng_seed))
        X_aug = feats(augmented)
        y_aug = augmented["target"]
        X_val = X.iloc[va_idx]
        oof[va_idx] = blend_rank(X_aug, y_aug, X_val, seed)
        synthetic_rows.append(len(augmented) - len(fold_frame))
    pred = quartile_bin(oof)
    return {
        "factor": float(factor),
        "seed": int(seed),
        "accuracy": float(accuracy_score(y, pred)),
        "synthetic_rows_per_fold": synthetic_rows,
        "mean_synthetic_rows": float(np.mean(synthetic_rows)),
        "prediction_counts": {str(k): int(v) for k, v in zip(*np.unique(pred, return_counts=True))},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--factors", nargs="+", type=float, default=None)
    args = ap.parse_args()
    seeds = tuple(args.seeds or ((42,) if args.mode == "smoke" else tuple(SEEDS)))
    factors = tuple(args.factors or ((0.0, 0.25, 0.5) if args.mode == "smoke" else (0.0, 0.25, 0.5, 1.0)))
    if any(f < 0 for f in factors):
        raise SystemExit("factors must be nonnegative")

    train = pd.read_csv(DATA / "train.csv")
    results = []
    for factor in factors:
        for seed in seeds:
            print(f"running factor={factor:.2f} seed={seed}", flush=True)
            result = evaluate_factor(train, seed, factor)
            results.append(result)
            print(f"  accuracy={result['accuracy']:.6f} synthetic/fold={result['mean_synthetic_rows']:.1f}", flush=True)

    grouped = {}
    for factor in factors:
        vals = [r["accuracy"] for r in results if r["factor"] == factor]
        grouped[str(factor)] = {
            "mean_accuracy": float(np.mean(vals)),
            "std_accuracy": float(np.std(vals)),
            "per_seed": vals,
        }
    report = {
        "experiment": "fold_safe_same_label_sequence_mixup",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "seeds": list(seeds),
        "factors": list(factors),
        "generator": "same-target donor interpolation; sequence-preserving; task-constraint repair",
        "results": results,
        "grouped": grouped,
        "baseline_factor": 0.0,
        "no_kaggle_submission_made": True,
    }
    out = ROOT / "outputs" / f"exp_synthetic_augmentation_{args.mode}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
