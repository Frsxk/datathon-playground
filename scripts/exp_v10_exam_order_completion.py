#!/usr/bin/env python3
"""V10: full v7 test of the new exam-order × completion symbolic feature.

Uses v7's exact ranker with one additional target-free feature. It compares to
the cached five-seed v7 OOF score cache and writes a submission candidate only
when the predeclared local promotion gate is met. No Kaggle submission.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cv_harness import SEEDS, quartile_bin  # noqa: E402
from run_production_v7 import blend_rank, feats  # noqa: E402

PROMOTION_MIN_MEAN_GAIN = 0.001
PROMOTION_MIN_IMPROVED_SEEDS = 3


def add_exam_order_completion_feature(frame: pd.DataFrame) -> np.ndarray:
    assigned = np.maximum(frame["tugas_diberikan"].to_numpy(float), 1.0)
    completion = frame["tugas_selesai"].to_numpy(float) / assigned
    return frame["urutan_ujian"].to_numpy(float) * completion


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    X = feats(frame).copy()
    X["exam_order_x_completion"] = add_exam_order_completion_feature(frame)
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def main():
    train = pd.read_csv(ROOT / "kaggle" / "train.csv")
    test = pd.read_csv(ROOT / "kaggle" / "test.csv")
    sample = pd.read_csv(ROOT / "kaggle" / "sample_submission.csv")
    y = train["target"].to_numpy(int)
    X = feature_matrix(train)
    Xtest = feature_matrix(test)
    assert list(X.columns) == list(Xtest.columns)

    cache = np.load(ROOT / "outputs" / "v8_v7_scores.npz")
    base_oof = cache["oof"]
    assert base_oof.shape == (len(SEEDS), len(train))
    base_per_seed = np.array([accuracy_score(y, quartile_bin(base_oof[i])) for i in range(len(SEEDS))])

    candidate_oof = np.zeros((len(SEEDS), len(train)))
    candidate_test = np.zeros((len(SEEDS), len(test)))
    candidate_per_seed = []
    for si, seed in enumerate(SEEDS):
        cv = StratifiedKFold(5, shuffle=True, random_state=int(seed))
        oof = np.zeros(len(train))
        for train_idx, valid_idx in cv.split(X, y):
            oof[valid_idx] = blend_rank(X.iloc[train_idx], y[train_idx], X.iloc[valid_idx], int(seed))
        candidate_oof[si] = oof
        candidate_test[si] = blend_rank(X, y, Xtest, int(seed))
        candidate_per_seed.append(accuracy_score(y, quartile_bin(oof)))
        print(f"seed {seed}: base={base_per_seed[si]:.6f} candidate={candidate_per_seed[-1]:.6f}")

    candidate_per_seed = np.asarray(candidate_per_seed)
    gain_per_seed = candidate_per_seed - base_per_seed
    mean_gain = float(candidate_per_seed.mean() - base_per_seed.mean())
    improved_seeds = int(np.sum(gain_per_seed > 0))
    eligible = bool(mean_gain >= PROMOTION_MIN_MEAN_GAIN and improved_seeds >= PROMOTION_MIN_IMPROVED_SEEDS)

    candidate_path = None
    counts = None
    if eligible:
        ranks = np.mean([pd.Series(s).rank(method="average").to_numpy() for s in candidate_test], axis=0)
        pred = quartile_bin(ranks).astype(int)
        submission = pd.DataFrame({"id": test["id"], "target": pred})
        assert submission.shape == sample.shape
        assert list(submission.columns) == ["id", "target"]
        assert submission["id"].equals(sample["id"])
        assert set(submission["target"]).issubset({0, 1, 2, 3})
        counts = {str(k): int(v) for k, v in submission["target"].value_counts().sort_index().items()}
        candidate_path = "outputs/submission_v10_exam_order_completion.csv"
        submission.to_csv(ROOT / candidate_path, index=False)

    result = {
        "experiment": "v10_exam_order_x_completion",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature": "exam_order_x_completion = urutan_ujian * tugas_selesai / max(tugas_diberikan, 1)",
        "feature_count": int(X.shape[1]),
        "seeds": [int(s) for s in SEEDS],
        "baseline_per_seed": base_per_seed.tolist(),
        "candidate_per_seed": candidate_per_seed.tolist(),
        "gain_per_seed": gain_per_seed.tolist(),
        "baseline_mean": float(base_per_seed.mean()),
        "candidate_mean": float(candidate_per_seed.mean()),
        "mean_gain": mean_gain,
        "improved_seed_count": improved_seeds,
        "promotion_gate": {
            "min_mean_gain": PROMOTION_MIN_MEAN_GAIN,
            "min_improved_seeds": PROMOTION_MIN_IMPROVED_SEEDS,
            "eligible": eligible,
        },
        "candidate_path": candidate_path,
        "prediction_counts": counts,
        "no_kaggle_submission_made": True,
    }
    output = ROOT / "outputs" / "exp_v10_exam_order_completion.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
