#!/usr/bin/env python3
"""V8 boundary refinement around the verified v7 latent-ranker pipeline.

This experiment does not submit to Kaggle. It regenerates v7 OOF/test latent
scores when no cache exists, learns small threshold offsets by cross-seed
transfer, and writes a small candidate pack whose changes are inspectable.
"""
from __future__ import annotations

import argparse
import itertools
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
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "scripts"))

from run_production_v7 import SEEDS, blend_rank, feats, quartile_bin  # noqa: E402

NC = 4
DEFAULT_SEEDS = tuple(SEEDS)


def balanced_quartile_thresholds(scores: np.ndarray) -> np.ndarray:
    """Return score cuts at the empirical quartiles."""
    values = np.sort(np.asarray(scores, dtype=float))
    return np.quantile(values, [0.25, 0.50, 0.75], method="linear")


def apply_thresholds(scores: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Map a monotone score vector into labels 0..3."""
    cuts = np.asarray(cuts, dtype=float)
    if cuts.shape != (3,) or not np.all(cuts[:-1] <= cuts[1:]):
        raise ValueError(f"cuts must be three nondecreasing values, got {cuts}")
    return np.digitize(np.asarray(scores, dtype=float), cuts, right=True).astype(int)


def candidate_offsets(values=(-64, -48, -32, -16, 0, 16, 32, 48, 64), step=16):
    """Return deterministic boundary offset triples, baseline first."""
    values = tuple(int(v) for v in values)
    if step <= 0 or any(v % step for v in values):
        raise ValueError("offset values must be divisible by a positive step")
    triples = list(itertools.product(values, repeat=3))
    return sorted(triples, key=lambda t: (sum(abs(v) for v in t), t))


def passes_balance_guard(counts: np.ndarray, target_per_class: int, max_drift: int) -> bool:
    """Accept only candidates whose class counts stay near the target quartiles."""
    counts = np.asarray(counts, dtype=int)
    return counts.shape == (NC,) and bool(np.all(np.abs(counts - target_per_class) <= max_drift))


def thresholds_from_offset(scores: np.ndarray, offsets: tuple[int, int, int]) -> np.ndarray:
    """Use empirical quartile positions shifted by row-count offsets."""
    values = np.sort(np.asarray(scores, dtype=float))
    n = len(values)
    base = (n // 4, n // 2, (3 * n) // 4)
    idx = [max(0, min(n - 1, b + int(o))) for b, o in zip(base, offsets)]
    # Sorting offsets is not enough when the score distribution has ties. Keep
    # cuts monotone so candidate labels remain ordered.
    cuts = np.maximum.accumulate(values[idx])
    return cuts.astype(float)


def learn_thresholds(scores: np.ndarray, y: pd.Series, offsets_list):
    """Select thresholds on one seed's OOF scores, returning score and cuts."""
    best = None
    yv = np.asarray(y, dtype=int)
    for offsets in offsets_list:
        cuts = thresholds_from_offset(scores, offsets)
        acc = float(accuracy_score(yv, apply_thresholds(scores, cuts)))
        key = (acc, -sum(abs(v) for v in offsets), tuple(-v for v in offsets))
        if best is None or key > best[0]:
            best = (key, offsets, cuts)
    assert best is not None
    return {"accuracy": best[0][0], "offsets": tuple(best[1]), "cuts": best[2]}


def cache_scores(seeds, cache_path: Path):
    """Generate v7 OOF and test scores, or load an existing cache."""
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        cached_seeds = tuple(int(v) for v in cached["seeds"])
        if cached_seeds == tuple(seeds):
            return cached["oof"], cached["test"], cached_seeds

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["target"]
    X = feats(train)
    Xtest = feats(test)
    if list(X.columns) != list(Xtest.columns):
        raise ValueError("v7 train/test feature columns differ")

    oof = np.zeros((len(seeds), len(train)), dtype=float)
    test_scores = np.zeros((len(seeds), len(test)), dtype=float)
    for si, seed in enumerate(seeds):
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tr_idx, va_idx in cv.split(X, y):
            oof[si, va_idx] = blend_rank(X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[va_idx], seed)
        test_scores[si] = blend_rank(X, y, Xtest, seed)
        print(f"score cache seed {seed}: OOF ready", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, seeds=np.asarray(seeds), oof=oof, test=test_scores)
    return oof, test_scores, tuple(seeds)


def rank_average(test_scores: np.ndarray) -> np.ndarray:
    """Match v7's per-seed rank averaging before quartile binning."""
    ranked = np.vstack([pd.Series(row).rank(method="first").to_numpy() for row in test_scores])
    return ranked.mean(axis=0)


def load_reference_predictions(reference_path: Path, pd_ids: np.ndarray) -> np.ndarray:
    """Load the exact submitted v7 labels and reject mismatched ID order."""
    reference = pd.read_csv(reference_path)
    if list(reference.columns) != ["id", "target"]:
        raise ValueError(f"unexpected reference columns: {list(reference.columns)}")
    if list(reference["id"].to_numpy()) != list(pd_ids):
        raise ValueError("submitted v7 ID order does not match current test IDs")
    pred = reference["target"].to_numpy(dtype=int)
    if not np.all(np.isin(pred, np.arange(NC))):
        raise ValueError("submitted v7 contains an invalid target label")
    return pred


def candidate_summary(pred: np.ndarray, v7_pred: np.ndarray):
    delta = pred.astype(int) - v7_pred.astype(int)
    return {
        "changed_rows": int(np.count_nonzero(delta)),
        "adjacent_changes": int(np.count_nonzero(np.abs(delta) == 1)),
        "non_adjacent_changes": int(np.count_nonzero(np.abs(delta) > 1)),
        "signed_delta_counts": {str(k): int(v) for k, v in zip(*np.unique(delta, return_counts=True)) if k != 0},
        "prediction_counts": {str(k): int(v) for k, v in zip(*np.unique(pred, return_counts=True))},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument("--offset-radius", type=int, default=64)
    ap.add_argument("--offset-step", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-class-drift", type=int, default=16,
                    help="maximum allowed deviation from the 25%% class target for safe candidates")
    ap.add_argument("--force-recompute", action="store_true")
    args = ap.parse_args()
    seeds = tuple(args.seeds)
    if len(seeds) < 3:
        raise SystemExit("Use at least three seeds for cross-seed threshold transfer")
    if args.offset_radius < 0 or args.offset_radius % args.offset_step:
        raise SystemExit("offset-radius must be nonnegative and divisible by offset-step")
    if args.max_class_drift < 0:
        raise SystemExit("max-class-drift must be nonnegative")

    OUT.mkdir(parents=True, exist_ok=True)
    cache_path = OUT / "v8_v7_scores.npz"
    if args.force_recompute and cache_path.exists():
        cache_path.unlink()
    offsets = tuple(range(-args.offset_radius, args.offset_radius + 1, args.offset_step))
    triples = candidate_offsets(offsets, args.offset_step)
    print(f"offset values={offsets}; candidates={len(triples)}", flush=True)

    oof, test_scores, cached_seeds = cache_scores(seeds, cache_path)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    y = train["target"]

    # Baseline follows v7 exactly: per-seed OOF quartiles for CV and averaged
    # per-seed test ranks for the final prediction.
    baseline_acc = np.array([
        accuracy_score(y, quartile_bin(oof[i])) for i in range(len(seeds))
    ])
    test_rank = rank_average(test_scores)
    rerun_v7_pred = quartile_bin(test_rank)
    submitted_v7_pred = load_reference_predictions(OUT / "submission_v7.csv", test["id"].to_numpy())

    # Honest transfer: learn cuts on one seed's OOF labels and apply them to
    # every other seed. This avoids selecting a threshold and evaluating it on
    # the same OOF vector.
    pair_rows = []
    for source in range(len(seeds)):
        learned = learn_thresholds(oof[source], y, triples)
        for target in range(len(seeds)):
            if source == target:
                continue
            pred = apply_thresholds(oof[target], learned["cuts"])
            pair_rows.append({
                "source_seed": int(seeds[source]),
                "target_seed": int(seeds[target]),
                "offsets": list(learned["offsets"]),
                "source_fit_accuracy": learned["accuracy"],
                "target_transfer_accuracy": float(accuracy_score(y, pred)),
            })

    # Score each offset by cross-seed transfer, with smaller movements winning
    # ties. This deliberately does not choose based on the target test labels.
    aggregate = []
    for triple in triples:
        vals = []
        for source in range(len(seeds)):
            cuts = thresholds_from_offset(oof[source], triple)
            for target in range(len(seeds)):
                if source != target:
                    vals.append(accuracy_score(y, apply_thresholds(oof[target], cuts)))
        aggregate.append({
            "offsets": list(triple),
            "transfer_mean": float(np.mean(vals)),
            "transfer_std": float(np.std(vals)),
            "transfer_min": float(np.min(vals)),
        })
    aggregate.sort(key=lambda r: (-r["transfer_mean"], r["transfer_std"], sum(abs(v) for v in r["offsets"])))

    # Keep only near-quartile candidates. The observed successful submissions
    # use a balanced 200/200/200/200 target distribution, so unconstrained
    # threshold shifts are logged but never promoted to the safe pack.
    target_per_class = len(test) // NC
    safe_ranked = []
    for row in aggregate:
        triple = tuple(row["offsets"])
        cuts = thresholds_from_offset(test_rank, triple)
        pred = apply_thresholds(test_rank, cuts)
        counts = np.bincount(pred, minlength=NC)
        if passes_balance_guard(counts, target_per_class, args.max_class_drift):
            safe_ranked.append((row, cuts, pred, counts))

    candidates = []
    for rank, (row, cuts, pred, counts) in enumerate(safe_ranked[: max(args.top_k, 1)], start=1):
        triple = tuple(row["offsets"])
        name = "submission_v8_safe_threshold_%02d.csv" % rank
        path = OUT / name
        pd.DataFrame({"id": test["id"], "target": pred}).to_csv(path, index=False)
        candidates.append({
            "rank": rank,
            "file": str(path.relative_to(ROOT)),
            "offsets": list(triple),
            "test_cuts": cuts.tolist(),
            "transfer_mean": row["transfer_mean"],
            "transfer_std": row["transfer_std"],
            "transfer_min": row["transfer_min"],
            **candidate_summary(pred, submitted_v7_pred),
        })

    # Always include the exact v7 baseline as a reference, without overwriting
    # the user's submitted file.
    baseline_path = OUT / "submission_v8_baseline_v7.csv"
    pd.DataFrame({"id": test["id"], "target": submitted_v7_pred}).to_csv(baseline_path, index=False)

    report = {
        "experiment": "v8_cross_seed_boundary_refinement",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seeds": list(cached_seeds),
        "offset_values": list(offsets),
        "candidate_offset_count": len(triples),
        "feature_source": "run_production_v7.feats + blend_rank",
        "baseline_v7_oof_mean": float(baseline_acc.mean()),
        "baseline_v7_oof_std": float(baseline_acc.std()),
        "baseline_v7_per_seed": baseline_acc.round(6).tolist(),
        "baseline_v7_submission": "outputs/submission_v7.csv",
        "rerun_vs_submitted_changed_rows": int(np.count_nonzero(rerun_v7_pred != submitted_v7_pred)),
        "score_cache": str(cache_path.relative_to(ROOT)),
        "max_class_drift": args.max_class_drift,
        "target_per_class": target_per_class,
        "safe_candidate_count": len(safe_ranked),
        "top_unconstrained_offsets": aggregate[: min(20, len(aggregate))],
        "top_balance_guarded_offsets": [
            {
                "offsets": row["offsets"],
                "transfer_mean": row["transfer_mean"],
                "transfer_std": row["transfer_std"],
                "transfer_min": row["transfer_min"],
                "prediction_counts": {str(i): int(v) for i, v in enumerate(counts)},
            }
            for row, _cuts, _pred, counts in safe_ranked[: min(20, len(safe_ranked))]
        ],
        "candidates": candidates,
        "no_kaggle_submission_made": True,
        "selection_rule": "cross-seed OOF threshold transfer; no test labels",
    }
    (OUT / "exp_v8_boundary_refinement.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "baseline_oof": report["baseline_v7_oof_mean"],
        "best_offsets": candidates[0]["offsets"] if candidates else None,
        "candidates": candidates,
        "report": "outputs/exp_v8_boundary_refinement.json",
    }, indent=2))


if __name__ == "__main__":
    main()
