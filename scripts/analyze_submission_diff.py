#!/usr/bin/env python3
"""Analyze differences between two Datathon submission CSVs.

Safe diagnostic only:
- no model training
- no CV
- no tuning
- no Kaggle calls/submission
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "submission_diff_audit.md"
RAW_TEST = ROOT / "kaggle" / "test.csv"

KEY_FEATURES = [
    "skor_tryout",
    "indeks_kehadiran",
    "skor_literasi",
    "skor_motivasi",
    "skor_minat_belajar",
    "skor_kedisiplinan",
    "tugas_diberikan",
    "tugas_selesai",
    "kelas",
]


def load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"id", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df[["id", "target"]].copy()


def transition_table(old: pd.Series, new: pd.Series) -> pd.DataFrame:
    table = pd.crosstab(old, new, rownames=["old"], colnames=["new"], dropna=False)
    return table.reindex(index=range(4), columns=range(4), fill_value=0)


def target_counts(df: pd.DataFrame) -> dict[int, int]:
    return {int(k): int(v) for k, v in df["target"].value_counts().sort_index().items()}


def feature_contrast(test: pd.DataFrame, changed_ids: set[int]) -> pd.DataFrame:
    if not changed_ids:
        return pd.DataFrame()
    tmp = test.copy()
    tmp["changed"] = tmp["id"].isin(changed_ids)
    rows: list[dict[str, float | str]] = []
    for col in KEY_FEATURES:
        if col not in tmp.columns:
            continue
        changed = tmp.loc[tmp["changed"], col]
        unchanged = tmp.loc[~tmp["changed"], col]
        rows.append({
            "feature": col,
            "changed_mean": float(changed.mean()),
            "unchanged_mean": float(unchanged.mean()),
            "mean_delta": float(changed.mean() - unchanged.mean()),
            "changed_median": float(changed.median()),
            "unchanged_median": float(unchanged.median()),
        })
    return pd.DataFrame(rows).sort_values("mean_delta", key=lambda s: s.abs(), ascending=False)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    return df.to_markdown(index=False)


def write_report(old_path: Path, new_path: Path, out_path: Path) -> None:
    old = load_submission(old_path)
    new = load_submission(new_path)
    if len(old) != len(new):
        raise ValueError(f"Length mismatch: {len(old)} vs {len(new)}")
    if not old["id"].equals(new["id"]):
        raise ValueError("Submission IDs/order do not match")

    diff_mask = old["target"] != new["target"]
    changed = pd.DataFrame({
        "id": old.loc[diff_mask, "id"].astype(int),
        "old_target": old.loc[diff_mask, "target"].astype(int),
        "new_target": new.loc[diff_mask, "target"].astype(int),
    })
    changed["delta"] = changed["new_target"] - changed["old_target"]
    changed_ids = set(changed["id"].tolist())

    transitions = transition_table(old["target"], new["target"])
    delta_counts = Counter(int(x) for x in changed["delta"])
    abs_delta_counts = Counter(abs(int(x)) for x in changed["delta"])

    contrast = pd.DataFrame()
    if RAW_TEST.exists():
        contrast = feature_contrast(pd.read_csv(RAW_TEST), changed_ids)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Submission Difference Audit",
        "",
        f"Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Inputs",
        "",
        f"- Old submission: `{old_path}`",
        f"- New submission: `{new_path}`",
        "",
        "## Summary",
        "",
        f"- Rows: `{len(old)}`",
        f"- IDs/order match: `true`",
        f"- Changed predictions: `{int(diff_mask.sum())}`",
        f"- Unchanged predictions: `{int((~diff_mask).sum())}`",
        f"- Old target counts: `{target_counts(old)}`",
        f"- New target counts: `{target_counts(new)}`",
        "",
        "## Transition matrix",
        "",
        "Rows are old targets; columns are new targets.",
        "",
        transitions.to_markdown(),
        "",
        "## Delta counts",
        "",
        f"- Signed deltas: `{dict(sorted(delta_counts.items()))}`",
        f"- Absolute deltas: `{dict(sorted(abs_delta_counts.items()))}`",
        "",
        "## First changed rows",
        "",
        markdown_table(changed.head(40)),
        "",
        "## Changed-vs-unchanged raw feature contrast",
        "",
        "This is descriptive only; it uses no labels and trains no model.",
        "",
        markdown_table(contrast.round(4) if not contrast.empty else contrast),
        "",
        "## Interpretation",
        "",
        "- The public LB tied despite changed predictions, so the changed rows likely did not affect the public split enough to move the rounded score, or gains/losses canceled out.",
        "- Use this audit to target boundary/threshold experiments rather than broad hyperparameter searches first.",
        "- Because both submissions preserve a 200/200/200/200 class balance, the next likely lever is rank/threshold quality near class boundaries, not global class distribution.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"changed_predictions={int(diff_mask.sum())}")
    print(f"delta_counts={dict(sorted(delta_counts.items()))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit differences between two submission CSVs")
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_report(args.old, args.new, args.out)


if __name__ == "__main__":
    main()
