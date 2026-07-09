#!/usr/bin/env python3
"""Safe experiment backlog viewer for Datathon Playground.

This tool intentionally does not execute experiment commands. It only lists the
planned CRISP-DM backlog and prints command templates for future approved runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKLOG = ROOT / "outputs" / "experiment_backlog.json"


def load_backlog() -> dict[str, Any]:
    if not BACKLOG.exists():
        raise FileNotFoundError(f"Missing backlog: {BACKLOG}")
    return json.loads(BACKLOG.read_text(encoding="utf-8"))


def list_experiments(backlog: dict[str, Any], priority: str | None = None) -> None:
    meta = backlog.get("metadata", {})
    print(f"Project: {meta.get('project')}")
    print(f"Current best local CV: {meta.get('current_best_local_cv')}")
    print(f"Current best public LB: {meta.get('current_best_public_lb')}")
    print("Guardrail:", meta.get("guardrail"))
    print()
    for exp in backlog.get("experiments", []):
        if priority and exp.get("priority") != priority:
            continue
        approval = "TRAINING APPROVAL REQUIRED" if exp.get("requires_explicit_training_approval") else "no training approval needed"
        submission = "SUBMISSION APPROVAL REQUIRED" if exp.get("requires_submission_approval") else "no submission approval needed"
        print(f"[{exp.get('priority')}] {exp.get('id')} — {exp.get('status')}")
        print(f"  phase: {exp.get('crisp_dm_phase')}")
        print(f"  compute: {exp.get('expected_compute')} | risk: {exp.get('risk')}")
        print(f"  approvals: {approval}; {submission}")
        print(f"  hypothesis: {exp.get('hypothesis')}")
        print(f"  command template: {exp.get('command_template')}")
        if exp.get("promotion_rule"):
            print(f"  promotion rule: {exp.get('promotion_rule')}")
        print()


def show_experiment(backlog: dict[str, Any], exp_id: str) -> None:
    for exp in backlog.get("experiments", []):
        if exp.get("id") == exp_id:
            print(json.dumps(exp, indent=2, sort_keys=True))
            return
    raise SystemExit(f"Experiment not found: {exp_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List Datathon experiment backlog without running compute")
    parser.add_argument("--list", action="store_true", help="List experiment backlog")
    parser.add_argument("--priority", choices=["P0", "P1", "P2", "P3"], help="Filter by priority")
    parser.add_argument("--show", metavar="EXPERIMENT_ID", help="Show one experiment as JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backlog = load_backlog()
    if args.show:
        show_experiment(backlog, args.show)
        return
    if args.list or args.priority:
        list_experiments(backlog, args.priority)
        return
    raise SystemExit("Nothing executed. Use --list or --show EXPERIMENT_ID. This CLI never runs training.")


if __name__ == "__main__":
    main()
