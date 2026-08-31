"""Diff two run artifacts: which engine won, and on what.

The point of keeping several cv-guestimator tags side by side is comparing
their artifacts/ JSON -- but until now that was done by eye, and nothing in
the repo read an artifact back at all. That absence is why RunArtifact's
schema_version drifted into claiming support for versions it could no longer
parse: with no reader, nothing ever exercised the claim.

So this deliberately reads artifacts as plain JSON rather than validating
them through RunArtifact. A comparison tool that only works on the current
schema is useless for the one job it exists to do -- judging a new build
against runs recorded weeks ago, under an older shape. Fields are looked up
defensively and reported as "-" when a given artifact predates them.

Usage:
    uv run python scripts/compare_runs.py <baseline.json> <candidate.json>
    uv run python scripts/compare_runs.py --latest          # last two runs
    uv run python scripts/compare_runs.py --task model_eval # last two of one task
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ARTIFACTS_DIR = Path("artifacts")


def load(path: Path) -> dict:
    # utf-8-sig: some artifacts were written with a BOM.
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dig(payload: dict, *keys: str, default: Any = None) -> Any:
    """payload["a"]["b"] without raising when an older artifact lacks it."""
    node: Any = payload
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def is_run_artifact(payload: dict) -> bool:
    """Scored runs only -- ingestion artifacts have no scorecard to compare."""
    return "scorecard" in payload


def run_artifacts(task: str | None = None) -> list[Path]:
    """Scored artifacts, oldest first, optionally filtered to one task name."""
    found = []
    for path in ARTIFACTS_DIR.glob("run-*.json"):
        try:
            payload = load(path)
        except (json.JSONDecodeError, OSError):
            continue
        if not is_run_artifact(payload):
            continue
        if task and dig(payload, "config", "task_name") != task:
            continue
        found.append(path)
    return sorted(found)


def fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:g}{suffix}"
    return f"{value}{suffix}"


def delta(baseline: Any, candidate: Any, suffix: str = "") -> str:
    """Signed change, or "-" when either side is missing/non-numeric."""
    if not isinstance(baseline, (int, float)) or not isinstance(candidate, (int, float)):
        return "-"
    change = candidate - baseline
    if change == 0:
        return "same"
    return f"{change:+g}{suffix}"


def row(label: str, left: Any, right: Any, suffix: str = "") -> str:
    return (
        f"  {label:<22} {fmt(left, suffix):>18}  {fmt(right, suffix):>18}"
        f"   {delta(left, right, suffix):>10}"
    )


def skill_sets(payload: dict) -> tuple[set[str], set[str]]:
    matched = set(dig(payload, "skills_evaluation", "matched_cv_skills", default=[]) or [])
    missing = set(dig(payload, "skills_evaluation", "missing_cv_skills", default=[]) or [])
    return matched, missing


def describe(payload: dict) -> str:
    engine = dig(payload, "metadata", "engine", default="?")
    task = dig(payload, "config", "task_name") or "(no task)"
    return f"{engine}  [{task}]"


def compare(baseline_path: Path, candidate_path: Path) -> None:
    baseline, candidate = load(baseline_path), load(candidate_path)

    print("=" * 78)
    print(f"BASELINE   {baseline_path.name}\n           {describe(baseline)}")
    print(f"CANDIDATE  {candidate_path.name}\n           {describe(candidate)}")
    print("=" * 78)

    print(f"\n  {'':<22} {'baseline':>18}  {'candidate':>18}   {'delta':>10}")
    print("  " + "-" * 74)
    print(row("final relevance", dig(baseline, "scorecard", "final_relevance"),
              dig(candidate, "scorecard", "final_relevance"), "%"))
    print(row("skills match", dig(baseline, "metrics", "match_percentage"),
              dig(candidate, "metrics", "match_percentage"), "%"))
    print(row("skills matched", dig(baseline, "metrics", "total_matched"),
              dig(candidate, "metrics", "total_matched")))
    print(row("requirements found", dig(baseline, "metrics", "total_requirements"),
              dig(candidate, "metrics", "total_requirements")))
    print(row("career pillar", dig(baseline, "scorecard", "pillar_b", "score"),
              dig(candidate, "scorecard", "pillar_b", "score"), "%"))
    print(row("roles counted", len(dig(baseline, "scorecard", "counted_roles", default=[]) or []),
              len(dig(candidate, "scorecard", "counted_roles", default=[]) or [])))
    print(row("seconds", dig(baseline, "metadata", "execution_time_seconds"),
              dig(candidate, "metadata", "execution_time_seconds"), "s"))

    # The number moving is the headline; *which* skills moved is the reason.
    base_matched, base_missing = skill_sets(baseline)
    cand_matched, cand_missing = skill_sets(candidate)
    for label, skills in (
        ("newly matched", sorted(cand_matched - base_matched)),
        ("newly missed", sorted(cand_missing - base_missing)),
    ):
        if skills:
            print(f"\n  {label}:")
            for skill in skills:
                print(f"    - {skill}")

    fell_back = [
        name
        for name, payload in (("baseline", baseline), ("candidate", candidate))
        if dig(payload, "config", "evaluation_model", "fallback_used")
    ]
    if fell_back:
        # Otherwise the comparison silently measures the fallback model.
        print(f"\n  WARNING: fallback model served the {', '.join(fell_back)} run.")

    versions = {
        dig(baseline, "schema_version", default="?"),
        dig(candidate, "schema_version", default="?"),
    }
    if len(versions) > 1:
        print(f"\n  note: different schema versions ({', '.join(sorted(versions))});"
              " fields absent from the older one show as '-'.")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path, help="baseline and candidate artifacts")
    parser.add_argument("--latest", action="store_true", help="compare the two most recent runs")
    parser.add_argument("--task", help="restrict --latest to one task_name")
    args = parser.parse_args()

    if args.paths:
        if len(args.paths) != 2:
            parser.error("give exactly two artifact paths, or use --latest")
        baseline, candidate = args.paths
    else:
        if not (args.latest or args.task):
            parser.error("give two artifact paths, or --latest / --task")
        found = run_artifacts(args.task)
        if len(found) < 2:
            where = f" for task {args.task!r}" if args.task else ""
            print(f"Need two scored runs{where}; found {len(found)}.", file=sys.stderr)
            return 1
        baseline, candidate = found[-2], found[-1]

    for path in (baseline, candidate):
        if not path.is_file():
            print(f"No such artifact: {path}", file=sys.stderr)
            return 1

    compare(baseline, candidate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
