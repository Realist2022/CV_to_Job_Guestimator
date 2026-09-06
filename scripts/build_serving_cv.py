"""Build the single PII-free RedactedCV that the public API serves.

The deployed site matches job listings against one fixed CV — the owner's
— so no visitor ever uploads a CV and no PII detection runs on the request
path. That CV is a build input, not runtime state: it is produced here,
once, on the machine that holds the raw document, and the result is
committed and copied into the image (see docker/Dockerfile.api).

Why this script exists rather than copying the store file directly:
redacted_cvs/*.json is NOT safe to ship. Its `text` is redacted, but its
`pii_spans` list carries the detected values verbatim — a tidy inventory
of exactly the name, address and identifiers that were taken out (see
RedactedCV.for_serving). That is fine in a gitignored local directory and
a disclosure anywhere else, and a Docker image layer cannot be un-pushed.

So this does two things the store cannot do for itself:

  1. Replaces every span value with "[withheld]", keeping the count and
     kind so the copy still describes its own redaction honestly.
  2. Verifies the result. Step 1 says nothing about whether the detector
     actually caught everything — a value it missed is still sitting in
     `text`. So every original span value is searched for in the finished
     JSON, and a hit aborts the build. That check is the whole point: it
     is the last gate before this file leaves the machine.

Usage:

    python scripts/build_serving_cv.py --list
    python scripts/build_serving_cv.py <cv_id>
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The project isn't installed as a package, so `python scripts/x.py` puts
# scripts/ on sys.path but not the repo root, and `import src` fails. This
# is a PII gate: it has to run correctly on the first try, from the
# documented command, rather than tempting whoever hits the ImportError
# into a workaround that skips it.
sys.path.insert(0, str(ROOT))

from src.schemas.ingestion import RedactedCV  # noqa: E402

STORE_DIR = ROOT / "redacted_cvs"
OUT_PATH = ROOT / "serving" / "redacted_cv.json"


def _mask(value: str) -> str:
    """A span value described well enough to find, without printing it.

    Terminal output gets scrolled back, screenshotted and pasted into bug
    reports, so a leak report must not itself leak the value it found.
    """
    return f"{value[:1]}...({len(value)} chars)"


def _load(path: Path) -> RedactedCV:
    return RedactedCV.model_validate_json(path.read_text(encoding="utf-8"))


def _list_store() -> int:
    paths = sorted(STORE_DIR.glob("*.json"))
    if not paths:
        print(f"No ingested CVs in {STORE_DIR}. Run an ingestion task first.")
        return 1
    print(f"Ingested CVs in {STORE_DIR}:\n")
    for path in paths:
        cv = _load(path)
        print(
            f"  {cv.cv_id}\n"
            f"    redacted {cv.redacted_at:%Y-%m-%d %H:%M}  "
            f"{len(cv.text)} chars  {len(cv.pii_spans)} span(s)  {cv.pii_engine}"
        )
    print("\nPick one:  python scripts/build_serving_cv.py <cv_id>")
    return 0


def _find_leaks(serving_json: str, original: RedactedCV) -> list[str]:
    """Original span values that still appear in the file about to be written.

    A hit means redaction missed that value in `text` — the span list
    recorded it, but the text it was supposed to be cut from still has it.
    """
    return [
        f"{span.kind}: {_mask(span.text)}"
        for span in original.pii_spans
        if span.text and span.text in serving_json
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cv_id", nargs="?", help="cv_id to publish (see --list).")
    parser.add_argument("--list", action="store_true", help="Show ingested CVs and exit.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Write even if the leak check fails. Only for a confirmed false "
        "positive — e.g. a two-letter span value that also occurs as an "
        "ordinary substring of the redacted text.",
    )
    args = parser.parse_args()

    if args.list or not args.cv_id:
        return _list_store()

    source = STORE_DIR / f"{args.cv_id}.json"
    if not source.exists():
        print(f"No ingested CV with cv_id '{args.cv_id}'. Try --list.", file=sys.stderr)
        return 1

    original = _load(source)
    serving = original.for_serving()
    serving_json = serving.model_dump_json(indent=2) + "\n"

    leaks = _find_leaks(serving_json, original)
    if leaks:
        print(
            "REFUSING TO WRITE: values the detector recorded as PII are still "
            f"present in the redacted text of {source.name}:",
            file=sys.stderr,
        )
        for leak in leaks:
            print(f"  - {leak}", file=sys.stderr)
        print(
            "\nRedaction did not fully apply. Re-ingest with an adjusted "
            "configs/pii_policy.yaml, or pass --force if you have confirmed "
            "these are coincidental substrings.",
            file=sys.stderr,
        )
        if not args.force:
            return 1
        print("--force given: writing anyway.", file=sys.stderr)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(serving_json, encoding="utf-8")

    kinds: dict[str, int] = {}
    for span in serving.pii_spans:
        kinds[span.kind] = kinds.get(span.kind, 0) + 1
    summary = ", ".join(f"{count}x {kind}" for kind, count in sorted(kinds.items())) or "none"

    print(f"Wrote {OUT_PATH.relative_to(ROOT)}")
    print(f"  cv_id       {serving.cv_id}")
    print(f"  redacted    {len(serving.text)} chars, {len(serving.pii_spans)} span(s) withheld")
    print(f"  span kinds  {summary}")
    print("  leak check  passed" if not leaks else "  leak check  FAILED (forced)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
