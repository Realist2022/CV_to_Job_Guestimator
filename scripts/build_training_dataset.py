"""Builds an SFT dataset for the cv-guestimator LoRA that mirrors the exact
prompts production sends -- Instructor's schema-stuffing suffix included.

Why this exists: retraining on hand-written (system_prompt, user_prompt)
pairs reproduces the train/inference mismatch that caused prompt edits in
src/prompts/templates.py to have zero effect on cv-guestimator's output --
Instructor's `Mode.JSON` appends a large literal JSON-schema block ("As a
genius expert...", the full Pydantic schema, "Make sure to return an
instance of the JSON...") to the end of whatever system_prompt is passed,
and that block is what the model actually sees last, immediately before the
CV/JD content. A LoRA trained without that block in its examples is being
fine-tuned on a different input distribution than the one it's served at
inference time. This script closes that gap by hooking Instructor's
`completion:kwargs` event to record the literal outbound `messages` for
each of the three matching-pipeline requests, instead of reconstructing
them by hand.

For each example under `--examples` (default training_data/examples/, laid
out as <domain>/<example_id>/{cv.txt,job.txt}) this queries a "teacher"
model (default: gemini-flash, already configured in configs/llm.yaml, and
strong enough to trust as a label source -- spot-check a sample before
trusting a new domain wholesale) for all three matching-pipeline requests --
job_requirements extraction, skill_matcher evaluation, overall_experience
extraction -- and writes one JSONL row per (example, stage) with the exact
captured messages plus the teacher's validated response as the assistant
turn, ready for an Unsloth SFT run.

IMPORTANT: system_prompt/user_prompt/response_model construction below is
re-derived from src/services/agents.py's three Agent classes rather than
calling their .run() methods, because two of the three post-process the raw
LLM output before returning it (SkillMatcherAgent collapses per-requirement
booleans into matched/missing lists; OverallExperienceAgent backfills
missing dates by regex over the CV text) -- training should target what the
model is actually asked to produce, not run()'s post-processed derivative.
If agents.py's prompt/schema construction changes, mirror the change here.

Usage:
    python scripts/build_training_dataset.py
    python scripts/build_training_dataset.py --teacher gemini-flash --examples training_data/examples --out training_data/dataset.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any

from src.config import load_model_config
from src.model.adapters import client_from_config
from src.prompts.templates import (
    JOB_REQUIREMENTS_SYSTEM_PROMPT,
    OVERALL_EXPERIENCE_SYSTEM_PROMPT,
    SKILL_MATCHER_SYSTEM_PROMPT,
)
from src.schemas.experience import OverallExperienceResponse
from src.schemas.requirements import JobRequirementsOutput
from src.services.agents import _constrained_evaluation_model
from src.services.document_parser import JobListing
from src.services.llm_client import InstructorClient


def discover_examples(examples_dir: Path) -> list[dict[str, str]]:
    """<examples_dir>/<domain>/<example_id>/{cv.txt,job.txt} -> example records."""
    examples = []
    for job_path in sorted(examples_dir.glob("*/*/job.txt")):
        example_dir = job_path.parent
        cv_path = example_dir / "cv.txt"
        if not cv_path.exists():
            print(f"skipping {example_dir}: no cv.txt")
            continue
        examples.append(
            {
                "domain": example_dir.parent.name,
                "example_id": example_dir.name,
                "job_text": job_path.read_text(encoding="utf-8"),
                "cv_text": cv_path.read_text(encoding="utf-8"),
            }
        )
    return examples


def capture(
    client: InstructorClient, system_prompt: str, user_prompt: str, response_model: type
) -> tuple[list[dict[str, str]], Any]:
    """Call client.complete(), returning (exact outbound messages, validated result).

    The outbound messages are captured via Instructor's completion:kwargs
    hook rather than reconstructed, so any Instructor-injected schema text
    is captured verbatim exactly as the real model will see it.
    """
    captured: dict[str, Any] = {}
    client.client.on("completion:kwargs", lambda **kw: captured.update(kw))
    result = client.complete(
        system_prompt=system_prompt, user_prompt=user_prompt, response_model=response_model
    )
    return captured["messages"], result


def build_rows(example: dict[str, str], client: InstructorClient) -> list[dict[str, Any]]:
    # These examples are synthetic/fictional CVs written directly as
    # "already redacted" text, not real candidate data run through the PII
    # pipeline -- so cv_text is used as-is, standing in for RedactedCV.text.
    cv_text = example["cv_text"]
    listing = JobListing(text=example["job_text"])

    rows = []

    # --- job_requirements (mirrors JobRequirementsAgent.run) ---
    jr_user = f"JOB DESCRIPTION:\n{listing.requirements_section}"
    jr_messages, jr_result = capture(
        client, JOB_REQUIREMENTS_SYSTEM_PROMPT, jr_user, JobRequirementsOutput
    )
    rows.append(_row(example, "job_requirements", jr_messages, jr_result))

    # --- skill_matcher (mirrors SkillMatcherAgent.run; schema is dynamic,
    # sized to this example's requirement count, so it must be built here) ---
    requirements_json = json.dumps(
        [
            {"requirement_id": i, "skill_name": r.skill_name}
            for i, r in enumerate(jr_result.job_requirements)
        ],
        ensure_ascii=False,
    )
    sm_user = f"JOB REQUIREMENTS:\n{requirements_json}\n\nCANDIDATE CV:\n{cv_text}"
    sm_model = _constrained_evaluation_model(jr_result.job_requirements)
    sm_messages, sm_result = capture(client, SKILL_MATCHER_SYSTEM_PROMPT, sm_user, sm_model)
    rows.append(_row(example, "skill_matcher", sm_messages, sm_result))

    # --- overall_experience (mirrors OverallExperienceAgent.run) ---
    oe_user = f"JOB DESCRIPTION:\n{listing.text}\n\nCANDIDATE CV:\n{cv_text}"
    oe_messages, oe_result = capture(
        client, OVERALL_EXPERIENCE_SYSTEM_PROMPT, oe_user, OverallExperienceResponse
    )
    rows.append(_row(example, "overall_experience", oe_messages, oe_result))

    return rows


def _row(example: dict[str, str], stage: str, messages: list[dict[str, str]], result: Any) -> dict:
    return {
        "domain": example["domain"],
        "example_id": example["example_id"],
        "stage": stage,
        "messages": [
            *({"role": m["role"], "content": m["content"]} for m in messages),
            {"role": "assistant", "content": result.model_dump_json()},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples", type=Path, default=Path("training_data/examples"))
    parser.add_argument("--teacher", default="gemini-flash", help="Model name from configs/llm.yaml")
    parser.add_argument("--out", type=Path, default=Path("training_data/dataset.jsonl"))
    args = parser.parse_args()

    examples = discover_examples(args.examples)
    if not examples:
        raise SystemExit(
            f"No examples found under {args.examples} "
            "(expected <domain>/<example_id>/{cv.txt,job.txt})"
        )

    client = client_from_config(load_model_config(args.teacher))
    if not isinstance(client, InstructorClient):
        raise SystemExit(f"--teacher must resolve to a plain InstructorClient, got {type(client).__name__}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as f:
        for example in examples:
            label = f"{example['domain']}/{example['example_id']}"
            print(f"[{label}] querying {args.teacher}...")
            for row in build_rows(example, client):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            print(f"[{label}] wrote 3 rows (job_requirements, skill_matcher, overall_experience)")

    print(f"\nWrote {written} rows from {len(examples)} example(s) to {args.out}")


if __name__ == "__main__":
    main()
