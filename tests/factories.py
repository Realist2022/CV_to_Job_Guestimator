"""Shared test doubles and builders.

A plain importable module rather than pytest-fixture-only conftest.py,
because the suite mixes unittest.TestCase classes (test_ingestion_split.py,
test_pipeline_privacy.py, test_artifact_logger.py, ...) with plain pytest
functions (test_harness.py, test_web_app.py) — unittest.TestCase methods
can't receive injected pytest fixtures directly, but both styles can import
from here the same way.

Extracted because the same few shapes — a fake LLM client that plays back
canned responses by system prompt, and a minimal-but-valid PipelineResult —
were independently hand-built, slightly differently, in four+ places. See
the audit note this closed: each was a near-duplicate of the others, so a
required-field change (e.g. redacted_cv_trace_id) meant editing all of them
by hand instead of one factory.
"""

from typing import Any, Callable, Type
from uuid import UUID

from pydantic import BaseModel

from src.schemas.experience import OverallExperienceOutput, WorkRole
from src.schemas.pii import TextSpan
from src.schemas.pipeline import PipelineMetrics, PipelineResult, uuid7
from src.schemas.requirements import SkillMatchResult
from src.schemas.scoring import Scorecard


class RecordingClient:
    """Fake InstructorClient: plays back a canned response per system prompt
    and records every (system_prompt, user_prompt) pair it was called with.

    `responses` maps a system prompt string to either a fixed response
    object, or a callable(response_model) for when the test needs to build
    the response using the caller's dynamically-constrained model (see
    SkillMatcherAgent's per-call ConstrainedSkillEvaluationDecision).
    """

    def __init__(self, model: str, responses: dict[str, Any], *, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.responses = responses
        self.requests: list[tuple[str, str]] = []
        self.last_attempts: int | None = None

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Callable,
        max_retries: int = 2,
    ) -> Any:
        self.requests.append((system_prompt, user_prompt))
        self.last_attempts = 1
        response = self.responses[system_prompt]
        return response(response_model) if callable(response) else response


class FailingClient:
    """Fake InstructorClient whose complete() always raises the given exception.

    Pair with RecordingClient inside a FallbackInstructorClient to simulate
    "primary backend is down/broken, fallback serves the request" without a
    live model — see test_fallback_client.py (the client's own unit tests)
    and test_fallback_integration.py (a real pipeline/agents routed through
    it end to end).
    """

    def __init__(self, model: str, exc: BaseException):
        self.model = model
        self.temperature = 0.0
        self.last_attempts: int | None = None
        self._exc = exc

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        max_retries: int = 2,
    ) -> Any:
        raise self._exc


def build_pipeline_result(
    *,
    engine: str = "fake-eval",
    pii_engine: str = "fake-pii",
    execution_seconds: float = 0.5,
    skill_name: str = "React",
    rationale: str | None = None,
    target_job_title: str = "Full Stack Developer",
    target_overall_years: float | None = 2.0,
    candidate_roles: list[WorkRole] | list[dict] | None = None,
    final_relevance: float = 45.0,
    match_percentage: float = 100.0,
    pillar_b_applicable: bool = True,
    pii_spans: list[TextSpan] | None = None,
    trace: list | None = None,
    redacted_cv_trace_id: UUID | None = None,
) -> PipelineResult:
    """A minimal-but-schema-valid PipelineResult, one matched skill by default.

    Every field is overridable, but the defaults are enough on their own for
    tests that only care about plumbing (artifact logging, harness
    dispatch, endpoint wiring) rather than scoring specifics.
    """
    return PipelineResult(
        engine=engine,
        pii_engine=pii_engine,
        execution_seconds=execution_seconds,
        skills_eval=SkillMatchResult(
            job_requirements=[{"skill_name": skill_name}],
            matched_cv_skills=[skill_name],
            missing_cv_skills=[],
            rationale=rationale or f"{skill_name} is present.",
        ),
        overall_experience=OverallExperienceOutput(
            target_job_title=target_job_title,
            target_overall_years=target_overall_years,
            candidate_roles=candidate_roles or [],
        ),
        scorecard=Scorecard(
            final_relevance=final_relevance,
            pillar_a={"score": match_percentage, "raw": "1/1 skills"},
            pillar_b={
                "score": 0.0,
                "raw": "No relevant roles",
                "applicable": pillar_b_applicable,
            },
            counted_roles=[],
        ),
        metrics=PipelineMetrics(
            total_requirements=1,
            total_matched=1,
            match_percentage=match_percentage,
            final_relevance=final_relevance,
        ),
        redacted_cv_trace_id=redacted_cv_trace_id or uuid7(),
        pii_spans=(
            pii_spans
            if pii_spans is not None
            else [TextSpan(kind="person_name", text="Jane Doe")]
        ),
        trace=trace or [],
    )
