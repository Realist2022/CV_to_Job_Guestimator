import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional
from src.config_loader import load_pii_detector_names
from src.schemas.pipeline import PipelineMetrics, PipelineResult, TraceSpan, uuid7
from src.schemas.requirements import SkillMatchResult
from src.services.document_parser import CandidateCV, JobListing
from src.services.llm_client import InstructorClient
from src.services.agents import (
    JobRequirementsAgent,
    OverallExperienceAgent,
    SkillMatcherAgent,
)
from src.services.pii_detector import PIIDetector, build_pii_detector
from src.services.scoring_engine import RelevanceScoringEngine


def _require(value, failure_message: str):
    if value is None:
        raise RuntimeError(failure_message)
    return value


@contextmanager
def _traced_step(trace: list[TraceSpan], step: str):
    """Time one pipeline step and append it to `trace` as a TraceSpan.

    Yields a dict the caller can write `attempts` into (see InstructorClient
    .last_attempts) to surface LLM retries in the trace.
    """
    started_at = datetime.now(timezone.utc)
    started = time.time()
    info: dict = {}
    yield info
    trace.append(
        TraceSpan(
            step=step,
            started_at=started_at,
            duration_seconds=round(time.time() - started, 3),
            attempts=info.get("attempts"),
        )
    )


class ExtractionPipeline:
    def __init__(
        self,
        client: InstructorClient,
        pii_detector: Optional[PIIDetector] = None,
        pii_client: Optional[InstructorClient] = None,
        scoring_engine: Optional[RelevanceScoringEngine] = None,
    ):
        self.client = client
        self.job_requirements_agent = JobRequirementsAgent(client)
        self.skill_matcher_agent = SkillMatcherAgent(client)
        self.overall_experience_agent = OverallExperienceAgent(client)
        self.scoring_engine = scoring_engine or RelevanceScoringEngine()
        self.pii_client = pii_client or _default_pii_client()
        self.pii_detector = pii_detector or build_pii_detector(
            load_pii_detector_names(), self.pii_client
        )

    def run(
        self, listing: JobListing, cv: CandidateCV, *, verbose: bool = True
    ) -> PipelineResult:
        started = time.time()
        trace_id = uuid7()
        trace: list[TraceSpan] = []
        say = print if verbose else (lambda *_: None)
        say(f" -> Trace ID: {trace_id}")

        # STEP 1 and STEP 2 are independent: PII redaction only needs the
        # CV, job requirement extraction only needs the listing (never CV
        # data), so their LLM calls run concurrently instead of stacking.
        # If PII redaction then fails, the run still aborts before any
        # CV-derived content reaches a model — but the requirements call
        # may already be in flight, since nothing about it depends on PII
        # succeeding.
        say(" -> [1/5] Detecting and redacting PII from Candidate CV...")
        say(" -> [2/5] Extracting job requirements...")

        def _detect_pii():
            with _traced_step(trace, "pii_redaction") as info:
                spans = self.pii_detector.detect(cv)
                info["attempts"] = self.pii_client.last_attempts
                return spans, cv.redacted(spans)

        def _extract_requirements():
            with _traced_step(trace, "job_requirements_extraction") as info:
                result = _require(
                    self.job_requirements_agent.run(listing),
                    "Job requirement extraction failed.",
                )
                info["attempts"] = self.client.last_attempts
                return result

        with ThreadPoolExecutor(max_workers=2) as executor:
            pii_future = executor.submit(_detect_pii)
            requirements_future = executor.submit(_extract_requirements)
            spans, redacted_cv = pii_future.result()
            requirements_result = requirements_future.result()

        say(f"       {len(spans)} PII spans redacted from CV.")

        # STEP 3: Classify only the extracted requirements against the redacted CV
        say(" -> [3/5] Evaluating extracted requirements against Candidate CV...")
        with _traced_step(trace, "skill_matching") as info:
            evaluation = _require(
                self.skill_matcher_agent.run(
                    job_requirements=requirements_result.job_requirements,
                    cv=redacted_cv,
                ),
                "Skill matching evaluation failed.",
            )
            info["attempts"] = self.client.last_attempts

        skills_result = SkillMatchResult(
            job_requirements=requirements_result.job_requirements,
            matched_cv_skills=evaluation.matched_cv_skills,
            missing_cv_skills=evaluation.missing_cv_skills,
            rationale=evaluation.rationale,
        )
        say(f"       Matched {skills_result.total_matched_skills}/{skills_result.total_job_requirements} skills "
            f"({skills_result.match_percentage}%)")

        # STEP 4: Extract and classify the candidate's roles from the redacted CV
        say(" -> [4/5] Evaluating overall relevant career experience...")
        with _traced_step(trace, "overall_experience_extraction") as info:
            overall_experience = _require(
                self.overall_experience_agent.run(listing, redacted_cv),
                "Overall experience extraction failed.",
            )
            info["attempts"] = self.client.last_attempts

        with _traced_step(trace, "scoring"):
            scorecard = self.scoring_engine.calculate_scorecard(
                skills_result,
                overall_experience,
            )

        execution_seconds = round(time.time() - started, 2)
        # pii_redaction and job_requirements_extraction ran concurrently, so
        # they may have been appended in either order; keep the trace
        # readable by start time regardless of which one finished first.
        trace.sort(key=lambda span: span.started_at)

        return PipelineResult(
            trace_id=trace_id,
            engine=self.client.model,
            pii_engine=self.pii_client.model,
            execution_seconds=execution_seconds,
            skills_eval=skills_result,
            overall_experience=overall_experience,
            scorecard=scorecard,
            metrics=PipelineMetrics(
                total_requirements=skills_result.total_job_requirements,
                total_matched=skills_result.total_matched_skills,
                match_percentage=skills_result.match_percentage,
                final_relevance=scorecard.final_relevance,
            ),
            redacted_cv=redacted_cv.text,
            pii_spans=spans,
            trace=trace,
        )


def _default_pii_client() -> InstructorClient:
    # Imported lazily to avoid a circular import through src.services.
    from src.model.adapters import client_for_role

    return client_for_role("pii")