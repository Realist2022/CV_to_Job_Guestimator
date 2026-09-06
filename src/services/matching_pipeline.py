"""Standalone skill-matching / experience pipeline.

Consumes an already-redacted CV (RedactedCV, produced once by
IngestionPipeline — see ingestion_pipeline.py) and a JobListing. This
module deliberately never imports CandidateCV or PIIDetector: there is no
code path here capable of loading or seeing an unredacted CV, so "matching
never touches raw PII" is enforced by the import graph, not just by
discipline about which argument gets passed where.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from src.schemas.ingestion import RedactedCV
from src.schemas.pipeline import PipelineMetrics, PipelineResult, TraceSpan, uuid7
from src.schemas.requirements import SkillMatchResult
from src.services.agents import JobRequirementsAgent, OverallExperienceAgent, SkillMatcherAgent
from src.services.document_parser import JobListing
from src.services.llm_client import CompletionClient
from src.services.pipeline_tracing import traced_step
from src.services.scoring_engine import RelevanceScoringEngine


def _require(value, failure_message: str):
    if value is None:
        raise RuntimeError(failure_message)
    return value


class MatchingPipeline:
    def __init__(
        self,
        client: CompletionClient,
        scoring_engine: Optional[RelevanceScoringEngine] = None,
        experience_client: Optional[CompletionClient] = None,
    ):
        """`experience_client`, when given, runs the overall-experience step
        concurrently with the requirements/matching pair.

        Those two halves have no data dependency on each other -- the
        experience agent reads only the listing and the CV, never the
        extracted requirements -- so the sequential order below was
        arrangement, not necessity. On the Modal deployment the experience
        step dominates a run (~40s of a ~53s warm total), so overlapping it
        with the other two is most of the wall-clock win available without
        touching the model.

        It takes a SECOND client rather than reusing `client` because
        InstructorClient documents that it is not safe to call concurrently:
        it registers and removes an attempt-counting hook around each call,
        so two racing calls double-count or drop `last_attempts`. Passing
        one client twice would produce quietly wrong `attempts` values in
        every artifact -- the kind of bug that looks like model flakiness.

        Default None keeps the sequential behaviour for the CLI harness and
        the tests, where the latency does not matter and a single client is
        simpler to reason about.
        """
        self.client = client
        self.experience_client = experience_client
        self.job_requirements_agent = JobRequirementsAgent(client)
        self.skill_matcher_agent = SkillMatcherAgent(client)
        self.overall_experience_agent = OverallExperienceAgent(experience_client or client)
        self.scoring_engine = scoring_engine or RelevanceScoringEngine()

    def _extract_experience(
        self, listing: JobListing, redacted_cv: RedactedCV, trace: list[TraceSpan]
    ):
        """The overall-experience step, with its own trace span and client.

        Appends to `trace` from a worker thread when run concurrently.
        list.append is atomic, so the list itself is safe; the spans can
        land out of order, which run() fixes by sorting on started_at.
        """
        client = self.experience_client or self.client
        with traced_step(trace, "overall_experience_extraction") as info:
            result = _require(
                self.overall_experience_agent.run(listing, redacted_cv),
                "Overall experience extraction failed.",
            )
            info["attempts"] = client.last_attempts
        return result

    def run(
        self, listing: JobListing, redacted_cv: RedactedCV, *, verbose: bool = True
    ) -> PipelineResult:
        started = time.time()
        trace_id = uuid7()
        trace: list[TraceSpan] = []
        say = print if verbose else (lambda *_: None)
        say(f" -> Trace ID: {trace_id}")

        # Started first and collected last: it takes no input from the two
        # steps below, and on Modal it is the long pole by a wide margin.
        # Its span is appended from the worker thread, so `trace` is sorted
        # by start time before it is returned.
        experience_pool: Optional[ThreadPoolExecutor] = None
        experience_future = None
        if self.experience_client is not None:
            say(" -> [1/4] Evaluating overall career experience (concurrently)...")
            experience_pool = ThreadPoolExecutor(max_workers=1)
            experience_future = experience_pool.submit(
                self._extract_experience, listing, redacted_cv, trace
            )

        try:
            say(" -> [2/4] Extracting job requirements...")
            with traced_step(trace, "job_requirements_extraction") as info:
                requirements_result = _require(
                    self.job_requirements_agent.run(listing),
                    "Job requirement extraction failed.",
                )
                info["attempts"] = self.client.last_attempts

            say(" -> [3/4] Evaluating extracted requirements against Candidate CV...")
            with traced_step(trace, "skill_matching") as info:
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
            say(
                f"       Matched {skills_result.total_matched_skills}/"
                f"{skills_result.total_job_requirements} skills "
                f"({skills_result.match_percentage}%)"
            )

            if experience_future is not None:
                overall_experience = experience_future.result()
            else:
                say(" -> [4/4] Evaluating overall relevant career experience...")
                overall_experience = self._extract_experience(listing, redacted_cv, trace)
        finally:
            # wait=False so a failure here doesn't block on an in-flight GPU
            # call nobody is waiting for any more.
            if experience_pool is not None:
                experience_pool.shutdown(wait=False)

        # The concurrent span is appended whenever the worker happens to
        # finish, so order the trace by when each step actually started --
        # otherwise an artifact reads as though experience ran last.
        trace.sort(key=lambda span: span.started_at)

        with traced_step(trace, "scoring"):
            scorecard = self.scoring_engine.calculate_scorecard(skills_result, overall_experience)

        execution_seconds = round(time.time() - started, 2)

        return PipelineResult(
            trace_id=trace_id,
            engine=self.client.model,
            # Sourced from the RedactedCV, not recomputed: this pipeline
            # never runs PII detection, it only reports which engine did.
            pii_engine=redacted_cv.pii_engine,
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
            redacted_cv_trace_id=redacted_cv.ingestion_trace_id,
            pii_spans=redacted_cv.pii_spans,
            trace=trace,
        )
