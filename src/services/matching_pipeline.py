"""Standalone skill-matching / experience pipeline.

Consumes an already-redacted CV (RedactedCV, produced once by
IngestionPipeline — see ingestion_pipeline.py) and a JobListing. This
module deliberately never imports CandidateCV or PIIDetector: there is no
code path here capable of loading or seeing an unredacted CV, so "matching
never touches raw PII" is enforced by the import graph, not just by
discipline about which argument gets passed where.
"""

import time
from typing import Optional

from src.schemas.ingestion import RedactedCV
from src.schemas.pipeline import PipelineMetrics, PipelineResult, TraceSpan, uuid7
from src.schemas.requirements import SkillMatchResult
from src.services.agents import JobRequirementsAgent, OverallExperienceAgent, SkillMatcherAgent
from src.services.document_parser import JobListing
from src.services.llm_client import InstructorClient
from src.services.pipeline_tracing import traced_step
from src.services.scoring_engine import RelevanceScoringEngine


def _require(value, failure_message: str):
    if value is None:
        raise RuntimeError(failure_message)
    return value


class MatchingPipeline:
    def __init__(
        self,
        client: InstructorClient,
        scoring_engine: Optional[RelevanceScoringEngine] = None,
    ):
        self.client = client
        self.job_requirements_agent = JobRequirementsAgent(client)
        self.skill_matcher_agent = SkillMatcherAgent(client)
        self.overall_experience_agent = OverallExperienceAgent(client)
        self.scoring_engine = scoring_engine or RelevanceScoringEngine()

    def run(
        self, listing: JobListing, redacted_cv: RedactedCV, *, verbose: bool = True
    ) -> PipelineResult:
        started = time.time()
        trace_id = uuid7()
        trace: list[TraceSpan] = []
        say = print if verbose else (lambda *_: None)
        say(f" -> Trace ID: {trace_id}")

        say(" -> [1/4] Extracting job requirements...")
        with traced_step(trace, "job_requirements_extraction") as info:
            requirements_result = _require(
                self.job_requirements_agent.run(listing),
                "Job requirement extraction failed.",
            )
            info["attempts"] = self.client.last_attempts

        say(" -> [2/4] Evaluating extracted requirements against Candidate CV...")
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
        say(f"       Matched {skills_result.total_matched_skills}/{skills_result.total_job_requirements} skills "
            f"({skills_result.match_percentage}%)")

        say(" -> [3/4] Evaluating overall relevant career experience...")
        with traced_step(trace, "overall_experience_extraction") as info:
            overall_experience = _require(
                self.overall_experience_agent.run(listing, redacted_cv),
                "Overall experience extraction failed.",
            )
            info["attempts"] = self.client.last_attempts

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
