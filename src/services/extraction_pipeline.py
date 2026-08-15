import time
from typing import Optional
from src.schemas.pipeline import PipelineMetrics, PipelineResult
from src.schemas.requirements import SkillMatchResult
from src.services.document_parser import CandidateCV, JobListing
from src.services.llm_client import InstructorClient
from src.services.agents import (
    JobRequirementsAgent,
    OverallExperienceAgent,
    PIIAgent,
    SkillMatcherAgent,
)
from src.services.pii_detector import (
    PIIDetector,
    CompositePIIDetector,
    RegexPIIDetector,
    ModelPIIDetector,
)
from src.services.scoring_engine import RelevanceScoringEngine


def _require(value, failure_message: str):
    if value is None:
        raise RuntimeError(failure_message)
    return value


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
        self.pii_detector = pii_detector or CompositePIIDetector(
            RegexPIIDetector(), ModelPIIDetector(PIIAgent(self.pii_client))
        )

    def run(
        self, listing: JobListing, cv: CandidateCV, *, verbose: bool = True
    ) -> PipelineResult:
        started = time.time()
        say = print if verbose else (lambda *_: None)

        # STEP 1: Detect and redact PII from the Candidate CV ONLY
        say(" -> [1/5] Detecting and redacting PII from Candidate CV...")
        spans = self.pii_detector.detect(cv)
        redacted_cv = cv.redacted(spans)
        say(f"       {len(spans)} PII spans redacted from CV.")

        # STEP 2: Extract an authoritative requirement list from the untouched listing
        say(" -> [2/5] Extracting job requirements...")
        requirements_result = _require(
            self.job_requirements_agent.run(listing),
            "Job requirement extraction failed.",
        )

        # STEP 3: Classify only the extracted requirements against the redacted CV
        say(" -> [3/5] Evaluating extracted requirements against Candidate CV...")
        evaluation = _require(
            self.skill_matcher_agent.run(
                job_requirements=requirements_result.job_requirements,
                cv=redacted_cv,
            ),
            "Skill matching evaluation failed.",
        )

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
        overall_experience = _require(
            self.overall_experience_agent.run(listing, redacted_cv),
            "Overall experience extraction failed.",
        )

        scorecard = self.scoring_engine.calculate_scorecard(
            skills_result,
            overall_experience,
        )

        execution_seconds = round(time.time() - started, 2)

        return PipelineResult(
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
        )


def _default_pii_client() -> InstructorClient:
    # Imported lazily to avoid a circular import through src.services.
    from src.model.adapters import client_for_role

    return client_for_role("pii")