import time
from typing import Optional
from src.schemas.experience import SkillTenureOutput
from src.schemas.pipeline import PipelineMetrics, PipelineResult
from src.schemas.requirements import SkillMatchResult
from src.services.document_parser import CandidateCV, JobListing
from src.services.llm_client import InstructorClient
from src.services.agents import (
    JobRequirementsAgent,
    OverallExperienceAgent,
    PIIAgent,
    SkillMatcherAgent,
    SkillTenureAgent,
)
from src.services.pii_detector import (
    PIIDetector,
    CompositePIIDetector,
    RegexPIIDetector,
    ModelPIIDetector,
)
from src.services.scoring_engine import RelevanceScoringEngine
from src.config import PII_MODEL_NAME, PII_MODEL_BASE_URL, PII_MODEL_API_KEY


class ExtractionPipeline:
    def __init__(
        self,
        client: InstructorClient,
        pii_detector: Optional[PIIDetector] = None,
        pii_client: Optional[InstructorClient] = None,
    ):
        self.client = client
        self.job_requirements_agent = JobRequirementsAgent(client)
        self.skill_matcher_agent = SkillMatcherAgent(client)
        self.skill_tenure_agent = SkillTenureAgent(client)
        self.overall_experience_agent = OverallExperienceAgent(client)
        self.scoring_engine = RelevanceScoringEngine()
        self.pii_client = pii_client or InstructorClient(
            model=PII_MODEL_NAME,
            base_url=PII_MODEL_BASE_URL,
            api_key=PII_MODEL_API_KEY,
        )
        self.pii_detector = pii_detector or CompositePIIDetector(
            RegexPIIDetector(), ModelPIIDetector(PIIAgent(self.pii_client))
        )

    def run(
        self, listing: JobListing, cv: CandidateCV, *, verbose: bool = True
    ) -> PipelineResult:
        started = time.time()

        # STEP 1: Detect and redact PII from the Candidate CV ONLY
        if verbose:
            print(" -> [1/5] Detecting and redacting PII from Candidate CV...")
        spans = self.pii_detector.detect(cv)
        redacted_cv = cv.redacted(spans)

        if verbose:
            print(f"       {len(spans)} PII spans redacted from CV.")

        # STEP 2: Extract an authoritative requirement list from the untouched listing
        if verbose:
            print(" -> [2/5] Extracting job requirements...")

        requirements_result = self.job_requirements_agent.run(listing)
        if requirements_result is None:
            raise RuntimeError("Job requirement extraction failed.")

        # STEP 3: Classify only the extracted requirements against the redacted CV
        if verbose:
            print(" -> [3/5] Evaluating extracted requirements against Candidate CV...")

        evaluation = self.skill_matcher_agent.run(
            job_requirements=requirements_result.job_requirements,
            cv=redacted_cv,
        )
        if evaluation is None:
            raise RuntimeError("Skill matching evaluation failed.")

        skills_result = SkillMatchResult(
            job_requirements=requirements_result.job_requirements,
            matched_cv_skills=evaluation.matched_cv_skills,
            missing_cv_skills=evaluation.missing_cv_skills,
            rationale=evaluation.rationale,
        )

        if verbose:
            print(f"       Matched {skills_result.total_matched_skills}/{skills_result.total_job_requirements} skills "
                  f"({skills_result.match_percentage}%)")

        # STEP 4: Extract and classify the candidate's roles from the redacted CV
        if verbose:
            print(" -> [4/5] Evaluating overall relevant career experience...")
        overall_experience = self.overall_experience_agent.run(listing, redacted_cv)
        if overall_experience is None:
            raise RuntimeError("Overall experience extraction failed.")

        # STEP 5: Associate matched requirements with the extracted dated roles
        if verbose:
            print(" -> [5/5] Measuring tenure for matched requirements...")
        if any(
            requirement.minimum_commercial_years is not None
            for requirement in skills_result.job_requirements
        ):
            skill_tenure = self.skill_tenure_agent.run(
                job_requirements=skills_result.job_requirements,
                overall_experience=overall_experience,
                cv=redacted_cv,
            )
            if skill_tenure is None:
                raise RuntimeError("Skill tenure extraction failed.")
        else:
            skill_tenure = SkillTenureOutput(skills=[])

        scorecard = self.scoring_engine.calculate_scorecard(
            skills_result,
            skill_tenure,
            overall_experience,
        )

        execution_seconds = round(time.time() - started, 2)

        return PipelineResult(
            engine=self.client.model,
            pii_engine=self.pii_client.model,
            execution_seconds=execution_seconds,
            skills_eval=skills_result,
            skill_tenure=skill_tenure,
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