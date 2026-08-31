"""Response models for the public API."""

from pydantic import BaseModel

from src.schemas.evaluation import EvaluationReport
from src.schemas.experience import OverallExperienceOutput
from src.schemas.pipeline import PipelineMetrics
from src.schemas.requirements import SkillMatchResult
from src.schemas.scoring import Scorecard


class CompareResponse(BaseModel):
    artifact_path: str
    engine: str
    pii_engine: str
    execution_seconds: float
    metrics: PipelineMetrics
    scorecard: Scorecard
    scoring_weights: dict[str, float]
    skills_evaluation: SkillMatchResult
    overall_experience: OverallExperienceOutput
    # PASS/FAIL against configs/pipeline.yaml's `api_evaluation` thresholds,
    # mirroring what a harness task's `evaluation:` block gives a CLI run.
    # None when no criteria are configured (or none apply to this result
    # shape) -- an absent verdict, not a failed one.
    evaluation: EvaluationReport | None = None


class IngestResponse(BaseModel):
    """Raw CV in, redacted CV persisted, cv_id out. Pass cv_id to /api/match
    to evaluate it against any number of job listings without re-uploading
    or re-redacting the CV."""

    cv_id: str
    artifact_path: str
    pii_engine: str
    execution_seconds: float
    pii_span_count: int
    # PASS/FAIL against configs/pipeline.yaml's `api_evaluation` thresholds,
    # mirroring what a harness task's `evaluation:` block gives a CLI run.
    # None when no criteria are configured (or none apply to this result
    # shape) -- an absent verdict, not a failed one.
    evaluation: EvaluationReport | None = None


class MatchResponse(BaseModel):
    """Job listing + a previously-ingested cv_id in, match result out. No
    PII model is called for this endpoint — the CV arrived pre-redacted."""

    artifact_path: str
    engine: str
    execution_seconds: float
    metrics: PipelineMetrics
    scorecard: Scorecard
    scoring_weights: dict[str, float]
    skills_evaluation: SkillMatchResult
    overall_experience: OverallExperienceOutput
    # PASS/FAIL against configs/pipeline.yaml's `api_evaluation` thresholds,
    # mirroring what a harness task's `evaluation:` block gives a CLI run.
    # None when no criteria are configured (or none apply to this result
    # shape) -- an absent verdict, not a failed one.
    evaluation: EvaluationReport | None = None
