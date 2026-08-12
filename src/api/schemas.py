"""Response models for the public API."""

from pydantic import BaseModel

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
