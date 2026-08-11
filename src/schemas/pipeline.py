from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.schemas.experience import OverallExperienceOutput
from src.schemas.pii import TextSpan
from src.schemas.requirements import SkillMatchResult
from src.schemas.scoring import Scorecard


class PipelineMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_requirements: int = Field(ge=0)
    total_matched: int = Field(ge=0)
    match_percentage: float = Field(ge=0.0, le=100.0)
    final_relevance: float = Field(ge=0.0, le=100.0)

    @model_validator(mode="after")
    def validate_match_counts(self):
        if self.total_matched > self.total_requirements:
            raise ValueError("total_matched cannot exceed total_requirements")
        return self


class PipelineResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    engine: str = Field(min_length=1)
    pii_engine: str = Field(min_length=1)
    execution_seconds: float = Field(ge=0.0)
    skills_eval: SkillMatchResult
    overall_experience: OverallExperienceOutput
    scorecard: Scorecard
    metrics: PipelineMetrics
    redacted_cv: str
    pii_spans: list[TextSpan]