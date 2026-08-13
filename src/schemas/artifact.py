from datetime import datetime, timezone
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.evaluation import EvaluationReport
from src.schemas.experience import OverallExperienceOutput
from src.schemas.pipeline import PipelineMetrics, PipelineResult
from src.schemas.requirements import SkillMatchResult
from src.schemas.scoring import Scorecard


class ArtifactMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_number: int = Field(gt=0)
    run_id: UUID = Field(default_factory=uuid4)
    engine: str = Field(min_length=1)
    pii_engine: str = Field(min_length=1)
    execution_time_seconds: float = Field(ge=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RunArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["3.0", "3.1"] = "3.1"
    metadata: ArtifactMetadata
    skills_evaluation: SkillMatchResult
    overall_experience: OverallExperienceOutput
    scorecard: Scorecard
    metrics: PipelineMetrics
    evaluation: EvaluationReport | None = None
    redacted_cv: str

    @classmethod
    def from_pipeline_result(
        cls,
        result: PipelineResult,
        run_number: int,
        evaluation: EvaluationReport | None = None,
    ) -> "RunArtifact":
        return cls(
            metadata=ArtifactMetadata(
                run_number=run_number,
                engine=result.engine,
                pii_engine=result.pii_engine,
                execution_time_seconds=result.execution_seconds,
            ),
            skills_evaluation=result.skills_eval,
            overall_experience=result.overall_experience,
            scorecard=result.scorecard,
            metrics=result.metrics,
            evaluation=evaluation,
            redacted_cv=result.redacted_cv,
        )