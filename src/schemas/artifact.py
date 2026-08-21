from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.evaluation import EvaluationReport
from src.schemas.experience import OverallExperienceOutput
from src.schemas.pipeline import PipelineMetrics, PipelineResult, TraceSpan, uuid7
from src.schemas.requirements import SkillMatchResult
from src.schemas.scoring import Scorecard


class RunModelConfig(BaseModel):
    """The resolved model config actually used for one role in a run."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Named config key from configs/llm.yaml.")
    engine: str = Field(min_length=1, description="Resolved model string sent to the provider.")
    temperature: float


class RunConfig(BaseModel):
    """Snapshot of the config that produced a run, for later reproducibility.

    Metrics and scores are meaningless in isolation once the config that
    produced them can no longer be reconstructed; this captures it alongside
    the result instead of leaving it to whatever configs/*.yaml happen to
    contain later.
    """

    model_config = ConfigDict(extra="forbid")

    task_name: str | None = None
    task_path: str | None = None
    pipeline: str = Field(min_length=1)
    scoring_weights: dict[str, float]
    pii_detectors: list[str] = Field(default_factory=list)
    evaluation_model: RunModelConfig
    pii_model: RunModelConfig


class ArtifactMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_number: int = Field(gt=0)
    trace_id: UUID = Field(default_factory=uuid7)
    engine: str = Field(min_length=1)
    pii_engine: str = Field(min_length=1)
    execution_time_seconds: float = Field(ge=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RunArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["3.0", "3.1", "3.2"] = "3.2"
    metadata: ArtifactMetadata
    config: RunConfig | None = None
    skills_evaluation: SkillMatchResult
    overall_experience: OverallExperienceOutput
    scorecard: Scorecard
    metrics: PipelineMetrics
    evaluation: EvaluationReport | None = None
    trace: list[TraceSpan] = Field(default_factory=list)
    redacted_cv: str

    @classmethod
    def from_pipeline_result(
        cls,
        result: PipelineResult,
        run_number: int,
        evaluation: EvaluationReport | None = None,
        config: RunConfig | None = None,
    ) -> "RunArtifact":
        return cls(
            metadata=ArtifactMetadata(
                run_number=run_number,
                trace_id=result.trace_id,
                engine=result.engine,
                pii_engine=result.pii_engine,
                execution_time_seconds=result.execution_seconds,
            ),
            config=config,
            skills_evaluation=result.skills_eval,
            overall_experience=result.overall_experience,
            scorecard=result.scorecard,
            metrics=result.metrics,
            evaluation=evaluation,
            trace=result.trace,
            redacted_cv=result.redacted_cv,
        )
