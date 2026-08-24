from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.evaluation import EvaluationReport
from src.schemas.experience import OverallExperienceOutput
from src.schemas.ingestion import IngestionResult
from src.schemas.pii import TextSpan
from src.schemas.pipeline import PipelineMetrics, PipelineResult, TraceSpan, uuid7
from src.schemas.requirements import SkillMatchResult
from src.schemas.scoring import Scorecard


class RunModelConfig(BaseModel):
    """The resolved model config actually used for one role in a run."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Named config key from configs/llm.yaml.")
    engine: str = Field(min_length=1, description="Resolved model string sent to the provider.")
    temperature: float
    fallback_used: bool = Field(
        default=False,
        description=(
            "True if `name`'s primary model failed for this run and `engine` "
            "reflects a fallback model instead (see configs/pipeline.yaml "
            "'fallback_models' and FallbackInstructorClient)."
        ),
    )


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
    prompt_versions: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Version of each system prompt (src/prompts/templates.py) that "
            "actually ran for this run, keyed by prompt name (e.g. "
            "'job_requirements', 'skill_matcher', 'overall_experience', "
            "'pii'). Keyed per prompt rather than per model role since "
            "evaluation_model alone covers three distinct prompts."
        ),
    )


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

    schema_version: Literal["3.0", "3.1", "3.2", "3.3", "3.4"] = "3.4"
    metadata: ArtifactMetadata
    config: RunConfig | None = None
    skills_evaluation: SkillMatchResult
    overall_experience: OverallExperienceOutput
    scorecard: Scorecard
    metrics: PipelineMetrics
    evaluation: EvaluationReport | None = None
    trace: list[TraceSpan] = Field(default_factory=list)
    redacted_cv_trace_id: UUID = Field(
        description="trace_id of the IngestionArtifact that already has this "
        "run's redacted CV text on record — see RedactedCV.ingestion_trace_id. "
        "Deliberately not the text itself: this run's artifact only needs to "
        "point at the one place that text is stored, not duplicate it."
    )

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
            redacted_cv_trace_id=result.redacted_cv_trace_id,
        )


class IngestionRunConfig(BaseModel):
    """Snapshot of the config that produced an ingestion-only run.

    Deliberately not RunConfig: an ingestion run has no evaluation model
    and no scoring weights at all, so reusing RunConfig would mean either
    making those fields optional there (weakening the guarantee for
    "extraction"/"matching" runs, which always have both) or fabricating
    placeholder values here. Neither is honest, hence a separate schema.
    """

    model_config = ConfigDict(extra="forbid")

    task_name: str | None = None
    task_path: str | None = None
    pii_detectors: list[str] = Field(default_factory=list)
    pii_model: RunModelConfig
    prompt_versions: dict[str, str] = Field(default_factory=dict)


class IngestionArtifactMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_number: int = Field(gt=0)
    trace_id: UUID = Field(default_factory=uuid7)
    pii_engine: str = Field(min_length=1)
    execution_time_seconds: float = Field(ge=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IngestionArtifact(BaseModel):
    """Logged run artifact for a standalone IngestionPipeline run.

    Distinct from RunArtifact for the same reason IngestionRunConfig is
    distinct from RunConfig: there's no skills/experience/scorecard to
    report, since matching never ran as part of this task.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0", "1.1"] = "1.1"
    metadata: IngestionArtifactMetadata
    config: IngestionRunConfig | None = None
    cv_id: str = Field(min_length=1)
    pii_spans: list[TextSpan]
    evaluation: EvaluationReport | None = None
    trace: list[TraceSpan] = Field(default_factory=list)
    redacted_cv: str

    @classmethod
    def from_ingestion_result(
        cls,
        result: IngestionResult,
        run_number: int,
        evaluation: EvaluationReport | None = None,
        config: IngestionRunConfig | None = None,
    ) -> "IngestionArtifact":
        return cls(
            metadata=IngestionArtifactMetadata(
                run_number=run_number,
                trace_id=result.trace_id,
                pii_engine=result.pii_engine,
                execution_time_seconds=result.execution_seconds,
            ),
            config=config,
            cv_id=result.cv_id,
            pii_spans=result.pii_spans,
            evaluation=evaluation,
            trace=result.trace,
            redacted_cv=result.redacted_cv.text,
        )
