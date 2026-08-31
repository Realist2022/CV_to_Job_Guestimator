from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import Field

from src.schemas.base import StrictBaseModel
from src.schemas.evaluation import EvaluationReport
from src.schemas.experience import OverallExperienceOutput
from src.schemas.ingestion import IngestionResult
from src.schemas.pii import TextSpan
from src.schemas.pipeline import PipelineMetrics, PipelineResult, TraceSpan, uuid7
from src.schemas.requirements import SkillMatchResult
from src.schemas.scoring import Scorecard


class RunModelConfig(StrictBaseModel):
    """The resolved model config actually used for one role in a run."""

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

    @classmethod
    def from_client(cls, client, *, name: str, fallback_used: bool = False) -> "RunModelConfig":
        """Build from the client that actually served a run's completions.

        `client` is typically an InstructorClient or FallbackInstructorClient
        (see src/services/llm_client.py) — anything duck-typing `.model` and
        `.temperature`. Centralizes the `engine`/`temperature` extraction that
        every caller (CLI harness, web API) otherwise repeated identically;
        `fallback_used` stays a plain kwarg since detecting it is caller-
        specific (e.g. routes.py's `_fallback_used`, which only a
        FallbackInstructorClient can report).
        """
        return cls(
            name=name,
            engine=client.model,
            temperature=client.temperature,
            fallback_used=fallback_used,
        )


class PIIRunConfig(StrictBaseModel):
    """Which PII detector is behind a run's redacted CV.

    Deliberately not RunModelConfig: PII redaction runs entirely through
    presidio with no LLM in the loop (see pii_base.py), so that schema's
    `temperature` and `fallback_used` were dead weight here -- hardcoded 0.0
    and always False, left over from when a `pii` model role existed and
    "model"/"regex" detectors were selectable (removed in 1b5e3cc, Aug 2026).
    Reporting them invited the reading that PII had a model that could fall
    back to the cloud. It cannot, and never could.

    `ran_this_run` is the field that distinction actually needs: on a
    "matching" run no detector executes at all -- the CV arrived pre-redacted
    -- so `engine` names whichever detector produced that *earlier*
    redaction. Previously that was only inferrable from `pii_detectors: []`
    two fields up.
    """

    name: str = Field(min_length=1)
    engine: str = Field(min_length=1, description="Detector string, e.g. 'presidio:en_core_web_sm'.")
    ran_this_run: bool = Field(
        description=(
            "True if redaction executed during this run ('extraction' / "
            "'ingestion'). False if the CV arrived already redacted "
            "('matching'), in which case `engine` is historical provenance "
            "read off the stored RedactedCV, not work this run performed."
        )
    )


class RunConfig(StrictBaseModel):
    """Snapshot of the config that produced a run, for later reproducibility.

    Metrics and scores are meaningless in isolation once the config that
    produced them can no longer be reconstructed; this captures it alongside
    the result instead of leaving it to whatever configs/*.yaml happen to
    contain later.
    """

    # Together these record what produced the run: both set = a CLI run from
    # a tasks/*.yaml file; task_name only = a programmatic runner.run(TaskSpec)
    # call (see HarnessRunner.run); an "api:" prefix = a web endpoint, which
    # has no task file at all, so task_path stays None rather than inventing
    # a path that never existed. Both None means the source predates that
    # labelling (any web-API artifact logged before 2026-08-30).
    task_name: str | None = None
    task_path: str | None = None
    pipeline: str = Field(min_length=1)
    scoring_weights: dict[str, float]
    pii_detectors: list[str] = Field(default_factory=list)
    evaluation_model: RunModelConfig
    pii_model: PIIRunConfig
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


class ArtifactMetadata(StrictBaseModel):
    run_number: int = Field(gt=0)
    trace_id: UUID = Field(default_factory=uuid7)
    engine: str = Field(min_length=1)
    pii_engine: str = Field(min_length=1)
    execution_time_seconds: float = Field(ge=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RunArtifact(StrictBaseModel):
    # A stamp on what this schema writes, NOT a claim to read older versions.
    # It used to list every version back to "3.0", which read as backward
    # compatibility that never existed: StrictBaseModel forbids extra keys, so
    # 3.0-3.2 artifacts (they carry a `redacted_cv` field, dropped in 3.3 for
    # `redacted_cv_trace_id`) fail validation regardless of what this accepts.
    # Nothing in the repo reads an artifact back except tests validating what
    # they just wrote, so the honest move is to stamp one version and let
    # scripts/compare_runs.py handle older files as plain JSON.
    schema_version: Literal["3.5"] = "3.5"
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


class IngestionRunConfig(StrictBaseModel):
    """Snapshot of the config that produced an ingestion-only run.

    Deliberately not RunConfig: an ingestion run has no evaluation model
    and no scoring weights at all, so reusing RunConfig would mean either
    making those fields optional there (weakening the guarantee for
    "extraction"/"matching" runs, which always have both) or fabricating
    placeholder values here. Neither is honest, hence a separate schema.
    """

    # Same provenance convention as RunConfig's pair — see the comment there.
    task_name: str | None = None
    task_path: str | None = None
    pii_detectors: list[str] = Field(default_factory=list)
    pii_model: PIIRunConfig
    prompt_versions: dict[str, str] = Field(default_factory=dict)


class IngestionArtifactMetadata(StrictBaseModel):
    run_number: int = Field(gt=0)
    trace_id: UUID = Field(default_factory=uuid7)
    pii_engine: str = Field(min_length=1)
    execution_time_seconds: float = Field(ge=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IngestionArtifact(StrictBaseModel):
    """Logged run artifact for a standalone IngestionPipeline run.

    Distinct from RunArtifact for the same reason IngestionRunConfig is
    distinct from RunConfig: there's no skills/experience/scorecard to
    report, since matching never ran as part of this task.
    """

    # One stamp, not a compatibility claim — see RunArtifact.schema_version.
    schema_version: Literal["1.2"] = "1.2"
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
