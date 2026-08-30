import os
import time
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import ConfigDict, Field, model_validator

from src.schemas.base import StrictBaseModel
from src.schemas.experience import OverallExperienceOutput
from src.schemas.pii import TextSpan
from src.schemas.requirements import SkillMatchResult
from src.schemas.scoring import Scorecard


def uuid7() -> UUID:
    """Generate a time-ordered UUID (RFC 9562 version 7).

    Unlike a random UUIDv4, a UUIDv7 sorts chronologically and encodes its
    creation time, so trace/run IDs generated with this can be compared,
    ordered, and correlated with logs without a separate timestamp lookup.
    `uuid.uuid7` only ships in the stdlib from Python 3.14; this project
    targets 3.12, so it's implemented directly here.
    """
    unix_ts_ms = int(time.time() * 1000)
    rand = int.from_bytes(os.urandom(10), "big")
    rand_a = (rand >> 68) & 0xFFF  # 12 random bits
    rand_b = rand & ((1 << 62) - 1)  # 62 random bits

    value = unix_ts_ms << 80
    value |= 0x7 << 76  # version 7
    value |= rand_a << 64
    value |= 0b10 << 62  # RFC 4122 variant
    value |= rand_b
    return UUID(int=value)


class TraceSpan(StrictBaseModel):
    """One timed step inside a pipeline run.

    Spans give a per-step latency breakdown and something concrete to
    correlate against structured logs, instead of only the run's final
    aggregate duration.
    """

    span_id: UUID = Field(default_factory=uuid4)
    step: str = Field(min_length=1)
    started_at: datetime
    duration_seconds: float = Field(ge=0.0)
    attempts: int | None = Field(
        default=None,
        ge=1,
        description="LLM call attempts (>1 means instructor retried after a "
        "validation failure). None for steps with no LLM call.",
    )


class PipelineMetrics(StrictBaseModel):
    total_requirements: int = Field(ge=0)
    total_matched: int = Field(ge=0)
    match_percentage: float = Field(ge=0.0, le=100.0)
    final_relevance: float = Field(ge=0.0, le=100.0)

    @model_validator(mode="after")
    def validate_match_counts(self):
        if self.total_matched > self.total_requirements:
            raise ValueError("total_matched cannot exceed total_requirements")
        return self


class PipelineResult(StrictBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    trace_id: UUID = Field(default_factory=uuid7)
    engine: str = Field(min_length=1)
    pii_engine: str = Field(min_length=1)
    execution_seconds: float = Field(ge=0.0)
    skills_eval: SkillMatchResult
    overall_experience: OverallExperienceOutput
    scorecard: Scorecard
    metrics: PipelineMetrics
    redacted_cv_trace_id: UUID = Field(
        description="trace_id of the IngestionPipeline run whose RedactedCV "
        "this result was matched against (RedactedCV.ingestion_trace_id) — "
        "not the redacted text itself, which already lives on that run's own "
        "artifact (see IngestionArtifact)."
    )
    pii_spans: list[TextSpan]
    trace: list[TraceSpan] = Field(default_factory=list)