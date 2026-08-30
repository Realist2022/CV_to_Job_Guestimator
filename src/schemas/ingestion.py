"""Schemas for the standalone ingestion pipeline (see services/ingestion_pipeline.py).

RedactedCV is deliberately a distinct type from CandidateCV/SourceDocument:
it is the only CV representation MatchingPipeline is allowed to consume, so
that boundary is a type distinction, not just a naming convention. Nothing
in this module can produce a RedactedCV from anything other than an
IngestionPipeline run.
"""

import hashlib
from datetime import datetime, timezone
from uuid import UUID

from pydantic import Field

from src.schemas.base import StrictBaseModel
from src.schemas.pii import TextSpan
from src.schemas.pipeline import TraceSpan, uuid7


class RedactedCV(StrictBaseModel):
    cv_id: str = Field(
        min_length=1,
        description="sha256 of the normalised raw CV text. Content-addressed "
        "so re-ingesting identical text is idempotent, and the id can be "
        "recomputed from a raw CV without the store ever retaining raw text.",
    )
    text: str = Field(description="The redacted CV text — the only content persisted.")
    pii_spans: list[TextSpan]
    pii_engine: str = Field(min_length=1)
    ingestion_trace_id: UUID = Field(
        default_factory=uuid7,
        description="trace_id of the IngestionPipeline run that produced this "
        "RedactedCV — see IngestionResult.trace_id. Downstream consumers "
        "(MatchingPipeline, RunArtifact) carry this instead of the redacted "
        "text itself, so they can point back at the ingestion run/artifact "
        "that already has it on record rather than duplicating it.",
    )
    redacted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_raw_text(
        cls,
        raw_text: str,
        redacted_text: str,
        pii_spans: list[TextSpan],
        pii_engine: str,
        ingestion_trace_id: UUID | None = None,
    ) -> "RedactedCV":
        # Imported lazily to avoid a schemas -> services import at module
        # load time; only needed for the id's normalisation step.
        from src.services.document_parser import normalise

        cv_id = hashlib.sha256(normalise(raw_text).encode("utf-8")).hexdigest()
        kwargs = {"cv_id": cv_id, "text": redacted_text, "pii_spans": pii_spans, "pii_engine": pii_engine}
        if ingestion_trace_id is not None:
            kwargs["ingestion_trace_id"] = ingestion_trace_id
        return cls(**kwargs)


class IngestionResult(StrictBaseModel):
    trace_id: UUID = Field(default_factory=uuid7)
    cv_id: str = Field(min_length=1)
    pii_engine: str = Field(min_length=1)
    execution_seconds: float = Field(ge=0.0)
    pii_spans: list[TextSpan]
    trace: list[TraceSpan] = Field(default_factory=list)
    redacted_cv: RedactedCV
