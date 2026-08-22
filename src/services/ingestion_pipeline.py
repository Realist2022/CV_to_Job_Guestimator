"""Standalone PII ingestion pipeline.

Redacts a raw CV exactly once, independent of any job listing or matching
run. See matching_pipeline.py for the downstream half — it only ever
consumes this pipeline's RedactedCV output (via CVIngestionStore), never a
raw CandidateCV, and doesn't import CandidateCV or PIIDetector at all. That
is the actual boundary: not that nobody currently passes raw text through,
but that there is no code path on the matching side capable of it.
"""

import time
from datetime import datetime, timezone
from typing import Optional

from src.config import load_pii_detector_names
from src.schemas.ingestion import IngestionResult, RedactedCV
from src.schemas.pipeline import TraceSpan, uuid7
from src.services.document_parser import CandidateCV
from src.services.llm_client import InstructorClient
from src.services.pii_detector import PIIDetector, build_pii_detector


class IngestionPipeline:
    def __init__(
        self,
        pii_detector: Optional[PIIDetector] = None,
        pii_client: Optional[InstructorClient] = None,
    ):
        self.pii_client = pii_client or _default_pii_client()
        self.pii_detector = pii_detector or build_pii_detector(
            load_pii_detector_names(), self.pii_client
        )

    def run(self, cv: CandidateCV, *, verbose: bool = True) -> IngestionResult:
        started = time.time()
        trace_id = uuid7()
        say = print if verbose else (lambda *_: None)
        say(f" -> Trace ID: {trace_id}")
        say(" -> [1/1] Detecting and redacting PII from Candidate CV...")

        started_at = datetime.now(timezone.utc)
        step_started = time.time()
        spans = self.pii_detector.detect(cv)
        redacted = cv.redacted(spans)
        trace = [
            TraceSpan(
                step="pii_redaction",
                started_at=started_at,
                duration_seconds=round(time.time() - step_started, 3),
                attempts=self.pii_client.last_attempts,
            )
        ]
        say(f"       {len(spans)} PII spans redacted from CV.")

        redacted_cv = RedactedCV.from_raw_text(
            raw_text=cv.text,
            redacted_text=redacted.text,
            pii_spans=spans,
            pii_engine=self.pii_client.model,
            ingestion_trace_id=trace_id,
        )

        return IngestionResult(
            trace_id=trace_id,
            cv_id=redacted_cv.cv_id,
            pii_engine=self.pii_client.model,
            execution_seconds=round(time.time() - started, 2),
            pii_spans=spans,
            trace=trace,
            redacted_cv=redacted_cv,
        )


def _default_pii_client() -> InstructorClient:
    # Imported lazily to avoid a circular import through src.services.
    from src.model.adapters import client_for_role

    return client_for_role("pii")
