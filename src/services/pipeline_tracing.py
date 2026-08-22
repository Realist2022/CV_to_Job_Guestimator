"""Shared step-timing helper for pipeline `run()` methods.

Extracted out of extraction_pipeline.py so IngestionPipeline and
MatchingPipeline (its two standalone successors — see those modules) can
each build a TraceSpan list the same way without duplicating this context
manager three times.
"""

import time
from contextlib import contextmanager
from datetime import datetime, timezone

from src.schemas.pipeline import TraceSpan


@contextmanager
def traced_step(trace: list[TraceSpan], step: str):
    """Time one pipeline step and append it to `trace` as a TraceSpan.

    Yields a dict the caller can write `attempts` into (see InstructorClient
    .last_attempts) to surface LLM retries in the trace.
    """
    started_at = datetime.now(timezone.utc)
    started = time.time()
    info: dict = {}
    yield info
    trace.append(
        TraceSpan(
            step=step,
            started_at=started_at,
            duration_seconds=round(time.time() - started, 3),
            attempts=info.get("attempts"),
        )
    )
