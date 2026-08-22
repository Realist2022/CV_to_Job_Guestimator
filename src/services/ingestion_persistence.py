"""Shared "save RedactedCV + log its IngestionArtifact" glue.

Used by both the CLI harness (HarnessRunner) and the web API (routes.py) so
the guarantee documented on RedactedCV.ingestion_trace_id / PipelineResult
.redacted_cv_trace_id — that it resolves to a real, persisted RedactedCV and
a real logged IngestionArtifact — is implemented exactly once, not
reimplemented at each call site with the same two calls in the same order.

Covers three callers today: the standalone "ingestion" task, the
"extraction" task's on_ingested hook, and /api/compare's and /api/ingest's
equivalents in routes.py. All four give the same on-disk guarantee.
"""

from pathlib import Path

from src.schemas.artifact import IngestionRunConfig
from src.schemas.evaluation import EvaluationReport
from src.schemas.ingestion import IngestionResult
from src.services.cv_store import CVIngestionStore
from src.utils.artifact_logger import ArtifactLogger


def persist_ingestion(
    result: IngestionResult,
    config: IngestionRunConfig,
    *,
    evaluation: EvaluationReport | None = None,
    redacted_cv_dir: str | Path = "redacted_cvs",
    artifacts_dir: str | Path = "artifacts",
) -> tuple[str, int | None]:
    """Save `result.redacted_cv` and log its IngestionArtifact.

    Returns (artifact_path, run_number) — the same pair every caller already
    needed to report back (HarnessRunReport.artifact_path/run_number,
    IngestResponse.artifact_path).
    """
    CVIngestionStore(output_dir=redacted_cv_dir).save(result.redacted_cv)
    logger = ArtifactLogger(output_dir=artifacts_dir)
    artifact_path = logger.log_ingestion_run(result, evaluation=evaluation, config=config)
    return artifact_path, logger.last_run_number
