"""HTTP layer only: parse the request, delegate execution to the harness.

Each endpoint does exactly three things — turn the multipart upload into
domain objects, ask harness_adapter to resolve a Resolved*Run from them,
and hand that to the shared HarnessRunner (one instance, on app.state; see
src/api/app.py). All pipeline execution, threshold evaluation, and artifact
persistence happens inside HarnessRunner.run_* — the same code path the CLI
harness (main.py -> run_task) goes through.
"""

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from src.api import harness_adapter
from src.api.schemas import CompareResponse, IngestResponse, MatchResponse
from src.harness.runner import HarnessRunner
from src.services import CandidateCV, JobListing, PDFTextExtractionError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

router = APIRouter()


def _runner(request: Request) -> HarnessRunner:
    return request.app.state.harness_runner


@router.post("/api/compare", response_model=CompareResponse)
async def compare_documents(
    request: Request,
    job_listing: UploadFile = File(...),
    candidate_cv: UploadFile = File(...),
    skills_weight: float | None = Form(None),
    work_experience_weight: float | None = Form(None),
) -> CompareResponse:
    try:
        listing = await _load_document(job_listing, JobListing, "job listing")
        cv = await _load_document(candidate_cv, CandidateCV, "candidate CV")

        resolved = harness_adapter.resolve_extraction(
            listing,
            cv,
            skills_weight=skills_weight,
            work_experience_weight=work_experience_weight,
        )
        report = _runner(request).run_extraction(resolved)
    except (PDFTextExtractionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Compare failed: {type(exc).__name__}: {exc}",
        ) from exc

    result = report.result
    return CompareResponse(
        artifact_path=report.artifact_path,
        engine=result.engine,
        pii_engine=result.pii_engine,
        execution_seconds=result.execution_seconds,
        metrics=result.metrics,
        scorecard=result.scorecard,
        scoring_weights=resolved.scoring_engine.weights,
        skills_evaluation=result.skills_eval,
        overall_experience=result.overall_experience,
    )


@router.post("/api/ingest", response_model=IngestResponse)
async def ingest_cv(
    request: Request, candidate_cv: UploadFile = File(...)
) -> IngestResponse:
    """Redact a raw CV once and persist it to CVIngestionStore. Returns a
    cv_id — pass it to /api/match to evaluate against any number of job
    listings without re-uploading or re-redacting the CV. The response
    never carries pii_spans/redacted text: only a count, since the actual
    detected values are exactly the PII this endpoint exists to keep off
    the wire once ingestion is done."""
    try:
        cv = await _load_document(candidate_cv, CandidateCV, "candidate CV")
        report = _runner(request).run_ingestion(harness_adapter.resolve_ingestion(cv))
    except (PDFTextExtractionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Ingest failed: {type(exc).__name__}: {exc}",
        ) from exc

    result = report.result
    return IngestResponse(
        cv_id=result.cv_id,
        artifact_path=report.artifact_path,
        pii_engine=result.pii_engine,
        execution_seconds=result.execution_seconds,
        pii_span_count=len(result.pii_spans),
    )


@router.post("/api/match", response_model=MatchResponse)
async def match_cv(
    request: Request,
    job_listing: UploadFile = File(...),
    cv_id: str = Form(...),
    skills_weight: float | None = Form(None),
    work_experience_weight: float | None = Form(None),
) -> MatchResponse:
    """Match a job listing against a CV previously ingested via /api/ingest.
    No PII model is called here — the CV arrives already redacted."""
    try:
        listing = await _load_document(job_listing, JobListing, "job listing")

        resolved = harness_adapter.resolve_matching(
            listing,
            cv_id,
            skills_weight=skills_weight,
            work_experience_weight=work_experience_weight,
        )
        report = _runner(request).run_matching(resolved)
    except (PDFTextExtractionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Match failed: {type(exc).__name__}: {exc}",
        ) from exc

    result = report.result
    return MatchResponse(
        artifact_path=report.artifact_path,
        engine=result.engine,
        execution_seconds=result.execution_seconds,
        metrics=result.metrics,
        scorecard=result.scorecard,
        scoring_weights=resolved.scoring_engine.weights,
        skills_evaluation=result.skills_eval,
        overall_experience=result.overall_experience,
    )


async def _load_document[DocumentT: (JobListing, CandidateCV)](
    upload: UploadFile, document_type: type[DocumentT], label: str
) -> DocumentT:
    content = await upload.read()
    if not content:
        raise ValueError(f"The {label} upload is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(f"The {label} upload exceeds the 10 MB limit.")

    suffix = Path(upload.filename or "").suffix.lower()
    if suffix == ".txt":
        return document_type.from_text_bytes(content, label=label)

    if suffix != ".pdf":
        raise ValueError(f"The {label} must be a PDF or TXT file.")

    temporary_path = _write_temporary_upload(content, suffix)
    try:
        return document_type.from_pdf(str(temporary_path))
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_temporary_upload(content: bytes, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(content)
        return Path(handle.name)
