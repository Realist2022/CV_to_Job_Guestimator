from pathlib import Path
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.api.schemas import CompareResponse
from src.config import (
    MODEL_API_KEY,
    MODEL_BASE_URL,
    MODEL_NAME,
    PII_MODEL_API_KEY,
    PII_MODEL_BASE_URL,
    PII_MODEL_NAME,
)
from src.services import (
    CandidateCV,
    ExtractionPipeline,
    InstructorClient,
    JobListing,
    PDFTextExtractionError,
    RelevanceScoringEngine,
)
from src.utils import ArtifactLogger

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

router = APIRouter()


@router.post("/api/compare", response_model=CompareResponse)
async def compare_documents(
    job_listing: UploadFile = File(...),
    candidate_cv: UploadFile = File(...),
    skills_weight: float = Form(0.60),
    work_experience_weight: float = Form(0.40),
) -> CompareResponse:
    try:
        scoring_engine = _scoring_engine(skills_weight, work_experience_weight)
        listing = await _load_document(job_listing, JobListing, "job listing")
        cv = await _load_document(candidate_cv, CandidateCV, "candidate CV")

        pipeline = ExtractionPipeline(
            _evaluation_client(),
            pii_client=_pii_client(),
            scoring_engine=scoring_engine,
        )
        result = pipeline.run(listing, cv, verbose=False)
        artifact_path = ArtifactLogger(output_dir="artifacts").log_run(result)
    except (PDFTextExtractionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Compare failed: {type(exc).__name__}: {exc}",
        ) from exc

    return CompareResponse(
        artifact_path=artifact_path,
        engine=result.engine,
        pii_engine=result.pii_engine,
        execution_seconds=result.execution_seconds,
        metrics=result.metrics,
        scorecard=result.scorecard,
        scoring_weights=scoring_engine.weights,
        skills_evaluation=result.skills_eval,
        overall_experience=result.overall_experience,
    )


def _scoring_engine(
    skills_weight: float,
    work_experience_weight: float,
) -> RelevanceScoringEngine:
    weights = {
        "skills_match": skills_weight,
        "work_experience": work_experience_weight,
    }
    if any(weight < 0 or weight > 1 for weight in weights.values()):
        raise ValueError("Scoring weights must be between 0.0 and 1.0.")
    return RelevanceScoringEngine(weights)


def _evaluation_client() -> InstructorClient:
    return InstructorClient(
        model=MODEL_NAME,
        base_url=MODEL_BASE_URL,
        api_key=MODEL_API_KEY,
    )


def _pii_client() -> InstructorClient:
    return InstructorClient(
        model=PII_MODEL_NAME,
        base_url=PII_MODEL_BASE_URL,
        api_key=PII_MODEL_API_KEY,
    )


async def _load_document(
    upload: UploadFile, document_type: type[JobListing] | type[CandidateCV], label: str
) -> JobListing | CandidateCV:
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
