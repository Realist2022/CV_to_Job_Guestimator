from pathlib import Path
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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
)
from src.utils import ArtifactLogger


STATIC_DIR = Path(__file__).resolve().parent / "web_static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(title="CV to Job Guestimator")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/compare")
async def compare_documents(
    job_listing: UploadFile = File(...),
    candidate_cv: UploadFile = File(...),
) -> dict:
    try:
        listing = await _load_document(job_listing, JobListing, "job listing")
        cv = await _load_document(candidate_cv, CandidateCV, "candidate CV")

        pipeline = ExtractionPipeline(_evaluation_client(), pii_client=_pii_client())
        result = pipeline.run(listing, cv, verbose=False)
        artifact_path = ArtifactLogger(output_dir="artifacts").log_run(result)
    except (PDFTextExtractionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Compare failed: {type(exc).__name__}: {exc}",
        ) from exc

    return {
        "artifact_path": artifact_path,
        "engine": result.engine,
        "pii_engine": result.pii_engine,
        "execution_seconds": result.execution_seconds,
        "metrics": result.metrics.model_dump(mode="json"),
        "scorecard": result.scorecard.model_dump(mode="json"),
        "skills_evaluation": result.skills_eval.model_dump(mode="json"),
        "skill_tenure": result.skill_tenure.model_dump(mode="json"),
        "overall_experience": result.overall_experience.model_dump(mode="json"),
    }


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
        text = _decode_text_upload(content, label)
        if not text.strip():
            raise ValueError(f"The {label} text file did not contain readable text.")
        return document_type(text)

    if suffix != ".pdf":
        raise ValueError(f"The {label} must be a PDF or TXT file.")

    temporary_path = _write_temporary_upload(content, suffix)
    try:
        return document_type.from_pdf(str(temporary_path))
    finally:
        temporary_path.unlink(missing_ok=True)


def _decode_text_upload(content: bytes, label: str) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"The {label} text file could not be decoded as text.")


def _write_temporary_upload(content: bytes, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(content)
        return Path(handle.name)