import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.api.schemas import CompareResponse, IngestResponse, MatchResponse
from src.config import (
    load_pii_detector_names,
    load_pipeline_model_names,
    load_scoring_weights,
)
from src.model.adapters import client_for_role
from src.schemas.artifact import IngestionRunConfig, RunConfig, RunModelConfig
from src.services import (
    CandidateCV,
    CVIngestionStore,
    CVNotFoundError,
    ExtractionPipeline,
    IngestionPipeline,
    JobListing,
    MatchingPipeline,
    PDFTextExtractionError,
    RelevanceScoringEngine,
)
from src.services.ingestion_persistence import persist_ingestion
from src.utils import ArtifactLogger

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

router = APIRouter()


@router.post("/api/compare", response_model=CompareResponse)
async def compare_documents(
    job_listing: UploadFile = File(...),
    candidate_cv: UploadFile = File(...),
    skills_weight: float | None = Form(None),
    work_experience_weight: float | None = Form(None),
) -> CompareResponse:
    try:
        scoring_engine = _scoring_engine(skills_weight, work_experience_weight)
        listing = await _load_document(job_listing, JobListing, "job listing")
        cv = await _load_document(candidate_cv, CandidateCV, "candidate CV")

        model_names = load_pipeline_model_names()
        eval_client = client_for_role("evaluation")
        pii_client = client_for_role("pii")

        pipeline = ExtractionPipeline(
            eval_client,
            pii_client=pii_client,
            scoring_engine=scoring_engine,
        )

        # So this run's redacted_cv_trace_id (see RunArtifact/PipelineResult)
        # always resolves to a real, persisted IngestionArtifact — the same
        # guarantee /api/ingest already gives, not something only the
        # ingest-then-match endpoint pair provides.
        ingestion_config = IngestionRunConfig(
            pii_detectors=load_pii_detector_names(),
            pii_model=RunModelConfig(
                name=model_names["pii"],
                engine=pii_client.model,
                temperature=pii_client.temperature,
            ),
        )

        result = pipeline.run(
            listing,
            cv,
            verbose=False,
            on_ingested=lambda ingestion_result: persist_ingestion(
                ingestion_result, ingestion_config
            ),
        )
        run_config = RunConfig(
            pipeline="extraction",
            scoring_weights=scoring_engine.weights,
            pii_detectors=load_pii_detector_names(),
            evaluation_model=RunModelConfig(
                name=model_names["evaluation"],
                engine=eval_client.model,
                temperature=eval_client.temperature,
            ),
            pii_model=RunModelConfig(
                name=model_names["pii"],
                engine=pii_client.model,
                temperature=pii_client.temperature,
            ),
        )
        artifact_path = ArtifactLogger(output_dir="artifacts").log_run(
            result, config=run_config
        )
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


@router.post("/api/ingest", response_model=IngestResponse)
async def ingest_cv(candidate_cv: UploadFile = File(...)) -> IngestResponse:
    """Redact a raw CV once and persist it to CVIngestionStore. Returns a
    cv_id — pass it to /api/match to evaluate against any number of job
    listings without re-uploading or re-redacting the CV. The response
    never carries pii_spans/redacted text: only a count, since the actual
    detected values are exactly the PII this endpoint exists to keep off
    the wire once ingestion is done."""
    try:
        cv = await _load_document(candidate_cv, CandidateCV, "candidate CV")

        model_names = load_pipeline_model_names()
        pii_client = client_for_role("pii")
        pipeline = IngestionPipeline(pii_client=pii_client)
        result = pipeline.run(cv, verbose=False)

        config = IngestionRunConfig(
            pii_detectors=load_pii_detector_names(),
            pii_model=RunModelConfig(
                name=model_names["pii"],
                engine=pii_client.model,
                temperature=pii_client.temperature,
            ),
        )
        artifact_path, _ = persist_ingestion(result, config)
    except (PDFTextExtractionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Ingest failed: {type(exc).__name__}: {exc}",
        ) from exc

    return IngestResponse(
        cv_id=result.cv_id,
        artifact_path=artifact_path,
        pii_engine=result.pii_engine,
        execution_seconds=result.execution_seconds,
        pii_span_count=len(result.pii_spans),
    )


@router.post("/api/match", response_model=MatchResponse)
async def match_cv(
    job_listing: UploadFile = File(...),
    cv_id: str = Form(...),
    skills_weight: float | None = Form(None),
    work_experience_weight: float | None = Form(None),
) -> MatchResponse:
    """Match a job listing against a CV previously ingested via /api/ingest.
    No PII model is called here — the CV arrives already redacted."""
    try:
        scoring_engine = _scoring_engine(skills_weight, work_experience_weight)
        listing = await _load_document(job_listing, JobListing, "job listing")
        try:
            redacted_cv = CVIngestionStore().load(cv_id)
        except CVNotFoundError as exc:
            raise ValueError(str(exc)) from exc

        eval_client = client_for_role("evaluation")
        pipeline = MatchingPipeline(eval_client, scoring_engine=scoring_engine)
        result = pipeline.run(listing, redacted_cv, verbose=False)

        run_config = RunConfig(
            pipeline="matching",
            scoring_weights=scoring_engine.weights,
            pii_detectors=[],
            evaluation_model=RunModelConfig(
                name=load_pipeline_model_names()["evaluation"],
                engine=eval_client.model,
                temperature=eval_client.temperature,
            ),
            pii_model=RunModelConfig(
                name=redacted_cv.pii_engine,
                engine=redacted_cv.pii_engine,
                temperature=0.0,
            ),
        )
        artifact_path = ArtifactLogger(output_dir="artifacts").log_run(
            result, config=run_config
        )
    except (PDFTextExtractionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Match failed: {type(exc).__name__}: {exc}",
        ) from exc

    return MatchResponse(
        artifact_path=artifact_path,
        engine=result.engine,
        execution_seconds=result.execution_seconds,
        metrics=result.metrics,
        scorecard=result.scorecard,
        scoring_weights=scoring_engine.weights,
        skills_evaluation=result.skills_eval,
        overall_experience=result.overall_experience,
    )


def _scoring_engine(
    skills_weight: float | None,
    work_experience_weight: float | None,
) -> RelevanceScoringEngine:
    weights = load_scoring_weights()
    if skills_weight is not None:
        weights["skills_match"] = skills_weight
    if work_experience_weight is not None:
        weights["work_experience"] = work_experience_weight
    if any(weight < 0 or weight > 1 for weight in weights.values()):
        raise ValueError("Scoring weights must be between 0.0 and 1.0.")
    return RelevanceScoringEngine(weights)


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
