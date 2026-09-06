import logging
import tempfile
import urllib.request
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from src.api.auth import require_api_key
from src.api.schemas import CompareResponse, IngestResponse, MatchResponse, WarmResponse
from src.config import (
    load_default_evaluation_criteria,
    load_pii_detector_names,
    load_pipeline_model_names,
    load_scoring_weights,
)
from src.harness.evaluator import ThresholdEvaluator, resolve_criteria
from src.model.adapters import client_for_role, endpoint_for_role
from src.prompts.templates import EXTRACTION_PROMPT_VERSIONS, MATCHING_PROMPT_VERSIONS
from src.schemas.artifact import IngestionRunConfig, RunConfig, RunModelConfig
from src.schemas.evaluation import EvaluationReport
from src.services import (
    CandidateCV,
    ExtractionPipeline,
    FallbackInstructorClient,
    IngestionPipeline,
    JobListing,
    MatchingPipeline,
    PDFTextExtractionError,
    RelevanceScoringEngine,
    ServingCVUnavailableError,
    load_serving_cv,
)
from src.services.ingestion_persistence import persist_ingestion
from src.services.pii_base import pii_run_model_config
from src.utils.gcs_artifact_logger import build_artifact_logger

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Generous: a Modal cold start is minutes, and the container only keeps
# booting while a request is in flight.
WARM_TIMEOUT_SECONDS = 300

logger = logging.getLogger(__name__)

# The public surface. One endpoint, one CV: a visitor supplies a job
# listing and nothing else. Nobody uploads a CV here, so no PII detection
# runs on the request path and this deployment never holds anyone else's
# personal data (see load_serving_cv).
router = APIRouter(dependencies=[Depends(require_api_key)])

# Everything that accepts a `candidate_cv` upload. NOT mounted by default
# -- create_app() includes it only when ALLOW_CV_UPLOAD is set, which the
# deployed site does not set. Kept rather than deleted because both
# endpoints are still useful locally, and the CLI harness exercises the
# same pipelines from tasks/*.yaml either way.
upload_router = APIRouter(dependencies=[Depends(require_api_key)])


def _fallback_used(client: object) -> bool:
    """Whether `client`'s most recent call was served by its fallback model
    rather than its configured primary (always False for a plain
    InstructorClient with no fallback configured)."""
    return isinstance(client, FallbackInstructorClient) and client.fallback_used


def _api_evaluation(result) -> EvaluationReport | None:
    """Judge an API run against configs/pipeline.yaml's `default_evaluation`.

    The same baseline a harness task inherits (see resolve_criteria), so an
    /api/compare and an equivalent `uv run main.py` are held to one bar.
    There's no task to layer on top here -- an API request has no task file,
    which is the whole reason these artifacts used to log a null evaluation.

    Stays None when nothing is configured, so the artifact keeps saying
    "evaluation": null rather than carrying an empty report that claims a
    verdict nobody asked for.
    """
    criteria = resolve_criteria(result, defaults=load_default_evaluation_criteria())
    if not criteria.model_fields_set:
        return None
    return ThresholdEvaluator(criteria).evaluate(result)


@upload_router.post("/api/compare", response_model=CompareResponse)
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

        pipeline = ExtractionPipeline(eval_client, scoring_engine=scoring_engine)

        # So this run's redacted_cv_trace_id (see RunArtifact/PipelineResult)
        # always resolves to a real, persisted IngestionArtifact — the same
        # guarantee /api/ingest already gives, not something only the
        # ingest-then-match endpoint pair provides. pii_model is built from
        # ingestion_result.pii_engine (set once redaction actually ran)
        # rather than read off `pipeline` directly, so this keeps working
        # against any pipeline.run() that honors the on_ingested contract —
        # e.g. a test double standing in for ExtractionPipeline.
        result = pipeline.run(
            listing,
            cv,
            verbose=False,
            on_ingested=lambda ingestion_result: persist_ingestion(
                ingestion_result,
                IngestionRunConfig(
                    # A web-API run has no task file behind it, so task_path
                    # stays None -- fabricating one would be a lie (same
                    # reasoning as IngestionRunConfig's own docstring). The
                    # "api:" prefix marks this as an endpoint rather than a
                    # tasks/*.yaml name, so the run's origin is still on
                    # record instead of reading as an anonymous null.
                    task_name="api:compare",
                    pii_detectors=load_pii_detector_names(),
                    pii_model=pii_run_model_config(ingestion_result.pii_engine),
                    prompt_versions={},
                ),
                # Judged the same way /api/ingest judges its own artifact --
                # this one is a side effect of /api/compare, not a lesser run.
                evaluation=_api_evaluation(ingestion_result),
            ),
        )
        run_config = RunConfig(
            task_name="api:compare",
            pipeline="extraction",
            scoring_weights=scoring_engine.weights,
            pii_detectors=load_pii_detector_names(),
            evaluation_model=RunModelConfig.from_client(
                eval_client,
                name=model_names["evaluation"],
                fallback_used=_fallback_used(eval_client),
            ),
            pii_model=pii_run_model_config(result.pii_engine),
            prompt_versions=EXTRACTION_PROMPT_VERSIONS,
        )
        evaluation = _api_evaluation(result)
        artifact_path = build_artifact_logger().log_run(
            result, evaluation=evaluation, config=run_config
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
        evaluation=evaluation,
        engine=result.engine,
        pii_engine=result.pii_engine,
        execution_seconds=result.execution_seconds,
        metrics=result.metrics,
        scorecard=result.scorecard,
        scoring_weights=scoring_engine.weights,
        skills_evaluation=result.skills_eval,
        overall_experience=result.overall_experience,
    )


@upload_router.post("/api/ingest", response_model=IngestResponse)
async def ingest_cv(candidate_cv: UploadFile = File(...)) -> IngestResponse:
    """Redact a raw CV once and persist it to CVIngestionStore. Returns a
    cv_id — pass it to /api/match to evaluate against any number of job
    listings without re-uploading or re-redacting the CV. The response
    never carries pii_spans/redacted text: only a count, since the actual
    detected values are exactly the PII this endpoint exists to keep off
    the wire once ingestion is done."""
    try:
        cv = await _load_document(candidate_cv, CandidateCV, "candidate CV")

        pipeline = IngestionPipeline()
        result = pipeline.run(cv, verbose=False)

        config = IngestionRunConfig(
            task_name="api:ingest",
            pii_detectors=load_pii_detector_names(),
            pii_model=pii_run_model_config(result.pii_engine),
            prompt_versions={},
        )
        evaluation = _api_evaluation(result)
        artifact_path, _ = persist_ingestion(result, config, evaluation=evaluation)
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
        evaluation=evaluation,
        pii_engine=result.pii_engine,
        execution_seconds=result.execution_seconds,
        pii_span_count=len(result.pii_spans),
    )


def _wake_model_backend() -> None:
    """Ping the model endpoint so a scale-to-zero GPU starts booting.

    Runs after the response is sent (see warm_model), so nothing here may
    raise into a request. A warm-up that fails is a missed optimisation,
    never an error the visitor should see -- the match itself will simply
    pay the cold start it would have paid anyway.

    Hits `<base_url>/models` rather than running a completion: on vLLM that
    returns the served model list without touching the GPU, while still
    being enough of a request to make Modal start the container.
    """
    try:
        base_url, api_key = endpoint_for_role("evaluation")
    except Exception as exc:  # unset env var, unknown model name, ...
        logger.warning("Warm-up skipped, could not resolve the model endpoint: %s", exc)
        return

    if not base_url:
        logger.info("Warm-up skipped: the configured model has no base_url to reach.")
        return

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = urllib.request.Request(f"{base_url.rstrip('/')}/models", headers=headers)
    try:
        # Long enough to cover a real cold start, since the container only
        # keeps booting while a request is in flight -- hanging up early can
        # leave it half-started.
        with urllib.request.urlopen(request, timeout=WARM_TIMEOUT_SECONDS) as response:
            logger.info("Warm-up finished with HTTP %s", response.status)
    except Exception as exc:
        logger.info("Warm-up did not complete (this is not fatal): %s", exc)


@router.post("/api/warm", response_model=WarmResponse, status_code=202)
async def warm_model(background_tasks: BackgroundTasks) -> WarmResponse:
    """Start waking the model backend, and return without waiting for it.

    The model runs on a GPU that scales to zero, so the first request after
    a quiet spell pays ~90s of boot time -- which lands entirely on the
    first step of a run and is most of what makes a cold match feel broken.

    A visitor spends time reading the dialog and picking a file before they
    submit anything. Calling this when that dialog opens spends the boot
    during those seconds instead of after the submit. It costs nothing
    extra: it is the same wake-up the match would have triggered, moved
    earlier.

    202, and returns immediately: the caller must not wait on this, and a
    failure to warm is deliberately not an error.
    """
    background_tasks.add_task(_wake_model_backend)
    return WarmResponse(warming=True, detail="Waking the model backend.")


@router.post("/api/match", response_model=MatchResponse)
async def match_cv(
    job_listing: UploadFile = File(...),
    skills_weight: float | None = Form(None),
    work_experience_weight: float | None = Form(None),
) -> MatchResponse:
    """Match a job listing against this deployment's one pinned CV.

    The only public endpoint. A visitor uploads a job listing and nothing
    else: the CV is fixed at build time (see load_serving_cv), so there is
    no `cv_id` for a caller to supply and no CV upload to accept. That is
    the whole lock-down -- no PII detector runs here, no visitor's personal
    data is ever received, and nothing in the request can select a
    different document to serve.
    """
    try:
        scoring_engine = _scoring_engine(skills_weight, work_experience_weight)
        listing = await _load_document(job_listing, JobListing, "job listing")
        redacted_cv = load_serving_cv()

        eval_client = client_for_role("evaluation")
        # A second, independent client so the overall-experience step can run
        # concurrently with requirements+matching (see MatchingPipeline). It
        # has to be its own instance: InstructorClient is not concurrency-safe,
        # and sharing one would corrupt the `attempts` recorded in artifacts.
        experience_client = client_for_role("evaluation")
        pipeline = MatchingPipeline(
            eval_client,
            scoring_engine=scoring_engine,
            experience_client=experience_client,
        )
        result = pipeline.run(listing, redacted_cv, verbose=False)

        run_config = RunConfig(
            task_name="api:match",
            pipeline="matching",
            scoring_weights=scoring_engine.weights,
            pii_detectors=[],
            evaluation_model=RunModelConfig.from_client(
                eval_client,
                name=load_pipeline_model_names()["evaluation"],
                # Either client can fall back independently now that they run
                # in parallel. OR-ing them means a run where only the
                # experience step fell back still says so, rather than
                # reporting a clean primary-only run that never happened.
                fallback_used=_fallback_used(eval_client) or _fallback_used(experience_client),
            ),
            # No PII detector runs for /api/match at all — the CV was
            # redacted offline, long before this process started — so this
            # reports the engine that actually produced that redaction,
            # flagged ran_this_run=False to say exactly that.
            pii_model=pii_run_model_config(redacted_cv.pii_engine, ran_this_run=False),
            prompt_versions=MATCHING_PROMPT_VERSIONS,
        )
        evaluation = _api_evaluation(result)
        artifact_path = build_artifact_logger().log_run(
            result, evaluation=evaluation, config=run_config
        )
    except ServingCVUnavailableError as exc:
        # This deployment has no CV to serve — every request will fail the
        # same way until the file is rebuilt. 503, not the 400 below: the
        # visitor's job listing was fine, and nothing they change will help.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (PDFTextExtractionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Match failed: {type(exc).__name__}: {exc}",
        ) from exc

    return MatchResponse(
        artifact_path=artifact_path,
        evaluation=evaluation,
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
