"""Layer 2 (API): already-parsed HTTP inputs -> ResolvedRun.

The web API's counterpart to src/harness/task_resolver.py: where the CLI
resolves components from a tasks/*.yaml TaskSpec (named model configs,
registries, file paths), this module resolves them the way a live request
needs — client_for_role() (role-based, with the configs/pipeline.yaml
fallback model), project-default scoring weights with per-request form
overrides, and the concrete pipeline classes constructed directly. Both
resolvers hand HarnessRunner the same Resolved*Run shapes; neither runs a
pipeline itself.

Every Resolved*Run built here carries empty EvaluationCriteria (a live call
has no ground truth to threshold against — the evaluator then reports
checks=[], passed=True and nothing is printed or returned for it) and
verbose=False (no per-request stdout chatter from the pipelines).
"""

from src.config import (
    load_pii_detector_names,
    load_pipeline_model_names,
    load_scoring_weights,
)
from src.harness.resolved import (
    ResolvedExtractionRun,
    ResolvedIngestionRun,
    ResolvedMatchingRun,
)
from src.model.adapters import client_for_role
from src.services import (
    CandidateCV,
    CVIngestionStore,
    CVNotFoundError,
    ExtractionPipeline,
    IngestionPipeline,
    JobListing,
    MatchingPipeline,
    RelevanceScoringEngine,
)


def resolve_extraction(
    listing: JobListing,
    cv: CandidateCV,
    *,
    skills_weight: float | None = None,
    work_experience_weight: float | None = None,
) -> ResolvedExtractionRun:
    scoring_engine = _scoring_engine(skills_weight, work_experience_weight)
    eval_client = client_for_role("evaluation")
    return ResolvedExtractionRun(
        listing=listing,
        cv=cv,
        pipeline=ExtractionPipeline(eval_client, scoring_engine=scoring_engine),
        eval_client=eval_client,
        eval_model_name=load_pipeline_model_names()["evaluation"],
        scoring_engine=scoring_engine,
        pii_detector_names=load_pii_detector_names(),
        verbose=False,
    )


def resolve_ingestion(cv: CandidateCV) -> ResolvedIngestionRun:
    return ResolvedIngestionRun(
        cv=cv,
        pipeline=IngestionPipeline(),
        pii_detector_names=load_pii_detector_names(),
        verbose=False,
    )


def resolve_matching(
    listing: JobListing,
    cv_id: str,
    *,
    skills_weight: float | None = None,
    work_experience_weight: float | None = None,
) -> ResolvedMatchingRun:
    scoring_engine = _scoring_engine(skills_weight, work_experience_weight)
    try:
        redacted_cv = CVIngestionStore().load(cv_id)
    except CVNotFoundError as exc:
        # ValueError is the routes' "client error" type -> HTTP 400.
        raise ValueError(str(exc)) from exc

    eval_client = client_for_role("evaluation")
    return ResolvedMatchingRun(
        listing=listing,
        redacted_cv=redacted_cv,
        pipeline=MatchingPipeline(eval_client, scoring_engine=scoring_engine),
        eval_client=eval_client,
        eval_model_name=load_pipeline_model_names()["evaluation"],
        scoring_engine=scoring_engine,
        verbose=False,
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
