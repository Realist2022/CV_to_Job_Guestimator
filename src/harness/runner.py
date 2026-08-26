"""Harness runner: the one orchestration core behind both entry points.

run_extraction/run_ingestion/run_matching each take a Resolved*Run (see
resolved.py) — documents already parsed, clients and pipelines already
constructed — and do only the source-agnostic part: run the pipeline,
evaluate thresholds, persist the artifact, assemble a HarnessRunReport.
This layer never sees a task YAML, a file path, or an UploadFile.

Who produces the Resolved*Run depends on the trigger:
  - CLI (main.py): run_task() loads tasks/*.yaml into a TaskSpec and hands
    it to TaskResolver (task_resolver.py), which resolves components via
    the pipelines/pii_detectors registries.
  - Web API (src/api/routes.py): src/api/harness_adapter.py resolves
    already-parsed HTTP inputs with client_for_role() and the concrete
    pipeline classes, then calls the same run_* methods here.

ingestion/matching keep their own run_* methods rather than sharing
run_extraction: IngestionPipeline.run(cv) and
MatchingPipeline.run(listing, redacted_cv) genuinely don't share
ExtractionPipeline's run(listing, cv) -> PipelineResult contract (that's
the point of the ingestion/matching split), so forcing them through one
generic call would just be a silent kwarg/shape mismatch waiting to happen.
"""

import inspect
from pathlib import Path

from pydantic import BaseModel

from src.harness.evaluator import EvaluationReport, ThresholdEvaluator
from src.harness.resolved import (
    ResolvedExtractionRun,
    ResolvedIngestionRun,
    ResolvedMatchingRun,
    ResolvedRun,
)
from src.harness.task_loader import TaskSpec, load_task
from src.harness.task_resolver import TaskResolver
from src.prompts.templates import EXTRACTION_PROMPT_VERSIONS, MATCHING_PROMPT_VERSIONS
from src.schemas.artifact import IngestionRunConfig, RunConfig, RunModelConfig
from src.schemas.ingestion import IngestionResult
from src.schemas.pipeline import PipelineResult
from src.services.ingestion_persistence import persist_ingestion
from src.services.llm_client import CompletionClient, FallbackInstructorClient
from src.services.pii_detector import pii_run_model_config
from src.utils.artifact_logger import ArtifactLogger


class HarnessRunReport(BaseModel):
    task_name: str | None
    artifact_path: str
    run_number: int | None
    result: PipelineResult | IngestionResult
    evaluation: EvaluationReport


def _fallback_used(client: CompletionClient) -> bool:
    """Whether `client`'s most recent call was served by its fallback model
    rather than its configured primary (always False for a plain
    InstructorClient with no fallback configured). Computed here, once, so
    every caller's RunConfig gets it — the CLI's pinned task clients are
    never FallbackInstructorClients, the API's client_for_role() clients
    may be."""
    return isinstance(client, FallbackInstructorClient) and client.fallback_used


class HarnessRunner:
    def __init__(
        self,
        configs_dir: str | Path = "configs",
        artifacts_dir: str | Path = "artifacts",
        redacted_cv_dir: str | Path = "redacted_cvs",
    ):
        self.artifacts_dir = artifacts_dir
        self.redacted_cv_dir = redacted_cv_dir
        self.resolver = TaskResolver(
            configs_dir=configs_dir, redacted_cv_dir=redacted_cv_dir
        )

    # -- CLI entry point: task YAML in, report out ------------------------

    def run_task(self, task: TaskSpec | str | Path) -> HarnessRunReport:
        task_path = None if isinstance(task, TaskSpec) else str(task)
        if not isinstance(task, TaskSpec):
            task = load_task(task)
        return self.run(self.resolver.resolve(task, task_path))

    # -- Orchestration core: Resolved*Run in, report out ------------------

    def run(self, resolved: ResolvedRun) -> HarnessRunReport:
        if isinstance(resolved, ResolvedIngestionRun):
            return self.run_ingestion(resolved)
        if isinstance(resolved, ResolvedMatchingRun):
            return self.run_matching(resolved)
        return self.run_extraction(resolved)

    def run_extraction(self, resolved: ResolvedExtractionRun) -> HarnessRunReport:
        run_kwargs = {}
        # Capability check, not isinstance(pipeline, ExtractionPipeline):
        # the pipeline may come from the `pipelines` registry, whose whole
        # point is that a second implementation can be registered under
        # "extraction" without editing this runner. What actually matters
        # here is whether pipeline.run() accepts the on_ingested hook, not
        # which concrete class produced it.
        if "on_ingested" in inspect.signature(resolved.pipeline.run).parameters:
            # So this one-shot run's redacted_cv_trace_id (see
            # RunArtifact/PipelineResult) always resolves to a real,
            # persisted IngestionArtifact — the same guarantee a standalone
            # ingestion run already gives, not something only the two-step
            # ingest-then-match flow provides. pii_model is built from
            # ingestion_result.pii_engine (set once redaction actually ran)
            # rather than read off the pipeline's detector directly, so this
            # keeps working against any pipeline.run() that honors the
            # on_ingested contract — e.g. a test double.
            run_kwargs["on_ingested"] = lambda ingestion_result: persist_ingestion(
                ingestion_result,
                IngestionRunConfig(
                    task_name=resolved.metadata.task_name,
                    task_path=resolved.metadata.task_path,
                    pii_detectors=resolved.pii_detector_names,
                    pii_model=pii_run_model_config(ingestion_result.pii_engine),
                    prompt_versions={},
                ),
                redacted_cv_dir=self.redacted_cv_dir,
                artifacts_dir=self.artifacts_dir,
            )

        result = resolved.pipeline.run(
            resolved.listing, resolved.cv, verbose=resolved.verbose, **run_kwargs
        )
        evaluation = ThresholdEvaluator(resolved.evaluation).evaluate(result)

        run_config = RunConfig(
            task_name=resolved.metadata.task_name,
            task_path=resolved.metadata.task_path,
            pipeline="extraction",
            scoring_weights=resolved.scoring_engine.weights,
            pii_detectors=resolved.pii_detector_names,
            evaluation_model=self._evaluation_model_config(resolved),
            pii_model=pii_run_model_config(result.pii_engine),
            prompt_versions=EXTRACTION_PROMPT_VERSIONS,
        )

        logger = ArtifactLogger(output_dir=self.artifacts_dir)
        artifact_path = logger.log_run(result, evaluation=evaluation, config=run_config)

        return HarnessRunReport(
            task_name=resolved.metadata.task_name,
            artifact_path=artifact_path,
            run_number=logger.last_run_number,
            result=result,
            evaluation=evaluation,
        )

    def run_ingestion(self, resolved: ResolvedIngestionRun) -> HarnessRunReport:
        result = resolved.pipeline.run(resolved.cv, verbose=resolved.verbose)
        evaluation = ThresholdEvaluator(resolved.evaluation).evaluate(result)

        ingestion_config = IngestionRunConfig(
            task_name=resolved.metadata.task_name,
            task_path=resolved.metadata.task_path,
            pii_detectors=resolved.pii_detector_names,
            pii_model=pii_run_model_config(result.pii_engine),
            prompt_versions={},
        )
        artifact_path, run_number = persist_ingestion(
            result,
            ingestion_config,
            evaluation=evaluation,
            redacted_cv_dir=self.redacted_cv_dir,
            artifacts_dir=self.artifacts_dir,
        )

        return HarnessRunReport(
            task_name=resolved.metadata.task_name,
            artifact_path=artifact_path,
            run_number=run_number,
            result=result,
            evaluation=evaluation,
        )

    def run_matching(self, resolved: ResolvedMatchingRun) -> HarnessRunReport:
        result = resolved.pipeline.run(
            resolved.listing, resolved.redacted_cv, verbose=resolved.verbose
        )
        evaluation = ThresholdEvaluator(resolved.evaluation).evaluate(result)

        run_config = RunConfig(
            task_name=resolved.metadata.task_name,
            task_path=resolved.metadata.task_path,
            pipeline="matching",
            scoring_weights=resolved.scoring_engine.weights,
            # No detector ran in this run; the CV arrived pre-redacted.
            pii_detectors=[],
            evaluation_model=self._evaluation_model_config(resolved),
            # No PII detector runs for a matching-only run at all — the CV
            # arrived pre-redacted — so this reports the engine that
            # actually produced that earlier redaction, off the RedactedCV
            # itself, same as everywhere else.
            pii_model=pii_run_model_config(resolved.redacted_cv.pii_engine),
            prompt_versions=MATCHING_PROMPT_VERSIONS,
        )
        logger = ArtifactLogger(output_dir=self.artifacts_dir)
        artifact_path = logger.log_run(result, evaluation=evaluation, config=run_config)

        return HarnessRunReport(
            task_name=resolved.metadata.task_name,
            artifact_path=artifact_path,
            run_number=logger.last_run_number,
            result=result,
            evaluation=evaluation,
        )

    @staticmethod
    def _evaluation_model_config(
        resolved: ResolvedExtractionRun | ResolvedMatchingRun,
    ) -> RunModelConfig:
        return RunModelConfig(
            name=resolved.eval_model_name,
            engine=resolved.eval_client.model,
            temperature=resolved.eval_client.temperature,
            fallback_used=_fallback_used(resolved.eval_client),
        )
