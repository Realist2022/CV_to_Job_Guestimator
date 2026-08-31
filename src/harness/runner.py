"""Harness runner: task YAML -> resolved components -> run -> evaluate -> artifact.

Three task shapes share this runner:
  - pipeline: extraction  — one-shot compatibility path (raw CV in, full
    result out). Goes through the generic `pipelines` registry, same as
    before the ingestion/matching split.
  - pipeline: ingestion    — PII redaction only. Persists the resulting
    RedactedCV to CVIngestionStore so a later matching task can consume it.
  - pipeline: matching      — skills/experience/scoring only, reading a
    RedactedCV out of CVIngestionStore via task.inputs.redacted_cv_id.

ingestion/matching get their own branches rather than going through the
`pipelines` registry: IngestionPipeline.run(cv) and
MatchingPipeline.run(listing, redacted_cv) genuinely don't share
ExtractionPipeline's run(listing, cv) -> PipelineResult contract that the
registry was built around (that's the point of the split), so forcing them
through the same generic call would just be a silent kwarg/shape mismatch
waiting to happen.
"""

import inspect
from pathlib import Path

from pydantic import BaseModel

from src.config import load_default_evaluation_criteria, read_yaml
from src.harness.evaluator import EvaluationReport, ThresholdEvaluator, resolve_criteria
from src.harness.registry import pii_detectors, pipelines
from src.harness.task_loader import EvaluationCriteria, TaskSpec, load_task
from src.model.adapters import client_from_config
from src.prompts.templates import EXTRACTION_PROMPT_VERSIONS, MATCHING_PROMPT_VERSIONS
from src.schemas.artifact import IngestionRunConfig, RunConfig, RunModelConfig
from src.schemas.ingestion import IngestionResult
from src.schemas.pipeline import PipelineResult
from src.services.cv_store import CVIngestionStore
from src.services.document_parser import CandidateCV, JobListing
from src.services.extraction_pipeline import ExtractionPipeline
from src.services.ingestion_persistence import persist_ingestion
from src.services.ingestion_pipeline import IngestionPipeline
from src.services.matching_pipeline import MatchingPipeline
from src.services.pii_detector import (
    PII_DETECTOR_FACTORIES,
    CompositePIIDetector,
    pii_run_model_config,
)
from src.services.scoring_engine import RelevanceScoringEngine
from src.utils.artifact_logger import ArtifactLogger

# Default component registrations. New implementations register alongside.
# The "presidio" factory comes from PII_DETECTOR_FACTORIES so this registry
# and ExtractionPipeline's/IngestionPipeline's own default (used outside
# the harness, e.g. the web API) agree on what each detector name means.
if "extraction" not in pipelines.names():
    pipelines.register("extraction", ExtractionPipeline)
for _name, _factory in PII_DETECTOR_FACTORIES.items():
    if _name not in pii_detectors.names():
        pii_detectors.register(_name, _factory)


class HarnessRunReport(BaseModel):
    task_name: str
    artifact_path: str
    run_number: int | None
    result: PipelineResult | IngestionResult
    evaluation: EvaluationReport


class HarnessRunner:
    def __init__(
        self,
        configs_dir: str | Path = "configs",
        artifacts_dir: str | Path = "artifacts",
        redacted_cv_dir: str | Path = "redacted_cvs",
    ):
        configs_dir = Path(configs_dir)
        self.artifacts_dir = artifacts_dir
        self.redacted_cv_dir = redacted_cv_dir
        self.model_configs: dict = read_yaml(configs_dir / "llm.yaml")["models"]
        self.default_weights: dict = read_yaml(configs_dir / "scoring.yaml")["weights"]
        self.default_pii_detector_names: list[str] = read_yaml(configs_dir / "pii_policy.yaml")[
            "detectors"
        ]
        self.verbose: bool = read_yaml(configs_dir / "pipeline.yaml").get("verbose", True)
        # Baseline thresholds every task inherits (see _criteria_for). Read
        # through the config package rather than configs_dir so it resolves
        # the same file the web API reads -- the point of the block is that
        # both entry points are judged by one definition.
        self.default_evaluation_criteria: dict = load_default_evaluation_criteria()

    def run(self, task: TaskSpec | str | Path) -> HarnessRunReport:
        task_path = None if isinstance(task, TaskSpec) else str(task)
        if not isinstance(task, TaskSpec):
            task = load_task(task)

        if task.pipeline == "ingestion":
            return self._run_ingestion(task, task_path)
        if task.pipeline == "matching":
            return self._run_matching(task, task_path)
        return self._run_extraction(task, task_path)

    def _run_extraction(self, task: TaskSpec, task_path: str | None) -> HarnessRunReport:
        evaluation_model_name = _require_model(task, "evaluation")
        eval_client = client_from_config(self._model_config(evaluation_model_name))
        pii_detector_names = task.pii_detectors or self.default_pii_detector_names

        detector = CompositePIIDetector(
            *[pii_detectors.create(name) for name in pii_detector_names]
        )
        scoring_weights = task.scoring_weights or self.default_weights
        scoring_engine = RelevanceScoringEngine(scoring_weights)
        pipeline = pipelines.create(
            task.pipeline,
            client=eval_client,
            pii_detector=detector,
            scoring_engine=scoring_engine,
        )

        listing = _load_document(JobListing, task.inputs.job_listing, "job listing")
        cv = _load_document(CandidateCV, task.inputs.candidate_cv, "candidate CV")

        run_kwargs = {}
        # Capability check, not isinstance(pipeline, ExtractionPipeline): the
        # pipeline was just resolved dynamically from the `pipelines`
        # registry, whose whole point is that a second implementation can be
        # registered under "extraction" without editing this runner. What
        # actually matters here is whether pipeline.run() accepts the
        # on_ingested hook, not which concrete class produced it.
        if "on_ingested" in inspect.signature(pipeline.run).parameters:
            # So this one-shot run's redacted_cv_trace_id (see
            # RunArtifact/PipelineResult) always resolves to a real,
            # persisted IngestionArtifact — the same guarantee the
            # standalone "ingestion" task already gives, not something
            # only the two-task ingest-then-match flow provides.
            # pii_model is built from ingestion_result.pii_engine (set once
            # redaction actually ran) rather than read off `detector`
            # directly, so this keeps working against any pipeline.run()
            # that honors the on_ingested contract — not just whichever
            # concrete pipeline built `detector` above.
            run_kwargs["on_ingested"] = lambda ingestion_result: persist_ingestion(
                ingestion_result,
                IngestionRunConfig(
                    task_name=task.name,
                    task_path=task_path,
                    pii_detectors=pii_detector_names,
                    pii_model=pii_run_model_config(ingestion_result.pii_engine),
                    prompt_versions={},
                ),
                redacted_cv_dir=self.redacted_cv_dir,
                artifacts_dir=self.artifacts_dir,
            )

        result = pipeline.run(listing, cv, verbose=self.verbose, **run_kwargs)
        evaluation = ThresholdEvaluator(self._criteria_for(task, result)).evaluate(result)

        run_config = RunConfig(
            task_name=task.name,
            task_path=task_path,
            pipeline=task.pipeline,
            scoring_weights=scoring_weights,
            pii_detectors=pii_detector_names,
            evaluation_model=RunModelConfig.from_client(eval_client, name=evaluation_model_name),
            pii_model=pii_run_model_config(result.pii_engine),
            prompt_versions=EXTRACTION_PROMPT_VERSIONS,
        )

        logger = ArtifactLogger(output_dir=self.artifacts_dir)
        artifact_path = logger.log_run(result, evaluation=evaluation, config=run_config)

        return HarnessRunReport(
            task_name=task.name,
            artifact_path=artifact_path,
            run_number=logger.last_run_number,
            result=result,
            evaluation=evaluation,
        )

    def _run_ingestion(self, task: TaskSpec, task_path: str | None) -> HarnessRunReport:
        pii_detector_names = task.pii_detectors or self.default_pii_detector_names
        detector = CompositePIIDetector(
            *[pii_detectors.create(name) for name in pii_detector_names]
        )
        pipeline = IngestionPipeline(pii_detector=detector)

        cv = _load_document(CandidateCV, task.inputs.candidate_cv, "candidate CV")
        result = pipeline.run(cv, verbose=self.verbose)
        evaluation = ThresholdEvaluator(self._criteria_for(task, result)).evaluate(result)

        ingestion_config = IngestionRunConfig(
            task_name=task.name,
            task_path=task_path,
            pii_detectors=pii_detector_names,
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
            task_name=task.name,
            artifact_path=artifact_path,
            run_number=run_number,
            result=result,
            evaluation=evaluation,
        )

    def _run_matching(self, task: TaskSpec, task_path: str | None) -> HarnessRunReport:
        evaluation_model_name = _require_model(task, "evaluation")
        eval_client = client_from_config(self._model_config(evaluation_model_name))
        scoring_weights = task.scoring_weights or self.default_weights
        scoring_engine = RelevanceScoringEngine(scoring_weights)
        pipeline = MatchingPipeline(eval_client, scoring_engine=scoring_engine)

        if not task.inputs.redacted_cv_id:
            raise ValueError(
                f"Task '{task.name}' uses pipeline: matching but sets no "
                "inputs.redacted_cv_id. Run an 'ingestion' task against the "
                "raw CV first, then pass the cv_id it produced here."
            )
        redacted_cv = CVIngestionStore(output_dir=self.redacted_cv_dir).load(
            task.inputs.redacted_cv_id
        )
        listing = _load_document(JobListing, task.inputs.job_listing, "job listing")

        result = pipeline.run(listing, redacted_cv, verbose=self.verbose)
        evaluation = ThresholdEvaluator(self._criteria_for(task, result)).evaluate(result)

        run_config = RunConfig(
            task_name=task.name,
            task_path=task_path,
            pipeline=task.pipeline,
            scoring_weights=scoring_weights,
            # No detector ran in this task; the CV arrived pre-redacted.
            pii_detectors=[],
            evaluation_model=RunModelConfig.from_client(eval_client, name=evaluation_model_name),
            # No PII detector runs for a matching-only task at all — the CV
            # arrived pre-redacted — so this reports the engine that
            # actually produced that earlier redaction, off the RedactedCV
            # itself, same as everywhere else. ran_this_run=False is what
            # says so in the artifact rather than leaving it inferrable
            # from the empty pii_detectors above.
            pii_model=pii_run_model_config(redacted_cv.pii_engine, ran_this_run=False),
            prompt_versions=MATCHING_PROMPT_VERSIONS,
        )
        logger = ArtifactLogger(output_dir=self.artifacts_dir)
        artifact_path = logger.log_run(result, evaluation=evaluation, config=run_config)

        return HarnessRunReport(
            task_name=task.name,
            artifact_path=artifact_path,
            run_number=logger.last_run_number,
            result=result,
            evaluation=evaluation,
        )

    def _criteria_for(self, task: TaskSpec, result) -> EvaluationCriteria:
        """The task's own thresholds over configs/pipeline.yaml's baseline.

        `result` is passed because resolve_criteria drops inherited defaults
        this result shape can't answer -- an ingestion run has no scorecard
        for a global min_final_relevance to read.
        """
        return resolve_criteria(
            result, task=task.evaluation, defaults=self.default_evaluation_criteria
        )

    def _model_config(self, name: str) -> dict:
        try:
            return self.model_configs[name]
        except KeyError:
            known = ", ".join(sorted(self.model_configs))
            raise KeyError(
                f"Unknown model config '{name}' in configs/llm.yaml. Known: {known}"
            ) from None


def _require_model(task: TaskSpec, role: str) -> str:
    name = getattr(task.models, role)
    if not name:
        raise ValueError(
            f"Task '{task.name}' uses pipeline: {task.pipeline} but sets no "
            f"models.{role}."
        )
    return name


def _load_document(document_type, candidate_paths: list[str], label: str):
    for raw_path in candidate_paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        if path.suffix.lower() == ".pdf":
            return document_type.from_pdf(str(path), cache_text=True)
        return document_type.from_path(str(path))
    raise FileNotFoundError(
        f"No {label} found. Tried: {', '.join(candidate_paths)}"
    )
