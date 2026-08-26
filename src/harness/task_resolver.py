"""Layer 2 (CLI): TaskSpec -> ResolvedRun.

Everything that turns declarative task config into live components lives
here: candidate file paths into parsed documents, named model configs
(configs/llm.yaml keys) into clients via client_from_config, detector and
pipeline names into instances via the registries. The orchestration core
(runner.py) never sees a TaskSpec or a file path — it receives the
Resolved*Run this module produces.

The web API has its own resolver (src/api/harness_adapter.py) that produces
the same Resolved*Run shapes from already-parsed HTTP inputs; the registries
below are deliberately a task-config concern only, since the API knows its
concrete pipeline classes and constructs them directly.
"""

from pathlib import Path

from src.config import read_yaml
from src.harness.registry import pii_detectors, pipelines
from src.harness.resolved import (
    ResolvedExtractionRun,
    ResolvedIngestionRun,
    ResolvedMatchingRun,
    ResolvedRun,
    RunMetadata,
)
from src.harness.task_loader import TaskSpec
from src.model.adapters import client_from_config
from src.services.cv_store import CVIngestionStore
from src.services.document_parser import CandidateCV, JobListing
from src.services.extraction_pipeline import ExtractionPipeline
from src.services.ingestion_pipeline import IngestionPipeline
from src.services.matching_pipeline import MatchingPipeline
from src.services.pii_detector import PII_DETECTOR_FACTORIES, CompositePIIDetector
from src.services.scoring_engine import RelevanceScoringEngine

# Default component registrations. New implementations register alongside.
# The "presidio" factory comes from PII_DETECTOR_FACTORIES so this registry
# and ExtractionPipeline's/IngestionPipeline's own default (used outside
# the harness, e.g. the web API) agree on what each detector name means.
if "extraction" not in pipelines.names():
    pipelines.register("extraction", ExtractionPipeline)
for _name, _factory in PII_DETECTOR_FACTORIES.items():
    if _name not in pii_detectors.names():
        pii_detectors.register(_name, _factory)


class TaskResolver:
    def __init__(
        self,
        configs_dir: str | Path = "configs",
        redacted_cv_dir: str | Path = "redacted_cvs",
    ):
        configs_dir = Path(configs_dir)
        self.redacted_cv_dir = redacted_cv_dir
        self.model_configs: dict = read_yaml(configs_dir / "llm.yaml")["models"]
        self.default_weights: dict = read_yaml(configs_dir / "scoring.yaml")["weights"]
        self.default_pii_detector_names: list[str] = read_yaml(
            configs_dir / "pii_policy.yaml"
        )["detectors"]
        self.verbose: bool = read_yaml(configs_dir / "pipeline.yaml").get("verbose", True)

    def resolve(self, task: TaskSpec, task_path: str | None = None) -> ResolvedRun:
        if task.pipeline == "ingestion":
            return self._resolve_ingestion(task, task_path)
        if task.pipeline == "matching":
            return self._resolve_matching(task, task_path)
        return self._resolve_extraction(task, task_path)

    def _resolve_extraction(
        self, task: TaskSpec, task_path: str | None
    ) -> ResolvedExtractionRun:
        evaluation_model_name = _require_model(task, "evaluation")
        eval_client = client_from_config(self._model_config(evaluation_model_name))
        pii_detector_names = task.pii_detectors or self.default_pii_detector_names

        detector = CompositePIIDetector(
            *[pii_detectors.create(name) for name in pii_detector_names]
        )
        scoring_engine = RelevanceScoringEngine(
            task.scoring_weights or self.default_weights
        )
        pipeline = pipelines.create(
            task.pipeline,
            client=eval_client,
            pii_detector=detector,
            scoring_engine=scoring_engine,
        )

        return ResolvedExtractionRun(
            listing=_load_document(JobListing, task.inputs.job_listing, "job listing"),
            cv=_load_document(CandidateCV, task.inputs.candidate_cv, "candidate CV"),
            pipeline=pipeline,
            eval_client=eval_client,
            eval_model_name=evaluation_model_name,
            scoring_engine=scoring_engine,
            pii_detector_names=pii_detector_names,
            evaluation=task.evaluation,
            metadata=RunMetadata(task_name=task.name, task_path=task_path),
            verbose=self.verbose,
        )

    def _resolve_ingestion(
        self, task: TaskSpec, task_path: str | None
    ) -> ResolvedIngestionRun:
        pii_detector_names = task.pii_detectors or self.default_pii_detector_names
        detector = CompositePIIDetector(
            *[pii_detectors.create(name) for name in pii_detector_names]
        )

        return ResolvedIngestionRun(
            cv=_load_document(CandidateCV, task.inputs.candidate_cv, "candidate CV"),
            pipeline=IngestionPipeline(pii_detector=detector),
            pii_detector_names=pii_detector_names,
            evaluation=task.evaluation,
            metadata=RunMetadata(task_name=task.name, task_path=task_path),
            verbose=self.verbose,
        )

    def _resolve_matching(
        self, task: TaskSpec, task_path: str | None
    ) -> ResolvedMatchingRun:
        evaluation_model_name = _require_model(task, "evaluation")
        eval_client = client_from_config(self._model_config(evaluation_model_name))
        scoring_engine = RelevanceScoringEngine(
            task.scoring_weights or self.default_weights
        )

        if not task.inputs.redacted_cv_id:
            raise ValueError(
                f"Task '{task.name}' uses pipeline: matching but sets no "
                "inputs.redacted_cv_id. Run an 'ingestion' task against the "
                "raw CV first, then pass the cv_id it produced here."
            )
        redacted_cv = CVIngestionStore(output_dir=self.redacted_cv_dir).load(
            task.inputs.redacted_cv_id
        )

        return ResolvedMatchingRun(
            listing=_load_document(JobListing, task.inputs.job_listing, "job listing"),
            redacted_cv=redacted_cv,
            pipeline=MatchingPipeline(eval_client, scoring_engine=scoring_engine),
            eval_client=eval_client,
            eval_model_name=evaluation_model_name,
            scoring_engine=scoring_engine,
            evaluation=task.evaluation,
            metadata=RunMetadata(task_name=task.name, task_path=task_path),
            verbose=self.verbose,
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
