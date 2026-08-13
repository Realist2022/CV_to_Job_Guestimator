"""Harness runner: task YAML -> resolved components -> run -> evaluate -> artifact."""

from pathlib import Path

from pydantic import BaseModel

from src.harness.evaluator import EvaluationReport, ThresholdEvaluator
from src.harness.registry import pipelines, pii_detectors
from src.harness.task_loader import TaskSpec, load_task, load_yaml
from src.model.adapters import client_from_config
from src.schemas.pipeline import PipelineResult
from src.services.agents import PIIAgent
from src.services.document_parser import CandidateCV, JobListing
from src.services.pii_detector import (
    CompositePIIDetector,
    ModelPIIDetector,
    RegexPIIDetector,
)
from src.services.extraction_pipeline import ExtractionPipeline
from src.services.scoring_engine import RelevanceScoringEngine
from src.utils.artifact_logger import ArtifactLogger

# Default component registrations. New implementations register alongside.
if "extraction" not in pipelines.names():
    pipelines.register("extraction", ExtractionPipeline)
if "regex" not in pii_detectors.names():
    pii_detectors.register("regex", lambda **_: RegexPIIDetector())
    pii_detectors.register(
        "model", lambda pii_client, **_: ModelPIIDetector(PIIAgent(pii_client))
    )


class HarnessRunReport(BaseModel):
    task_name: str
    artifact_path: str
    run_number: int | None
    result: PipelineResult
    evaluation: EvaluationReport


class HarnessRunner:
    def __init__(
        self,
        configs_dir: str | Path = "configs",
        artifacts_dir: str | Path = "artifacts",
    ):
        configs_dir = Path(configs_dir)
        self.artifacts_dir = artifacts_dir
        self.model_configs: dict = load_yaml(configs_dir / "llm.yaml")["models"]
        self.default_weights: dict = load_yaml(configs_dir / "scoring.yaml")["weights"]
        self.pii_detector_names: list[str] = load_yaml(configs_dir / "pii_policy.yaml")[
            "detectors"
        ]
        pipeline_config = load_yaml(configs_dir / "pipeline.yaml")
        self.verbose: bool = pipeline_config.get("verbose", True)

    def run(self, task: TaskSpec | str | Path) -> HarnessRunReport:
        if not isinstance(task, TaskSpec):
            task = load_task(task)

        eval_client = client_from_config(self._model_config(task.models.evaluation))
        pii_client = client_from_config(self._model_config(task.models.pii))

        detector = CompositePIIDetector(
            *[
                pii_detectors.create(name, pii_client=pii_client)
                for name in self.pii_detector_names
            ]
        )
        scoring_engine = RelevanceScoringEngine(
            task.scoring_weights or self.default_weights
        )
        pipeline = pipelines.create(
            task.pipeline,
            client=eval_client,
            pii_client=pii_client,
            pii_detector=detector,
            scoring_engine=scoring_engine,
        )

        listing = _load_document(JobListing, task.inputs.job_listing, "job listing")
        cv = _load_document(CandidateCV, task.inputs.candidate_cv, "candidate CV")

        result = pipeline.run(listing, cv, verbose=self.verbose)
        evaluation = ThresholdEvaluator(task.evaluation).evaluate(result)

        logger = ArtifactLogger(output_dir=self.artifacts_dir)
        artifact_path = logger.log_run(result, evaluation=evaluation)

        return HarnessRunReport(
            task_name=task.name,
            artifact_path=artifact_path,
            run_number=logger.last_run_number,
            result=result,
            evaluation=evaluation,
        )

    def _model_config(self, name: str) -> dict:
        try:
            return self.model_configs[name]
        except KeyError:
            known = ", ".join(sorted(self.model_configs))
            raise KeyError(
                f"Unknown model config '{name}' in configs/llm.yaml. Known: {known}"
            ) from None


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
