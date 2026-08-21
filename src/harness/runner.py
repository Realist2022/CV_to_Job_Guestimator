"""Harness runner: task YAML -> resolved components -> run -> evaluate -> artifact."""

from pathlib import Path

from pydantic import BaseModel

from src.harness.evaluator import EvaluationReport, ThresholdEvaluator
from src.harness.registry import pipelines, pii_detectors
from src.config_loader import read_yaml
from src.harness.task_loader import TaskSpec, load_task
from src.model.adapters import client_from_config
from src.schemas.artifact import RunConfig, RunModelConfig
from src.schemas.pipeline import PipelineResult
from src.services.document_parser import CandidateCV, JobListing
from src.services.pii_detector import CompositePIIDetector, PII_DETECTOR_FACTORIES
from src.services.extraction_pipeline import ExtractionPipeline
from src.services.scoring_engine import RelevanceScoringEngine
from src.utils.artifact_logger import ArtifactLogger

# Default component registrations. New implementations register alongside.
# The "regex"/"model" factories come from PII_DETECTOR_FACTORIES so this
# registry and ExtractionPipeline's own default (used outside the harness,
# e.g. the web API) agree on what each detector name means.
if "extraction" not in pipelines.names():
    pipelines.register("extraction", ExtractionPipeline)
for _name, _factory in PII_DETECTOR_FACTORIES.items():
    if _name not in pii_detectors.names():
        pii_detectors.register(_name, lambda pii_client=None, _factory=_factory, **_: _factory(pii_client))


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
        self.model_configs: dict = read_yaml(configs_dir / "llm.yaml")["models"]
        self.default_weights: dict = read_yaml(configs_dir / "scoring.yaml")["weights"]
        self.pii_detector_names: list[str] = read_yaml(configs_dir / "pii_policy.yaml")[
            "detectors"
        ]
        self.verbose: bool = read_yaml(configs_dir / "pipeline.yaml").get("verbose", True)

    def run(self, task: TaskSpec | str | Path) -> HarnessRunReport:
        task_path = None if isinstance(task, TaskSpec) else str(task)
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
        scoring_weights = task.scoring_weights or self.default_weights
        scoring_engine = RelevanceScoringEngine(scoring_weights)
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

        run_config = RunConfig(
            task_name=task.name,
            task_path=task_path,
            pipeline=task.pipeline,
            scoring_weights=scoring_weights,
            pii_detectors=self.pii_detector_names,
            evaluation_model=RunModelConfig(
                name=task.models.evaluation,
                engine=eval_client.model,
                temperature=eval_client.temperature,
            ),
            pii_model=RunModelConfig(
                name=task.models.pii,
                engine=pii_client.model,
                temperature=pii_client.temperature,
            ),
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
