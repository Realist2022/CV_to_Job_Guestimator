"""Loads declarative task definitions (tasks/*.yaml) into validated specs."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.config import read_yaml

PipelineKind = Literal["extraction", "ingestion", "matching"]


class ModelSelection(BaseModel):
    """Named model config (a key in configs/llm.yaml) for the evaluation role.

    Optional because not every pipeline needs it: an "ingestion" task never
    calls an evaluation model. There is no "pii" role at all — PII
    redaction runs entirely through presidio (see pii_detector.py), with no
    LLM in the loop and so nothing to select a model config for.
    """

    model_config = ConfigDict(extra="forbid")

    evaluation: str | None = None


class TaskInputs(BaseModel):
    """Candidate paths per document; the first existing path wins.

    candidate_cv is the raw-CV path list, used by "extraction" and
    "ingestion" tasks. A "matching" task instead sets redacted_cv_id and
    reads a RedactedCV out of CVIngestionStore — it never takes a raw CV
    path, matching MatchingPipeline's own signature.
    """

    model_config = ConfigDict(extra="forbid")

    job_listing: list[str] = []
    candidate_cv: list[str] = []
    redacted_cv_id: str | None = None


class EvaluationCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_final_relevance: float | None = None
    min_skills_match: float | None = None
    min_pii_spans: int | None = None
    max_execution_seconds: float | None = None


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    pipeline: PipelineKind = "extraction"
    models: ModelSelection
    inputs: TaskInputs
    scoring_weights: dict[str, float] | None = None
    # Overrides configs/pii_policy.yaml's detector list for this task only,
    # e.g. to A/B "presidio" against the default "model" detector without
    # touching the project-wide policy. None means "use the global default".
    pii_detectors: list[str] | None = None
    evaluation: EvaluationCriteria = Field(default_factory=EvaluationCriteria)


def load_task(path: str | Path) -> TaskSpec:
    return TaskSpec.model_validate(read_yaml(path))
