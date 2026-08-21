"""Loads declarative task definitions (tasks/*.yaml) into validated specs."""

from pathlib import Path

from pydantic import BaseModel, Field

from src.config_loader import read_yaml


class ModelSelection(BaseModel):
    """Named model configs (keys in configs/llm.yaml) for each role."""

    evaluation: str
    pii: str


class TaskInputs(BaseModel):
    """Candidate paths per document; the first existing path wins."""

    job_listing: list[str]
    candidate_cv: list[str]


class EvaluationCriteria(BaseModel):
    min_final_relevance: float | None = None
    min_skills_match: float | None = None
    min_pii_spans: int | None = None
    max_execution_seconds: float | None = None


class TaskSpec(BaseModel):
    name: str
    description: str = ""
    pipeline: str = "extraction"
    models: ModelSelection
    inputs: TaskInputs
    scoring_weights: dict[str, float] | None = None
    evaluation: EvaluationCriteria = Field(default_factory=EvaluationCriteria)


def load_task(path: str | Path) -> TaskSpec:
    return TaskSpec.model_validate(read_yaml(path))
