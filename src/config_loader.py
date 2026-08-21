"""Shared accessors for YAML-backed project configuration."""

from pathlib import Path
import math

import yaml
from dotenv import load_dotenv

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

load_dotenv(override=True)


def read_yaml(path: str | Path) -> dict:
    """Read a YAML mapping from an explicit path (no CONFIG_DIR resolution)."""
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return data


def load_yaml(name: str | Path) -> dict:
    path = Path(name)
    if not path.is_absolute():
        path = CONFIG_DIR / path
    return read_yaml(path)


def load_scoring_weights() -> dict[str, float]:
    weights = load_yaml("scoring.yaml").get("weights")
    if not isinstance(weights, dict):
        raise ValueError("configs/scoring.yaml must contain a weights mapping.")
    if not math.isclose(sum(weights.values()), 1.0):
        raise ValueError("configs/scoring.yaml weights must sum to 1.0.")
    return dict(weights)


def load_model_config(name: str) -> dict:
    models = load_yaml("llm.yaml").get("models")
    if not isinstance(models, dict):
        raise ValueError("configs/llm.yaml must contain a models mapping.")
    try:
        return dict(models[name])
    except KeyError:
        known = ", ".join(sorted(models))
        raise KeyError(f"Unknown model config '{name}' in configs/llm.yaml. Known: {known}") from None


def load_pipeline_model_names() -> dict[str, str]:
    pipeline_config = load_yaml("pipeline.yaml")
    models = pipeline_config.get("models")
    if not isinstance(models, dict) or "evaluation" not in models or "pii" not in models:
        raise ValueError(
            "configs/pipeline.yaml must define models.evaluation and models.pii."
        )
    return {"evaluation": models["evaluation"], "pii": models["pii"]}


def load_pii_detector_names() -> list[str]:
    detectors = load_yaml("pii_policy.yaml").get("detectors")
    if not isinstance(detectors, list) or not detectors:
        raise ValueError("configs/pii_policy.yaml must define a non-empty detectors list.")
    return list(detectors)