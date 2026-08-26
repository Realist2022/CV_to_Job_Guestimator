"""Shared accessors for YAML-backed project configuration."""

import math
from pathlib import Path

import yaml
from dotenv import load_dotenv

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

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
    # Only "evaluation" is a real model role now: PII redaction runs
    # entirely through presidio (see pii_detector.py), with no LLM in the
    # loop and so nothing to select a model config for.
    pipeline_config = load_yaml("pipeline.yaml")
    models = pipeline_config.get("models")
    if not isinstance(models, dict) or "evaluation" not in models:
        raise ValueError("configs/pipeline.yaml must define models.evaluation.")
    return {"evaluation": models["evaluation"]}


def load_pipeline_fallback_names() -> dict[str, str]:
    """Optional `fallback_models` mapping from configs/pipeline.yaml.

    A role (e.g. "evaluation") with no entry here has no fallback, which is
    the default: `client_for_role()` returns that role's primary client
    unwrapped. A role listed here gets its primary client wrapped in a
    FallbackInstructorClient that falls back to the named model config on
    failure — e.g. a fine-tuned local SLM falling back to a hosted model.
    """
    fallback_models = load_yaml("pipeline.yaml").get("fallback_models") or {}
    if not isinstance(fallback_models, dict):
        raise ValueError("configs/pipeline.yaml 'fallback_models' key must be a mapping.")
    return dict(fallback_models)


def load_pii_detector_names() -> list[str]:
    detectors = load_yaml("pii_policy.yaml").get("detectors")
    if not isinstance(detectors, list) or not detectors:
        raise ValueError("configs/pii_policy.yaml must define a non-empty detectors list.")
    return list(detectors)


def load_presidio_config() -> dict:
    """Optional tuning block for the "presidio" detector (see pii_policy.yaml).

    Absent entirely when that detector isn't enabled; defaults are applied
    by PresidioPIIDetector itself, not here, so an empty/missing block is
    always valid.
    """
    config = load_yaml("pii_policy.yaml").get("presidio") or {}
    if not isinstance(config, dict):
        raise ValueError("configs/pii_policy.yaml 'presidio' key must be a mapping.")
    return config
