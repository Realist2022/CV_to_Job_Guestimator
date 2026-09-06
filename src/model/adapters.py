"""Adapts a model config mapping (from configs/llm.yaml) to an InstructorClient."""

import os

from src.config import load_model_config, load_pipeline_fallback_names, load_pipeline_model_names
from src.model.model_registry import get_provider_class
from src.services.llm_client import CompletionClient, FallbackInstructorClient, InstructorClient


def client_for_role(role: str) -> CompletionClient:
    """Build the client configured in configs/pipeline.yaml for 'evaluation' or 'pii'.

    If configs/pipeline.yaml's `fallback_models` names a fallback for this
    role, the returned client is a FallbackInstructorClient that tries the
    primary model first and only calls the fallback model if the primary
    fails (see FallbackInstructorClient for what counts as a failure).
    Callers don't need to care either way: both expose the same
    `complete()` / `.model` / `.temperature` / `.last_attempts` interface.
    """
    primary = client_from_config(load_model_config(load_pipeline_model_names()[role]))
    fallback_name = load_pipeline_fallback_names().get(role)
    if fallback_name is None:
        return primary
    fallback = client_from_config(load_model_config(fallback_name))
    return FallbackInstructorClient(primary, fallback)


def _resolve_with_env(config: dict, key: str) -> str | None:
    """Pop `key` from a model config, falling back to the env var named by `<key>_env`.

    Both `api_key`/`api_key_env` and `base_url`/`base_url_env` work this way,
    so a config can name a value inline where it is harmless to commit (an
    Ollama URL on localhost) and name an environment variable where it is not
    — a secret, or a per-workspace endpoint like the Modal deployment, which
    differs between developers and so cannot be written into a shared config
    at all.

    A literal wins over the env var when both are given. A `<key>_env` naming
    an unset variable raises rather than returning None: it was written down
    precisely because the value is needed, and letting it through surfaces
    later as a 401 or a connection refused against a provider default, well
    away from the config that actually caused it.
    """
    value = config.pop(key, None)
    env_name = config.pop(f"{key}_env", None)
    if value is not None or not env_name:
        return value
    value = os.getenv(env_name)
    if not value:
        raise ValueError(
            f"Environment variable '{env_name}' is not set but is required "
            f"by the model config's {key}_env."
        )
    return value


def client_from_config(config: dict) -> InstructorClient:
    config = dict(config)
    provider_name = config.pop("provider", "openai_compatible")

    provider_class = get_provider_class(provider_name)
    provider = provider_class(
        model=config.pop("model"),
        base_url=_resolve_with_env(config, "base_url"),
        api_key=_resolve_with_env(config, "api_key"),
        temperature=config.pop("temperature", None),
    )
    if config:
        raise ValueError(f"Unrecognised model config keys: {sorted(config)}")
    return provider.create_client()
