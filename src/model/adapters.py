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


def client_from_config(config: dict) -> InstructorClient:
    config = dict(config)
    provider_name = config.pop("provider", "openai_compatible")

    api_key = config.pop("api_key", None)
    api_key_env = config.pop("api_key_env", None)
    if api_key is None and api_key_env:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(
                f"Environment variable '{api_key_env}' is not set but is "
                "required by the model config."
            )

    provider_class = get_provider_class(provider_name)
    provider = provider_class(
        model=config.pop("model"),
        base_url=config.pop("base_url", None),
        api_key=api_key,
        temperature=config.pop("temperature", None),
    )
    if config:
        raise ValueError(f"Unrecognised model config keys: {sorted(config)}")
    return provider.create_client()
