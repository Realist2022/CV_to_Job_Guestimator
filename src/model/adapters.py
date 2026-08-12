"""Adapts a model config mapping (from configs/llm.yaml) to an InstructorClient."""

import os

from src.model.model_registry import get_provider_class
from src.services.llm_client import InstructorClient


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
