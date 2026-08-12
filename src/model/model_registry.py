"""Registry mapping provider names used in configs/llm.yaml to classes."""

from src.model.providers import ModelProvider, OllamaProvider, OpenAICompatibleProvider

_PROVIDERS: dict[str, type[ModelProvider]] = {
    "ollama": OllamaProvider,
    "openai_compatible": OpenAICompatibleProvider,
}


def register_provider(name: str, provider_class: type[ModelProvider]) -> None:
    if name in _PROVIDERS:
        raise ValueError(f"Provider '{name}' is already registered.")
    _PROVIDERS[name] = provider_class


def get_provider_class(name: str) -> type[ModelProvider]:
    try:
        return _PROVIDERS[name]
    except KeyError:
        known = ", ".join(sorted(_PROVIDERS))
        raise KeyError(f"Unknown model provider '{name}'. Known: {known}") from None
