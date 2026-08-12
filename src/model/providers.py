"""Model providers: each knows how to build an InstructorClient for one backend.

The pipeline only ever sees an InstructorClient, so swapping a local Ollama
model (including a fine-tuned LoRA build) for a cloud endpoint is purely a
config change.
"""

from abc import ABC

from src.services.llm_client import InstructorClient


class ModelProvider(ABC):
    default_base_url: str | None = None
    default_api_key: str | None = None

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
    ):
        self.model = model
        self.base_url = base_url or self.default_base_url
        self.api_key = api_key or self.default_api_key
        self.temperature = temperature
        if not self.base_url:
            raise ValueError(f"{type(self).__name__} requires a base_url.")
        if self.api_key is None:
            raise ValueError(
                f"{type(self).__name__} requires an api_key (or api_key_env)."
            )

    def create_client(self) -> InstructorClient:
        return InstructorClient(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature,
        )


class OllamaProvider(ModelProvider):
    default_base_url = "http://localhost:11434/v1"
    default_api_key = "ollama"


class OpenAICompatibleProvider(ModelProvider):
    """Any OpenAI-compatible endpoint (Gemini OpenAI shim, OpenAI, vLLM, ...)."""
