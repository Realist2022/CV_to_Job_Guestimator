import instructor
from openai import OpenAI
from pydantic import BaseModel
from typing import Type, Any


class InstructorClient:
    """LLM Transport layer backed by Instructor for structured outputs."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float | None = None,
    ):
        self.model = model
        self.temperature = 0.0 if temperature is None else temperature
        # Wrap standard OpenAI client with Instructor
        self.client = instructor.from_openai(
            OpenAI(base_url=base_url, api_key=api_key),
            mode=instructor.Mode.JSON,
        )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        max_retries: int = 2,
    ) -> Any:
        return self.client.chat.completions.create(
            model=self.model,
            response_model=response_model,
            max_retries=max_retries,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
        )