import instructor
from openai import OpenAI
from pydantic import BaseModel
from typing import Type, Any
from src.config import MODEL_NAME, MODEL_BASE_URL, MODEL_API_KEY


class InstructorClient:
    """LLM Transport layer backed by Instructor for structured outputs."""

    def __init__(
        self,
        model: str = MODEL_NAME,
        base_url: str = MODEL_BASE_URL,
        api_key: str = MODEL_API_KEY,
    ):
        self.model = model
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
            temperature=0.0,
        )