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
        # Set by complete() after each call: how many attempts instructor
        # actually made (1 = succeeded first try, >1 = it retried after a
        # validation failure). None until the first call completes.
        self.last_attempts: int | None = None
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
        attempts = 0

        def _count_attempt(*_args, **_kwargs) -> None:
            nonlocal attempts
            attempts += 1

        # Not safe to call concurrently on the same InstructorClient: the
        # hook is registered/removed around a single call, so two calls on
        # this client racing would double-count or drop attempts.
        self.client.on("completion:kwargs", _count_attempt)
        try:
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
        finally:
            self.client.off("completion:kwargs", _count_attempt)
            self.last_attempts = attempts