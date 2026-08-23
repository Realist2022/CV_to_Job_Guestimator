from typing import Any, Protocol, Sequence, Type, runtime_checkable

import instructor
from instructor.core import InstructorRetryException
from openai import APIError, OpenAI
from pydantic import BaseModel


@runtime_checkable
class CompletionClient(Protocol):
    """Structural type for anything agents/pipelines can call for a completion.

    InstructorClient and FallbackInstructorClient both satisfy this without
    declaring it explicitly (that's what makes FallbackInstructorClient a
    drop-in replacement for InstructorClient wherever one is accepted).
    Agents and pipelines type their `client`/`pii_client` parameters as this
    instead of InstructorClient specifically, so that either can be passed
    without a type-checker complaint — see RecordingClient in
    tests/factories.py for a third implementation, used in tests.
    """

    @property
    def model(self) -> str: ...

    @property
    def temperature(self) -> float: ...

    @property
    def last_attempts(self) -> int | None: ...

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        max_retries: int = 2,
    ) -> Any: ...


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


class FallbackInstructorClient:
    """Tries a primary InstructorClient, falling back to the next one on failure.

    Duck-types InstructorClient (`complete()`, `.model`, `.temperature`,
    `.last_attempts`) so it's a drop-in replacement anywhere an
    InstructorClient is used today — agents/pipelines never need to know a
    fallback happened. `.model`/`.temperature`/`.last_attempts` always
    reflect whichever client actually served the most recent `complete()`
    call, which is what callers read after a run to report which engine
    produced the result (see matching_pipeline.py's `engine=self.client.model`).

    A "failure" here means the primary is unreachable or broken, not merely
    slow to agree with the response schema on one attempt — Instructor
    already retries validation failures internally up to `max_retries`
    before it gives up, so by the time that raises, the primary has had its
    fair shot:

    - `InstructorRetryException`: Instructor exhausted its retries without
      getting output that validated against `response_model`.
    - `openai.APIError` (and subclasses `APIConnectionError`,
      `APITimeoutError`, `APIStatusError`, `RateLimitError`, ...): the
      backend itself failed — e.g. a local Ollama server that isn't
      running, or hasn't had the model `ollama pull`ed/`ollama create`d yet.

    Any other exception (a bug in our own code, a bad response_model, ...)
    is left to propagate rather than silently masked by a fallback attempt.
    """

    _FAILURE_TYPES: tuple[type[BaseException], ...] = (InstructorRetryException, APIError)

    def __init__(self, primary: CompletionClient, *fallbacks: CompletionClient):
        if not fallbacks:
            raise ValueError("FallbackInstructorClient requires at least one fallback client.")
        self._clients: Sequence[CompletionClient] = (primary, *fallbacks)
        self._active: CompletionClient = primary
        self.last_attempts: int | None = None

    @property
    def model(self) -> str:
        return self._active.model

    @property
    def temperature(self) -> float:
        return self._active.temperature

    @property
    def fallback_used(self) -> bool:
        """Whether the most recent `complete()` call was served by anything
        other than the primary client."""
        return self._active is not self._clients[0]

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        max_retries: int = 2,
    ) -> Any:
        errors: list[str] = []
        for client in self._clients:
            try:
                result = client.complete(
                    system_prompt, user_prompt, response_model, max_retries=max_retries
                )
            except self._FAILURE_TYPES as exc:
                errors.append(f"{client.model}: {type(exc).__name__}: {exc}")
                continue
            self._active = client
            self.last_attempts = client.last_attempts
            return result

        raise RuntimeError(
            "All configured models failed for this request: " + "; ".join(errors)
        )