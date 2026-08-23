"""Tests for FallbackInstructorClient (see src/services/llm_client.py).

Uses fake clients rather than a live model/network: FailingClient raises one
of the exception types FallbackInstructorClient treats as "the backend
failed" (see its docstring for why those specific types), RecordingClient
(from tests/factories.py) plays back a canned response.
"""

import unittest
from typing import Any, Type

from httpx import Request
from instructor.core import InstructorRetryException
from openai import APIConnectionError
from pydantic import BaseModel

from src.services.llm_client import FallbackInstructorClient
from tests.factories import RecordingClient


class Answer(BaseModel):
    text: str


class FailingClient:
    """Fake InstructorClient whose complete() always raises the given exception."""

    def __init__(self, model: str, exc: BaseException):
        self.model = model
        self.temperature = 0.0
        self.last_attempts: int | None = None
        self._exc = exc

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        max_retries: int = 2,
    ) -> Any:
        raise self._exc


def _connection_error(message: str) -> APIConnectionError:
    return APIConnectionError(request=Request("POST", "http://localhost:11434/v1"))


class FallbackInstructorClientTest(unittest.TestCase):
    def test_uses_primary_when_it_succeeds(self):
        primary = RecordingClient(model="primary", responses={"sys": Answer(text="ok")})
        fallback = FailingClient(model="fallback", exc=_connection_error("unused"))

        client = FallbackInstructorClient(primary, fallback)
        result = client.complete("sys", "user", Answer)

        self.assertEqual(result.text, "ok")
        self.assertEqual(client.model, "primary")
        self.assertFalse(client.fallback_used)

    def test_falls_back_on_connection_error(self):
        primary = FailingClient(model="primary", exc=_connection_error("ollama down"))
        fallback = RecordingClient(model="fallback", responses={"sys": Answer(text="ok")})

        client = FallbackInstructorClient(primary, fallback)
        result = client.complete("sys", "user", Answer)

        self.assertEqual(result.text, "ok")
        self.assertEqual(client.model, "fallback")
        self.assertTrue(client.fallback_used)

    def test_falls_back_on_instructor_retry_exhaustion(self):
        retry_exc = InstructorRetryException(
            "gave up", n_attempts=2, total_usage=0, last_completion=None, messages=[]
        )
        primary = FailingClient(model="primary", exc=retry_exc)
        fallback = RecordingClient(model="fallback", responses={"sys": Answer(text="ok")})

        client = FallbackInstructorClient(primary, fallback)
        result = client.complete("sys", "user", Answer)

        self.assertEqual(result.text, "ok")
        self.assertTrue(client.fallback_used)

    def test_raises_when_every_client_fails(self):
        primary = FailingClient(model="primary", exc=_connection_error("down"))
        fallback = FailingClient(model="fallback", exc=_connection_error("also down"))

        client = FallbackInstructorClient(primary, fallback)

        with self.assertRaises(RuntimeError) as ctx:
            client.complete("sys", "user", Answer)
        self.assertIn("primary", str(ctx.exception))
        self.assertIn("fallback", str(ctx.exception))

    def test_unrecognised_exception_is_not_swallowed(self):
        primary = FailingClient(model="primary", exc=TypeError("bug, not a backend failure"))
        fallback = RecordingClient(model="fallback", responses={"sys": Answer(text="ok")})

        client = FallbackInstructorClient(primary, fallback)

        with self.assertRaises(TypeError):
            client.complete("sys", "user", Answer)

    def test_requires_at_least_one_fallback(self):
        primary = RecordingClient(model="primary", responses={})
        with self.assertRaises(ValueError):
            FallbackInstructorClient(primary)

    def test_reports_last_attempts_from_whichever_client_served_the_call(self):
        primary = RecordingClient(model="primary", responses={"sys": Answer(text="ok")})
        fallback = FailingClient(model="fallback", exc=_connection_error("unused"))
        client = FallbackInstructorClient(primary, fallback)

        client.complete("sys", "user", Answer)

        self.assertEqual(client.last_attempts, primary.last_attempts)


if __name__ == "__main__":
    unittest.main()
