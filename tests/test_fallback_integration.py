"""Integration coverage for FallbackInstructorClient plugged into a real
pipeline/agent chain, not just the client's own retry loop.

test_fallback_client.py proves FallbackInstructorClient's fallback logic in
isolation (fake clients, no pipeline). test_web_app.py proves the API routes
work, but every test there monkeypatches client_for_role to a bare fake and
fakes out the pipeline classes entirely, so .complete() never actually runs.
Neither proves FallbackInstructorClient behaves correctly when
JobRequirementsAgent -> SkillMatcherAgent -> OverallExperienceAgent (the real
agents, via a real MatchingPipeline) call it. This file closes that gap.
"""

import unittest

from httpx import Request
from openai import APIConnectionError

from src.api.routes import _fallback_used
from src.schemas.experience import OverallExperienceOutput
from src.schemas.ingestion import RedactedCV
from src.schemas.pii import TextSpan
from src.schemas.requirements import JobRequirementsOutput
from src.services.document_parser import JobListing
from src.services.llm_client import FallbackInstructorClient
from src.services.matching_pipeline import MatchingPipeline
from tests.factories import FailingClient, RecordingClient


def _unreachable_backend() -> APIConnectionError:
    # Simulates the primary Ollama server being down/unreachable — one of
    # the two failure types FallbackInstructorClient treats as "the backend
    # failed" (see its docstring in src/services/llm_client.py).
    return APIConnectionError(request=Request("POST", "http://localhost:11434/v1"))


class FallbackThroughRealPipelineTest(unittest.TestCase):
    def test_matching_pipeline_completes_via_fallback_when_primary_is_down(self):
        primary = FailingClient(model="cv-guestimator:latest", exc=_unreachable_backend())
        fallback = RecordingClient(
            model="gemini-flash",
            responses={
                "job-requirements-prompt": JobRequirementsOutput(
                    job_requirements=[{"skill_name": "Python"}]
                ),
                "skill-matcher-prompt": lambda model: model(
                    evaluations=[{"requirement_id": 0, "matched": True}]
                ),
                "overall-experience-prompt": OverallExperienceOutput(
                    target_job_title="Python Developer",
                    target_overall_years=2.0,
                    candidate_roles=[],
                ),
            },
        )
        client = FallbackInstructorClient(primary, fallback)

        # Patch the three agents' system prompts to simple keys so this test
        # doesn't depend on the exact prose in prompts/templates.py (same
        # technique as test_ingestion_split.py's MatchingPipelineTest).
        pipeline = MatchingPipeline(client)
        pipeline.job_requirements_agent.system_prompt = "job-requirements-prompt"
        pipeline.skill_matcher_agent.system_prompt = "skill-matcher-prompt"
        pipeline.overall_experience_agent.system_prompt = "overall-experience-prompt"

        redacted_cv = RedactedCV.from_raw_text(
            raw_text="Jane Doe\nPython developer",
            redacted_text="[PERSON_NAME]\nPython developer",
            pii_spans=[TextSpan(kind="person_name", text="Jane Doe")],
            pii_engine="llama3.2:latest",
        )

        result = pipeline.run(
            JobListing("Requirements\nPython"), redacted_cv, verbose=False
        )

        # The pipeline produced a correct result despite every call to the
        # primary failing — proves the fallback path works through the real
        # agent chain, not just FallbackInstructorClient's own retry loop.
        self.assertEqual(result.skills_eval.matched_cv_skills, ["Python"])
        self.assertEqual(result.overall_experience.target_job_title, "Python Developer")

        # engine=self.client.model in matching_pipeline.py must report the
        # model that actually served the run, not the configured primary.
        self.assertEqual(result.engine, "gemini-flash")
        self.assertTrue(client.fallback_used)

        # The piece the API response's RunModelConfig.fallback_used field
        # depends on (src/api/routes.py) agrees with what actually happened.
        self.assertTrue(_fallback_used(client))

    def test_matching_pipeline_never_touches_fallback_when_primary_succeeds(self):
        primary = RecordingClient(
            model="cv-guestimator:latest",
            responses={
                "job-requirements-prompt": JobRequirementsOutput(
                    job_requirements=[{"skill_name": "Python"}]
                ),
                "skill-matcher-prompt": lambda model: model(
                    evaluations=[{"requirement_id": 0, "matched": True}]
                ),
                "overall-experience-prompt": OverallExperienceOutput(
                    target_job_title="Python Developer",
                    target_overall_years=2.0,
                    candidate_roles=[],
                ),
            },
        )
        # A fallback that would raise if ever called, proving it wasn't.
        fallback = FailingClient(model="gemini-flash", exc=_unreachable_backend())
        client = FallbackInstructorClient(primary, fallback)

        pipeline = MatchingPipeline(client)
        pipeline.job_requirements_agent.system_prompt = "job-requirements-prompt"
        pipeline.skill_matcher_agent.system_prompt = "skill-matcher-prompt"
        pipeline.overall_experience_agent.system_prompt = "overall-experience-prompt"

        redacted_cv = RedactedCV.from_raw_text(
            raw_text="Jane Doe\nPython developer",
            redacted_text="[PERSON_NAME]\nPython developer",
            pii_spans=[TextSpan(kind="person_name", text="Jane Doe")],
            pii_engine="llama3.2:latest",
        )

        result = pipeline.run(
            JobListing("Requirements\nPython"), redacted_cv, verbose=False
        )

        self.assertEqual(result.engine, "cv-guestimator:latest")
        self.assertFalse(client.fallback_used)
        self.assertFalse(_fallback_used(client))


if __name__ == "__main__":
    unittest.main()
