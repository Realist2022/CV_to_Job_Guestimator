import unittest

from src.prompts.templates import (
    JOB_REQUIREMENTS_SYSTEM_PROMPT,
    OVERALL_EXPERIENCE_SYSTEM_PROMPT,
    PII_SYSTEM_PROMPT,
    SKILL_MATCHER_SYSTEM_PROMPT,
    SKILL_TENURE_SYSTEM_PROMPT,
)
from src.schemas.experience import OverallExperienceOutput
from src.schemas.pii import PIIOutput
from src.schemas.requirements import JobRequirementsOutput
from src.services.document_parser import CandidateCV, JobListing
from src.services.pipeline import ExtractionPipeline


class RecordingClient:
    def __init__(self, model, responses):
        self.model = model
        self.responses = responses
        self.requests = []

    def complete(self, system_prompt, user_prompt, response_model, max_retries=2):
        self.requests.append((system_prompt, user_prompt))
        response = self.responses[system_prompt]
        if callable(response):
            return response(response_model)
        return response


class ExtractionPipelinePrivacyTest(unittest.TestCase):
    def test_pii_prompt_forbids_invented_kinds_for_excluded_content(self):
        self.assertIn("Every span.kind must use one of these exact enum values", PII_SYSTEM_PROMPT)
        self.assertIn("including job_title", PII_SYSTEM_PROMPT)
        self.assertIn("Omit it from spans", PII_SYSTEM_PROMPT)

    def test_raw_cv_pii_is_sent_only_to_local_pii_client(self):
        pii_client = RecordingClient(
            "llama3.2:latest",
            {
                PII_SYSTEM_PROMPT: PIIOutput(
                    spans=[{"kind": "person_name", "text": "Jane Doe"}]
                )
            },
        )
        evaluation_client = RecordingClient(
            "gemini-3.1-flash-lite",
            {
                JOB_REQUIREMENTS_SYSTEM_PROMPT: JobRequirementsOutput(
                    job_requirements=[
                        {"capability": "Python", "minimum_commercial_years": 1.0}
                    ]
                ),
                SKILL_MATCHER_SYSTEM_PROMPT: lambda model: model(
                    evaluations=[{"requirement_id": 0, "matched": True}]
                ),
                SKILL_TENURE_SYSTEM_PROMPT: lambda model: model(
                    skills=[
                        {
                            "requirement_id": 0,
                            "role_ids": [0],
                            "evidence": "Python developer",
                        }
                    ]
                ),
                OVERALL_EXPERIENCE_SYSTEM_PROMPT: OverallExperienceOutput(
                    target_job_title="Python Developer",
                    target_overall_years=2.0,
                    candidate_roles=[
                        {
                            "role_title": "Python Developer",
                            "start_date": "2020-01",
                            "end_date": "2022-01",
                            "match_rationale": "Directly relevant development role.",
                            "is_relevant": True,
                        }
                    ],
                ),
            },
        )

        result = ExtractionPipeline(
            evaluation_client,
            pii_client=pii_client,
        ).run(
            JobListing("Requirements\nPython"),
            CandidateCV("Jane Doe\nPython developer"),
            verbose=False,
        )

        self.assertIn("Jane Doe", pii_client.requests[0][1])
        cloud_prompts = "\n".join(prompt for _, prompt in evaluation_client.requests)
        self.assertNotIn("Jane Doe", cloud_prompts)
        self.assertIn("[PERSON_NAME]", cloud_prompts)
        self.assertEqual(result.pii_engine, "llama3.2:latest")
        self.assertEqual(result.engine, "gemini-3.1-flash-lite")
        self.assertEqual(result.scorecard.final_relevance, 100.0)
        self.assertFalse(result.scorecard.pillar_b.applicable)
        self.assertEqual(len(evaluation_client.requests), 3)

    def test_local_pii_failure_prevents_cloud_requests(self):
        pii_client = RecordingClient(
            "llama3.2:latest",
            {PII_SYSTEM_PROMPT: None},
        )
        evaluation_client = RecordingClient("gemini-3.1-flash-lite", {})

        with self.assertRaisesRegex(RuntimeError, "cloud evaluation was not started"):
            ExtractionPipeline(
                evaluation_client,
                pii_client=pii_client,
            ).run(
                JobListing("Requirements\nPython"),
                CandidateCV("Jane Doe\nPython developer"),
                verbose=False,
            )

        self.assertEqual(evaluation_client.requests, [])


if __name__ == "__main__":
    unittest.main()