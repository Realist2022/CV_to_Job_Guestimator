import unittest

from src.prompts.templates import (
    JOB_REQUIREMENTS_SYSTEM_PROMPT,
    OVERALL_EXPERIENCE_SYSTEM_PROMPT,
    PII_SYSTEM_PROMPT,
    SKILL_MATCHER_SYSTEM_PROMPT,
)
from src.schemas.experience import OverallExperienceOutput
from src.schemas.pii import PIIOutput
from src.schemas.requirements import JobRequirementsOutput
from src.services.agents import PIIAgent
from src.services.document_parser import CandidateCV, JobListing
from src.services.pii_detector import (
    ModelPIIDetector,
    RegexPIIDetector,
    is_valid_pii_span,
)
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
                    job_requirements=[{"skill_name": "Python"}]
                ),
                SKILL_MATCHER_SYSTEM_PROMPT: lambda model: model(
                    evaluations=[{"requirement_id": 0, "matched": True}]
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
        self.assertTrue(result.scorecard.pillar_b.applicable)
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

    def test_model_pii_guard_keeps_employer_and_employment_dates(self):
        pii_client = RecordingClient(
            "llama3.2:latest",
            {
                PII_SYSTEM_PROMPT: PIIOutput(
                    spans=[
                        {
                            "kind": "other_identifier",
                            "text": "FOODSTUFFS · July 2025 – October 2025",
                        },
                        {"kind": "other_identifier", "text": "Email: jane@example.com"},
                    ]
                )
            },
        )
        detector = ModelPIIDetector(PIIAgent(pii_client))

        spans = detector.detect(
            CandidateCV(
                "Experience\n"
                "FOODSTUFFS · July 2025 – October 2025\n"
                "Role: Software Engineer\n"
                "Email: jane@example.com"
            )
        )

        self.assertEqual([span.text for span in spans], ["Email: jane@example.com"])
        self.assertEqual(
            detector.rejections,
            [
                {
                    "reason": "validation_guard_rejected",
                    "kind": "other_identifier",
                    "text": "FOODSTUFFS · July 2025 – October 2025",
                }
            ],
        )

    def test_model_pii_guard_allows_single_date_dob_formats(self):
        valid_dobs = [
            "1998-05-20",
            "20-05-1998",
            "20/05/1998",
            "20 May 1998",
            "May 20, 1998",
        ]

        for dob in valid_dobs:
            with self.subTest(dob=dob):
                self.assertTrue(is_valid_pii_span(dob, "date_of_birth"))

    def test_model_pii_guard_rejects_date_of_birth_ranges(self):
        invalid_ranges = [
            "2018 - 2022",
            "July 2025 – October 2025",
            "2020 to Present",
            "2015 2019",
        ]

        for date_range in invalid_ranges:
            with self.subTest(date_range=date_range):
                self.assertFalse(is_valid_pii_span(date_range, "date_of_birth"))

    def test_regex_detector_extracts_strict_email_addresses(self):
        spans = RegexPIIDetector().detect(
            CandidateCV(
                "Reach us at support@example.com, "
                "o'connor.tech+tag@sub.domain.co.uk, "
                "or john_doe!@company.org. "
                "Avoid invalid emails like user..name@test.com, "
                "user@bad_domain.com, user@-domain.com, "
                "user@domain-.com, and user@example.c."
            )
        )

        self.assertEqual(
            [span.text for span in spans if span.kind == "email"],
            [
                "support@example.com",
                "o'connor.tech+tag@sub.domain.co.uk",
                "john_doe!@company.org",
            ],
        )


if __name__ == "__main__":
    unittest.main()