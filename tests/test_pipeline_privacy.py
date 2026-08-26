import unittest

from src.prompts.templates import (
    JOB_REQUIREMENTS_SYSTEM_PROMPT,
    OVERALL_EXPERIENCE_SYSTEM_PROMPT,
    SKILL_MATCHER_SYSTEM_PROMPT,
)
from src.schemas.experience import OverallExperienceOutput
from src.schemas.requirements import JobRequirementsOutput
from src.services.document_parser import CandidateCV, JobListing
from src.services.extraction_pipeline import ExtractionPipeline
from src.services.pii_detector import (
    CompositePIIDetector,
    PIIDetector,
    build_pii_detector,
    is_valid_pii_span,
)
from src.services.presidio_detector import PresidioPIIDetector
from tests.factories import RecordingClient


class _FailingDetector(PIIDetector):
    """Simulates PII detection failing outright (e.g. a bug in a
    recognizer), for test_pii_detection_failure_prevents_cv_data_from_reaching_cloud."""

    def detect(self, cv: CandidateCV):
        raise RuntimeError("PII detection failed.")


class ExtractionPipelinePrivacyTest(unittest.TestCase):
    def test_raw_cv_pii_never_reaches_cloud_evaluation_client(self):
        # Real detector, not mocked: presidio runs entirely locally (no
        # network call at all), so there's no "local client" to record
        # requests against the way the old LLM-based detector had — the
        # guarantee under test is simply that raw PII never reaches the
        # cloud evaluation client, only its [KIND] redaction token does.
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
            pii_detector=PresidioPIIDetector(),
        ).run(
            JobListing("Requirements\nPython"),
            CandidateCV("Jane Doe\n\nPython developer"),
            verbose=False,
        )

        cloud_prompts = "\n".join(prompt for _, prompt in evaluation_client.requests)
        self.assertNotIn("Jane Doe", cloud_prompts)
        self.assertIn("[PERSON_NAME]", cloud_prompts)
        self.assertEqual(result.pii_engine, "presidio:en_core_web_sm")
        self.assertEqual(result.engine, "gemini-3.1-flash-lite")
        self.assertEqual(result.scorecard.final_relevance, 100.0)
        self.assertTrue(result.scorecard.pillar_b.applicable)
        self.assertEqual(len(evaluation_client.requests), 3)
        # Deterministic now: ingestion (ExtractionPipeline's compatibility
        # wrapper, see extraction_pipeline.py) always runs to completion
        # before matching starts, so pii_redaction is always first.
        self.assertEqual(
            [span.step for span in result.trace],
            [
                "pii_redaction",
                "job_requirements_extraction",
                "skill_matching",
                "overall_experience_extraction",
                "scoring",
            ],
        )
        self.assertTrue(all(span.duration_seconds >= 0.0 for span in result.trace))

    def test_pii_detection_failure_prevents_cv_data_from_reaching_cloud(self):
        # ExtractionPipeline runs ingestion (PII) to completion before
        # matching starts (see extraction_pipeline.py's module docstring
        # for the trade-off this makes vs. the old concurrent version).
        # That means a PII detection failure aborts before *any* cloud call
        # happens — not just before the CV-dependent ones — since job
        # requirements extraction is part of MatchingPipeline, which never
        # runs at all if ingestion raises first.
        evaluation_client = RecordingClient(
            "gemini-3.1-flash-lite",
            {
                JOB_REQUIREMENTS_SYSTEM_PROMPT: JobRequirementsOutput(
                    job_requirements=[{"skill_name": "Python"}]
                )
            },
        )

        with self.assertRaisesRegex(RuntimeError, "PII detection failed"):
            ExtractionPipeline(
                evaluation_client,
                pii_detector=_FailingDetector(),
            ).run(
                JobListing("Requirements\nPython"),
                CandidateCV("Jane Doe\n\nPython developer"),
                verbose=False,
            )

        self.assertEqual(evaluation_client.requests, [])

    def test_pii_guard_allows_single_date_dob_formats(self):
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

    def test_pii_guard_rejects_date_of_birth_ranges(self):
        invalid_ranges = [
            "2018 - 2022",
            "July 2025 – October 2025",
            "2020 to Present",
            "2015 2019",
        ]

        for date_range in invalid_ranges:
            with self.subTest(date_range=date_range):
                self.assertFalse(is_valid_pii_span(date_range, "date_of_birth"))

    def test_build_pii_detector_composes_from_configured_names(self):
        detector = build_pii_detector(["presidio"])

        self.assertIsInstance(detector, CompositePIIDetector)
        self.assertIsInstance(detector.detectors[0], PresidioPIIDetector)

    def test_build_pii_detector_rejects_unknown_names(self):
        with self.assertRaisesRegex(KeyError, "Unknown PII detector 'bogus'"):
            build_pii_detector(["bogus"])


if __name__ == "__main__":
    unittest.main()
