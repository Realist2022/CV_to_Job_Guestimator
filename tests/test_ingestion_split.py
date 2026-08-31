"""Tests for the standalone ingestion/matching split.

See src/services/ingestion_pipeline.py, matching_pipeline.py, cv_store.py.
End-to-end behavior through the real harness (tasks/cv_ingest.yaml ->
tasks/cv_match_from_redacted.yaml, against a live local Ollama model) was
verified manually while building this; these tests cover what can be
checked without a live model, using the same RecordingClient-fake pattern
as test_pipeline_privacy.py.
"""

import tempfile
import unittest

from src.schemas.experience import OverallExperienceOutput
from src.schemas.ingestion import RedactedCV
from src.schemas.pii import TextSpan
from src.schemas.requirements import JobRequirementsOutput
from src.services.cv_store import CVIngestionStore, CVNotFoundError
from src.services.document_parser import CandidateCV, JobListing
from src.services.ingestion_pipeline import IngestionPipeline
from src.services.matching_pipeline import MatchingPipeline
from src.services.pii_base import PIIDetector
from tests.factories import RecordingClient


class _FakeEmailDetector(PIIDetector):
    """Minimal PIIDetector for testing IngestionPipeline's own plumbing —
    this test isn't concerned with detection quality, so there's no need
    to pay presidio's spaCy load cost for it."""

    def detect(self, cv: CandidateCV) -> list[TextSpan]:
        if "jane@example.com" not in cv.text:
            return []
        return [TextSpan(kind="other_identifier", text="jane@example.com")]


class CVIngestionStoreTest(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CVIngestionStore(output_dir=tmp)
            redacted = RedactedCV.from_raw_text(
                raw_text="Jane Doe\njane@example.com",
                redacted_text="[PERSON_NAME]\n[OTHER_IDENTIFIER]",
                pii_spans=[TextSpan(kind="person_name", text="Jane Doe")],
                pii_engine="llama3.2:latest",
            )

            store.save(redacted)
            loaded = store.load(redacted.cv_id)

            self.assertEqual(loaded.cv_id, redacted.cv_id)
            self.assertEqual(loaded.text, redacted.text)
            self.assertEqual(loaded.pii_engine, "llama3.2:latest")

    def test_load_missing_cv_id_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CVIngestionStore(output_dir=tmp)
            with self.assertRaises(CVNotFoundError):
                store.load("does-not-exist")

    def test_same_raw_text_is_content_addressed(self):
        # Same raw text (even with a formatting-only whitespace
        # difference, since the id is derived from normalise()) always
        # produces the same cv_id, so re-ingesting is idempotent.
        first = RedactedCV.from_raw_text("Jane Doe\n\ntext", "[X]", [], "engine-a")
        second = RedactedCV.from_raw_text("Jane Doe  text", "[Y]", [], "engine-b")
        different = RedactedCV.from_raw_text("John Smith text", "[Z]", [], "engine-a")

        self.assertEqual(first.cv_id, second.cv_id)
        self.assertNotEqual(first.cv_id, different.cv_id)


class IngestionPipelineTest(unittest.TestCase):
    def test_run_produces_redacted_cv_with_matching_cv_id(self):
        pipeline = IngestionPipeline(pii_detector=_FakeEmailDetector())

        result = pipeline.run(CandidateCV("Contact: jane@example.com"), verbose=False)

        self.assertEqual(result.cv_id, result.redacted_cv.cv_id)
        self.assertIn("jane@example.com", [s.text for s in result.pii_spans])
        self.assertNotIn("jane@example.com", result.redacted_cv.text)
        self.assertEqual([span.step for span in result.trace], ["pii_redaction"])
        self.assertEqual(result.pii_engine, "_FakeEmailDetector")


class MatchingPipelineTest(unittest.TestCase):
    def test_never_imports_candidate_cv_or_pii_detector(self):
        # The actual boundary: not that nobody currently passes raw text
        # in, but that this module's namespace has no bound reference to
        # these types at all — checking bound names, not source text,
        # since the module's own docstring legitimately *talks about* them
        # by name without importing them.
        import src.services.matching_pipeline as module

        module_names = set(vars(module))
        self.assertNotIn("CandidateCV", module_names)
        self.assertNotIn("PIIDetector", module_names)

    def test_run_produces_pipeline_result_from_redacted_cv(self):
        client = RecordingClient(
            "gemini-3.1-flash-lite",
            {
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
        # Patch the three agents' system prompts to simple keys so this
        # test doesn't depend on the exact prose in prompts/templates.py.
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

        self.assertEqual(result.engine, "gemini-3.1-flash-lite")
        self.assertEqual(result.pii_engine, "llama3.2:latest")
        self.assertEqual(result.redacted_cv_trace_id, redacted_cv.ingestion_trace_id)
        self.assertEqual(result.pii_spans, [TextSpan(kind="person_name", text="Jane Doe")])
        self.assertEqual(
            [span.step for span in result.trace],
            ["job_requirements_extraction", "skill_matching", "overall_experience_extraction", "scoring"],
        )
        # No PII call happened: MatchingPipeline has no pii_client at all.
        self.assertFalse(hasattr(pipeline, "pii_client"))


if __name__ == "__main__":
    unittest.main()
