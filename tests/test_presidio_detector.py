"""Tests for PresidioPIIDetector (src/services/presidio_detector.py).

These are scoped to what real testing against the project's sample CV
(dataSet/tradeMeCV/Sonny H Tapara CV.txt) and synthetic CVs actually showed
to be reliable: the custom regex-based recognizers (marital/nationality),
the shared guard reuse, overlap suppression, and config/registry wiring.

Deliberately NOT asserted here: person_name/date_of_birth/street_address
recall on fragmented, colon-heavy CV text. Real runs against en_core_web_sm
showed those NER-dependent categories are unreliable on this document style
(label words get glued into adjacent PERSON spans, DATE_TIME/LOCATION spans
are frequently missed entirely or truncated) — see the module docstring in
presidio_detector.py. Encoding that unreliability as pass/fail assertions
would make this suite flaky against the very thing it should be honest
about. Track that gap via tasks/pii_presidio_eval.yaml artifacts instead.
"""

import unittest

from src.services.document_parser import CandidateCV
from src.services.pii_detector import PII_DETECTOR_FACTORIES, build_pii_detector
from src.services.presidio_detector import PresidioPIIDetector, _looks_like_street_address


class PresidioDetectorTest(unittest.TestCase):
    def test_registered_in_factories_and_composes_with_regex(self):
        self.assertIn("presidio", PII_DETECTOR_FACTORIES)

        detector = build_pii_detector(["regex", "presidio"])

        self.assertEqual(len(detector.detectors), 2)
        self.assertIsInstance(detector.detectors[1], PresidioPIIDetector)

    def test_contact_fields_detected_on_real_cv(self):
        # en_core_web_sm also produces some person_name noise on this CV's
        # fragmented bullet layout (see the module docstring) — this only
        # asserts the real PII is *among* the results, not that nothing
        # else fires; the noise itself is out of scope for this suite.
        cv = CandidateCV.from_path("dataSet/tradeMeCV/Sonny H Tapara CV.txt")
        detector = PresidioPIIDetector()

        spans = detector.detect(cv)
        texts_by_kind = {}
        for span in spans:
            texts_by_kind.setdefault(span.kind, []).append(span.text)

        self.assertIn("Sonny Tapara", texts_by_kind.get("person_name", []))
        self.assertIn("0211428396", texts_by_kind.get("other_identifier", []))
        self.assertIn("s.h.tapara@gmail.com", texts_by_kind.get("other_identifier", []))
        self.assertIn(
            "Residency: New Zealand Citizen", texts_by_kind.get("nationality", [])
        )

    def test_phone_number_not_misclassified_as_date_of_birth(self):
        # Regression check: spaCy's DATE_TIME entity fired on the bare
        # mobile digit string, and is_valid_pii_span's permissive
        # unparseable-defaults-to-valid DOB fallback (tuned for curated LLM
        # spans) let it straight through until the parse_possible_dob guard
        # was added.
        cv = CandidateCV.from_path("dataSet/tradeMeCV/Sonny H Tapara CV.txt")
        detector = PresidioPIIDetector()

        spans = detector.detect(cv)

        self.assertNotIn("date_of_birth", {span.kind for span in spans})
        self.assertIn(
            {"reason": "not_a_parseable_date", "kind": "date_of_birth", "text": "0211428396"},
            detector.rejections,
        )

    def test_email_duplicate_domain_span_is_suppressed(self):
        # Presidio's URL recognizer can additionally match just the domain
        # portion of an already-matched email; that nested span should be
        # suppressed rather than redacted/reported twice.
        cv = CandidateCV("Contact\nEmail: s.h.tapara@gmail.com\n")
        detector = PresidioPIIDetector()

        spans = [s.text for s in detector.detect(cv) if s.kind == "other_identifier"]

        self.assertEqual(spans, ["s.h.tapara@gmail.com"])

    def test_marital_and_nationality_custom_recognizers(self):
        cv = CandidateCV("Jane Smith\nMarital Status: Married, 2 children\nResidency: New Zealand Citizen\n")
        detector = PresidioPIIDetector()

        spans = detector.detect(cv)
        marital_texts = [s.text for s in spans if s.kind == "marital_or_family"]
        nationality_texts = [s.text for s in spans if s.kind == "nationality"]

        self.assertIn("Marital Status: Married", marital_texts)
        self.assertIn("2 children", marital_texts)
        self.assertIn("Residency: New Zealand Citizen", nationality_texts)

    def test_referee_reclassification_after_references_heading(self):
        cv = CandidateCV("Jane Doe\n\nReferences\nSarah Johnson - ref@example.com\n")
        detector = PresidioPIIDetector()

        spans = detector.detect(cv)

        self.assertIn("Jane Doe", [s.text for s in spans if s.kind == "person_name"])
        self.assertTrue(
            any(s.kind == "referee" and s.text.startswith("Sarah Johnson") for s in spans),
            spans,
        )
        self.assertFalse(
            any(s.kind == "person_name" and s.text.startswith("Sarah Johnson") for s in spans)
        )

    def test_employer_and_employment_dates_not_flagged(self):
        # Same exclusion the PII_SYSTEM_PROMPT enforces for the LLM
        # detector: org/company names and employment date ranges are not
        # PII. spaCy's ORG label is never mapped in ENTITY_TO_KIND, and
        # date ranges fail the parseable-date guard.
        cv = CandidateCV(
            "Jane Doe\n\nExperience\nACME CORP - Software Engineer\n2018 - 2022\n"
        )
        detector = PresidioPIIDetector()

        spans = detector.detect(cv)

        self.assertNotIn("ACME CORP", [s.text for s in spans])
        self.assertNotIn("date_of_birth", {s.kind for s in spans})

    def test_looks_like_street_address_requires_a_digit(self):
        self.assertFalse(_looks_like_street_address("Auckland, New Zealand"))
        self.assertTrue(_looks_like_street_address("12 Queen Street, Auckland 1010"))

    def test_score_threshold_is_configurable(self):
        detector = PresidioPIIDetector(score_threshold=0.9)
        self.assertEqual(detector.score_threshold, 0.9)


if __name__ == "__main__":
    unittest.main()
