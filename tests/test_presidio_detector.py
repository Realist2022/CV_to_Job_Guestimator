"""Tests for PresidioPIIDetector (src/services/presidio_detector.py).

These are scoped to what real testing against the project's sample CV
(dataSet/tradeMeCV/Sonny H Tapara CV.txt) and synthetic CVs actually showed
to be reliable: the custom regex-based recognizers (marital/nationality,
NZ IRD number/driver's licence/postcode line), the shared guard reuse,
overlap suppression, and config/registry wiring.

Deliberately NOT asserted here: person_name/date_of_birth/street_address
recall on fragmented, colon-heavy CV text. Real runs against en_core_web_sm
showed those NER-dependent categories are unreliable on this document style
(label words get glued into adjacent PERSON spans, DATE_TIME/LOCATION spans
are frequently missed entirely or truncated) — see the module docstring in
presidio_detector.py. Encoding that unreliability as pass/fail assertions
would make this suite flaky against the very thing it should be honest
about. Track that gap by running tasks/cv_ingest.yaml against a wider set
of real CVs and inspecting the resulting artifacts' pii_spans/redacted_cv
instead.
"""

import unittest

from src.schemas.pii import TextSpan
from src.services.document_parser import CandidateCV
from src.services.pii_base import PII_DETECTOR_FACTORIES, build_pii_detector
from src.services.presidio_detector import (
    PresidioPIIDetector,
    _content_section_start,
    _looks_like_person_name,
    _looks_like_personal_link,
    _looks_like_street_address,
    _person_name_fragment,
)


class PresidioDetectorTest(unittest.TestCase):
    def test_registered_in_factories_and_composed_via_build_pii_detector(self):
        self.assertIn("presidio", PII_DETECTOR_FACTORIES)

        detector = build_pii_detector(["presidio"])

        self.assertEqual(len(detector.detectors), 1)
        self.assertIsInstance(detector.detectors[0], PresidioPIIDetector)

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
        # Org/company names and employment date ranges are not PII.
        # spaCy's ORG label is never mapped in ENTITY_TO_KIND, and date
        # ranges fail the parseable-date guard.
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

    def test_looks_like_person_name_rejects_glued_bullet_lines(self):
        self.assertFalse(_looks_like_person_name("• Git / GitHub\n• CI"))
        self.assertFalse(_looks_like_person_name("• Skill"))
        self.assertTrue(_looks_like_person_name("Sonny Tapara"))
        self.assertTrue(_looks_like_person_name("O'Connor-Smith"))

    def test_content_section_start_anchors_on_body_headings(self):
        self.assertIsNone(_content_section_start("Jane Doe\nSoftware Engineer\n"))
        text = "Jane Doe\n\nEducation\nBSc Computer Science\n"
        start = _content_section_start(text)
        self.assertIsNotNone(start)
        self.assertEqual(text[start:].strip(), "BSc Computer Science")
        # A sub-bullet like "Key Skills:" must not false-trigger — only a
        # line that *starts* with one of the heading words counts.
        self.assertIsNone(_content_section_start("Jane Doe\n• Key Skills: Python\n"))

    def test_tech_terms_in_body_sections_not_misredacted_as_person_name(self):
        # Regression for a real run: "Cloud Storage", "Sharing App", and
        # "Truck Driver" — all inside Skills/Projects/Experience sections,
        # nowhere near the header/contact block — were misclassified as
        # PERSON by en_core_web_sm and redacted, corrupting real content
        # ("Google Cloud Storage" -> "Google [PERSON_NAME]").
        cv = CandidateCV.from_path("dataSet/tradeMeCV/Sonny H Tapara CV.txt")
        detector = PresidioPIIDetector()

        spans = detector.detect(cv)
        person_texts = [s.text for s in spans if s.kind == "person_name"]

        self.assertEqual(person_texts, ["Sonny Tapara"])
        redacted = cv.redacted(spans).text
        self.assertIn("Google Cloud Storage", redacted)
        self.assertIn("Sharing App", redacted)
        self.assertIn("Truck Driver", redacted)

    def test_bulleted_skill_lines_not_misredacted_as_person_name(self):
        # Regression for a real run against this exact fixture: spaCy glued
        # "• Git / GitHub\n• CI" into one PERSON span, which then split
        # "CI/CD pipelines" into "[PERSON_NAME]/CD pipelines" on redaction;
        # "• Skill" (as its own PERSON span) ate the bullet off "Skill
        # set:", leaving "[PERSON_NAME] set:". A synthetic snippet doesn't
        # reliably reproduce spaCy's context-dependent misclassification
        # (see the module docstring on why this suite tests against the
        # real CV rather than minimal repros), so this asserts directly on
        # the fixture the corruption was observed on.
        cv = CandidateCV.from_path("dataSet/tradeMeCV/Sonny H Tapara CV.txt")
        detector = PresidioPIIDetector()

        spans = detector.detect(cv)

        self.assertFalse(any(s.text.startswith("• ") for s in spans if s.kind == "person_name"))
        redacted = cv.redacted(spans).text
        self.assertIn("CI/CD pipelines", redacted)
        self.assertIn("Skill set:", redacted)

    def test_looks_like_personal_link_requires_scheme_or_nearby_context(self):
        # "Cortex.io" sits well outside the 40-char lookback window from
        # "LinkedIn:", same as in the real CV where the two are pages apart.
        text = "LinkedIn: sonnytapara.dev" + (" " * 60) + "Cortex.io: Ingested data"
        linkedin_start = text.index("sonnytapara.dev")
        cortex_start = text.index("Cortex.io")

        self.assertTrue(
            _looks_like_personal_link(text, linkedin_start, "sonnytapara.dev")
        )
        self.assertFalse(_looks_like_personal_link(text, cortex_start, "Cortex.io"))
        # Span itself being unambiguous (scheme/www./known host) is enough
        # even with zero surrounding context.
        self.assertTrue(
            _looks_like_personal_link("visit https://sonnytapara.dev today", 6, "https://sonnytapara.dev")
        )

    def test_third_party_tool_domain_not_misredacted_as_other_identifier(self):
        # Regression for a real run: Presidio's "Non schema URL" pattern
        # matches any bare word.tld — including "Cortex.io", a third-party
        # tool named in a skills bullet ("Cortex.io: Ingested and extracted
        # data for scorecards and EDP"), not a link the candidate is
        # sharing. URL is in PRE_VALIDATED_ENTITY_TYPES (skips the shared
        # CONTACT_OR_ID_PATTERN guard), so this needed its own check.
        cv = CandidateCV.from_path("dataSet/tradeMeCV/Sonny H Tapara CV.txt")
        detector = PresidioPIIDetector()

        spans = detector.detect(cv)

        self.assertNotIn("Cortex.io", [s.text for s in spans])
        self.assertIn(
            {"reason": "not_a_personal_link", "kind": "other_identifier", "text": "Cortex.io"},
            detector.rejections,
        )
        redacted = cv.redacted(spans).text
        self.assertIn("Cortex.io", redacted)

    def test_score_threshold_is_configurable(self):
        detector = PresidioPIIDetector(score_threshold=0.9)
        self.assertEqual(detector.score_threshold, 0.9)

    def test_nz_ird_number_and_drivers_licence_detected_natively(self):
        # Ported from RegexPIIDetector.PATTERNS (pii_base.py) so
        # "presidio" alone covers every PIIKind without needing "regex"
        # composed alongside it — see the module docstring's coverage
        # notes, now closed for these two.
        cv = CandidateCV("Jane Doe\nIRD: 123-456-789\nLicence: AB123456\n")
        detector = PresidioPIIDetector()

        spans = detector.detect(cv)
        kinds_by_text = {s.text: s.kind for s in spans}

        self.assertEqual(kinds_by_text.get("123-456-789"), "other_identifier")
        self.assertEqual(kinds_by_text.get("AB123456"), "other_identifier")

    def test_person_name_fragment_trims_glued_spans_without_widening_recall(self):
        # A clean span passes through untouched.
        self.assertEqual(_person_name_fragment("Sonny Tapara"), "Sonny Tapara")
        self.assertEqual(_person_name_fragment("O'Connor-Smith"), "O'Connor-Smith")
        # A glued span is trimmed back to the name, not thrown away.
        self.assertEqual(_person_name_fragment("Taylor Developer\nName"), "Taylor Developer")
        self.assertEqual(
            _person_name_fragment("Taylor Developer\nLocation: Auckland"), "Taylor Developer"
        )
        # What the guard exists to stop still stops: nothing precedes the
        # first bullet, and a label glued on before the newline is not
        # name-shaped.
        self.assertIsNone(_person_name_fragment("• Git / GitHub\n• CI"))
        self.assertIsNone(_person_name_fragment("• Skill"))
        self.assertIsNone(_person_name_fragment("Name: Taylor\nLocation"))

    def test_content_section_start_matches_qualified_experience_headings(self):
        # Regression: "Work Experience:" is the commonest spelling of this
        # heading and matched none of the alternatives, so content_start
        # was None and the header/body guard never ran at all.
        for heading in ("Work Experience:", "Professional Experience", "Relevant Skills"):
            text = f"Jane Doe\n\n{heading}\nLead Engineer at Acme\n"
            self.assertIsNotNone(_content_section_start(text), heading)
        # Still matched with no qualifier in front.
        self.assertIsNotNone(_content_section_start("Jane Doe\n\nEmployment History\nAcme\n"))
        # A qualifier alone is not a body heading.
        self.assertIsNone(_content_section_start("Jane Doe\nProfessional Summary: Versatile\n"))

    def test_header_name_recovered_and_body_false_positive_not_redacted(self):
        # Regression for run-000232: on this CV the only pii_span produced
        # was person_name "Java" (from the Work Experience bullets), while
        # the candidate's actual name went out in the clear. Three separate
        # causes, all exercised here — spaCy glued the following line onto
        # both "Taylor Developer" PERSON hits so both were dropped whole;
        # "Work Experience:" did not match CONTENT_SECTION_HEADING so the
        # body guard never armed; and redaction was a bare substring
        # replace, so "Java" rewrote "JavaScript" to "[PERSON_NAME]Script".
        cv = CandidateCV(
            "Taylor Developer\n"
            "Name: Taylor Developer\n"
            "Location: Auckland, New Zealand\n"
            "Role: Senior Full Stack Engineer\n"
            "Work Experience:\n"
            "Lead Software Engineer | CloudTech Solutions > Jan 2020 - Present\n"
            "• Led the migration of our legacy frontend from vanilla JavaScript to TypeScript.\n"
            "• Maintained a legacy Java codebase for an internal CRM tool.\n"
        )
        detector = PresidioPIIDetector()

        spans = detector.detect(cv)
        person_texts = [s.text for s in spans if s.kind == "person_name"]
        redacted = cv.redacted(spans).text

        self.assertEqual(person_texts, ["Taylor Developer"])
        self.assertNotIn("Taylor Developer", redacted)
        self.assertIn("JavaScript", redacted)

    def test_redaction_does_not_match_inside_a_longer_word(self):
        cv = CandidateCV("Java Smith\nWorked in JavaScript and Java.\n")

        redacted = cv.redacted([TextSpan("person_name", "Java")]).text

        self.assertIn("JavaScript", redacted)
        self.assertEqual(redacted.count("[PERSON_NAME]"), 2)


if __name__ == "__main__":
    unittest.main()
