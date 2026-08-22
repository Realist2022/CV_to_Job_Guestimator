"""Presidio + spaCy NER PII detector.

Fully local, in-process alternative to ModelPIIDetector: no LLM/Ollama
roundtrip, no network call, deterministic given the same spaCy model. It is
registered as the "presidio" name in PII_DETECTOR_FACTORIES (see
pii_detector.py) so it composes with the existing detectors the same way
"regex" and "model" do — nothing about the composite/guardrail machinery
changes to support it.

Coverage vs. the model-based detector (see PII_SYSTEM_PROMPT):
  - person_name, other_identifier (email/phone/url): spaCy PERSON NER and
    Presidio's built-in pattern recognizers cover these well.
  - referee: no NER label for "this name sits in a References section", so
    PERSON spans found after a References/Referees heading are reclassified
    to "referee" (see _referee_section_start).
  - date_of_birth: spaCy's DATE_TIME entity fires on any date-shaped text,
    including employment/education ranges. Reuses is_valid_pii_span's
    existing date-range and age-sanity guard rather than re-implementing it.
  - nationality: NRP (spaCy's NORP label) misses common NZ phrasing like
    "Residency: New Zealand Citizen" (tagged GPE, not NORP), so a small
    custom keyword recognizer (_nationality_recognizer) backs it up.
  - marital_or_family: not a standard entity type at all; handled entirely
    by a custom keyword recognizer (_marital_family_recognizer).
  - street_address: Presidio's LOCATION conflates cities/countries with
    actual street addresses (e.g. "Auckland, New Zealand" has no street
    number). Only accept a LOCATION span if it contains a digit, so a bare
    city name is not treated as a specific residential address.
  - NZ-specific IDs (IRD, driver's licence): not attempted here — the
    existing "regex" detector already covers these; run "presidio" composed
    alongside "regex" rather than as a replacement for it.

The heavy AnalyzerEngine/spaCy model load only happens if this module is
actually imported, which only happens if "presidio" is in
configs/pii_policy.yaml's detectors list (see the lazy import in
pii_detector.py's PII_DETECTOR_FACTORIES).
"""

import re
from typing import Dict, Final, List, Optional

from src.schemas.pii import TextSpan
from src.services.document_parser import CandidateCV, normalise
from src.services.pii_detector import PIIDetector, is_valid_pii_span, parse_possible_dob

# Presidio entity types whose own recognizer already validates shape
# (regex/checksum-based, not spaCy NER) well enough that the shared
# CONTACT_OR_ID_PATTERN guard shouldn't second-guess them. That guard was
# tuned for LLM output, which typically copies the label along with the
# value (e.g. "Mobile: 021...") — Presidio isolates just the value, so a
# bare phone number never matches CONTACT_OR_ID_PATTERN's keyword branch
# and would otherwise be rejected despite being a correct, well-typed hit.
PRE_VALIDATED_ENTITY_TYPES: Final = {"EMAIL_ADDRESS", "PHONE_NUMBER", "URL"}

DEFAULT_SCORE_THRESHOLD: Final = 0.4
DEFAULT_MODEL_NAME: Final = "en_core_web_sm"

# Presidio entity_type -> PIIKind value. Anything Presidio finds that isn't
# a key here (IP_ADDRESS, CREDIT_CARD, US_SSN, etc. from its default
# recognizer set) is simply not requested — see ANALYZE_ENTITIES below —
# rather than silently redacted under the wrong kind.
ENTITY_TO_KIND: Final[Dict[str, str]] = {
    "PERSON": "person_name",
    "NRP": "nationality",
    "NATIONALITY_STATUS": "nationality",
    "LOCATION": "street_address",
    "DATE_TIME": "date_of_birth",
    "EMAIL_ADDRESS": "other_identifier",
    "PHONE_NUMBER": "other_identifier",
    "URL": "other_identifier",
    "MARITAL_FAMILY": "marital_or_family",
}

ANALYZE_ENTITIES: Final = list(ENTITY_TO_KIND)

REFEREE_HEADING: Final = re.compile(r"^\s*(?:referees?|references?)\b.*$", re.I | re.M)

# Best-effort keyword recognizers for the two PIIKind categories that have
# no standard NER equivalent. These are intentionally simple/regex-based —
# tune against real CVs via tasks/pii_presidio_eval.yaml rather than
# treating this as a finished model.
MARITAL_FAMILY_REGEX: Final = (
    r"(?i)\bmarital status\s*[:\-]?\s*\w[\w\s]{0,20}"
    r"|(?i)\b(?:married|single|divorced|widowed|separated|de facto|"
    r"in a relationship)\b(?:,?\s*(?:with\s+)?\d+\s+(?:children|kids|dependents))?"
    r"|(?i)\b\d+\s+(?:children|kids|dependents)\b"
)

NATIONALITY_STATUS_REGEX: Final = (
    r"(?i)\b(?:nationality|residency|citizenship)\s*[:\-]?\s*[^\n,.;]{2,40}"
    r"|(?i)\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?\s+"
    r"(?:citizen|permanent resident|national)\b"
    r"|(?i)\bwork visa\b|(?i)\bpermanent residency\b"
)


def _referee_section_start(text: str) -> Optional[int]:
    match = REFEREE_HEADING.search(text)
    return match.end() if match else None


def _looks_like_street_address(text: str) -> bool:
    """LOCATION is only treated as street_address when it has a house/unit
    number in it; a bare city or country name is not a specific address."""
    return bool(re.search(r"\d", text))


def _build_analyzer(model_name: str):
    # Imported lazily: this is the one place presidio/spacy get pulled in,
    # so a policy that never enables "presidio" never pays their import or
    # model-load cost.
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    nlp_engine = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": model_name}],
        }
    ).create_engine()

    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="MARITAL_FAMILY",
            patterns=[Pattern(name="marital_or_family_status", regex=MARITAL_FAMILY_REGEX, score=0.6)],
            supported_language="en",
        )
    )
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="NATIONALITY_STATUS",
            patterns=[Pattern(name="nationality_or_residency_status", regex=NATIONALITY_STATUS_REGEX, score=0.6)],
            supported_language="en",
        )
    )
    return analyzer


class PresidioPIIDetector(PIIDetector):
    def __init__(self, score_threshold: float = DEFAULT_SCORE_THRESHOLD, model_name: str = DEFAULT_MODEL_NAME):
        self.score_threshold = score_threshold
        self._analyzer = _build_analyzer(model_name)
        self._rejections: List[Dict[str, str]] = []

    @property
    def rejections(self) -> List[Dict[str, str]]:
        return self._rejections

    def detect(self, cv: CandidateCV) -> List[TextSpan]:
        self._rejections = []
        text = cv.text
        results = self._analyzer.analyze(
            text=text,
            language="en",
            entities=ANALYZE_ENTITIES,
            score_threshold=self.score_threshold,
        )
        referee_start = _referee_section_start(text)

        spans: List[TextSpan] = []
        seen: set[str] = set()
        accepted_ranges: List[tuple[int, int]] = []
        # Longest-span-first so e.g. a full email is accepted before an
        # overlapping recognizer's weaker match on just its domain, letting
        # the containment check below suppress the narrower duplicate.
        for result in sorted(results, key=lambda r: (r.start, -(r.end - r.start))):
            span_text = text[result.start:result.end].strip()
            if not span_text:
                continue
            if any(result.start >= s and result.end <= e for s, e in accepted_ranges):
                continue

            kind = ENTITY_TO_KIND[result.entity_type]
            if kind == "person_name" and referee_start is not None and result.start >= referee_start:
                kind = "referee"

            if kind == "street_address" and not _looks_like_street_address(span_text):
                self._rejections.append({"reason": "not_street_shaped", "kind": kind, "text": span_text})
                continue

            # spaCy's DATE_TIME fires on any date-shaped text (bare years,
            # month names, and non-dates like phone numbers that just don't
            # parse). is_valid_pii_span's DOB guard defaults unparseable
            # text to "valid" — a safe assumption for curated LLM output,
            # not for raw NER candidates — so require an actual parse here
            # first, on top of that guard's range/age-sanity checks.
            if kind == "date_of_birth" and parse_possible_dob(span_text) is None:
                self._rejections.append({"reason": "not_a_parseable_date", "kind": kind, "text": span_text})
                continue

            # Same grounding + heuristic guards ModelPIIDetector applies, so
            # rejections are directly comparable across detectors regardless
            # of which one produced the candidate span.
            if not cv.contains(span_text):
                self._rejections.append({"reason": "not_in_document", "kind": kind, "text": span_text})
                continue
            if result.entity_type not in PRE_VALIDATED_ENTITY_TYPES and not is_valid_pii_span(span_text, kind):
                self._rejections.append({"reason": "validation_guard_rejected", "kind": kind, "text": span_text})
                continue

            key = normalise(span_text)
            if not key or key in seen:
                continue
            seen.add(key)
            accepted_ranges.append((result.start, result.end))
            spans.append(TextSpan(kind, span_text))

        return spans
