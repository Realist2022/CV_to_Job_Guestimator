"""Presidio + spaCy NER PII detector — the only PII detector this project
has: fully local, in-process, no LLM/Ollama roundtrip, no network call,
deterministic given the same spaCy model. Registered as the "presidio" name
in PII_DETECTOR_FACTORIES (see pii_detector.py).

Per-PIIKind coverage:
  - person_name, other_identifier (email/phone/url): spaCy PERSON NER and
    Presidio's built-in pattern recognizers cover these well. Two shape
    guards (_looks_like_person_name, _content_section_start) reject PERSON
    hits that are either malformed (glued across a bullet/newline by
    spaCy's NER on fragmented CV text) or sit outside the header/contact
    block (a candidate's own name is written once, at the very top, before
    any Education/Experience/Skills heading — see CONTENT_SECTION_HEADING).
  - other_identifier (URL specifically): Presidio's "Non schema URL"
    pattern matches any bare word.tld, including a third-party tool named
    in prose ("Cortex.io: Ingested..."). _looks_like_personal_link requires
    either an unambiguous span (scheme/www./known host) or a link word
    (LinkedIn/GitHub/portfolio/...) immediately before the match.
  - other_identifier (NZ IRD number, driver's licence, postcode line): no
    Presidio built-in for these NZ-specific formats, so three custom
    PatternRecognizers cover them (see _build_analyzer).
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

The heavy AnalyzerEngine/spaCy model load only happens once this module is
actually imported (see the lazy import in pii_detector.py's
PII_DETECTOR_FACTORIES).
"""

import re
from typing import Dict, Final, List, Optional

from src.schemas.pii import TextSpan
from src.services.document_parser import CandidateCV, normalise
from src.services.pii_detector import (
    CONTACT_OR_ID_PATTERN,
    PIIDetector,
    is_valid_pii_span,
    parse_possible_dob,
)

# Presidio entity types whose own recognizer already validates shape
# (regex/checksum-based, not spaCy NER) well enough that the shared
# CONTACT_OR_ID_PATTERN guard shouldn't second-guess them. That guard was
# tuned for LLM output, which typically copies the label along with the
# value (e.g. "Mobile: 021...") — Presidio isolates just the value, so a
# bare phone number never matches CONTACT_OR_ID_PATTERN's keyword branch
# and would otherwise be rejected despite being a correct, well-typed hit.
# NZ_IRD_NUMBER/NZ_DRIVERS_LICENCE are the same kind of precise, fully
# deterministic regex match (ported from RegexPIIDetector's own patterns —
# see pii_detector.py) rather than a NER guess, so they get the same
# treatment as EMAIL_ADDRESS/PHONE_NUMBER, not URL's extra context guard.
PRE_VALIDATED_ENTITY_TYPES: Final = {
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "URL",
    "NZ_IRD_NUMBER",
    "NZ_DRIVERS_LICENCE",
}

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
    "NZ_IRD_NUMBER": "other_identifier",
    "NZ_DRIVERS_LICENCE": "other_identifier",
    # A city-plus-postcode line is address-identifying rather than a
    # contact/ID value, so it goes through the same street_address branch
    # (and _looks_like_street_address guard) as LOCATION — trivially
    # satisfied, since the postcode digits are part of the match itself.
    "NZ_POSTCODE_LINE": "street_address",
}

ANALYZE_ENTITIES: Final = list(ENTITY_TO_KIND)

REFEREE_HEADING: Final = re.compile(r"^\s*(?:referees?|references?)\b.*$", re.I | re.M)

# Headings that mark the CV body (education/experience/skills/projects)
# rather than the header/contact block. A candidate's own name is written
# once, at the very top of the document, before any of these; PERSON hits
# found after this point are overwhelmingly section content — job titles
# ("Truck Driver"), tech/product names ("Cloud Storage", "Sharing App") —
# that en_core_web_sm's small NER model mistakes for a name on this kind of
# fragmented, bullet-heavy text (see the module docstring). Anchored to
# start-of-line so a sub-bullet like "Key Skills:" doesn't false-trigger.
CONTENT_SECTION_HEADING: Final = re.compile(
    r"^\s*(?:education|experience|employment(?: history)?|work history|"
    r"skills?|technical skills|key projects|projects?|portfolio)\b.*$",
    re.I | re.M,
)

# Best-effort keyword recognizers for the two PIIKind categories that have
# no standard NER equivalent. These are intentionally simple/regex-based —
# tune against real CVs (run tasks/cv_ingest.yaml against varied documents
# and inspect pii_spans/redacted_cv in the resulting artifact) rather than
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

# NZ-specific ID formats Presidio's own built-in recognizers don't attempt
# (see the module docstring) — the two formats these patterns cover are
# what makes "presidio" self-sufficient across every PIIKind on its own.
NZ_IRD_NUMBER_REGEX: Final = r"\b\d{2,3}-\d{3}-\d{3}\b"
NZ_DRIVERS_LICENCE_REGEX: Final = r"\b[A-Z]{2}\d{6}\b"
NZ_POSTCODE_LINE_REGEX: Final = (
    r"(?i)\b(?:Auckland|Wellington|Christchurch|Hamilton|Dunedin|Tauranga|"
    r"Napier|Nelson)\s+\d{4}\b"
)


def _referee_section_start(text: str) -> Optional[int]:
    match = REFEREE_HEADING.search(text)
    return match.end() if match else None


def _content_section_start(text: str) -> Optional[int]:
    match = CONTENT_SECTION_HEADING.search(text)
    return match.end() if match else None


def _looks_like_street_address(text: str) -> bool:
    """LOCATION is only treated as street_address when it has a house/unit
    number in it; a bare city or country name is not a specific address."""
    return bool(re.search(r"\d", text))


# Presidio's URL recognizer's "Non schema URL" pattern matches ANY bare
# word.tld against a list of ~700 TLDs (including .io, .co, .dev, ...) with
# no scheme, no "www.", nothing — scoring 0.5, above our default threshold
# regardless of context. That correctly catches a bare personal link like
# "sonnytapara.dev", but identically catches a third-party tool mentioned
# in prose ("Cortex.io: Ingested and extracted data..."). URL is in
# PRE_VALIDATED_ENTITY_TYPES below (skips the shared CONTACT_OR_ID_PATTERN
# guard, since that guard expects a label alongside the value the way LLM
# output is written, and Presidio isolates just the value) — so a
# schemeless match gets no secondary check at all otherwise.
PERSONAL_LINK_CONTEXT_PATTERN: Final = re.compile(
    r"(?i)\b(?:linkedin|github|portfolio|website|profile|blog)\b"
)
PERSONAL_LINK_CONTEXT_WINDOW: Final = 40


def _looks_like_personal_link(text: str, start: int, span_text: str) -> bool:
    """Accept a URL match either because it's unambiguously a link (has a
    scheme, "www.", or a known profile host — see CONTACT_OR_ID_PATTERN),
    or because a link-indicating word sits immediately before it in the
    source text. A bare domain named in prose with no such context is
    rejected rather than assumed to be a link the candidate is sharing."""
    if CONTACT_OR_ID_PATTERN.search(span_text):
        return True
    window_start = max(0, start - PERSONAL_LINK_CONTEXT_WINDOW)
    return bool(PERSONAL_LINK_CONTEXT_PATTERN.search(text[window_start:start]))


# spaCy's PERSON recognizer on fragmented bullet-list CV text sometimes
# glues a leading bullet marker, or an entire adjacent bullet line, onto a
# PERSON span (e.g. "• Git / GitHub\n• CI" as one "name") — see the module
# docstring. Redacting that span then corrupts real content around it
# (splits "CI/CD" into "[PERSON_NAME]/CD", eats the bullet off "Skill
# set:"). No real person's name contains a bullet, a newline, or a slash,
# so this is a cheap, safe shape guard — unlike score_threshold, which
# can't help here: Presidio's spaCy-backed recognizer assigns PERSON hits a
# fixed confidence rather than grading each instance, so a malformed span
# scores the same as a clean one.
PERSON_NAME_REJECT_PATTERN: Final = re.compile(r"[•\n/]")


def _looks_like_person_name(text: str) -> bool:
    return not PERSON_NAME_REJECT_PATTERN.search(text)


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
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="NZ_IRD_NUMBER",
            patterns=[Pattern(name="nz_ird_number", regex=NZ_IRD_NUMBER_REGEX, score=0.6)],
            supported_language="en",
        )
    )
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="NZ_DRIVERS_LICENCE",
            patterns=[Pattern(name="nz_drivers_licence", regex=NZ_DRIVERS_LICENCE_REGEX, score=0.6)],
            supported_language="en",
        )
    )
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="NZ_POSTCODE_LINE",
            patterns=[Pattern(name="nz_postcode_line", regex=NZ_POSTCODE_LINE_REGEX, score=0.6)],
            supported_language="en",
        )
    )
    return analyzer


class PresidioPIIDetector(PIIDetector):
    def __init__(self, score_threshold: float = DEFAULT_SCORE_THRESHOLD, model_name: str = DEFAULT_MODEL_NAME):
        self.score_threshold = score_threshold
        self.model_name = model_name
        self._analyzer = _build_analyzer(model_name)
        self._rejections: List[Dict[str, str]] = []

    @property
    def rejections(self) -> List[Dict[str, str]]:
        return self._rejections

    @property
    def engine_name(self) -> str:
        return f"presidio:{self.model_name}"

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
        content_start = _content_section_start(text)

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

            if result.entity_type == "PERSON" and not _looks_like_person_name(span_text):
                self._rejections.append(
                    {"reason": "not_name_shaped", "kind": "person_name", "text": span_text}
                )
                continue

            if result.entity_type == "URL" and not _looks_like_personal_link(
                text, result.start, span_text
            ):
                self._rejections.append(
                    {"reason": "not_a_personal_link", "kind": "other_identifier", "text": span_text}
                )
                continue

            kind = ENTITY_TO_KIND[result.entity_type]
            if kind == "person_name" and referee_start is not None and result.start >= referee_start:
                kind = "referee"
            elif kind == "person_name" and content_start is not None and result.start >= content_start:
                self._rejections.append(
                    {"reason": "outside_header_block", "kind": kind, "text": span_text}
                )
                continue

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

            # Same grounding + heuristic guards defined in pii_detector.py
            # (is_valid_pii_span, CONTACT_OR_ID_PATTERN) — shared rather
            # than reimplemented here.
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
