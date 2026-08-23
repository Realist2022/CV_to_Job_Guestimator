import re
from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Callable, Dict, Final, List, Optional

from src.schemas.pii import TextSpan
from src.services.agents import PIIAgent
from src.services.document_parser import CandidateCV, normalise
from src.services.llm_client import CompletionClient

# Keywords that should never be tagged under marital_or_family
FORBIDDEN_REDACTION_KEYWORDS: Final = [
    "DIPLOMA", "DEGREE", "UNIVERSITY", "LIMITED", "LTD", 
    "CERTIFICATE", "COLLEGE", "INSTITUTE"
]

EMAIL_PATTERN: Final = re.compile(
    r"(?<![a-zA-Z0-9!#$%&'*+/=?^_`{|}~.-])"
    r"[a-zA-Z0-9!#$%&'*+/=?^_`{|}~-]+"
    r"(?:\.[a-zA-Z0-9!#$%&'*+/=?^_`{|}~-]+)*"
    r"@"
    r"(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+"
    r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])"
    r"\b"
)

CONTACT_OR_ID_PATTERN: Final = re.compile(
    rf"(?:"
    rf"{EMAIL_PATTERN.pattern}"
    r"|\bhttps?://[^\s,;]+|\b(?:www\.|linkedin\.com/|github\.com/)[^\s,;]+"
    r"|\b(?:email|phone|mobile|linkedin|github|driver'?s? licen[cs]e|ird|passport|facebook|twitter|instagram)\b"
    r"|\b\d{2,3}-\d{3}-\d{3}\b"
    r"|\b[A-Z]{2}\d{6}\b"
    r")",
    re.I,
)

RANGE_INDICATORS: Final = re.compile(
    r"\b(?:TO|THRU|THROUGH|UNTIL|TIL|PRESENT|CURRENT|NOW)\b",
    re.I,
)

DOB_DATE_FORMATS: Final = [
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d %B %Y",
    "%B %d, %Y",
    "%d %b %Y",
    "%b %d, %Y",
]


def is_date_range(text: str) -> bool:
    """Return True for likely employment/education ranges, not single dates."""
    clean_text = text.strip().upper()

    if re.search(r"\s+[-–—]\s+", clean_text):
        return True

    if RANGE_INDICATORS.search(clean_text):
        return True

    years = re.findall(r"\b(?:19|20)\d{2}\b", clean_text)
    return len(years) >= 2


def parse_possible_dob(text: str) -> Optional[date]:
    clean_text = text.strip()
    for date_format in DOB_DATE_FORMATS:
        try:
            return datetime.strptime(clean_text, date_format).date()
        except ValueError:
            continue
    return None


def is_valid_dob_span(span_text: str) -> bool:
    if is_date_range(span_text):
        return False

    dob = parse_possible_dob(span_text)
    if dob is None:
        return True

    today = date.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return 0 <= age <= 120


def is_valid_pii_span(span_text: str, kind: str) -> bool:
    """
    Validates whether an extracted PII span from the model is a legitimate match
    or a known false positive.
    """
    clean_text = span_text.strip().upper()

    # 1. Protect educational / organizational keywords from marital_or_family misclassification
    if kind == "marital_or_family":
        if any(kw in clean_text for kw in FORBIDDEN_REDACTION_KEYWORDS):
            return False

    # 2. Protect date ranges (employment/education durations) from being tagged as date_of_birth
    if kind == "date_of_birth":
        if not is_valid_dob_span(span_text):
            return False

    # 3. Only redact other_identifier when it is actually contact or ID material.
    if kind == "other_identifier" and not CONTACT_OR_ID_PATTERN.search(span_text):
        return False

    return True


class PIIDetector(ABC):
    @abstractmethod
    def detect(self, cv: CandidateCV) -> List[TextSpan]:
        ...

    @property
    def rejections(self) -> List[Dict[str, str]]:
        return []


class RegexPIIDetector(PIIDetector):
    PATTERNS = [
        ("email", EMAIL_PATTERN),
        ("phone", re.compile(r"(?:\+?64[\s-]?|\b0)(?:2\d{1,2}|[3-9])[\s-]?\d{3}[\s-]?\d{3,4}\b")),
        ("ird_number", re.compile(r"\b\d{2,3}-\d{3}-\d{3}\b")),
        ("nz_drivers_licence", re.compile(r"\b[A-Z]{2}\d{6}\b")),
        ("url", re.compile(r"\bhttps?://[^\s,;]+|\b(?:www\.|linkedin\.com/|github\.com/)[^\s,;]+")),
        ("postcode_line", re.compile(r"\b(?:Auckland|Wellington|Christchurch|Hamilton|Dunedin|Tauranga|Napier|Nelson)\s+\d{4}\b", re.I)),
    ]

    def detect(self, cv: CandidateCV) -> List[TextSpan]:
        spans, seen = [], set()
        for kind, pattern in self.PATTERNS:
            for match in pattern.finditer(cv.text):
                text = match.group(0).strip().rstrip(".,;")
                if text and text.lower() not in seen:
                    seen.add(text.lower())
                    spans.append(TextSpan(kind, text))
        return spans


class ModelPIIDetector(PIIDetector):
    def __init__(self, agent: PIIAgent):
        self.agent = agent
        self._rejections: List[Dict[str, str]] = []

    @property
    def rejections(self) -> List[Dict[str, str]]:
        return self._rejections

    def detect(self, cv: CandidateCV) -> List[TextSpan]:
        self._rejections = []
        result = self.agent.run(cv=cv)
        if result is None:
            raise RuntimeError("Local PII analysis failed; cloud evaluation was not started.")
        
        spans, seen = [], set()
        for span in result.spans:
            kind_val = span.kind.value
            key = normalise(span.text)
            
            if not key or key in seen:
                continue

            # 1. Document verification check
            if not cv.contains(span.text):
                self._rejections.append({
                    "reason": "not_in_document", 
                    "kind": kind_val, 
                    "text": span.text
                })
                continue

            # 2. Heuristic false-positive check (protects dates & qualifications)
            if not is_valid_pii_span(span.text, kind_val):
                self._rejections.append({
                    "reason": "validation_guard_rejected", 
                    "kind": kind_val, 
                    "text": span.text
                })
                continue

            seen.add(key)
            spans.append(TextSpan(kind_val, span.text.strip()))
            
        return spans


class CompositePIIDetector(PIIDetector):
    def __init__(self, *detectors: PIIDetector):
        self.detectors = detectors

    @property
    def rejections(self) -> List[Dict[str, str]]:
        return [r for d in self.detectors for r in d.rejections]

    def detect(self, cv: CandidateCV) -> List[TextSpan]:
        spans, seen = [], set()
        for detector in self.detectors:
            for span in detector.detect(cv):
                key = normalise(span.text)
                if key and key not in seen:
                    seen.add(key)
                    spans.append(span)
        return spans


def _build_presidio_detector(_pii_client: Optional[CompletionClient]) -> PIIDetector:
    # Imported lazily so importing this module never pulls in spacy/presidio
    # (and their model-load cost) unless "presidio" is actually enabled in
    # configs/pii_policy.yaml.
    from src.config import load_presidio_config
    from src.services.presidio_detector import DEFAULT_SCORE_THRESHOLD, PresidioPIIDetector

    config = load_presidio_config()
    return PresidioPIIDetector(score_threshold=config.get("score_threshold", DEFAULT_SCORE_THRESHOLD))


# Canonical name -> factory for each known detector. This is the single
# place that maps a configs/pii_policy.yaml detector name to a concrete
# implementation; both the harness's pluggable registry (see
# src/harness/runner.py) and ExtractionPipeline's own default (used when
# nothing calls the harness, e.g. the web API) build from this map so a
# name means the same thing everywhere instead of two hardcoded defaults
# drifting apart.
#
# "presidio" ignores the pii_client argument entirely: it's a local
# NER+regex detector (see presidio_detector.py) with no LLM in the loop, an
# alternative to "model" rather than a replacement for it. Composed with
# "regex" via configs/pii_policy.yaml or a task's `pii_detectors` override
# (see tasks/pii_presidio_eval.yaml) to A/B against the model-based run.
PII_DETECTOR_FACTORIES: Dict[str, Callable[[Optional[CompletionClient]], PIIDetector]] = {
    "regex": lambda _pii_client: RegexPIIDetector(),
    "model": lambda pii_client: ModelPIIDetector(PIIAgent(pii_client)),
    "presidio": _build_presidio_detector,
}


def build_pii_detector(
    names: List[str], pii_client: Optional[CompletionClient] = None
) -> CompositePIIDetector:
    """Compose a CompositePIIDetector from configs/pii_policy.yaml detector names."""
    try:
        detectors = [PII_DETECTOR_FACTORIES[name](pii_client) for name in names]
    except KeyError as exc:
        known = ", ".join(sorted(PII_DETECTOR_FACTORIES))
        raise KeyError(f"Unknown PII detector '{exc.args[0]}'. Known: {known}") from None
    return CompositePIIDetector(*detectors)