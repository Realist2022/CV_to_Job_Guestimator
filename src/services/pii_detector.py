import re
from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Callable, Dict, Final, List, Optional

from src.schemas.artifact import PIIRunConfig
from src.schemas.pii import TextSpan
from src.services.document_parser import CandidateCV, normalise

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

    @property
    def engine_name(self) -> str:
        """Identifies which detector (and, where relevant, which underlying
        model) actually produced a run's redaction — recorded verbatim as
        pii_engine/pii_model.engine in RedactedCV/IngestionResult/RunConfig
        artifacts. Defaults to the class name; PresidioPIIDetector overrides
        this with its actual spaCy model name."""
        return type(self).__name__


class CompositePIIDetector(PIIDetector):
    def __init__(self, *detectors: PIIDetector):
        self.detectors = detectors

    @property
    def rejections(self) -> List[Dict[str, str]]:
        return [r for d in self.detectors for r in d.rejections]

    @property
    def engine_name(self) -> str:
        return "+".join(d.engine_name for d in self.detectors)

    def detect(self, cv: CandidateCV) -> List[TextSpan]:
        spans, seen = [], set()
        for detector in self.detectors:
            for span in detector.detect(cv):
                key = normalise(span.text)
                if key and key not in seen:
                    seen.add(key)
                    spans.append(span)
        return spans


def _build_presidio_detector() -> PIIDetector:
    # Imported lazily so importing this module never pulls in spacy/presidio
    # (and their model-load cost) unless something actually asks for a PII
    # detector.
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
PII_DETECTOR_FACTORIES: Dict[str, Callable[[], PIIDetector]] = {
    "presidio": _build_presidio_detector,
}


def build_pii_detector(names: List[str]) -> CompositePIIDetector:
    """Compose a CompositePIIDetector from configs/pii_policy.yaml detector names."""
    try:
        detectors = [PII_DETECTOR_FACTORIES[name]() for name in names]
    except KeyError as exc:
        known = ", ".join(sorted(PII_DETECTOR_FACTORIES))
        raise KeyError(f"Unknown PII detector '{exc.args[0]}'. Known: {known}") from None
    return CompositePIIDetector(*detectors)


def pii_run_model_config(engine_name: str, *, ran_this_run: bool = True) -> PIIRunConfig:
    """PIIRunConfig for the PII role, given an already-run result's engine
    name (e.g. IngestionResult.pii_engine/PipelineResult.pii_engine).

    No LLM client or configs/llm.yaml key applies to PII redaction (see
    PIIDetector.engine_name) — `name` is just "presidio" and `engine` is
    whatever the detector actually reported. Callers derive `engine_name`
    from a result object rather than reading `detector.engine_name`
    directly, so this still works when the pipeline that produced the
    result isn't the exact object that built the detector (e.g. a test
    double standing in for IngestionPipeline/ExtractionPipeline).

    Pass ran_this_run=False on a "matching" run, where no detector executed
    and `engine_name` comes off a RedactedCV redacted by some earlier run.
    The default is True because that is the common case (extraction and
    ingestion both redact), and because a caller that forgets it on a
    matching run overstates what happened rather than understating it —
    so keep it explicit at those two call sites."""
    return PIIRunConfig(name="presidio", engine=engine_name, ran_this_run=ran_this_run)