import re
from abc import ABC, abstractmethod
from typing import List, Dict, Final
from src.schemas.pii import TextSpan
from src.services.document_parser import CandidateCV, normalise
from src.services.agents import PIIAgent

# Keywords that should never be tagged under marital_or_family
FORBIDDEN_REDACTION_KEYWORDS: Final = [
    "DIPLOMA", "DEGREE", "UNIVERSITY", "LIMITED", "LTD", 
    "CERTIFICATE", "COLLEGE", "INSTITUTE"
]


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
        # Checks for hyphens, en-dashes, em-dashes, or standalone "TO"
        if re.search(r'[-–—]|(\bTO\b)', clean_text):
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
        ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
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