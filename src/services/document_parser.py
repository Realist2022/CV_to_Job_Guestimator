import re
from typing import Sequence, List
from src.schemas.pii import TextSpan


def normalise(text: str) -> str:
    return re.sub(r"[\s\u00a0]+", " ", text or "").strip().lower()


class SourceDocument:
    def __init__(self, text: str):
        self.text = text
        self._normalised = normalise(text)

    @classmethod
    def from_pdf(cls, path: str) -> "SourceDocument":
        from pypdf import PdfReader

        reader = PdfReader(path)
        raw = "\n".join(page.extract_text() or "" for page in reader.pages)
        raw = raw.replace("\u00a0", " ")
        raw = re.sub(r"[ \t]+", " ", raw)
        return cls(re.sub(r"\n{3,}", "\n\n", raw).strip())

    def contains(self, snippet: str) -> bool:
        needle = normalise(snippet)
        return bool(needle) and needle in self._normalised

    def __len__(self) -> int:
        return len(self.text)


class JobListing(SourceDocument):
    REQUIREMENTS_HEADING = re.compile(
        r"^\s*(?:required|requirements|qualifications|key requirements|about you|"
        r"what you.ll need|skills? (?:and|&) experience|required qualifications.*|"
        r"essential)\b.*$", re.I | re.M,
    )
    STOP_HEADING = re.compile(
        r"^\s*(?:how to apply|apply now|benefits|what we offer|"
        r"about (?:us|the school|the company)|remuneration|salary)\b.*$", re.I | re.M,
    )

    @property
    def requirements_section(self) -> str:
        start = self.REQUIREMENTS_HEADING.search(self.text)
        if not start:
            return self.text
        stop = self.STOP_HEADING.search(self.text, start.end())
        section = self.text[start.start():stop.start() if stop else len(self.text)]
        return section if len(section) > 120 else self.text


class CandidateCV(SourceDocument):
    REDACTION_TOKEN = "[{kind}]"

    def redacted(self, spans: Sequence[TextSpan]) -> "CandidateCV":
        result = self.text
        for span in sorted(spans, key=lambda s: len(s.text), reverse=True):
            if span.text.strip():
                result = re.sub(
                    re.escape(span.text),
                    self.REDACTION_TOKEN.format(kind=span.kind.upper()),
                    result, flags=re.IGNORECASE,
                )
        return CandidateCV(result)

    def residual_fragments(self, spans: Sequence[TextSpan]) -> List[str]:
        leaks = {
            token for span in spans for token in span.text.split()
            if len(token) > 3 and token.lower() in self._normalised
        }
        return sorted(leaks)