import re
import io
import contextlib
from typing import Sequence, List
from src.schemas.pii import TextSpan


class PDFTextExtractionError(RuntimeError):
    pass


def _clean_pdf_text(text: str) -> str:
    text = (text or "").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _extract_with_pypdf(path: str) -> str:
    from pypdf import PdfReader

    with contextlib.redirect_stderr(io.StringIO()):
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_with_pymupdf(path: str) -> str:
    import fitz

    fitz.TOOLS.mupdf_display_errors(False)
    fitz.TOOLS.mupdf_display_warnings(False)
    with contextlib.redirect_stderr(io.StringIO()):
        with fitz.open(path) as document:
            return "\n".join(page.get_text() for page in document)


def _extract_with_ocr(path: str) -> str:
    import fitz
    import pytesseract
    from PIL import Image

    fitz.TOOLS.mupdf_display_errors(False)
    fitz.TOOLS.mupdf_display_warnings(False)
    text_pages = []
    with contextlib.redirect_stderr(io.StringIO()):
        with fitz.open(path) as document:
            for page in document:
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                image = Image.open(io.BytesIO(pixmap.tobytes("png")))
                text_pages.append(pytesseract.image_to_string(image))
    return "\n".join(text_pages)


def normalise(text: str) -> str:
    return re.sub(r"[\s\u00a0]+", " ", text or "").strip().lower()


class SourceDocument:
    def __init__(self, text: str):
        self.text = text
        self._normalised = normalise(text)

    @classmethod
    def from_pdf(cls, path: str) -> "SourceDocument":
        failures = []
        extractors = (
            ("pypdf", _extract_with_pypdf),
            ("pymupdf", _extract_with_pymupdf),
            ("ocr", _extract_with_ocr),
        )
        for name, extractor in extractors:
            try:
                raw = extractor(path)
            except Exception as exc:
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
                continue

            cleaned = _clean_pdf_text(raw)
            if normalise(cleaned):
                return cls(cleaned)
            failures.append(f"{name}: extracted no text")

        detail = "; ".join(failures)
        raise PDFTextExtractionError(
            f"Could not extract readable text from PDF '{path}'. {detail}. "
            "If this is a scanned or image-only PDF, install Tesseract OCR and make "
            "sure tesseract.exe is available on PATH, or re-export the PDF with selectable text."
        )

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