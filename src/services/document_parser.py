import re
import io
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from src.schemas.pii import TextSpan


class PDFTextExtractionError(RuntimeError):
    pass


@dataclass(frozen=True)
class TextQuality:
    char_count: int
    word_count: int
    line_count: int
    weird_char_ratio: float
    is_likely_readable: bool


def assess_text_quality(text: str) -> TextQuality:
    content = text or ""
    non_whitespace = re.findall(r"\S", content)
    words = re.findall(r"\b[\w+#.]+\b", content)
    weird_chars = re.findall(
        r"[^\w\s.,;:!?()&/@+#'\-\[\]{}|*\"\u2022\u2013\u2014]",
        content,
    )
    weird_char_ratio = len(weird_chars) / max(len(non_whitespace), 1)

    return TextQuality(
        char_count=len(content),
        word_count=len(words),
        line_count=len(content.splitlines()),
        weird_char_ratio=weird_char_ratio,
        is_likely_readable=bool(words) and weird_char_ratio < 0.25,
    )


def _clean_pdf_text(text: str) -> str:
    text = (text or "").replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"(?im)^\s*(?:page\s*)?\d+\s*(?:of\s*\d+)?\s*$", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _decode_text_bytes(content: bytes, label: str) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"The {label} text could not be decoded as UTF-8 or cp1252.")


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
    def from_text(cls, text: str, *, label: str = "document") -> "SourceDocument":
        cleaned = _clean_pdf_text(text)
        quality = assess_text_quality(cleaned)
        if quality.is_likely_readable:
            return cls(cleaned)
        if quality.word_count == 0:
            raise ValueError(f"The {label} did not contain readable text.")
        raise ValueError(
            f"The {label} text looks unreadable "
            f"({quality.word_count} words, {quality.weird_char_ratio:.1%} unusual characters)."
        )

    @classmethod
    def from_text_bytes(cls, content: bytes, *, label: str = "document") -> "SourceDocument":
        return cls.from_text(_decode_text_bytes(content, label), label=label)

    @classmethod
    def from_text_file(cls, path: str) -> "SourceDocument":
        file_path = Path(path)
        return cls.from_text_bytes(file_path.read_bytes(), label=str(file_path))

    @classmethod
    def from_path(cls, path: str) -> "SourceDocument":
        suffix = Path(path).suffix.lower()
        if suffix == ".txt":
            return cls.from_text_file(path)
        if suffix == ".pdf":
            return cls.from_pdf(path)
        raise ValueError(f"Document must be a PDF or TXT file: '{path}'")

    @classmethod
    def from_pdf(cls, path: str, *, cache_text: bool = False) -> "SourceDocument":
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
            quality = assess_text_quality(cleaned)
            if quality.is_likely_readable:
                if cache_text:
                    Path(path).with_suffix(".txt").write_text(cleaned, encoding="utf-8")
                return cls(cleaned)
            if quality.word_count == 0:
                failures.append(f"{name}: extracted no text")
                continue
            failures.append(
                f"{name}: extracted unreadable text "
                f"({quality.word_count} words, {quality.weird_char_ratio:.1%} unusual characters)"
            )

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