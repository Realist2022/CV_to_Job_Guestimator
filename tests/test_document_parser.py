import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

from src.services.document_parser import (
    PDFTextExtractionError,
    SourceDocument,
    _clean_pdf_text,
    assess_text_quality,
)


class PdfTextCleanupTest(unittest.TestCase):
    def test_clean_pdf_text_preserves_structure_and_fixes_common_artifacts(self):
        raw = " Skills\u00a0\u00a0\r\nexperi-\nence\r\n\r\n  2 of 4  \r\n\r\n\r\n- Python\t\tSQL "

        cleaned = _clean_pdf_text(raw)

        self.assertEqual(cleaned, "Skills\nexperience\n\n- Python SQL")

    def test_assess_text_quality_flags_unreadable_extraction_noise(self):
        readable = assess_text_quality("Requirements\nPython, SQL, and REST APIs")
        noisy = assess_text_quality("@@@@ @@@@ $$$$ %%%%")

        self.assertTrue(readable.is_likely_readable)
        self.assertFalse(noisy.is_likely_readable)

    def test_from_text_file_cleans_and_loads_readable_text(self):
        with NamedTemporaryFile("wb", suffix=".txt", delete=False) as handle:
            handle.write("Requirements\r\nReact\u00a0developer".encode("utf-8"))
            path = handle.name

        try:
            document = SourceDocument.from_text_file(path)
        finally:
            Path(path).unlink(missing_ok=True)

        self.assertEqual(document.text, "Requirements\nReact developer")

    def test_from_path_routes_txt_without_pdf_extractors(self):
        with NamedTemporaryFile("wb", suffix=".txt", delete=False) as handle:
            handle.write(b"Requirements\nPython")
            path = handle.name

        try:
            with patch("src.services.document_parser._extract_with_pypdf") as pypdf:
                document = SourceDocument.from_path(path)
        finally:
            Path(path).unlink(missing_ok=True)

        self.assertEqual(document.text, "Requirements\nPython")
        pypdf.assert_not_called()


class SourceDocumentPdfExtractionTest(unittest.TestCase):
    def test_from_pdf_falls_back_when_primary_extractor_returns_empty_text(self):
        with patch(
            "src.services.document_parser._extract_with_pypdf", return_value="\n\n"
        ), patch(
            "src.services.document_parser._extract_with_pymupdf",
            return_value="Requirements\nPython and REST APIs",
        ), patch(
            "src.services.document_parser._extract_with_ocr"
        ) as ocr:
            document = SourceDocument.from_pdf("job.pdf")

        self.assertEqual(document.text, "Requirements\nPython and REST APIs")
        ocr.assert_not_called()

    def test_from_pdf_raises_actionable_error_when_no_backend_extracts_text(self):
        with patch(
            "src.services.document_parser._extract_with_pypdf", return_value=""
        ), patch(
            "src.services.document_parser._extract_with_pymupdf", return_value=""
        ), patch(
            "src.services.document_parser._extract_with_ocr",
            side_effect=RuntimeError("tesseract unavailable"),
        ):
            with self.assertRaisesRegex(
                PDFTextExtractionError,
                "Could not extract readable text.*Tesseract OCR.*selectable text",
            ):
                SourceDocument.from_pdf("job.pdf")


if __name__ == "__main__":
    unittest.main()