import unittest
from unittest.mock import patch

from src.services.document_parser import PDFTextExtractionError, SourceDocument


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