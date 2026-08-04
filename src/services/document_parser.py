import fitz  # PyMuPDF
import logging

logger = logging.getLogger(__name__)

class DocumentParser:
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """
        Extracts all text from a given PDF file using PyMuPDF.
        Assumes the PDF is cleanly encoded.
        """
        try:
            doc = fitz.open(pdf_path)
            extracted_text = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                if text:
                    extracted_text.append(text)
                
            doc.close()
            final_text = "\n".join(extracted_text).strip()
            
            if not final_text:
                logger.warning(f"Warning: Extracted text from {pdf_path} is empty. The PDF may be corrupted or image-based.")
                
            return final_text
            
        except Exception as e:
            raise IOError(f"Failed to read PDF at {pdf_path}: {e}")