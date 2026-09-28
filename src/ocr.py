import pytesseract
import pdfplumber
from PIL import Image


class OCRError(Exception):
    pass


def _check_tesseract() -> None:
    try:
        pytesseract.get_tesseract_version()
    except Exception as error:
        raise OCRError(
            "Tesseract OCR is not installed or not on PATH. "
            "Install it (see README) and restart the app."
        ) from error


def run_ocr_on_image(file_path: str) -> str:
    _check_tesseract()
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
    except OCRError:
        raise
    except Exception as error:
        raise OCRError(f"OCR failed on this image: {error}") from error
    return text.strip()


def run_ocr_on_pdf(file_path: str, max_pages: int = 20) -> str:
    _check_tesseract()
    text_parts = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= max_pages:
                    break
                im = page.to_image(resolution=300).original
                page_text = pytesseract.image_to_string(im)
                if page_text.strip():
                    text_parts.append(page_text.strip())
    except OCRError:
        raise
    except Exception as error:
        raise OCRError(f"OCR failed on this PDF: {error}") from error
    return "\n".join(text_parts)
