import pytesseract
import pdfplumber
from PIL import Image

def run_ocr_on_image(file_path: str) -> str:
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)
    return text.strip()

def run_ocr_on_pdf(file_path: str, max_pages: int = 20) -> str:
    text_parts = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            if i >= max_pages:
                break
            im = page.to_image(resolution=300).original
            page_text = pytesseract.image_to_string(im)
            if page_text.strip():
                text_parts.append(page_text.strip())
    return "\n".join(text_parts)