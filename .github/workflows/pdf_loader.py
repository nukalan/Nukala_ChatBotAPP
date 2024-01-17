# pdf_loader.py
import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    print("Extracted PDF Text:", text)
    return text
