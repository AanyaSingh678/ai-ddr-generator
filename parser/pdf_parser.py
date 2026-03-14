import fitz  # PyMuPDF


def extract_text(pdf_path):
    """
    Extract all text from a PDF file
    """

    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        full_text += page.get_text()

    return full_text