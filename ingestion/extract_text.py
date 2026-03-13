import pdfplumber
import os
from tqdm import tqdm


RAW_FOLDER = "../data/raw"
OUTPUT_FOLDER = "../data/cleaned"


def extract_text_from_pdf(pdf_path):

    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


def process_all_pdfs():

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    pdf_files = [f for f in os.listdir(RAW_FOLDER) if f.endswith(".pdf")]

    for pdf_file in tqdm(pdf_files):

        pdf_path = os.path.join(RAW_FOLDER, pdf_file)

        text = extract_text_from_pdf(pdf_path)

        output_file = pdf_file.replace(".pdf", ".txt")
        output_path = os.path.join(OUTPUT_FOLDER, output_file)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    process_all_pdfs()