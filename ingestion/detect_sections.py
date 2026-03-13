import os
import re
import json
from tqdm import tqdm


CLEAN_FOLDER = "../data/cleaned"
OUTPUT_FILE = "../data/processed/laws_dataset.json"


def extract_sections(text):

    """
    Detect different section formats used in Nigerian Acts
    """

    pattern = r'(Section\s+\d+[A-Za-z]*|\n\d+\.\s|\n\d+\s|S\.\d+)'

    parts = re.split(pattern, text)

    sections = []

    for i in range(1, len(parts), 2):

        section_number = parts[i].strip()
        section_body = parts[i + 1].strip()

        sections.append({
            "section": section_number,
            "text": section_body
        })

    return sections


def process_all_files():

    os.makedirs("../data/processed", exist_ok=True)

    dataset = []

    files = [f for f in os.listdir(CLEAN_FOLDER) if f.endswith(".txt")]

    for file in tqdm(files):

        filepath = os.path.join(CLEAN_FOLDER, file)

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        sections = extract_sections(text)

        act_name = file.replace(".txt", "")

        for s in sections:

            dataset.append({
                "act": act_name,
                "section": s["section"],
                "text": s["text"]
            })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)


if __name__ == "__main__":
    process_all_files()