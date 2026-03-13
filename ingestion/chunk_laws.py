import json
import os
import re


INPUT_FILE = "../data/processed/laws_dataset.json"
OUTPUT_FILE = "../data/processed/laws_chunks.json"


MAX_CHUNK_SIZE = 1200


def clean_text(text):

    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def split_large_text(text):

    if len(text) <= MAX_CHUNK_SIZE:
        return [text]

    chunks = []

    sentences = re.split(r'(?<=[.!?]) +', text)

    current_chunk = ""

    for sentence in sentences:

        if len(current_chunk) + len(sentence) < MAX_CHUNK_SIZE:
            current_chunk += " " + sentence

        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def process_dataset():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    processed_chunks = []

    for entry in dataset:

        act = entry["act"]
        section = entry["section"]

        text = clean_text(entry["text"])

        text_chunks = split_large_text(text)

        for i, chunk in enumerate(text_chunks):

            processed_chunks.append({
                "act": act,
                "section": section,
                "chunk_id": i,
                "text": chunk
            })

    os.makedirs("../data/processed", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(processed_chunks, f, indent=2)


if __name__ == "__main__":
    process_dataset()