import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle


INPUT_FILE = "../data/processed/laws_chunks.json"

FAISS_INDEX_FILE = "../data/embeddings/faiss_index.index"
METADATA_FILE = "../data/embeddings/metadata.pkl"


MODEL_NAME = "all-MiniLM-L6-v2"


def load_dataset():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_embeddings(dataset):

    model = SentenceTransformer(MODEL_NAME)

    texts = [entry["text"] for entry in dataset]

    embeddings = model.encode(texts, show_progress_bar=True)

    return embeddings


def build_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    return index


def save_index(index, metadata):

    faiss.write_index(index, FAISS_INDEX_FILE)

    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)


def main():

    print("Loading dataset...")

    dataset = load_dataset()

    print("Generating embeddings...")

    embeddings = build_embeddings(dataset)

    print("Building FAISS index...")

    index = build_faiss_index(embeddings)

    print("Saving index and metadata...")

    metadata = dataset

    save_index(index, metadata)

    print("Done!")


if __name__ == "__main__":
    main()