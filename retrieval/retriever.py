import faiss
import pickle
import numpy as np
import os

from sentence_transformers import SentenceTransformer, CrossEncoder


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAISS_INDEX_FILE = os.path.join(BASE_DIR, "data", "embeddings", "faiss_index.index")
METADATA_FILE = os.path.join(BASE_DIR, "data", "embeddings", "metadata.pkl")

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class LegalRetriever:

    def __init__(self):

        print("Loading embedding model...")
        self.embedder = SentenceTransformer(EMBED_MODEL)

        print("Loading reranker...")
        self.reranker = CrossEncoder(RERANK_MODEL)

        print("Loading FAISS index...")
        self.index = faiss.read_index(FAISS_INDEX_FILE)

        print("Loading metadata...")
        with open(METADATA_FILE, "rb") as f:
            self.metadata = pickle.load(f)


    def search(self, query, top_k=3):

        # Step 1: embed question
        query_vector = self.embedder.encode([query])

        # Step 2: retrieve candidates
        distances, indices = self.index.search(np.array(query_vector), 10)

        candidates = [self.metadata[i] for i in indices[0] if i < len(self.metadata)]

        # Step 3: rerank
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )

        results = [r[0] for r in ranked[:top_k]]

        return results