from __future__ import annotations
from typing import List
from sentence_transformers import SentenceTransformer


class textEmbedding:

    def __init__(self, model_name: str="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def wordChunker(self, text: str, chunk_size: int= 300, overlap: int= 50, ) -> List[str]:

        words = text.split()
        if not words:
            return []
        chunks = []
        step = chunk_size - overlap

        for i in range(0, len(words), step):
            chunk_words = words[i: i + chunk_size]
            chunks.append(" ".join(chunk_words))
        return chunks

    def generateEmbeddings(self, rawText: str):
        chunks = self.wordChunker(rawText)

        embeddings = self.model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
        return chunks, embeddings