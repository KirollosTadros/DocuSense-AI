from __future__ import annotations
import chromadb
from typing import List
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__ (self, collection_name: str="pdf_collection", db_path: str="./doc_db"):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name)

    def reset_collection(self):
            """Deletes the current collection and recreates an empty one."""
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_chunks(self, chunks: List[str], embeddings):

        ids = [f"chunk_{i}" for i in range(len(chunks))]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist())
        print(f"Added {len(chunks)} chunks to DB")

    def search(self, query: str, embedding_model: SentenceTransformer, k: int=3) -> List[str]:
        query = embedding_model.encode(query, normalize_embeddings=True).tolist()
        results = self.collection.query(query_embeddings=[query], n_results=k)

        return results["documents"][0]