from pathlib import Path
from typing import Sequence

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from app.core.models import Document, RetrievedChunk


class ChromaVectorStore:
    """Local persistent vector store using Chroma's bundled ONNX MiniLM embeddings."""

    def __init__(self, path: Path, collection: str) -> None:
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=collection,
            embedding_function=DefaultEmbeddingFunction(),
            configuration={"hnsw": {"space": "cosine"}},
        )

    def upsert(self, documents: Sequence[Document]) -> None:
        if not documents:
            return
        self._collection.upsert(
            ids=[d.id for d in documents],
            documents=[d.text for d in documents],
        )

    def query(self, text: str, top_k: int) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_texts=[text],
            n_results=min(top_k, max(self.count(), 1)),
            include=["documents", "distances"],
        )
        ids = result["ids"][0]
        docs = result["documents"][0]
        distances = result["distances"][0]
        # Cosine distance -> similarity, so callers can reason in "higher is better".
        return [
            RetrievedChunk(id=i, text=d, score=round(1.0 - dist, 4))
            for i, d, dist in zip(ids, docs, distances)
        ]

    def count(self) -> int:
        return self._collection.count()
