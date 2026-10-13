"""Abstractions the RAG pipeline depends on (Dependency Inversion).

Concrete implementations (Chroma, Anthropic, a future Ollama client) live in
their own packages and are wired together in ``app.main``.
"""

from typing import Protocol, Sequence

from app.core.models import Document, LLMResult, RetrievedChunk


class VectorStore(Protocol):
    def upsert(self, documents: Sequence[Document]) -> None: ...

    def query(self, text: str, top_k: int) -> list[RetrievedChunk]: ...

    def count(self) -> int: ...


class LLMClient(Protocol):
    @property
    def model(self) -> str: ...

    def generate(self, system: str, user: str) -> LLMResult: ...


class LLMError(RuntimeError):
    """Raised by LLMClient implementations when the provider call fails."""
