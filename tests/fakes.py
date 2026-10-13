from typing import Sequence

from app.core.interfaces import LLMError
from app.core.models import Document, LLMResult, RetrievedChunk


class FakeVectorStore:
    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self.chunks = chunks or []
        self.documents: dict[str, Document] = {}

    def upsert(self, documents: Sequence[Document]) -> None:
        self.documents.update({d.id: d for d in documents})

    def query(self, text: str, top_k: int) -> list[RetrievedChunk]:
        return self.chunks[:top_k]

    def count(self) -> int:
        return len(self.documents)


class FakeLLM:
    def __init__(self, reply: str = "fake answer", fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    @property
    def model(self) -> str:
        return "fake-model"

    def generate(self, system: str, user: str) -> LLMResult:
        self.calls.append((system, user))
        if self.fail:
            raise LLMError("boom")
        return LLMResult(text=self.reply, model=self.model)
