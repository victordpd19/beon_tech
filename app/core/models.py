from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    id: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    score: float  # cosine similarity in [-1, 1]; higher is more relevant


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    sources: list[RetrievedChunk]
    model: str | None  # None when the LLM was not called (no relevant context)
