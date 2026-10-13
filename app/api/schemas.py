from pydantic import BaseModel, ConfigDict, Field

from app.core.models import Answer


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=1000, examples=["What services does BEON.tech offer?"])
    top_k: int | None = Field(default=None, ge=1, le=10, description="Override the number of retrieved chunks.")


class SourceChunk(BaseModel):
    id: str
    text: str
    score: float = Field(description="Cosine similarity between question and chunk (higher is more relevant).")


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
    model: str | None = Field(description="LLM that produced the answer; null when no relevant context was found.")

    @classmethod
    def from_domain(cls, answer: Answer) -> "AskResponse":
        return cls(
            question=answer.question,
            answer=answer.text,
            sources=[SourceChunk(id=s.id, text=s.text, score=s.score) for s in answer.sources],
            model=answer.model,
        )


class HealthResponse(BaseModel):
    status: str
    documents: int
