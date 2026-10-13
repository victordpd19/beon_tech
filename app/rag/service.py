from app.core.interfaces import LLMClient, VectorStore
from app.core.models import Answer
from app.rag.prompt import NO_ANSWER, PromptBuilder


class RAGService:
    def __init__(
        self,
        store: VectorStore,
        llm: LLMClient,
        prompt_builder: PromptBuilder,
        top_k: int,
        min_similarity: float,
    ) -> None:
        self._store = store
        self._llm = llm
        self._prompt_builder = prompt_builder
        self._top_k = top_k
        self._min_similarity = min_similarity

    def answer(self, question: str, top_k: int | None = None) -> Answer:
        chunks = self._store.query(question, top_k or self._top_k)
        relevant = [c for c in chunks if c.score >= self._min_similarity]

        # Nothing relevant retrieved: skip the LLM call rather than invite a hallucination.
        if not relevant:
            return Answer(question=question, text=NO_ANSWER, sources=[], model=None)

        result = self._llm.generate(
            system=self._prompt_builder.system_prompt,
            user=self._prompt_builder.build_user_message(question, relevant),
        )
        return Answer(question=question, text=result.text, sources=relevant, model=result.model)
