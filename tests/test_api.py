import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router
from app.core.models import RetrievedChunk
from app.rag.prompt import PromptBuilder
from app.rag.service import RAGService
from tests.fakes import FakeLLM, FakeVectorStore


def make_client(llm: FakeLLM) -> TestClient:
    store = FakeVectorStore([RetrievedChunk(id="kb-services", text="Staff augmentation.", score=0.8)])
    app = FastAPI()
    app.include_router(router)
    app.state.vector_store = store
    app.state.rag_service = RAGService(store, llm, PromptBuilder(), top_k=3, min_similarity=0.25)
    return TestClient(app)


def test_ask_returns_structured_answer():
    res = make_client(FakeLLM(reply="We do staff augmentation.")).post(
        "/ask", json={"question": "  What do you offer?  "}
    )

    assert res.status_code == 200
    assert res.json() == {
        "question": "What do you offer?",
        "answer": "We do staff augmentation.",
        "sources": [{"id": "kb-services", "text": "Staff augmentation.", "score": 0.8}],
        "model": "fake-model",
    }


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "   "}, {}, {"question": "hi", "top_k": 0}])
def test_ask_rejects_invalid_input(payload):
    assert make_client(FakeLLM()).post("/ask", json=payload).status_code == 422


def test_ask_maps_llm_failure_to_502():
    assert make_client(FakeLLM(fail=True)).post("/ask", json={"question": "hi"}).status_code == 502
