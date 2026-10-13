from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_rag_service, get_vector_store
from app.api.schemas import AskRequest, AskResponse, HealthResponse
from app.core.interfaces import LLMError, VectorStore
from app.rag.service import RAGService

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, service: Annotated[RAGService, Depends(get_rag_service)]) -> AskResponse:
    try:
        answer = service.answer(req.question, top_k=req.top_k)
    except LLMError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return AskResponse.from_domain(answer)


@router.get("/health", response_model=HealthResponse)
def health(store: Annotated[VectorStore, Depends(get_vector_store)]) -> HealthResponse:
    return HealthResponse(status="ok", documents=store.count())
