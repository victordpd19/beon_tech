from fastapi import Request

from app.core.interfaces import VectorStore
from app.rag.service import RAGService


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store
