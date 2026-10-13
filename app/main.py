import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import Settings, get_settings
from app.llm.factory import build_llm
from app.rag.ingestion import ingest
from app.rag.prompt import PromptBuilder
from app.rag.service import RAGService
from app.vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger("uvicorn.error")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = ChromaVectorStore(path=settings.chroma_path, collection=settings.chroma_collection)
        ingested = ingest(settings.knowledge_base_path, store)
        logger.info("Knowledge base ingested: %d documents (%d in store)", ingested, store.count())

        app.state.vector_store = store
        app.state.rag_service = RAGService(
            store=store,
            llm=build_llm(settings),
            prompt_builder=PromptBuilder(),
            top_k=settings.top_k,
            min_similarity=settings.min_similarity,
        )
        yield

    app = FastAPI(title="BEON.tech RAG API", version="1.0.0", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
