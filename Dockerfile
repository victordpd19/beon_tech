FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /data/chroma \
    && chown appuser:appuser /data/chroma

WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

USER appuser

# Bake the ONNX embedding model into the image so startup needs no network.
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['warmup'])"

COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser data ./data

ENV CHROMA_PATH=/data/chroma

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
