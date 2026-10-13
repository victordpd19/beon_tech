# BEON.tech RAG API

A FastAPI service that answers questions about BEON.tech using retrieval-augmented generation:

1. **Retrieval:** the knowledge base in [data/knowledge_base.json](data/knowledge_base.json) is embedded locally into a persistent **ChromaDB** collection, using the bundled ONNX `all-MiniLM-L6-v2` model, so retrieval needs no API key.
2. **Relevance gate:** chunks scoring below `MIN_SIMILARITY` are dropped. If none remain, the API answers "I don't know" without calling the LLM.
3. **Generation:** **Claude** answers through the `anthropic` SDK, under a strict prompt:
   - answer from the context only, in at most 2–3 sentences;
   - say "I don't know" when the answer isn't there;
   - context and question are wrapped in XML tags, and the question is escaped against prompt injection.

## Run

```bash
echo "CLAUDE_API_KEY=sk-ant-..." > .env
docker compose up --build            # API_PORT=8010 docker compose up ... if 8000 is taken
```

```bash
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "What services does BEON.tech offer?"}'
```

```json
{
  "question": "What services does BEON.tech offer?",
  "answer": "BEON.tech offers IT staff augmentation services for every modern tech need, from backend and frontend to AI, machine learning, DevOps and QA.",
  "sources": [{"id": "kb-services", "text": "...", "score": 0.7873}],
  "model": "claude-sonnet-5-5"
}
```

Endpoints:
- `POST /ask` takes `{question, top_k?}`.
- `GET /health` returns the status and the number of stored documents.
- `GET /docs` serves the Swagger UI.

## Configuration (env)

| Variable | Default | |
|---|---|---|
| `CLAUDE_API_KEY` | — | required for `LLM_PROVIDER=anthropic` |
| `LLM_PROVIDER` | `anthropic` | the provider is chosen in [app/llm/factory.py](app/llm/factory.py) |
| `LLM_MODEL` | `claude-sonnet-5-5` | |
| `LLM_MAX_TOKENS` | `300` | |
| `TOP_K` | `3` | the number of chunks retrieved |
| `MIN_SIMILARITY` | `0.2` | the cosine similarity cutoff |
| `CHROMA_PATH` | `/data/chroma` in Docker | stored in the `chroma_data` volume |

## Architecture

```
app/
  core/         interfaces.py (VectorStore, LLMClient Protocols), models.py (domain dataclasses)
  rag/          service.py (RAGService), prompt.py (PromptBuilder), ingestion.py
  vectorstore/  chroma_store.py (ChromaVectorStore)
  llm/          anthropic_client.py, factory.py
  api/          routes.py, schemas.py (Pydantic), dependencies.py
  main.py       app factory: wires the concrete implementations together in the lifespan
```

`RAGService` depends only on the `VectorStore` and `LLMClient` protocols. To add a provider such as Ollama:
1. Write a class with `model` and `generate(system, user)`.
2. Register it in `build_llm`.
3. Uncomment the `ollama` service in [docker-compose.yml](docker-compose.yml).

A Streamlit frontend slot is also stubbed there.

## Postman

To use the collection in Postman:
1. Import [postman/beon_rag.postman_collection.json](postman/beon_rag.postman_collection.json).
2. Set the `baseUrl` collection variable (default `http://localhost:8000`).
3. Run the collection.

To run it from the CLI instead:

```bash
npx newman run postman/beon_rag.postman_collection.json --env-var baseUrl=http://localhost:8000
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use fake stores and LLMs, so they need no API key and no network.
