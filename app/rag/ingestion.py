import json
from pathlib import Path

from app.core.interfaces import VectorStore
from app.core.models import Document


def load_documents(path: Path) -> list[Document]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Document(id=item["id"], text=item["text"]) for item in raw]


def ingest(path: Path, store: VectorStore) -> int:
    """Upsert the knowledge base into the store. Stable ids keep this idempotent."""
    documents = load_documents(path)
    store.upsert(documents)
    return len(documents)
