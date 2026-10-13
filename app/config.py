import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-5-5"
    llm_max_tokens: int = Field(default=300, gt=0)
    claude_api_key: str | None = None

    knowledge_base_path: Path = PROJECT_ROOT / "data" / "knowledge_base.json"
    chroma_path: Path = PROJECT_ROOT / ".chroma"
    chroma_collection: str = "beon_kb"

    top_k: int = Field(default=3, gt=0)
    # Calibrated for all-MiniLM-L6-v2: on-topic questions score >= ~0.27, off-topic <= ~0.08.
    min_similarity: float = Field(default=0.2, ge=-1.0, le=1.0)

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        overrides = {
            field: os.environ[field.upper()]
            for field in cls.model_fields
            if os.environ.get(field.upper())
        }
        return cls(**overrides)


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
