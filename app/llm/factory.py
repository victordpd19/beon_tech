from app.config import Settings
from app.core.interfaces import LLMClient


def build_llm(settings: Settings) -> LLMClient:
    """Pick the LLM implementation from settings. Add new providers (e.g. Ollama) here."""
    if settings.llm_provider == "anthropic":
        from app.llm.anthropic_client import AnthropicLLMClient

        if not settings.claude_api_key:
            raise RuntimeError("CLAUDE_API_KEY is not set; it is required when LLM_PROVIDER=anthropic.")
        return AnthropicLLMClient(
            api_key=settings.claude_api_key,
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider!r}")
