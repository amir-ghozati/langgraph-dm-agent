"""Provider selection. One place that knows which implementations exist."""

from __future__ import annotations

from app.config import LLMProviderName, Settings, get_settings
from app.llm.base import LLMProvider


def build_provider(settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    match settings.llm_provider:
        case LLMProviderName.GEMINI:
            from app.llm.gemini import GeminiProvider

            return GeminiProvider(settings)
        case LLMProviderName.OLLAMA:
            from app.llm.ollama import OllamaProvider

            return OllamaProvider(settings)
        case LLMProviderName.SCRIPTED:
            from app.llm.scripted import ScriptedProvider

            return ScriptedProvider(settings)
    raise ValueError(f"unknown provider {settings.llm_provider}")
