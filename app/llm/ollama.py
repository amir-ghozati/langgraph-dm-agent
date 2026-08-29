"""Ollama provider — DOCUMENTED BUT UNTESTED.

This exists so someone cloning the repo can run the system with no API key and
no foreign service. It has never been executed against a live Ollama and no
number in the README came from it. That is stated in the README too; a provider
claimed to work but never run is worse than one honestly marked.

It is written against Ollama's /api/chat contract:
https://github.com/ollama/ollama/blob/main/docs/api.md

To use it: set LLM_PROVIDER=ollama, run `ollama pull qwen2.5:7b-instruct`, and
expect materially worse structured-output validity than Gemini — a 7B model is
the case the repair ladder was built for.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import BaseModel

from app.config import SchemaMode, Settings
from app.llm.base import (
    LLMResult,
    Message,
    ProviderError,
    Role,
    parse_into,
    render_schema_instruction,
    split_system,
)
from app.logging import get_logger

log = get_logger(__name__)


class OllamaProvider:
    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._host = settings.ollama_host.rstrip("/")
        self.model = settings.ollama_model
        self._sem = asyncio.Semaphore(settings.llm_max_concurrency)

    def supports_native_schema(self) -> bool:
        # Ollama's `format` parameter accepts a JSON schema, which is the rough
        # equivalent of Gemini's responseSchema. Constraint quality at 7B is
        # not comparable, which is the point of measuring both modes.
        return True

    async def complete(
        self,
        messages: list[Message],
        schema: type[BaseModel] | None = None,
        *,
        schema_mode: SchemaMode | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        import httpx  # transitive via langchain-core; not a direct dependency

        mode = schema_mode or self._settings.schema_mode
        system, turns = split_system(messages)
        if schema is not None and mode is SchemaMode.PROMPT:
            system = "\n\n".join(filter(None, [system, render_schema_instruction(schema)]))

        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": (
                ([{"role": "system", "content": system}] if system else [])
                + [
                    {
                        "role": "assistant" if m.role == Role.ASSISTANT else "user",
                        "content": m.content,
                    }
                    for m in turns
                ]
            ),
        }
        if temperature is not None:
            payload["options"] = {"temperature": temperature}
        if schema is not None and mode is SchemaMode.NATIVE:
            payload["format"] = schema.model_json_schema()

        started = time.perf_counter()
        try:
            timeout = self._settings.llm_timeout_seconds
            async with self._sem, httpx.AsyncClient(timeout=timeout) as c:
                response = await c.post(f"{self._host}/api/chat", json=payload)
                response.raise_for_status()
                body = response.json()
        except Exception as exc:
            raise ProviderError(f"ollama call failed: {exc}") from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        text = (body.get("message", {}).get("content") or "").strip()
        parsed: BaseModel | None = None
        error: str | None = None
        if schema is not None:
            parsed, error = parse_into(schema, text)

        return LLMResult(
            text=text,
            parsed=parsed,
            parse_ok=schema is None or parsed is not None,
            parse_error=error,
            schema_name=schema.__name__ if schema else None,
            schema_mode=mode if schema else None,
            provider=self.name,
            model=self.model,
            latency_ms=latency_ms,
            tokens_in=int(body.get("prompt_eval_count") or 0),
            tokens_out=int(body.get("eval_count") or 0),
            raw={"done_reason": body.get("done_reason")},
        )

    async def list_models(self) -> list[str]:
        import httpx

        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{self._host}/api/tags")
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
