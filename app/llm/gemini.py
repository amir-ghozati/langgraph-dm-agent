"""Gemini provider — the default, working implementation.

Fidelity note: the source n8n workflow ran on Gemini (five
`lmChatGoogleGemini` nodes). Using Gemini here keeps the model family constant
and changes only the orchestration, which isolates the variable the port is
actually about. See decision D6.
"""

from __future__ import annotations

import asyncio
import re
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

# Transient conditions worth another attempt. 400/403 are not here on purpose:
# a malformed request or a bad key will not fix itself and retrying only burns
# the rate limit we are trying to protect.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Both shapes the API uses to say how long to wait. The structured one is
# authoritative; the prose one appears in the human-readable message and is
# read only as a fallback, because a quota error that carries neither should
# fall back to the exponential schedule rather than to zero.
_RETRY_DELAY = re.compile(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'")
_RETRY_PROSE = re.compile(r"[Pp]lease retry in (\d+(?:\.\d+)?)s")


def server_retry_delay(exc: BaseException | None) -> float | None:
    """How long the server asked us to wait, or None if it did not say."""
    if exc is None:
        return None
    text = str(exc)
    for pattern in (_RETRY_DELAY, _RETRY_PROSE):
        m = pattern.search(text)
        if m:
            return float(m.group(1))
    return None


class GeminiProvider:
    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        from google import genai  # imported lazily so tests need no SDK import cost

        if not settings.gemini_api_key:
            raise ProviderError("GEMINI_API_KEY is not set")
        self._genai = genai
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._settings = settings
        self.model = settings.gemini_model
        # Bounds concurrent in-flight requests. The free tier caps requests per
        # minute and an eval run is a few hundred calls; without this the run
        # dies partway through rather than slowing down.
        self._sem = asyncio.Semaphore(settings.llm_max_concurrency)

    def supports_native_schema(self) -> bool:
        return True

    async def complete(
        self,
        messages: list[Message],
        schema: type[BaseModel] | None = None,
        *,
        schema_mode: SchemaMode | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        from google.genai import types

        mode = schema_mode or self._settings.schema_mode
        system, turns = split_system(messages)

        if schema is not None and mode is SchemaMode.PROMPT:
            # Portable path: the schema is described in the prompt, and the
            # reply is validated afterwards. Nothing constrains generation.
            system = "\n\n".join(filter(None, [system, render_schema_instruction(schema)]))

        config: dict[str, Any] = {}
        if system:
            config["system_instruction"] = system
        if temperature is not None:
            config["temperature"] = temperature
        if schema is not None and mode is SchemaMode.NATIVE:
            # GEMINI-SPECIFIC: constrained decoding against the schema. This is
            # the behaviour OllamaProvider approximates with `format`, and that
            # PROMPT mode does without entirely.
            config["response_mime_type"] = "application/json"
            config["response_schema"] = schema

        contents = [
            types.Content(
                role="model" if m.role == Role.ASSISTANT else "user",
                parts=[types.Part.from_text(text=m.content)],
            )
            for m in turns
        ]

        started = time.perf_counter()
        response = await self._call_with_backoff(
            contents=contents, config=types.GenerateContentConfig(**config)
        )
        latency_ms = int((time.perf_counter() - started) * 1000)

        text = (response.text or "").strip()
        usage = response.usage_metadata
        tokens_in = getattr(usage, "prompt_token_count", 0) or 0
        tokens_out = (getattr(usage, "candidates_token_count", 0) or 0) + (
            getattr(usage, "thoughts_token_count", 0) or 0
        )

        parsed: BaseModel | None = None
        error: str | None = None
        if schema is not None:
            # Validate in both modes. Native constrained decoding is not a
            # guarantee — it can still return an empty candidate on a safety
            # stop — and a metric that trusts the provider measures nothing.
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
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    async def _call_with_backoff(self, **kwargs: Any):
        """Exponential backoff with a cap, on transient statuses only.

        A 429 from the Gemini API carries a `RetryInfo.retryDelay` saying how
        long the quota window actually has left. Doubling from one second
        ignores that: 1+2+4+8 is fifteen seconds of waiting against a
        per-minute limit that wanted fifty-four, and the run dies having
        guessed when it was told. The server's number wins when it is larger.
        """
        from google.genai import errors as genai_errors

        settings = self._settings
        delay = settings.llm_backoff_base_seconds
        last: Exception | None = None

        for attempt in range(1, settings.llm_max_attempts + 1):
            try:
                async with self._sem:
                    return await asyncio.wait_for(
                        self._client.aio.models.generate_content(model=self.model, **kwargs),
                        timeout=settings.llm_timeout_seconds,
                    )
            except genai_errors.APIError as exc:
                last = exc
                if getattr(exc, "code", None) not in _RETRYABLE_STATUS:
                    raise ProviderError(f"gemini call failed: {exc}") from exc
            except TimeoutError as exc:
                last = exc
            log.warning(
                "llm.retry",
                provider=self.name,
                attempt=attempt,
                of=settings.llm_max_attempts,
                sleeping=round(min(max(delay, server_retry_delay(last) or 0.0),
                                   settings.llm_backoff_max_seconds), 2),
                server_asked_for=server_retry_delay(last),
                error=str(last),
            )
            if attempt < settings.llm_max_attempts:
                wait = max(delay, server_retry_delay(last) or 0.0)
                wait = min(wait, settings.llm_backoff_max_seconds)
                await asyncio.sleep(wait)
                delay = min(delay * 2, settings.llm_backoff_max_seconds)

        raise ProviderError(
            f"gemini call failed after {settings.llm_max_attempts} attempts: {last}"
        ) from last

    async def list_models(self) -> list[str]:
        """Used by `python -m app.cli doctor` so a stale model id is caught at
        setup rather than mid-conversation."""
        out: list[str] = []
        for m in await self._client.aio.models.list():
            name = getattr(m, "name", "")
            if name:
                out.append(name.removeprefix("models/"))
        return out
