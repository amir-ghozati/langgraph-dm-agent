"""Provider-agnostic LLM interface.

Two reasons this is an interface rather than a direct SDK call:

1. Gemini access may not be stable, and the whole project must not be hostage
   to one vendor's availability.
2. `SCHEMA_MODE` (see below) is only meaningful if there is a seam where the
   two strategies can be swapped and compared.

Anything that depends on Gemini-specific behaviour is marked GEMINI-SPECIFIC in
app/llm/gemini.py so the coupling stays visible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from app.config import SchemaMode


class Role:
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str


@dataclass(slots=True)
class LLMResult:
    """The outcome of exactly one provider call.

    Deliberately *not* the outcome of the repair ladder: the ladder (Phase 2)
    makes several of these and aggregates them. Keeping one call = one result
    is what lets `structured_output_validity_rate` mean "first attempt parsed"
    rather than "eventually parsed".
    """

    text: str
    parsed: BaseModel | None
    parse_ok: bool
    parse_error: str | None
    schema_name: str | None
    schema_mode: SchemaMode | None
    provider: str
    model: str
    latency_ms: int
    tokens_in: int
    tokens_out: int
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model: str

    async def complete(
        self,
        messages: list[Message],
        schema: type[BaseModel] | None = None,
        *,
        schema_mode: SchemaMode | None = None,
        temperature: float | None = None,
    ) -> LLMResult: ...

    def supports_native_schema(self) -> bool: ...


class ProviderError(RuntimeError):
    """Transport, auth or quota failure — not a schema failure.

    Kept distinct so the repair ladder does not waste attempts repairing a
    prompt when the real problem is a 429.
    """


# ---------------------------------------------------------------------------
# Schema-in-prompt helpers, shared by every provider
# ---------------------------------------------------------------------------

_SCHEMA_INSTRUCTION = """\
Return ONLY a single JSON object conforming to this JSON Schema. No prose, no
explanation, no markdown fence. Every required property must be present.

{schema}
"""


def render_schema_instruction(schema: type[BaseModel]) -> str:
    return _SCHEMA_INSTRUCTION.format(schema=json.dumps(schema.model_json_schema(), indent=2))


def unwrap_json_text(text: str) -> str:
    """Strip a markdown code fence if the whole reply is wrapped in one.

    This is the *only* text munging permitted before `json.loads`. It removes a
    fence that delimits the entire payload; it does not scan prose looking for a
    brace. If a model buries JSON inside commentary, that is a parse failure and
    it goes to the repair ladder — which is the behaviour we want to measure,
    not paper over. The source system's regex-extraction step is exactly what
    this refuses to reimplement.
    """
    s = text.strip()
    if not s.startswith("```"):
        return s
    newline = s.find("\n")
    if newline == -1:
        return s
    body = s[newline + 1 :]
    if body.rstrip().endswith("```"):
        body = body.rstrip()[: -len("```")]
    return body.strip()


def parse_into(schema: type[BaseModel], text: str) -> tuple[BaseModel | None, str | None]:
    """Validate `text` as an instance of `schema`. Returns (instance, error)."""
    try:
        payload = json.loads(unwrap_json_text(text))
    except json.JSONDecodeError as exc:
        return None, f"not valid JSON: {exc}"
    try:
        return schema.model_validate(payload), None
    except ValidationError as exc:
        return None, exc.json(include_url=False)


def split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
    """Most providers take the system prompt out of band rather than in the turn list."""
    system = "\n\n".join(m.content for m in messages if m.role == Role.SYSTEM) or None
    rest = [m for m in messages if m.role != Role.SYSTEM]
    return system, rest
