"""The structured-output repair ladder.

Every LLM call that returns data goes through here. One helper, four rungs:

| Attempt | Strategy |
|---|---|
| 1 | request with the schema, native or in-prompt per `SCHEMA_MODE` |
| 2 | feed back the raw output **and** the Pydantic error, ask for a correction |
| 3 | retry against a flattened schema — no nesting, no unions — then widen back |
| exhausted | raise `StructuredOutputFailure` to the call site's typed fallback |

`first_attempt_valid` from attempt 1 aggregates to the headline
structured-output validity rate. That number is only meaningful because
attempt 1 is recorded separately from whether the ladder eventually succeeded.

Gemini passes attempt 1 nearly always, which is exactly why the ladder is
tested by deliberate fault injection rather than by waiting for a natural
failure. Shipping a code path that has never executed is the thing this
project exists not to do.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from app.config import SchemaMode, Settings, get_settings
from app.llm.base import LLMProvider, LLMResult, Message, ProviderError, Role
from app.logging import get_logger

log = get_logger(__name__)


class StructuredOutputFailure(RuntimeError):
    """The ladder is exhausted. The call site must fall back to a typed default
    and must never fabricate a value."""

    def __init__(self, schema: type[BaseModel], attempts: list[LLMResult]) -> None:
        self.schema = schema
        self.attempts = attempts
        last = attempts[-1].parse_error if attempts else "no attempts"
        super().__init__(
            f"{schema.__name__} did not validate after {len(attempts)} attempts: {last}"
        )


@dataclass(slots=True)
class StructuredResult:
    """What the ladder produced, and what it cost."""

    value: BaseModel
    attempts: int
    first_attempt_valid: bool
    schema_mode: SchemaMode
    provider: str
    model: str
    latency_ms: int
    tokens_in: int
    tokens_out: int
    trace: list[LLMResult] = field(default_factory=list)

    @property
    def repair_attempts(self) -> int:
        return max(0, self.attempts - 1)


def flatten_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """A nesting- and union-free view of the schema, for attempt 3.

    Small models fail hardest on `$ref`, `anyOf` and nested objects. Widening
    to a flat map of primitives usually gets *something* valid back, which the
    caller then re-validates against the real schema — so a flattened success
    is still a real success, not a relaxed one.
    """
    resolved = schema.model_json_schema()
    defs = resolved.get("$defs", {})

    def simplify(prop: dict[str, Any]) -> dict[str, Any]:
        if "$ref" in prop:
            ref = prop["$ref"].rsplit("/", 1)[-1]
            return simplify(defs.get(ref, {"type": "string"}))
        if "anyOf" in prop:
            # Prefer the first non-null branch; the model only has to produce
            # one of them and choosing is where it gets confused.
            branches = [b for b in prop["anyOf"] if b.get("type") != "null"]
            return simplify(branches[0]) if branches else {"type": "string"}
        if prop.get("type") == "object":
            return {"type": "string"}
        if prop.get("type") == "array":
            return {"type": "array", "items": {"type": "string"}}
        return {k: v for k, v in prop.items() if k in {"type", "enum", "description"}}

    return {
        "type": "object",
        "properties": {name: simplify(p) for name, p in resolved.get("properties", {}).items()},
        "required": resolved.get("required", []),
    }


_REPAIR_TEMPLATE = """\
Your previous reply did not validate against the required schema.

--- your reply ---
{raw}

--- validation error ---
{error}

Return a corrected JSON object. Change only what the error requires; keep every
value that was already correct. Reply with the JSON object and nothing else.
"""


async def generate[T: BaseModel](
    provider: LLMProvider,
    messages: list[Message],
    schema: type[T],
    *,
    settings: Settings | None = None,
    schema_mode: SchemaMode | None = None,
    temperature: float | None = None,
    fault_injector: Callable[[int, LLMResult], LLMResult] | None = None,
) -> StructuredResult:
    """Run the ladder. Returns a validated instance or raises.

    `fault_injector` is how the failure path is tested on purpose: it receives
    (attempt_number, result) and may return a corrupted result. Production code
    never passes it; `tests/test_structured.py` does.
    """
    settings = settings or get_settings()
    mode = schema_mode or settings.schema_mode
    trace: list[LLMResult] = []
    latency = tokens_in = tokens_out = 0

    for attempt in range(1, 4):
        turns = list(messages)
        active_mode = mode

        if attempt == 2 and trace:
            turns.append(Message(Role.ASSISTANT, trace[-1].text))
            turns.append(
                Message(
                    Role.USER,
                    _REPAIR_TEMPLATE.format(
                        raw=trace[-1].text or "(empty)",
                        error=trace[-1].parse_error or "no output",
                    ),
                )
            )
        elif attempt == 3:
            # Drop to the portable path with a flattened schema: if native
            # constrained decoding has failed twice, the constraint is not the
            # thing that is going to rescue it.
            active_mode = SchemaMode.PROMPT
            turns.append(
                Message(
                    Role.USER,
                    "Return a flat JSON object with exactly these keys and no nesting:\n"
                    f"{flatten_schema(schema)}",
                )
            )

        try:
            result = await provider.complete(
                turns, schema=schema, schema_mode=active_mode, temperature=temperature
            )
        except ProviderError:
            # Transport, quota or auth. Not a schema failure, so repairing the
            # prompt would waste attempts on the wrong problem.
            raise

        if fault_injector is not None:
            result = fault_injector(attempt, result)

        trace.append(result)
        latency += result.latency_ms
        tokens_in += result.tokens_in
        tokens_out += result.tokens_out

        if result.parse_ok and result.parsed is not None:
            log.info(
                "structured.ok",
                schema=schema.__name__,
                attempt=attempt,
                first_attempt_valid=attempt == 1,
                schema_mode=str(active_mode),
            )
            return StructuredResult(
                value=result.parsed,
                attempts=attempt,
                first_attempt_valid=trace[0].parse_ok,
                schema_mode=active_mode,
                provider=result.provider,
                model=result.model,
                latency_ms=latency,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                trace=trace,
            )

        log.warning(
            "structured.invalid",
            schema=schema.__name__,
            attempt=attempt,
            schema_mode=str(active_mode),
            error=(result.parse_error or "")[:200],
        )

    raise StructuredOutputFailure(schema, trace)
