"""The repair ladder, exercised by deliberate fault injection.

Gemini passes attempt 1 nearly always. Waiting for a natural failure to
exercise attempts 2 and 3 means shipping a path that has never run, so every
rung here is forced. "I tested the failure path on purpose" is the answer this
suite exists to make true.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from app.config import SchemaMode, Settings
from app.llm.base import LLMResult, Message, ProviderError, Role
from app.llm.structured import (
    StructuredOutputFailure,
    flatten_schema,
    generate,
)


class Slot(BaseModel):
    day: str
    time: str


class Plan(BaseModel):
    action: str = Field(description="what to do next")
    confidence: float
    slot: Slot | None = None
    topics: list[str] = Field(default_factory=list)


def _result(text: str, parsed: BaseModel | None, error: str | None = None) -> LLMResult:
    return LLMResult(
        text=text,
        parsed=parsed,
        parse_ok=parsed is not None,
        parse_error=error,
        schema_name="Plan",
        schema_mode=SchemaMode.NATIVE,
        provider="fake",
        model="fake-1",
        latency_ms=10,
        tokens_in=5,
        tokens_out=7,
    )


class FakeProvider:
    """Replays a scripted sequence of results, one per attempt."""

    name = "fake"
    model = "fake-1"

    def __init__(self, *results: LLMResult) -> None:
        self._results = list(results)
        self.calls: list[list[Message]] = []
        self.modes: list[SchemaMode] = []

    def supports_native_schema(self) -> bool:
        return True

    async def complete(self, messages, schema=None, *, schema_mode=None, temperature=None):
        self.calls.append(list(messages))
        self.modes.append(schema_mode)
        if not self._results:
            raise AssertionError("provider called more times than the ladder allows")
        return self._results.pop(0)


@pytest.fixture
def settings(settings_kwargs) -> Settings:
    return Settings(**settings_kwargs)


GOOD = Plan(action="ask", confidence=0.9)
MESSAGES = [Message(Role.USER, "hi")]


async def test_a_valid_first_attempt_needs_no_repair(settings):
    provider = FakeProvider(_result('{"action":"ask","confidence":0.9}', GOOD))
    out = await generate(provider, MESSAGES, Plan, settings=settings)
    assert out.value.action == "ask"
    assert out.attempts == 1
    assert out.first_attempt_valid
    assert out.repair_attempts == 0


async def test_attempt_two_feeds_back_the_error_and_the_raw_output(settings):
    """The repair prompt has to describe the problem, not merely signal one —
    otherwise attempt 2 is just a retry with a different seed."""
    bad = _result("not json at all", None, "not valid JSON: line 1")
    provider = FakeProvider(bad, _result('{"action":"ask","confidence":0.9}', GOOD))

    out = await generate(provider, MESSAGES, Plan, settings=settings)
    assert out.attempts == 2
    assert out.repair_attempts == 1
    assert not out.first_attempt_valid

    repair_turn = provider.calls[1][-1].content
    assert "not json at all" in repair_turn
    assert "not valid JSON: line 1" in repair_turn


async def test_attempt_three_flattens_the_schema_and_drops_to_the_portable_path(settings):
    """If native constrained decoding has failed twice, the constraint is not
    what is going to rescue it."""
    bad = _result("{}", None, "field required: confidence")
    provider = FakeProvider(bad, bad, _result('{"action":"ask","confidence":0.9}', GOOD))

    out = await generate(provider, MESSAGES, Plan, settings=settings)
    assert out.attempts == 3
    assert out.schema_mode is SchemaMode.PROMPT
    assert provider.modes[:2] == [SchemaMode.NATIVE, SchemaMode.NATIVE]
    assert provider.modes[2] is SchemaMode.PROMPT
    assert "no nesting" in provider.calls[2][-1].content


async def test_an_exhausted_ladder_raises_rather_than_returning_something(settings):
    """The one behaviour that matters. A fallback that fabricates a value is
    how "never a hallucinated confirmation" fails in practice."""
    bad = _result("nope", None, "not valid JSON")
    provider = FakeProvider(bad, bad, bad)

    with pytest.raises(StructuredOutputFailure) as exc:
        await generate(provider, MESSAGES, Plan, settings=settings)
    assert exc.value.schema is Plan
    assert len(exc.value.attempts) == 3


async def test_the_ladder_is_bounded(settings):
    """Three attempts, never four — an unbounded repair loop against a paid API
    is a bill, not a retry policy."""
    bad = _result("nope", None, "not valid JSON")
    provider = FakeProvider(bad, bad, bad)
    with pytest.raises(StructuredOutputFailure):
        await generate(provider, MESSAGES, Plan, settings=settings)
    assert len(provider.calls) == 3


async def test_a_provider_error_is_not_repaired(settings):
    """A 429 is not a schema problem. Repairing the prompt would spend the
    ladder on the wrong failure and hide the real one."""

    class Failing(FakeProvider):
        async def complete(self, *a, **kw):
            raise ProviderError("429 rate limited")

    with pytest.raises(ProviderError):
        await generate(Failing(), MESSAGES, Plan, settings=settings)


async def test_first_attempt_valid_is_recorded_independently_of_success(settings):
    """The headline validity rate must count attempt 1, not "eventually
    parsed" — otherwise it reads ~100% and measures nothing."""
    bad = _result("{}", None, "field required")
    provider = FakeProvider(bad, _result('{"action":"ask","confidence":0.9}', GOOD))
    out = await generate(provider, MESSAGES, Plan, settings=settings)
    assert out.value is not None
    assert out.first_attempt_valid is False


async def test_cost_accumulates_across_the_whole_ladder(settings):
    """A turn that repaired twice cost three calls, and the metrics must say
    so — otherwise repair looks free."""
    bad = _result("{}", None, "field required")
    provider = FakeProvider(bad, bad, _result('{"action":"ask","confidence":0.9}', GOOD))
    out = await generate(provider, MESSAGES, Plan, settings=settings)
    assert out.tokens_in == 15 and out.tokens_out == 21
    assert out.latency_ms == 30


async def test_fault_injection_can_corrupt_an_otherwise_good_reply(settings):
    """The hook the acceptance criterion depends on: Gemini rarely fails
    naturally, so the failure must be forced in a test rather than waited for."""
    good = _result('{"action":"ask","confidence":0.9}', GOOD)
    provider = FakeProvider(good, good)

    def corrupt(attempt: int, result: LLMResult) -> LLMResult:
        if attempt == 1:
            return _result("<not json>", None, "not valid JSON: injected")
        return result

    out = await generate(provider, MESSAGES, Plan, settings=settings, fault_injector=corrupt)
    assert out.attempts == 2
    assert not out.first_attempt_valid


# ---------------------------------------------------------------------------
# schema flattening
# ---------------------------------------------------------------------------


def test_flattening_removes_nesting_and_unions():
    flat = flatten_schema(Plan)
    assert flat["type"] == "object"
    assert set(flat["properties"]) == {"action", "confidence", "slot", "topics"}
    assert "$ref" not in str(flat)
    assert "anyOf" not in str(flat)
    assert flat["properties"]["topics"]["items"] == {"type": "string"}


def test_flattening_keeps_the_required_fields():
    """A flattened success is still validated against the real schema, so
    dropping `required` would turn attempt 3 into a lower bar rather than an
    easier phrasing of the same one."""
    assert set(flatten_schema(Plan)["required"]) == {"action", "confidence"}
