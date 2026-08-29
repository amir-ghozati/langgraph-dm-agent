"""A deterministic provider. `LLM_PROVIDER=scripted`.

Not a mock hidden in the tests — a third `LLMProvider` implementation, so a
reader who clones this repo with no API key can still run `docker compose up`,
hold a conversation, and watch the funnel advance. The quickstart has to work
from a clean clone; a reviewer who cannot start it stops reading.

It also makes the restart demonstration reproducible. The property being shown
there is that the funnel state survives a process death, and a deterministic
provider makes that a repeatable transcript rather than an anecdote about one
model call.

**No number in the README comes from this.** It answers by pattern-matching the
inbound text, so it demonstrates the machinery and measures nothing. The eval
harness refuses to run against it.
"""

from __future__ import annotations

import json
import re
import time

from pydantic import BaseModel

from app.config import SchemaMode, Settings
from app.llm.base import LLMResult, Message, Role, parse_into

BOOKING_WORDS = re.compile(r"\b(book|call|consult|slot|appointment|chat with)\b", re.I)
QUESTION_WORDS = re.compile(r"\?|^(how|what|when|where|who|why|is|are|do|does|can|could)\b", re.I)
OPT_OUT_WORDS = re.compile(r"\b(nvm|never ?mind|not ready|stop|unsubscribe|forget it)\b", re.I)
PHONE = re.compile(r"[\d][\d\s()+/-]{5,}")
NAME = re.compile(r"\b(?:i'?m|im|my name'?s|this is)\s+([A-Za-z][a-z]{1,20})", re.I)
DAY = re.compile(
    r"\b(mon|tues?|wed(nes)?|thur?s?|fri|sat(ur)?|sun)(day)?\b|\bnext week\b|\btomorrow\b", re.I
)
TIME = re.compile(r"\b(\d{1,2})\s*(am|pm)\b|\b(\d{1,2}):(\d{2})\b", re.I)


class ScriptedProvider:
    name = "scripted"

    def __init__(self, settings: Settings | None = None) -> None:
        self.model = "scripted-rules-1"
        self._settings = settings

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
        started = time.perf_counter()
        inbound = _latest_inbound(messages)
        payload = _respond(schema.__name__ if schema else "", inbound, messages)
        parsed, error = (None, None)
        if schema is not None:
            parsed, error = parse_into(schema, payload)
        return LLMResult(
            text=payload,
            parsed=parsed,
            parse_ok=schema is None or parsed is not None,
            parse_error=error,
            schema_name=schema.__name__ if schema else None,
            schema_mode=schema_mode or SchemaMode.NATIVE,
            provider=self.name,
            model=self.model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            tokens_in=0,
            tokens_out=0,
        )

    async def list_models(self) -> list[str]:
        return [self.model]


def _latest_inbound(messages: list[Message]) -> str:
    for m in reversed(messages):
        if m.role == Role.USER:
            tail = m.content.rsplit("latest message(s):", 1)
            return (tail[-1] if len(tail) > 1 else m.content).strip()
    return ""


def _respond(schema_name: str, inbound: str, messages: list[Message]) -> str:
    match schema_name:
        case "TurnPlan":
            return json.dumps(
                {
                    "intent": _intent(inbound),
                    "knowledge_query": (
                        inbound[:120] if QUESTION_WORDS.search(inbound) else None
                    ),
                    "needs_strategy": True,
                    "user_declined": bool(OPT_OUT_WORDS.search(inbound)),
                    "engaged_with_offer": bool(BOOKING_WORDS.search(inbound)),
                    "notes": "scripted plan",
                }
            )
        case "KnowledgeAnswer":
            # Refuses by default, which is the safe direction: a scripted
            # provider that invented answers would make the fabrication tests
            # pass for the wrong reason.
            return json.dumps(
                {"grounding": "not_in_corpus", "answer": "", "cited": [], "confidence": 0.0}
            )
        case "StageProposal":
            return json.dumps(
                {
                    "proposed_stage": "STAY",
                    "intent": _intent(inbound),
                    "engaged_with_offer": bool(BOOKING_WORDS.search(inbound)),
                    "declined": False,
                    "opt_out": bool(OPT_OUT_WORDS.search(inbound)),
                    "slots": _slots(inbound),
                    "next_action": "reply to what they said",
                    "reason": "scripted",
                }
            )
        case "ComposedReply":
            return json.dumps(
                {
                    "text": _reply_text(messages, inbound),
                    # High enough to clear the confidence gate on a clean turn. A
                    # scripted provider that tripped the guardrail every turn
                    # would demonstrate the interrupt, not the funnel.
                    "self_confidence": 0.8,
                    "mentions_call": bool(BOOKING_WORDS.search(inbound)),
                    "asks_for_slot": True,
                }
            )
    return "{}"


def _intent(inbound: str) -> str:
    if BOOKING_WORDS.search(inbound):
        return "booking_request"
    if QUESTION_WORDS.search(inbound):
        return "direct_question"
    return "none"


def _slots(inbound: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if (m := NAME.search(inbound)) is not None:
        out["name"] = m.group(1).title()
    if (m := PHONE.search(inbound)) is not None:
        out["phone_raw"] = m.group(0).strip()
    if (m := DAY.search(inbound)) is not None:
        out["day_expression"] = m.group(0)
    if (m := TIME.search(inbound)) is not None:
        out["time_expression"] = m.group(0)
    return out


def _reply_text(messages: list[Message], inbound: str) -> str:
    context = messages[-1].content if messages else ""
    stage = ""
    for line in context.splitlines():
        if line.startswith("Funnel stage:"):
            stage = line.split(":", 1)[1].strip()
            break

    if OPT_OUT_WORDS.search(inbound):
        return "No problem at all, I'll leave it there 🙏 shout if you change your mind."
    match stage:
        case "NEW" | "OPENER":
            return "Hey! If you had to pick one fitness goal right now, what would it be? 🙂"
        case "TRANSITION":
            return "Happy to get you booked in for the free call 📅 what's your name?"
        case "SLOT_FILLING":
            missing = ""
            for line in context.splitlines():
                if line.startswith("Slots still needed:"):
                    missing = line.split(":", 1)[1].strip()
                    break
            first = missing.split(",")[0].strip() if missing else "name"
            return f"Great 👍 could you send your {first} as well?"
    return "Got it 👍 what would you like to focus on first?"
