"""The graph, end to end, against a scripted provider and a real database.

No live model. Every LLM call is replayed from a script, so these assert the
wiring and the persistence rather than the model's judgement — which is what
the eval harness is for.
"""

from __future__ import annotations

import json
import uuid

import pytest

from app.channels.base import InboundMessage
from app.channels.envelope import InboundEnvelope
from app.config import SchemaMode, Settings
from app.funnel.stages import FunnelStage
from app.graph.build import build_graph, conversation_snapshot, run_turn
from app.llm.base import LLMResult, parse_into
from app.memory.summary import SlotStatus

pytestmark = pytest.mark.db


class ScriptedProvider:
    """Replays a payload per schema, so a turn can be driven deterministically."""

    name = "scripted"
    model = "scripted-1"

    def __init__(self, **by_schema: str) -> None:
        self.by_schema = by_schema
        self.calls: list[str] = []

    def supports_native_schema(self) -> bool:
        return True

    async def complete(self, messages, schema=None, *, schema_mode=None, temperature=None):
        name = getattr(schema, "__name__", "none")
        self.calls.append(name)
        text = self.by_schema.get(name, "{}")
        parsed, error = (None, "no schema")
        if schema is not None:
            parsed, error = parse_into(schema, text)
        return LLMResult(
            text=text,
            parsed=parsed,
            parse_ok=parsed is not None,
            parse_error=error,
            schema_name=name,
            schema_mode=schema_mode or SchemaMode.NATIVE,
            provider=self.name,
            model=self.model,
            latency_ms=3,
            tokens_in=11,
            tokens_out=7,
        )


def _plan(**over):
    base = {
        "intent": "none",
        "knowledge_query": None,
        "needs_strategy": True,
        "user_declined": False,
        "engaged_with_offer": False,
        "notes": "",
    }
    return json.dumps(base | over)


def _proposal(**over):
    base = {
        "proposed_stage": "STAY",
        "intent": "none",
        "engaged_with_offer": False,
        "declined": False,
        "opt_out": False,
        "slots": {},
        "next_action": "reply",
        "reason": "because",
    }
    return json.dumps(base | over)


def _reply(text="Hey! What's your main fitness goal right now? 🙂", **over):
    base = {
        "text": text,
        "self_confidence": 0.8,
        "mentions_call": False,
        "asks_for_slot": False,
    }
    return json.dumps(base | over)


def _knowledge(**over):
    base = {"grounding": "not_in_corpus", "answer": "", "cited": [], "confidence": 0.0}
    return json.dumps(base | over)


@pytest.fixture
def user() -> str:
    return f"graph-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def settings(settings_kwargs, database_url) -> Settings:
    """Disclosure off by default here.

    With `on_first_contact` the guardrail correctly re-composes any turn-1
    reply that omits the disclosure line — which is right, and makes every
    unrelated test spend three compose calls. Disclosure has its own tests in
    tests/test_interrupt.py.
    """
    return Settings(**settings_kwargs, database_url=database_url, disclosure_mode="none")


def envelope(user: str, *lines: str) -> InboundEnvelope:
    return InboundEnvelope.of(
        [InboundMessage(channel="cli", channel_user_id=user, text=line) for line in lines]
    )


async def _turn(db, provider, settings, env):
    graph = build_graph().compile()
    return await run_turn(graph, env, db=db, provider=provider, settings=settings)


# ---------------------------------------------------------------------------


async def test_one_turn_runs_end_to_end_and_persists(db_sessionmaker, settings, user):
    provider = ScriptedProvider(
        TurnPlan=_plan(), StageProposal=_proposal(), ComposedReply=_reply()
    )
    async with db_sessionmaker() as db:
        state = await _turn(db, provider, settings, envelope(user, "hey"))

    assert state["reply"]
    assert state["decided_stage"] == FunnelStage.OPENER

    async with db_sessionmaker() as db:
        snap = await conversation_snapshot(db, "cli", user)
    assert snap["stage"] == "OPENER"
    assert snap["agent_turns_sent"] == 1
    assert snap["next_turn_index"] == 2


async def test_a_burst_is_one_turn(db_sessionmaker, settings, user):
    """Three messages, one reply, index advances by one — the property D24
    exists for."""
    provider = ScriptedProvider(
        TurnPlan=_plan(), StageProposal=_proposal(), ComposedReply=_reply()
    )
    async with db_sessionmaker() as db:
        await _turn(
            db, provider, settings, envelope(user, "hey", "saw ur reel", "the knee one")
        )
        snap = await conversation_snapshot(db, "cli", user)
    assert snap["agent_turns_sent"] == 1
    assert provider.calls.count("ComposedReply") == 1


async def test_the_knowledge_agent_is_skipped_when_the_plan_asks_nothing(
    db_sessionmaker, settings, user
):
    """The delegation delta, as behaviour. The source called this every turn
    with the raw webhook text because it had no way not to."""
    provider = ScriptedProvider(
        TurnPlan=_plan(knowledge_query=None),
        StageProposal=_proposal(),
        ComposedReply=_reply(),
    )
    async with db_sessionmaker() as db:
        await _turn(db, provider, settings, envelope(user, "hey"))
    assert "KnowledgeAnswer" not in provider.calls


async def test_the_knowledge_agent_runs_when_the_plan_asks_something(
    db_sessionmaker, settings, user
):
    provider = ScriptedProvider(
        TurnPlan=_plan(knowledge_query="does he coach beginners?", intent="direct_question"),
        KnowledgeAnswer=_knowledge(
            grounding="grounded", answer="He coaches beginners too.", confidence=0.9
        ),
        StageProposal=_proposal(),
        ComposedReply=_reply(),
    )
    async with db_sessionmaker() as db:
        await _turn(db, provider, settings, envelope(user, "does he do beginners?"))
    assert "KnowledgeAnswer" in provider.calls


async def test_an_opening_booking_request_skips_the_opener(db_sessionmaker, settings, user):
    """Q1/D18. The source would have asked this lead for their fitness goal."""
    provider = ScriptedProvider(
        TurnPlan=_plan(intent="booking_request"),
        StageProposal=_proposal(intent="booking_request"),
        ComposedReply=_reply(),
    )
    async with db_sessionmaker() as db:
        state = await _turn(db, provider, settings, envelope(user, "can i grab that free call"))
    assert state["decided_stage"] == FunnelStage.TRANSITION


async def test_a_proposal_naming_a_system_only_stage_cannot_reach_the_state_machine(
    db_sessionmaker, settings, user
):
    """The two-lock guarantee. `ProposableStage` has no value for BOOKED, so
    the proposal fails schema validation and the strategy agent falls back to
    STAY rather than the guard having to catch it."""
    provider = ScriptedProvider(
        TurnPlan=_plan(),
        StageProposal=_proposal(proposed_stage="BOOKED"),
        ComposedReply=_reply(),
    )
    async with db_sessionmaker() as db:
        state = await _turn(db, provider, settings, envelope(user, "are we booked?"))
    assert state["decided_stage"] != FunnelStage.BOOKED
    assert state["decided_stage"] == FunnelStage.OPENER


async def test_an_unparseable_supervisor_still_produces_a_safe_turn(
    db_sessionmaker, settings, user
):
    """Ladder exhausted on the planner. The turn must still complete, and must
    not advance anything."""
    provider = ScriptedProvider(
        TurnPlan="not json at all",
        StageProposal=_proposal(),
        ComposedReply=_reply(),
    )
    async with db_sessionmaker() as db:
        state = await _turn(db, provider, settings, envelope(user, "hey"))
    assert state["reply"]
    assert state["decided_stage"] == FunnelStage.OPENER


async def test_an_unparseable_composer_hands_off_rather_than_inventing(
    db_sessionmaker, settings, user
):
    provider = ScriptedProvider(
        TurnPlan=_plan(), StageProposal=_proposal(), ComposedReply="{{{"
    )
    async with db_sessionmaker() as db:
        state = await _turn(db, provider, settings, envelope(user, "hey"))
    assert state["reply_meta"]["self_confidence"] == 0.0
    assert "Jan" in state["reply"]


async def test_the_tool_decides_slot_validity_not_the_model(db_sessionmaker, settings, user):
    """The model reports a raw string; `validate_phone` returns the verdict.

    `SlotReading` has no field for an E.164 number precisely so the model
    cannot supply one — a model that could mark a phone valid could book on a
    number nobody can call.
    """
    provider = ScriptedProvider(
        TurnPlan=_plan(),
        StageProposal=_proposal(slots={"name": "Dan", "phone_raw": "0341 9876543"}),
        ComposedReply=_reply(),
    )
    async with db_sessionmaker() as db:
        await _turn(db, provider, settings, envelope(user, "im dan, 0341 9876543"))
        snap = await conversation_snapshot(db, "cli", user)

    assert snap["slots"]["name"]["value"] == "Dan"
    # A landline, so the tool rejects it and names the kind — which is what the
    # next turn's correction is built from.
    assert snap["slots"]["phone"]["raw"] == "0341 9876543"
    assert snap["slots"]["phone"]["status"] == SlotStatus.INVALID
    assert snap["slots"]["phone"]["error"] == "landline"
    assert snap["slots"]["phone"]["value"] is None


async def test_a_valid_mobile_is_normalised_to_e164(db_sessionmaker, settings, user):
    provider = ScriptedProvider(
        TurnPlan=_plan(),
        StageProposal=_proposal(slots={"phone_raw": "0151 23456789"}),
        ComposedReply=_reply(),
    )
    async with db_sessionmaker() as db:
        await _turn(db, provider, settings, envelope(user, "0151 23456789"))
        snap = await conversation_snapshot(db, "cli", user)

    assert snap["slots"]["phone"]["status"] == SlotStatus.VALID
    assert snap["slots"]["phone"]["value"] == "+4915123456789"


async def test_metrics_are_written_for_every_llm_call(db_sessionmaker, settings, user):
    from sqlalchemy import text as sql

    provider = ScriptedProvider(
        TurnPlan=_plan(knowledge_query="beginners?"),
        KnowledgeAnswer=_knowledge(),
        StageProposal=_proposal(),
        ComposedReply=_reply(),
    )
    async with db_sessionmaker() as db:
        await _turn(db, provider, settings, envelope(user, "beginners?"))
        snap = await conversation_snapshot(db, "cli", user)
        rows = (
            await db.execute(
                sql("select node from turn_metrics where conversation_id = :c"),
                {"c": uuid.UUID(snap["conversation_id"])},
            )
        ).scalars().all()

    assert set(rows) == {"supervisor", "knowledge_agent", "strategy_agent", "compose"}


async def test_the_transition_is_logged_with_proposal_and_decision(
    db_sessionmaker, settings, user
):
    """`proposal_override_rate` only means something if the rejected proposal
    survives."""
    from sqlalchemy import text as sql

    provider = ScriptedProvider(
        TurnPlan=_plan(),
        StageProposal=_proposal(proposed_stage="TRANSITION"),
        ComposedReply=_reply(),
    )
    async with db_sessionmaker() as db:
        await _turn(db, provider, settings, envelope(user, "hey"))
        snap = await conversation_snapshot(db, "cli", user)
        row = (
            await db.execute(
                sql(
                    "select from_stage, proposed_stage, decided_stage, accepted "
                    "from funnel_transitions where conversation_id = :c"
                ),
                {"c": uuid.UUID(snap["conversation_id"])},
            )
        ).one()

    assert row.from_stage == "NEW"
    assert row.proposed_stage == "TRANSITION"
    assert row.decided_stage == "OPENER"
    assert row.accepted is False


async def test_a_duplicate_inbound_message_id_is_not_replied_to_twice(
    db_sessionmaker, settings, user
):
    """At-least-once webhook delivery. The source has no dedup at all."""
    from sqlalchemy import text as sql

    provider = ScriptedProvider(
        TurnPlan=_plan(), StageProposal=_proposal(), ComposedReply=_reply()
    )
    # Unique per run: dedup is global on (channel, channel_message_id), so a
    # fixed id would collide with earlier runs and make the first turn look
    # like the duplicate.
    message_id = f"m-{uuid.uuid4().hex[:10]}"
    env = InboundEnvelope.of(
        [
            InboundMessage(
                channel="cli", channel_user_id=user, text="hey", channel_message_id=message_id
            )
        ]
    )
    async with db_sessionmaker() as db:
        first = await _turn(db, provider, settings, env)
        second = await _turn(db, provider, settings, env)
        snap = await conversation_snapshot(db, "cli", user)
        counts = (
            await db.execute(
                sql(
                    "select direction, count(*) from messages where conversation_id = :c "
                    "group by direction"
                ),
                {"c": uuid.UUID(snap["conversation_id"])},
            )
        ).all()

    stored = dict(counts)
    assert stored["INBOUND"] == 1, "the redelivered message was stored twice"
    assert stored.get("OUTBOUND", 0) == 1, "the redelivery produced a second reply"

    assert first["gate_ok"] is True
    assert second["gate_ok"] is False
    assert "duplicate inbound" in second["drop_reason"]
    # And it cost nothing: the gate short-circuits before any model call.
    assert provider.calls.count("ComposedReply") == 1
    assert snap["agent_turns_sent"] == 1
