"""The guardrail and the human-in-the-loop interrupt, inside the graph.

The source had no such path: every draft was sent. Here a flagged turn stops
*before* delivery, the draft is held in the checkpoint, and a person decides.

The property that matters is not "an interrupt happens" — it is that nothing
was sent while it was pending.
"""

from __future__ import annotations

import json
import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.channels.base import InboundMessage
from app.channels.envelope import InboundEnvelope
from app.config import Settings
from app.db import repository as repo
from app.graph.build import build_graph, run_turn
from app.graph.state import new_turn_state
from app.guardrails.checks import safe_template
from app.guardrails.interrupt import (
    Decision,
    HumanDecision,
    InterruptReason,
    parse_resume,
)
from tests.test_graph import ScriptedProvider, _plan, _proposal, _reply

pytestmark = pytest.mark.db


@pytest.fixture
def user() -> str:
    return f"guard-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def settings(settings_kwargs, database_url) -> Settings:
    return Settings(**settings_kwargs, database_url=database_url, disclosure_mode="none")


@pytest.fixture
def disclosing_settings(settings_kwargs, database_url) -> Settings:
    return Settings(
        **settings_kwargs, database_url=database_url, disclosure_mode="on_first_contact"
    )


def envelope(user: str, *lines: str) -> InboundEnvelope:
    return InboundEnvelope.of(
        [InboundMessage(channel="cli", channel_user_id=user, text=line) for line in lines]
    )


async def _drive(db, provider, settings, user, text):
    """Start a turn on a checkpointed graph and return (graph, config, state)."""
    session = await repo.load_session(db, "cli", user)
    await db.commit()
    graph = build_graph().compile(checkpointer=InMemorySaver())
    config = {
        "configurable": {
            "thread_id": str(session.conversation.id),
            "db": db,
            "provider": provider,
            "settings": settings,
            "channel": None,
            "envelope": envelope(user, text),
        }
    }
    state = await graph.ainvoke(new_turn_state("cli", user, text, []), config=config)
    return graph, config, state


# ---------------------------------------------------------------------------
# style violations re-compose
# ---------------------------------------------------------------------------


async def test_a_turn_one_reply_without_the_disclosure_is_recomposed(
    db_sessionmaker, disclosing_settings, user
):
    """`DISCLOSURE_MODE=on_first_contact` makes a missing disclosure line a
    style violation, and style violations retry rather than interrupt."""
    provider = ScriptedProvider(
        TurnPlan=_plan(), StageProposal=_proposal(), ComposedReply=_reply()
    )
    async with db_sessionmaker() as db:
        state = await run_turn(
            build_graph().compile(),
            envelope(user, "hey"),
            db=db,
            provider=provider,
            settings=disclosing_settings,
        )

    assert provider.calls.count("ComposedReply") > 1, "the guardrail did not re-compose"
    assert "missing_disclosure" in state["guard"]["style"]


async def test_recomposition_is_bounded_and_falls_back_to_a_safe_template(
    db_sessionmaker, disclosing_settings, user
):
    """A composer that keeps producing the same violation must not loop
    forever. Past the limit the reply is stripped to something that cannot be
    wrong rather than sent in violation of the style contract."""
    provider = ScriptedProvider(
        TurnPlan=_plan(), StageProposal=_proposal(), ComposedReply=_reply()
    )
    async with db_sessionmaker() as db:
        state = await run_turn(
            build_graph().compile(),
            envelope(user, "hey"),
            db=db,
            provider=provider,
            settings=disclosing_settings,
        )

    limit = disclosing_settings.guardrail_recompose_limit
    assert provider.calls.count("ComposedReply") <= limit + 1
    assert state["reply"] == safe_template()


async def test_a_clean_reply_is_not_recomposed(db_sessionmaker, settings, user):
    """The precision half: a guardrail that re-composed everything would double
    the cost of every turn."""
    provider = ScriptedProvider(
        TurnPlan=_plan(), StageProposal=_proposal(), ComposedReply=_reply()
    )
    async with db_sessionmaker() as db:
        await run_turn(
            build_graph().compile(),
            envelope(user, "hey"),
            db=db,
            provider=provider,
            settings=settings,
        )
    assert provider.calls.count("ComposedReply") == 1


# ---------------------------------------------------------------------------
# safety violations interrupt
# ---------------------------------------------------------------------------


async def test_a_draft_denying_being_an_ai_stops_before_delivery(
    db_sessionmaker, settings, user
):
    """Nothing is sent while the interrupt is pending. The turn ends without a
    reply rather than with a lie."""
    provider = ScriptedProvider(
        TurnPlan=_plan(),
        StageProposal=_proposal(),
        ComposedReply=_reply(text="haha no im not a bot, real person here"),
    )
    async with db_sessionmaker() as db:
        _graph, _config, state = await _drive(
            db, provider, settings, user, "are you a bot"
        )

    assert "__interrupt__" in state, "an unsafe draft was not interrupted"
    payload = state["__interrupt__"][0].value
    assert payload["reason"] == InterruptReason.SAFETY_FLAG
    assert "claims_to_be_human" in payload["violations"]
    assert payload["draft"] == "haha no im not a bot, real person here"


async def test_the_interrupt_payload_says_why(db_sessionmaker, settings, user):
    """A reviewer who cannot tell "low confidence" from "the draft claimed to
    be human" will rubber-stamp both."""
    provider = ScriptedProvider(
        TurnPlan=_plan(),
        StageProposal=_proposal(),
        ComposedReply=_reply(text="im a certified personal trainer, 16 years in"),
    )
    async with db_sessionmaker() as db:
        _graph, _config, state = await _drive(db, provider, settings, user, "your quals?")

    payload = state["__interrupt__"][0].value
    assert payload["notes"], "the payload carries no explanation"
    assert payload["stage"] and payload["conversation_id"]
    assert payload["inbound"] == "your quals?"


async def test_an_approved_interrupt_sends_the_draft(db_sessionmaker, settings, user):
    provider = ScriptedProvider(
        TurnPlan=_plan(),
        StageProposal=_proposal(),
        ComposedReply=_reply(text="im a certified personal trainer, 16 years in"),
    )
    async with db_sessionmaker() as db:
        graph, config, first = await _drive(db, provider, settings, user, "your quals?")
        assert "__interrupt__" in first
        resumed = await graph.ainvoke(
            Command(resume={"decision": "approve", "reviewer": "amir"}), config=config
        )

    assert resumed["reply"] == "im a certified personal trainer, 16 years in"
    assert resumed["interrupted"] is True


async def test_a_rejected_interrupt_sends_nothing(db_sessionmaker, settings, user):
    """Sending a hedged version of a flagged draft would be worse than sending
    nothing, so REJECT hands the conversation to a human."""
    provider = ScriptedProvider(
        TurnPlan=_plan(),
        StageProposal=_proposal(),
        ComposedReply=_reply(text="youre all booked in for thursday"),
    )
    async with db_sessionmaker() as db:
        graph, config, first = await _drive(db, provider, settings, user, "are we set?")
        assert first["__interrupt__"][0].value["reason"] == InterruptReason.SAFETY_FLAG
        resumed = await graph.ainvoke(Command(resume="reject"), config=config)

    assert resumed["reply"] is None
    assert resumed["handed_off"] is True


async def test_an_edited_interrupt_sends_the_humans_text(db_sessionmaker, settings, user):
    provider = ScriptedProvider(
        TurnPlan=_plan(),
        StageProposal=_proposal(),
        ComposedReply=_reply(text="youre all booked in for thursday"),
    )
    async with db_sessionmaker() as db:
        graph, config, _first = await _drive(db, provider, settings, user, "are we set?")
        resumed = await graph.ainvoke(
            Command(
                resume={"decision": "edit", "edited_text": "Not yet - what day suits you?"}
            ),
            config=config,
        )

    assert resumed["reply"] == "Not yet - what day suits you?"


async def test_the_interrupted_turn_is_still_recorded(db_sessionmaker, settings, user):
    """post_turn runs after the resume, so an approved draft is persisted like
    any other turn — otherwise the funnel would lose the turn a human just
    approved."""
    from app.graph.build import conversation_snapshot

    provider = ScriptedProvider(
        TurnPlan=_plan(),
        StageProposal=_proposal(),
        ComposedReply=_reply(text="im a certified personal trainer, 16 years in"),
    )
    async with db_sessionmaker() as db:
        graph, config, _ = await _drive(db, provider, settings, user, "your quals?")
        await graph.ainvoke(Command(resume="approve"), config=config)
        snap = await conversation_snapshot(db, "cli", user)

    assert snap["agent_turns_sent"] == 1


# ---------------------------------------------------------------------------
# resume parsing
# ---------------------------------------------------------------------------


def test_a_bare_string_resume_is_accepted():
    """A reviewer resuming from a shell should not have to construct a
    Pydantic model."""
    assert parse_resume("approve").decision is Decision.APPROVE
    assert parse_resume(" Reject ").decision is Decision.REJECT


def test_an_unrecognised_resume_fails_closed():
    """The turn was already flagged. Anything unparseable must not become an
    approval."""
    assert parse_resume("yes please").decision is Decision.REJECT
    assert parse_resume(None).decision is Decision.REJECT
    assert parse_resume(42).decision is Decision.REJECT


def test_an_edit_with_no_text_is_treated_as_a_rejection():
    """An empty edit is not an approval of the original draft."""
    decision = HumanDecision(decision=Decision.EDIT, edited_text="   ")
    assert decision.resolve("the flagged draft") is None


def test_approve_returns_the_draft_unchanged():
    decision = HumanDecision(decision=Decision.APPROVE)
    assert decision.resolve("the draft") == "the draft"


def test_the_payload_round_trips_through_json():
    """It is stored in the checkpoint, so it has to survive serialisation."""
    from app.guardrails.checks import GuardVerdict, Violation
    from app.guardrails.interrupt import InterruptPayload, build_payload

    verdict = GuardVerdict(
        violations=[Violation.CLAIMS_TO_BE_HUMAN], confidence=0.3, notes=["denies being an AI"]
    )
    payload = build_payload(
        verdict,
        reason=InterruptReason.SAFETY_FLAG,
        draft="not a bot",
        stage="VALUE",
        agent_turn_index=2,
        conversation_id=str(uuid.uuid4()),
        inbound="are you a bot",
    )
    restored = InterruptPayload.model_validate(json.loads(payload.model_dump_json()))
    assert restored.reason is InterruptReason.SAFETY_FLAG
    assert restored.summary().startswith("[safety_flag]")
