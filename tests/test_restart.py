"""Resume after a mid-conversation restart.

The acceptance criterion is "kill the process mid-conversation and restart, and
it resumes". The easy version of that test — check the conversation continues
coherently — would pass while the funnel had silently reset to `NEW`, because
LangGraph's checkpointer restores its own message list without help. The
conversation would look fine and the failure would read as a model problem.

So this asserts what D4 actually claims: the app tables are the source of
truth. Nothing here touches the checkpointer. Every reload goes through
`app.db.repository`, in a *new engine and new session*, and checks the four
things that must survive:

    * the persisted funnel stage, not one re-inferred from the transcript
    * the collected slots, with the same validity status
    * the typed summary, including facts_delivered and objections
    * agent_turn_index — continuing, not restarting
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from app.db import repository as repo
from app.funnel.stages import FunnelStage
from app.funnel.transitions import Decision, Trigger
from app.memory.summary import Slot, Slots, SlotStatus, TypedSummary

pytestmark = pytest.mark.db

CHANNEL = "cli"


@pytest.fixture
def user() -> str:
    """A fresh lead per test.

    Fixed ids made the suite pass once and fail on every rerun: the second run
    resumed the first run's live conversation and saw six turns instead of
    three. A test that only passes against an empty database is not testing
    resumption, it is testing that nobody ran it twice.
    """
    return f"restart-{uuid.uuid4().hex[:12]}"


async def _drive_three_turns(sessionmaker, user: str) -> None:
    """A conversation up to mid-slot-filling, committed turn by turn the way
    the graph does — so a 'crash' is just the absence of the next call."""
    async with sessionmaker() as db:
        s = await repo.load_session(db, CHANNEL, user)
        turn = await repo.open_turn(db, s.conversation, s.agent_turn_index)
        await repo.record_transition(
            db,
            s.conversation,
            turn,
            Decision(
                from_stage=FunnelStage.NEW,
                decided_stage=FunnelStage.VALUE,
                proposed_stage=FunnelStage.VALUE,
                accepted=True,
                trigger=Trigger.LLM_PROPOSAL,
                guard_result="first_inbound_intent=direct_question",
            ),
        )
        summary = TypedSummary(stated_goals=["lose 10kg"], last_updated_turn=1)
        await repo.complete_turn(
            db,
            s.conversation,
            turn,
            reply="What's your main goal?",
            summary=summary,
            slots=Slots(),
            channel=CHANNEL,
        )
        await db.commit()

    async with sessionmaker() as db:
        s = await repo.load_session(db, CHANNEL, user)
        turn = await repo.open_turn(db, s.conversation, s.agent_turn_index)
        await repo.record_transition(
            db,
            s.conversation,
            turn,
            Decision(
                from_stage=FunnelStage.VALUE,
                decided_stage=FunnelStage.TRANSITION,
                proposed_stage=FunnelStage.TRANSITION,
                accepted=True,
                trigger=Trigger.SYSTEM,
                guard_result="user_asked_to_book",
            ),
        )
        summary = s.summary.merged_with(
            TypedSummary(
                constraints=["bad knee"],
                objections=["worried about time"],
                facts_delivered=["10 minutes of stretching each morning"],
            ),
            at_turn=2,
        )
        await repo.complete_turn(
            db,
            s.conversation,
            turn,
            reply="Fancy a free 30-minute call? What's your name?",
            summary=summary,
            slots=Slots(),
            channel=CHANNEL,
        )
        await db.commit()

    async with sessionmaker() as db:
        s = await repo.load_session(db, CHANNEL, user)
        turn = await repo.open_turn(db, s.conversation, s.agent_turn_index)
        await repo.record_transition(
            db,
            s.conversation,
            turn,
            Decision(
                from_stage=FunnelStage.TRANSITION,
                decided_stage=FunnelStage.SLOT_FILLING,
                proposed_stage=FunnelStage.SLOT_FILLING,
                accepted=True,
                trigger=Trigger.LLM_PROPOSAL,
                guard_result="slot_value_supplied",
            ),
        )
        slots = Slots(
            name=Slot(raw="Dan", value="Dan", status=SlotStatus.VALID, attempts=1),
            phone=Slot(
                raw="0341 9876543",
                status=SlotStatus.INVALID,
                error="landline",
                attempts=1,
            ),
        )
        await repo.complete_turn(
            db,
            s.conversation,
            turn,
            reply="That looks like a landline — got a mobile?",
            summary=s.summary,
            slots=slots,
            channel=CHANNEL,
        )
        await db.commit()


async def test_the_funnel_stage_survives_a_restart(db_sessionmaker, fresh_engine, user):
    """Not re-inferred from the transcript — read from `conversations.stage`."""
    await _drive_three_turns(db_sessionmaker, user)

    async with fresh_engine() as reborn:
        s = await repo.load_session(reborn, CHANNEL, user)
        assert s.stage is FunnelStage.SLOT_FILLING, "the funnel reset on restart"


async def test_the_turn_index_continues_rather_than_restarting(
    db_sessionmaker, fresh_engine, user
):
    """A reset index would re-run the value window and push the consultation
    ask past the hard cap — visible only as the agent never asking."""
    await _drive_three_turns(db_sessionmaker, user)

    async with fresh_engine() as reborn:
        s = await repo.load_session(reborn, CHANNEL, user)
        assert s.conversation.agent_turn_count == 3
        assert s.agent_turn_index == 4


async def test_the_slots_survive_with_their_validity_status(db_sessionmaker, fresh_engine, user):
    """Status, not just value. A phone that comes back as SUPPLIED rather than
    INVALID would let the next turn advance to AWAITING_CONFIRMATION on a
    number nobody can call."""
    await _drive_three_turns(db_sessionmaker, user)

    async with fresh_engine() as reborn:
        s = await repo.load_session(reborn, CHANNEL, user)
        assert s.slots.name.is_valid and s.slots.name.value == "Dan"
        assert s.slots.phone.status is SlotStatus.INVALID
        assert s.slots.phone.error == "landline"
        assert s.slots.phone.attempts == 1
        assert not s.slots.all_valid
        assert s.slots.missing == ["phone", "day", "time"]


async def test_the_typed_summary_survives_intact(db_sessionmaker, fresh_engine, user):
    """The goal stated on turn one is the thing most likely to be lost, and the
    reason the summary is typed at all."""
    await _drive_three_turns(db_sessionmaker, user)

    async with fresh_engine() as reborn:
        s = await repo.load_session(reborn, CHANNEL, user)
        assert s.summary.stated_goals == ["lose 10kg"]
        assert s.summary.constraints == ["bad knee"]
        assert s.summary.objections == ["worried about time"]
        assert s.summary.facts_delivered == ["10 minutes of stretching each morning"]


async def test_resuming_does_not_start_a_second_conversation(db_sessionmaker, fresh_engine, user):
    """The partial unique index guarantees one live conversation per lead, and
    a reload must find it rather than open another."""
    await _drive_three_turns(db_sessionmaker, user)

    async with fresh_engine() as reborn:
        first = await repo.load_session(reborn, CHANNEL, user)
        again = await repo.load_session(reborn, CHANNEL, user)
        assert first.conversation.id == again.conversation.id

        count = (
            await reborn.execute(
                text("select count(*) from conversations where lead_id = :lead"),
                {"lead": first.lead.id},
            )
        ).scalar_one()
        assert count == 1


async def test_the_transition_log_records_every_turn(db_sessionmaker, fresh_engine, user):
    """Three turns, three rows — the audit trail proposal_override_rate is
    computed from."""
    await _drive_three_turns(db_sessionmaker, user)

    async with fresh_engine() as reborn:
        s = await repo.load_session(reborn, CHANNEL, user)
        rows = (
            await reborn.execute(
                text(
                    "select from_stage, decided_stage, accepted from funnel_transitions "
                    "where conversation_id = :c order by created_at"
                ),
                {"c": s.conversation.id},
            )
        ).all()
        assert [r[0] for r in rows] == ["NEW", "VALUE", "TRANSITION"]
        assert [r[1] for r in rows] == ["VALUE", "TRANSITION", "SLOT_FILLING"]


async def test_a_crash_before_complete_turn_loses_only_that_turn(
    db_sessionmaker, fresh_engine, user
):
    """The bounded failure D4 accepts. An uncommitted turn is gone, the inbound
    message that triggered it is not, and the funnel state is whatever the last
    completed turn left."""
    await _drive_three_turns(db_sessionmaker, user)

    async with db_sessionmaker() as db:
        s = await repo.load_session(db, CHANNEL, user)
        await repo.open_turn(db, s.conversation, s.agent_turn_index)
        await db.rollback()  # the crash

    async with fresh_engine() as reborn:
        s = await repo.load_session(reborn, CHANNEL, user)
        assert s.conversation.agent_turn_count == 3
        assert s.agent_turn_index == 4
        assert s.stage is FunnelStage.SLOT_FILLING
