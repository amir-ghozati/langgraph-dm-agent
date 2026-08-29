"""Reading and writing the conversation's authoritative state.

This module is what makes D4 true rather than aspirational. Everything the
funnel depends on — the persisted stage, the slots and their validity, the
typed summary, the turn index — lives in `conversations` and is loaded from
there on every turn.

The distinction matters most on restart. LangGraph's checkpointer will restore
its own message list without help, so a resumed conversation *looks* fine while
the funnel has silently reset to `NEW`. That failure then reads as a model
problem rather than a persistence one, which is the worst kind. Nothing here
reads the checkpointer, and `tests/test_restart.py` reloads through this module
alone.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.channels.envelope import InboundEnvelope
from app.db.models import Conversation, FunnelTransition, Lead, Message, Turn, TurnMetric
from app.funnel.stages import TERMINAL_STAGES, FunnelStage
from app.funnel.transitions import Decision
from app.logging import get_logger
from app.memory.summary import Slots, TypedSummary

log = get_logger(__name__)


@dataclass(slots=True)
class Session:
    """Everything a turn needs, loaded from the tables and nowhere else."""

    lead: Lead
    conversation: Conversation
    summary: TypedSummary
    slots: Slots

    @property
    def stage(self) -> FunnelStage:
        return self.conversation.stage

    @property
    def agent_turn_index(self) -> int:
        """The 1-based index of the turn about to be produced.

        `conversations.agent_turn_count` is the number already sent, so this is
        that plus one. The convention is documented on `TransitionContext`.
        """
        return self.conversation.agent_turn_count + 1

    @property
    def is_terminal(self) -> bool:
        return self.conversation.stage in TERMINAL_STAGES


async def find_lead(
    session: AsyncSession, channel: str, channel_user_id: str
) -> Lead | None:
    return (
        await session.execute(
            select(Lead).where(
                Lead.channel == channel, Lead.channel_user_id == channel_user_id
            )
        )
    ).scalar_one_or_none()


async def upsert_lead(
    session: AsyncSession, channel: str, channel_user_id: str, *, source: str | None = None
) -> Lead:
    lead = await find_lead(session, channel, channel_user_id)
    if lead is None:
        lead = Lead(channel=channel, channel_user_id=channel_user_id, source=source)
        session.add(lead)
        await session.flush()
    return lead


async def live_conversation(session: AsyncSession, lead: Lead) -> Conversation | None:
    """The one non-terminal conversation, if any.

    The partial unique index guarantees there is at most one, so this returning
    two rows would be a schema failure rather than something to handle here.
    """
    return (
        await session.execute(
            select(Conversation)
            .where(Conversation.lead_id == lead.id)
            .where(Conversation.stage.not_in([s.value for s in TERMINAL_STAGES]))
        )
    ).scalar_one_or_none()


async def load_session(
    session: AsyncSession,
    channel: str,
    channel_user_id: str,
    *,
    create: bool = True,
    locale: str = "en",
) -> Session | None:
    """Load the lead and their live conversation, creating them if allowed.

    `create=False` is how `RequireLeadRecord` eligibility is enforced without
    this module knowing about the policy.
    """
    lead = await find_lead(session, channel, channel_user_id)
    if lead is None:
        if not create:
            return None
        lead = await upsert_lead(session, channel, channel_user_id)

    conversation = await live_conversation(session, lead)
    if conversation is None:
        if not create:
            return None
        # A new conversation, not a resumed one: a lead who booked six weeks
        # ago and messages again starts fresh rather than resuming a finished
        # funnel. That split is why `lead` and `conversation` are separate
        # tables at all.
        conversation = Conversation(lead_id=lead.id, stage=FunnelStage.NEW, locale=locale)
        session.add(conversation)
        await session.flush()

    return Session(
        lead=lead,
        conversation=conversation,
        summary=TypedSummary.model_validate(conversation.summary or {}),
        slots=Slots.model_validate(conversation.slots or {}),
    )


async def already_seen(session: AsyncSession, channel: str, ids: list[str]) -> set[str]:
    """Which of these channel message ids are already stored.

    Read-only, for the gate. Dedup is global on `(channel, channel_message_id)`
    rather than per conversation, matching the partial unique index — a
    redelivered webhook is the same message wherever it lands.
    """
    if not ids:
        return set()
    rows = await session.execute(
        select(Message.channel_message_id).where(
            Message.channel == channel, Message.channel_message_id.in_(ids)
        )
    )
    return {r for (r,) in rows if r is not None}


async def record_inbound(
    session: AsyncSession, conv: Conversation, envelope: InboundEnvelope
) -> list[Message]:
    """Persist a burst. Returns only the messages that were new.

    Dedup is by `(channel, channel_message_id)`, which is what makes an
    at-least-once webhook redelivery a no-op rather than a second reply.
    """
    existing: set[str] = set()
    ids = envelope.channel_message_ids
    if ids:
        rows = await session.execute(
            select(Message.channel_message_id).where(
                Message.channel == envelope.channel,
                Message.channel_message_id.in_(ids),
            )
        )
        existing = {r for (r,) in rows if r is not None}

    written: list[Message] = []
    for m in envelope.messages:
        if m.channel_message_id is not None and m.channel_message_id in existing:
            log.info("inbound.duplicate", channel_message_id=m.channel_message_id)
            continue
        row = Message(
            conversation_id=conv.id,
            channel=m.channel,
            channel_message_id=m.channel_message_id,
            direction="INBOUND",
            body=m.text,
            created_at=m.received_at,
        )
        session.add(row)
        written.append(row)

    conv.last_inbound_at = envelope.received_at
    await session.flush()
    return written


async def open_turn(session: AsyncSession, conv: Conversation, index: int) -> Turn:
    turn = Turn(conversation_id=conv.id, agent_turn_index=index)
    session.add(turn)
    await session.flush()
    return turn


async def record_transition(
    session: AsyncSession, conv: Conversation, turn: Turn | None, decision: Decision
) -> FunnelTransition:
    """Write the proposal AND the decision, always.

    Recording only the accepted stage would make `proposal_override_rate`
    unmeasurable, which is the metric that tells you a prompt edit made things
    worse.
    """
    row = FunnelTransition(
        conversation_id=conv.id,
        turn_id=turn.id if turn else None,
        from_stage=decision.from_stage,
        proposed_stage=decision.proposed_stage,
        decided_stage=decision.decided_stage,
        accepted=decision.accepted,
        guard_result=decision.guard_result,
        override_reason=decision.override_reason,
    )
    session.add(row)
    conv.stage = decision.decided_stage
    if decision.decided_stage in TERMINAL_STAGES and conv.ended_at is None:
        conv.ended_at = dt.datetime.now(dt.UTC)
    await session.flush()
    return row


async def complete_turn(
    session: AsyncSession,
    conv: Conversation,
    turn: Turn,
    *,
    reply: str | None,
    summary: TypedSummary,
    slots: Slots,
    channel: str,
    channel_message_id: str | None = None,
) -> None:
    """The single write that makes a turn durable.

    Everything the next turn reads is set here: the stage was set by
    `record_transition`, and the summary, the slots and the turn count are set
    now. A crash before this point loses the turn and re-drives it from the
    inbound message, which the `messages` table still holds.
    """
    if reply is not None:
        session.add(
            Message(
                conversation_id=conv.id,
                turn_id=turn.id,
                channel=channel,
                channel_message_id=channel_message_id,
                direction="OUTBOUND",
                body=reply,
                sent_at=dt.datetime.now(dt.UTC),
            )
        )
    conv.summary = summary.model_dump(mode="json")
    conv.slots = slots.model_dump(mode="json")
    conv.agent_turn_count = turn.agent_turn_index
    turn.completed_at = dt.datetime.now(dt.UTC)
    await session.flush()


async def record_metrics(
    session: AsyncSession,
    conv: Conversation,
    turn: Turn | None,
    *,
    node: str,
    provider: str | None = None,
    model: str | None = None,
    schema_mode: str | None = None,
    latency_ms: int = 0,
    tokens_in: int = 0,
    tokens_out: int = 0,
    tool_calls: int = 0,
    repair_attempts: int = 0,
    first_attempt_valid: bool | None = None,
    interrupted: bool = False,
) -> TurnMetric:
    """What the eval harness reads. In Postgres, not in a tracing backend, so
    the numbers are reproducible from a database dump."""
    row = TurnMetric(
        conversation_id=conv.id,
        turn_id=turn.id if turn else None,
        node=node,
        provider=provider,
        model=model,
        schema_mode=schema_mode,
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        tool_calls=tool_calls,
        repair_attempts=repair_attempts,
        first_attempt_valid=first_attempt_valid,
        interrupted=interrupted,
        locale=conv.locale,
    )
    session.add(row)
    await session.flush()
    return row


async def transcript(session: AsyncSession, conversation_id: uuid.UUID) -> list[Message]:
    return list(
        (
            await session.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at, Message.id)
            )
        ).scalars()
    )
