"""Relational state.

Replaces the source system's Google Sheets layer. The source had 48 sheets
nodes, but only 13 of them touched real data across 2 sheets; the other 35 were
error logging. So this is not "48 nodes become 8 tables" -- it is "2 sheets and
a pile of missing structure become 8 tables with real constraints".

Table list follows brief v2 section 4. The v1 design's `conversation_summaries`
and `slots_collected` were both strictly 1:1 with a conversation, so they are
JSONB columns here rather than tables; see decision D8 for the tradeoff.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.funnel.stages import FunnelStage

# native_enum=False renders VARCHAR rather than a Postgres ENUM type. A
# Postgres enum cannot have a value removed, and adding one is a DDL migration
# with locking implications; a CHECK constraint is a one-line ALTER. The funnel
# vocabulary will still change during this project, so the cheaper form wins.
# See decision D7.
#
# The CHECK is written out explicitly below rather than left to
# `create_constraint=True`. Two reasons: SQLAlchemy would name every generated
# constraint after the shared type, which collides on `funnel_transitions`
# where three columns use it; and the stage is the one piece of state the whole
# design treats as trustworthy, so its constraint should be visible in the
# migration rather than implied by a keyword argument.
StageEnum = Enum(
    FunnelStage, native_enum=False, create_constraint=False, length=32, name="funnel_stage"
)

STAGES_SQL = ", ".join(f"'{s.value}'" for s in FunnelStage)
TERMINAL_SQL = "'BOOKED', 'ABANDONED', 'OPTED_OUT', 'HANDED_OFF'"


def stage_check(column: str, name: str, nullable: bool = False) -> CheckConstraint:
    predicate = f"{column} IN ({STAGES_SQL})"
    if nullable:
        predicate = f"{column} IS NULL OR {predicate}"
    return CheckConstraint(predicate, name=name)


class Lead(Base):
    """A person. Stable across conversations."""

    __tablename__ = "leads"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    channel: Mapped[str] = mapped_column(String(32))
    channel_user_id: Mapped[str] = mapped_column(String(128))
    source: Mapped[str | None] = mapped_column(String(64))
    display_name: Mapped[str | None] = mapped_column(String(128))
    opted_out_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    conversations: Mapped[list[Conversation]] = relationship(back_populates="lead")

    __table_args__ = (
        # The eligibility gate looks a sender up by this pair on every inbound
        # message, so it is both the identity and the hot read path.
        UniqueConstraint("channel", "channel_user_id", name="uq_leads_channel_user"),
    )


class Conversation(Base):
    """A bounded episode with a lifecycle and an outcome.

    Split from Lead deliberately. The source keyed its memory buffers on the
    Instagram sender id, so a lead who booked, finished, and messaged again six
    weeks later resumed the same buffer and the same finished state.

    `thread_id` for the LangGraph checkpointer is this row's id.
    """

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    lead_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("leads.id", ondelete="CASCADE"))
    stage: Mapped[FunnelStage] = mapped_column(StageEnum, default=FunnelStage.NEW)
    locale: Mapped[str] = mapped_column(String(8), default="en")

    # Typed conversation summary (D3). Written only through the Pydantic model
    # in app/memory/, never as free text: a paragraph a model rewrites each
    # turn can silently drop the user's stated goal, which is the source
    # system's actual failure. A record cannot lose a field it has a slot for.
    summary: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")

    # The four booking slots -- name, phone, day, time -- each with a
    # validation state. The SLOT_FILLING -> AWAITING_CONFIRMATION guard reads
    # this and nothing else.
    slots: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")

    agent_turn_count: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_inbound_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    lead: Mapped[Lead] = relationship(back_populates="conversations")

    __table_args__ = (
        # At most one live conversation per lead. A partial unique index is the
        # only way to say this in SQL, and saying it in SQL rather than in
        # application code is what makes a concurrent second inbound message
        # fail loudly instead of quietly forking the funnel.
        Index(
            "uq_conversations_one_live_per_lead",
            "lead_id",
            unique=True,
            postgresql_where=f"stage NOT IN ({TERMINAL_SQL})",
        ),
        CheckConstraint("agent_turn_count >= 0", name="ck_conversations_turn_count"),
        stage_check("stage", "ck_conversations_stage"),
    )


class Turn(Base):
    """One agent turn: the inbound message(s) that triggered it, and the reply.

    `agent_turn_index` is 1-based and is what every funnel timing rule counts
    against (FUNNEL_TRANSITION_TURN, FUNNEL_HARD_CAP).
    """

    __tablename__ = "turns"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    agent_turn_index: Mapped[int] = mapped_column(Integer)
    started_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("conversation_id", "agent_turn_index", name="uq_turns_conv_index"),
        CheckConstraint("agent_turn_index >= 1", name="ck_turns_index_positive"),
    )


class Message(Base):
    """Every inbound and outbound message, durably.

    A turn owns N inbound messages, not one. Real users send bursts -- five of
    the eight golden conversations open with two inbound messages before any
    reply -- so the link is message -> turn, not turn -> message.
    """

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    turn_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("turns.id", ondelete="SET NULL"))
    channel: Mapped[str] = mapped_column(String(32))
    channel_message_id: Mapped[str | None] = mapped_column(String(128))
    direction: Mapped[str] = mapped_column(String(8))
    body: Mapped[str] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    sent_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        # Inbound dedup. Instagram webhooks are at-least-once and the source
        # has no dedup at all, so a redelivered webhook there produces a second
        # reply. Partial, because outbound CLI messages have no channel id and
        # would otherwise all collide on NULL.
        Index(
            "uq_messages_channel_msgid",
            "channel",
            "channel_message_id",
            unique=True,
            postgresql_where="channel_message_id IS NOT NULL",
        ),
        Index("ix_messages_conversation_created", "conversation_id", "created_at"),
        CheckConstraint("direction IN ('INBOUND', 'OUTBOUND')", name="ck_messages_direction"),
    )


class FunnelTransition(Base):
    """What the LLM proposed, what the machine decided, and why they differed.

    This table is the headline metric. `proposal_override_rate` -- how often the
    state machine rejected the model -- is what tells you a prompt edit made
    things worse, and it exists only because the proposal is recorded even when
    it is discarded.
    """

    __tablename__ = "funnel_transitions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    turn_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("turns.id", ondelete="SET NULL"))
    from_stage: Mapped[FunnelStage] = mapped_column(StageEnum)
    proposed_stage: Mapped[FunnelStage | None] = mapped_column(StageEnum)
    decided_stage: Mapped[FunnelStage] = mapped_column(StageEnum)
    accepted: Mapped[bool] = mapped_column(Boolean)
    guard_result: Mapped[str | None] = mapped_column(String(64))
    override_reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("ix_funnel_transitions_conversation", "conversation_id"),
        stage_check("from_stage", "ck_funnel_transitions_from_stage"),
        stage_check("proposed_stage", "ck_funnel_transitions_proposed_stage", nullable=True),
        stage_check("decided_stage", "ck_funnel_transitions_decided_stage"),
    )


class Availability(Base):
    """The coach's bookable slots. Replaces the Google Calendar integration.

    Free-slot computation is a SQL query against this table. The source made
    the model compute free blocks and subtract a 30-minute tail arithmetically,
    which is the first thing that would break.
    """

    __tablename__ = "availability"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    coach_id: Mapped[str] = mapped_column(String(64))
    start_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    duration_min: Mapped[int] = mapped_column(Integer, default=30)
    state: Mapped[str] = mapped_column(String(16), default="FREE")

    __table_args__ = (
        UniqueConstraint("coach_id", "start_at", name="uq_availability_coach_start"),
        Index("ix_availability_lookup", "coach_id", "state", "start_at"),
        CheckConstraint("state IN ('FREE', 'HELD', 'BOOKED')", name="ck_availability_state"),
        CheckConstraint("duration_min > 0", name="ck_availability_duration"),
    )


class Booking(Base):
    """A committed booking.

    Two constraints do the real work. `uq_bookings_coach_start` makes
    double-booking impossible at the database, rather than in the
    check-then-write the source performs non-transactionally.
    `uq_bookings_idempotency` makes a retry a no-op that returns the same
    receipt, which is what lets the booking tool be retried safely after a
    crash.
    """

    __tablename__ = "bookings"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    coach_id: Mapped[str] = mapped_column(String(64))
    start_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    duration_min: Mapped[int] = mapped_column(Integer, default=30)
    customer_name: Mapped[str] = mapped_column(String(128))
    phone_e164: Mapped[str] = mapped_column(String(24))
    idempotency_key: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("coach_id", "start_at", name="uq_bookings_coach_start"),
        UniqueConstraint("idempotency_key", name="uq_bookings_idempotency"),
    )


class TurnMetric(Base):
    """What the eval harness reads.

    Metrics live in Postgres and not in a tracing backend on purpose: the
    harness must not depend on an observability service being up, and the
    numbers in the README must be reproducible from a database dump.

    `schema_mode` is on every row because the native-versus-prompt comparison
    is a headline result, and a validity rate without the mode that produced it
    is not interpretable.
    """

    __tablename__ = "turn_metrics"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    turn_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("turns.id", ondelete="SET NULL"))
    node: Mapped[str] = mapped_column(String(48))
    provider: Mapped[str | None] = mapped_column(String(32))
    model: Mapped[str | None] = mapped_column(String(64))
    schema_mode: Mapped[str | None] = mapped_column(String(16))
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    tool_calls: Mapped[int] = mapped_column(Integer, default=0)
    repair_attempts: Mapped[int] = mapped_column(Integer, default=0)
    first_attempt_valid: Mapped[bool | None] = mapped_column(Boolean)
    interrupted: Mapped[bool] = mapped_column(Boolean, default=False)
    locale: Mapped[str] = mapped_column(String(8), default="en")
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("ix_turn_metrics_conversation", "conversation_id"),
        Index("ix_turn_metrics_node", "node"),
        CheckConstraint(
            "schema_mode IS NULL OR schema_mode IN ('native', 'prompt')",
            name="ck_turn_metrics_schema_mode",
        ),
    )


__all__ = [
    "Availability",
    "Booking",
    "Conversation",
    "FunnelTransition",
    "Lead",
    "Message",
    "Turn",
    "TurnMetric",
]
