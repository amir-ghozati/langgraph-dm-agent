"""Constraint tests that need no database.

The constraints in app/db/models.py are not decoration -- three of them are the
only thing standing between the system and a class of bug the source has:
double booking, duplicate replies to a redelivered webhook, and two live
funnels for one lead. These assert the constraints exist and are shaped
correctly, so a later model edit that drops one fails here rather than in
production.
"""

from __future__ import annotations

from sqlalchemy.schema import CreateIndex, CreateTable

from app.db.base import Base
from app.db.models import Availability, Booking, Conversation, Lead, Message, Turn


def _table(model):
    return Base.metadata.tables[model.__tablename__]


def _constraint_names(model) -> set[str]:
    return {c.name for c in _table(model).constraints if c.name}


def _index_names(model) -> set[str]:
    return {i.name for i in _table(model).indexes}


def test_all_eight_tables_are_registered():
    assert set(Base.metadata.tables) == {
        "leads",
        "conversations",
        "turns",
        "messages",
        "funnel_transitions",
        "availability",
        "bookings",
        "turn_metrics",
    }


def test_a_lead_is_unique_per_channel_identity():
    assert "uq_leads_channel_user" in _constraint_names(Lead)


def test_only_one_live_conversation_per_lead():
    """A partial unique index, because the constraint is "one *non-terminal*
    conversation", which a plain unique index cannot express. Without it a
    second inbound message arriving concurrently forks the funnel silently."""
    index = next(
        i
        for i in _table(Conversation).indexes
        if i.name == "uq_conversations_one_live_per_lead"
    )
    assert index.unique
    predicate = str(index.dialect_options["postgresql"]["where"])
    for terminal in ("BOOKED", "ABANDONED", "OPTED_OUT", "HANDED_OFF"):
        assert terminal in predicate


def test_inbound_messages_dedup_on_channel_message_id():
    """Instagram webhooks are at-least-once. The source has no dedup at all, so
    a redelivered webhook there produces a second reply to the lead."""
    index = next(i for i in _table(Message).indexes if i.name == "uq_messages_channel_msgid")
    assert index.unique
    assert [c.name for c in index.columns] == ["channel", "channel_message_id"]
    # Partial, or every outbound CLI message (no channel id) collides on NULL.
    assert "IS NOT NULL" in str(index.dialect_options["postgresql"]["where"])


def test_a_turn_index_is_unique_within_a_conversation():
    assert "uq_turns_conv_index" in _constraint_names(Turn)


def test_messages_link_to_a_turn_so_a_burst_can_share_one():
    """Five of the eight golden conversations open with two inbound messages
    before any reply, so a turn owns N inbound messages rather than one."""
    turn_fk = _table(Message).c.turn_id
    assert turn_fk.nullable
    assert {fk.column.table.name for fk in turn_fk.foreign_keys} == {"turns"}


def test_double_booking_is_impossible_at_the_database():
    assert "uq_bookings_coach_start" in _constraint_names(Booking)


def test_a_retried_booking_is_a_no_op():
    """The idempotency key is what makes commit_booking safe to retry after a
    crash -- the second attempt returns the same receipt instead of a second
    booking."""
    assert "uq_bookings_idempotency" in _constraint_names(Booking)


def test_availability_slots_are_unique_per_coach_and_start():
    assert "uq_availability_coach_start" in _constraint_names(Availability)
    assert "ix_availability_lookup" in _index_names(Availability)


def test_stage_columns_render_as_varchar_with_a_check_not_a_postgres_enum():
    """A Postgres ENUM cannot have a value removed and adding one is a locking
    DDL migration. The funnel vocabulary will still change, so the cheaper form
    is deliberate."""
    from sqlalchemy.dialects import postgresql

    ddl = str(CreateTable(_table(Conversation)).compile(dialect=postgresql.dialect()))
    assert "VARCHAR(32)" in ddl
    assert "CREATE TYPE" not in ddl
    assert "AWAITING_CONFIRMATION" in ddl


def _load_initial_migration():
    """Loaded by path: the module name starts with a digit, so it is not
    importable by name."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "migrations" / "versions" / "0001_core_tables.py"
    spec = importlib.util.spec_from_file_location("initial_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_migration_and_models_agree_on_the_stage_vocabulary():
    """The migration hardcodes the stage list, because a migration must not
    change meaning when the application enum is later edited. That makes drift
    possible, so it is asserted rather than hoped for."""
    from app.funnel.stages import FunnelStage

    assert _load_initial_migration().STAGES == tuple(s.value for s in FunnelStage)


def test_migration_and_models_agree_on_the_partial_index_predicate():
    """The migration writes the terminal stage list as a SQL string literal. If
    the two drift, a freshly migrated database gets a different index from the
    one the models describe, and nothing else would notice."""
    from app.db import models

    assert _load_initial_migration().TERMINAL_SQL == models.TERMINAL_SQL


def test_indexes_compile_against_postgres():
    from sqlalchemy.dialects import postgresql

    for model in (Conversation, Message):
        for index in _table(model).indexes:
            sql = str(CreateIndex(index).compile(dialect=postgresql.dialect()))
            assert sql.startswith("CREATE")
