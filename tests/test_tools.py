"""The four tools: argument validation, ambiguity, and the booking transaction.

Every case here is either a golden-set scenario or a defect in the source that
the port fixes. Nothing calls a model — these are the deterministic half, which
is why they are the ones producing headline numbers.
"""

from __future__ import annotations

import datetime as dt
import uuid

import pytest

from app.tools.booking import (
    BookingOutcome,
    check_availability,
    commit_booking,
    list_free_slots,
    seed_availability,
)
from app.tools.datetime_tool import Resolution, resolve_datetime
from app.tools.phone import PhoneError, validate_phone

# A Tuesday, matching every golden conversation's `context.today`.
NOW = dt.datetime(2026, 3, 10, 12, 0, tzinfo=dt.UTC)


async def make_conversation(db) -> uuid.UUID:
    """A real conversation row.

    `bookings.conversation_id` is a foreign key, so a random UUID is rejected —
    which is the constraint doing its job. A booking that references no
    conversation is a booking nobody can trace back to a lead.
    """
    from app.db import repository as repo

    session = await repo.load_session(db, "cli", f"tools-{uuid.uuid4().hex[:12]}")
    await db.commit()
    return session.conversation.id


# ---------------------------------------------------------------------------
# resolve_datetime
# ---------------------------------------------------------------------------


def test_next_tuesday_said_on_a_tuesday_is_ambiguous():
    """smoke-01. The source's rule gives no answer here and picks one anyway."""
    r = resolve_datetime("can we do next tuesday", now=NOW)
    assert r.status is Resolution.AMBIGUOUS
    assert r.subject == "day"
    assert r.candidates == ["2026-03-17", "2026-03-24"]


def test_a_bare_hour_inside_business_hours_is_ambiguous():
    """The hole in the source's time rule. It resolves "2:30" to 14:30 using
    09:00-21:00 as context, which works only when one reading is in range —
    "9" is 09:00 and 21:00 and both are."""
    r = resolve_datetime("can we do 9", now=NOW)
    assert r.status is Resolution.AMBIGUOUS
    assert r.subject == "time_of_day"
    assert r.candidates == ["09:00", "21:00"]


def test_a_bare_hour_with_only_one_valid_reading_resolves():
    """"friday 6" can only be 18:00 — 06:00 is before business hours. This is
    the case the source's rule actually handles, and it must keep working."""
    r = resolve_datetime("ok what about friday 6 then", now=NOW)
    assert r.status is Resolution.RESOLVED
    assert r.time == dt.time(18, 0)


def test_an_explicit_meridiem_is_never_ambiguous():
    r = resolve_datetime("thursday 5pm work?", now=NOW)
    assert r.status is Resolution.RESOLVED
    assert r.date == dt.date(2026, 3, 12)
    assert r.time == dt.time(17, 0)


def test_a_weekday_without_next_means_the_coming_one():
    r = resolve_datetime("tuesday", now=NOW)
    assert r.status is Resolution.RESOLVED
    assert r.date == dt.date(2026, 3, 17)


def test_next_weekday_is_ambiguous_even_on_a_different_day():
    """"next friday" said on a Wednesday is genuinely contested in English.
    Asking costs one turn; guessing costs a missed appointment."""
    r = resolve_datetime("next friday", now=NOW)
    assert r.status is Resolution.AMBIGUOUS
    assert len(r.candidates) == 2


def test_a_vague_expression_is_unparseable_not_guessed():
    """smoke-01's closing message. "early next week" has no defensible
    resolution, and inventing one is the failure."""
    r = resolve_datetime("or early next week could work too idk", now=NOW)
    assert r.status is not Resolution.RESOLVED or r.date is None


def test_now_is_injected_so_results_are_reproducible():
    """The golden set pins `context.today`; a tool reading the wall clock would
    make every dated assertion flake."""
    later = resolve_datetime("tuesday", now=NOW + dt.timedelta(days=1))
    assert later.date == dt.date(2026, 3, 17)


# ---------------------------------------------------------------------------
# phone validation
# ---------------------------------------------------------------------------


def test_a_landline_is_rejected_as_a_landline():
    """smoke-03 turn 3. The correction has to name the actual problem, or the
    lead has nothing to act on."""
    r = validate_phone("0341 9876543")
    assert not r.ok
    assert r.error is PhoneError.LANDLINE
    assert "landline" in r.message


def test_letters_are_rejected_before_parsing():
    """smoke-03 turn 5. `phonenumbers` treats letters as a vanity number and
    maps them to digits — "0151 23oh4a78" becomes a real, valid, wrong number.
    In a DM that is a typo, not a vanity number."""
    r = validate_phone("0151 23oh4a78")
    assert not r.ok
    assert r.error is PhoneError.NON_NUMERIC


def test_a_national_format_number_is_valid_with_a_default_region():
    """D38, and a divergence from smoke-03's expectation. A default region IS
    the country code, so requiring "+49" would reject the way a German lead
    actually types their number."""
    r = validate_phone("0151 23456789")
    assert r.ok
    assert r.e164 == "+4915123456789"


def test_an_international_number_is_valid_too():
    assert validate_phone("+49 151 23456789").e164 == "+4915123456789"


def test_a_too_short_number_is_rejected():
    r = validate_phone("123")
    assert not r.ok


def test_every_error_kind_has_a_distinct_message():
    """`phone_corrections_are_distinct` compares these. Identical messages
    would make that assertion unfalsifiable."""
    from app.tools.phone import MESSAGES

    assert len(set(MESSAGES.values())) == len(MESSAGES)


def test_a_valid_number_never_carries_an_error():
    r = validate_phone("0151 23456789")
    assert r.ok and r.error is None and r.message is None


# ---------------------------------------------------------------------------
# availability and booking
# ---------------------------------------------------------------------------

pytestmark_db = pytest.mark.db


@pytest.mark.db
async def test_seeding_availability_is_idempotent(db_sessionmaker):
    coach = f"coach-{uuid.uuid4().hex[:8]}"
    async with db_sessionmaker() as db:
        first = await seed_availability(db, coach_id=coach, days=1, now=NOW)
        await db.commit()
    async with db_sessionmaker() as db:
        second = await seed_availability(db, coach_id=coach, days=1, now=NOW)
        await db.commit()
    assert first > 0
    assert second == 0, "re-seeding created duplicate slots"


@pytest.mark.db
async def test_free_slots_respect_the_minimum_lead_time(db_sessionmaker):
    """A divergence from the source, which permits booking a slot five minutes
    out that the coach cannot see or prepare for."""
    coach = f"coach-{uuid.uuid4().hex[:8]}"
    async with db_sessionmaker() as db:
        await seed_availability(db, coach_id=coach, days=2, now=NOW)
        await db.commit()
        slots = await list_free_slots(db, coach_id=coach, min_lead_minutes=120, now=NOW)

    assert slots
    assert all(s.start_at >= NOW + dt.timedelta(minutes=120) for s in slots)


@pytest.mark.db
async def test_a_slot_inside_the_lead_time_is_not_bookable(db_sessionmaker):
    coach = f"coach-{uuid.uuid4().hex[:8]}"
    async with db_sessionmaker() as db:
        await seed_availability(db, coach_id=coach, days=2, now=NOW)
        await db.commit()
        soon = NOW + dt.timedelta(minutes=30)
        ok, reason = await check_availability(
            db, soon, coach_id=coach, min_lead_minutes=120, now=NOW
        )
    assert not ok
    assert "notice" in reason


@pytest.mark.db
async def test_a_naive_datetime_is_refused(db_sessionmaker):
    """Timezone-aware or nothing. The source's timezone argument has an empty
    description and no validation at all."""
    async with db_sessionmaker() as db:
        ok, reason = await check_availability(db, dt.datetime(2026, 3, 12, 10, 0), now=NOW)
    assert not ok
    assert "timezone-aware" in reason


@pytest.mark.db
async def test_a_booking_commits_and_takes_the_slot(db_sessionmaker):
    coach = f"coach-{uuid.uuid4().hex[:8]}"
    async with db_sessionmaker() as db:
        conversation = await make_conversation(db)
        await seed_availability(db, coach_id=coach, days=2, now=NOW)
        await db.commit()
        slot = (await list_free_slots(db, coach_id=coach, now=NOW))[0]

        receipt = await commit_booking(
            db,
            conversation_id=conversation,
            start_at=slot.start_at,
            customer_name="Dan",
            phone_e164="+4915123456789",
            coach_id=coach,
            now=NOW,
        )
        await db.commit()

        ok, reason = await check_availability(db, slot.start_at, coach_id=coach, now=NOW)

    assert receipt.outcome is BookingOutcome.COMMITTED
    assert receipt.committed
    assert not ok and reason == "already taken"


@pytest.mark.db
async def test_a_retried_booking_returns_the_same_receipt(db_sessionmaker):
    """The idempotency key is what makes commit_booking safe to retry after a
    crash. Without it the retry double-books the same lead."""
    coach = f"coach-{uuid.uuid4().hex[:8]}"
    async with db_sessionmaker() as db:
        conversation = await make_conversation(db)
        await seed_availability(db, coach_id=coach, days=2, now=NOW)
        await db.commit()
        slot = (await list_free_slots(db, coach_id=coach, now=NOW))[0]

        first = await commit_booking(
            db,
            conversation_id=conversation,
            start_at=slot.start_at,
            customer_name="Dan",
            phone_e164="+4915123456789",
            coach_id=coach,
            now=NOW,
        )
        await db.commit()
        second = await commit_booking(
            db,
            conversation_id=conversation,
            start_at=slot.start_at,
            customer_name="Dan",
            phone_e164="+4915123456789",
            coach_id=coach,
            now=NOW,
        )
        await db.commit()

    assert first.outcome is BookingOutcome.COMMITTED
    assert second.outcome is BookingOutcome.ALREADY_BOOKED
    assert second.booking_id == first.booking_id


@pytest.mark.db
async def test_a_taken_slot_offers_alternatives_rather_than_failing_flat(db_sessionmaker):
    """smoke-02. A bare rejection is the naive behaviour; the correct system
    proposes a concrete alternative without being asked twice."""
    coach = f"coach-{uuid.uuid4().hex[:8]}"
    async with db_sessionmaker() as db:
        await seed_availability(db, coach_id=coach, days=2, now=NOW)
        await db.commit()
        slot = (await list_free_slots(db, coach_id=coach, now=NOW))[0]

        await commit_booking(
            db,
            conversation_id=await make_conversation(db),
            start_at=slot.start_at,
            customer_name="Dan",
            phone_e164="+4915123456789",
            coach_id=coach,
            now=NOW,
        )
        await db.commit()

        clash = await commit_booking(
            db,
            conversation_id=await make_conversation(db),
            start_at=slot.start_at,
            customer_name="Eve",
            phone_e164="+4915123456780",
            coach_id=coach,
            now=NOW,
        )

    assert clash.outcome is BookingOutcome.SLOT_TAKEN
    assert clash.alternatives, "no alternatives offered"
    assert all(a.start_at != slot.start_at for a in clash.alternatives)


@pytest.mark.db
async def test_a_booking_refuses_a_non_e164_phone(db_sessionmaker):
    """Arguments are validated before the transaction. A model that could pass
    a raw string here could book on a number nobody can call."""
    coach = f"coach-{uuid.uuid4().hex[:8]}"
    async with db_sessionmaker() as db:
        await seed_availability(db, coach_id=coach, days=2, now=NOW)
        await db.commit()
        slot = (await list_free_slots(db, coach_id=coach, now=NOW))[0]
        receipt = await commit_booking(
            db,
            conversation_id=await make_conversation(db),
            start_at=slot.start_at,
            customer_name="Dan",
            phone_e164="0151 23456789",
            coach_id=coach,
            now=NOW,
        )
    assert receipt.outcome is BookingOutcome.INVALID_ARGUMENTS
    assert not receipt.committed


@pytest.mark.db
async def test_booking_a_slot_that_does_not_exist_offers_alternatives(db_sessionmaker):
    coach = f"coach-{uuid.uuid4().hex[:8]}"
    async with db_sessionmaker() as db:
        await seed_availability(db, coach_id=coach, days=2, now=NOW)
        await db.commit()
        receipt = await commit_booking(
            db,
            conversation_id=await make_conversation(db),
            start_at=NOW + dt.timedelta(days=1, minutes=7),
            customer_name="Dan",
            phone_e164="+4915123456789",
            coach_id=coach,
            now=NOW,
        )
    assert receipt.outcome is BookingOutcome.NO_SUCH_SLOT
    assert receipt.alternatives
