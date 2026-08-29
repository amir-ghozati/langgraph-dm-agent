"""Availability and booking.

Free-slot computation is a SQL query, not arithmetic by a language model. The
source made the model compute free blocks and subtract a 30-minute tail; that
node is the first thing that would break, and nothing about it was checkable.
Here the model receives slots and only phrases them.

`commit_booking` is one transaction with two constraints behind it:

* `uq_bookings_coach_start` — double-booking is impossible at the database,
  rather than unlikely in application code. The source does a non-transactional
  check-then-create.
* `uq_bookings_idempotency` — a retried commit returns the same receipt instead
  of a second booking, which is what makes the tool safe to retry after a crash.

The booking is also the only thing that can move the funnel to `BOOKED`, and
the confirmation sentence is emitted by that transition. A model cannot reach
it: `ProposableStage` has no value for it.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass
from enum import StrEnum
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Availability, Booking
from app.logging import get_logger

log = get_logger(__name__)

DEFAULT_COACH = "jan"


class SlotStatus(StrEnum):
    FREE = "FREE"
    HELD = "HELD"
    BOOKED = "BOOKED"


class BookingOutcome(StrEnum):
    COMMITTED = "committed"
    ALREADY_BOOKED = "already_booked"
    """The idempotency key matched: same receipt, no second row."""

    SLOT_TAKEN = "slot_taken"
    NO_SUCH_SLOT = "no_such_slot"
    INVALID_ARGUMENTS = "invalid_arguments"


@dataclass(frozen=True, slots=True)
class TimeSlot:
    start_at: dt.datetime
    duration_min: int

    def label(self, tz: str = "Europe/Berlin") -> str:
        local = self.start_at.astimezone(ZoneInfo(tz))
        return local.strftime("%A %d %B, %H:%M")


@dataclass(frozen=True, slots=True)
class BookingReceipt:
    outcome: BookingOutcome
    booking_id: uuid.UUID | None = None
    start_at: dt.datetime | None = None
    alternatives: list[TimeSlot] | None = None
    error: str | None = None

    @property
    def committed(self) -> bool:
        return self.outcome in (BookingOutcome.COMMITTED, BookingOutcome.ALREADY_BOOKED)


async def list_free_slots(
    db: AsyncSession,
    *,
    day: dt.date | None = None,
    coach_id: str = DEFAULT_COACH,
    tz: str = "Europe/Berlin",
    horizon_days: int = 14,
    min_lead_minutes: int = 120,
    now: dt.datetime | None = None,
    limit: int = 20,
) -> list[TimeSlot]:
    """Free slots, computed in SQL.

    `min_lead_minutes` is a deliberate divergence from the source, which
    permits booking a slot five minutes out that the coach cannot see or
    prepare for. That is a defect, not a design choice.
    """
    now = now or dt.datetime.now(dt.UTC)
    earliest = now + dt.timedelta(minutes=min_lead_minutes)
    latest = now + dt.timedelta(days=horizon_days)

    query = (
        select(Availability)
        .where(Availability.coach_id == coach_id)
        .where(Availability.state == SlotStatus.FREE)
        .where(Availability.start_at >= earliest)
        .where(Availability.start_at <= latest)
        .order_by(Availability.start_at)
        .limit(limit)
    )
    rows = (await db.execute(query)).scalars().all()

    zone = ZoneInfo(tz)
    slots = [TimeSlot(r.start_at, r.duration_min) for r in rows]
    if day is not None:
        slots = [s for s in slots if s.start_at.astimezone(zone).date() == day]
    return slots


async def check_availability(
    db: AsyncSession,
    start_at: dt.datetime,
    *,
    coach_id: str = DEFAULT_COACH,
    min_lead_minutes: int = 120,
    horizon_days: int = 14,
    now: dt.datetime | None = None,
) -> tuple[bool, str | None]:
    """Is this exact slot bookable? Returns (ok, reason-if-not)."""
    if start_at.tzinfo is None:
        return False, "start_at must be timezone-aware"

    now = now or dt.datetime.now(dt.UTC)
    if start_at < now + dt.timedelta(minutes=min_lead_minutes):
        return False, f"less than {min_lead_minutes} minutes' notice"
    if start_at > now + dt.timedelta(days=horizon_days):
        return False, f"beyond the {horizon_days}-day booking horizon"

    row = (
        await db.execute(
            select(Availability).where(
                Availability.coach_id == coach_id, Availability.start_at == start_at
            )
        )
    ).scalar_one_or_none()

    if row is None:
        return False, "no such slot"
    if row.state != SlotStatus.FREE:
        return False, "already taken"
    return True, None


async def commit_booking(
    db: AsyncSession,
    *,
    conversation_id: uuid.UUID,
    start_at: dt.datetime,
    customer_name: str,
    phone_e164: str,
    coach_id: str = DEFAULT_COACH,
    duration_min: int = 30,
    min_lead_minutes: int = 120,
    now: dt.datetime | None = None,
) -> BookingReceipt:
    """One transaction: re-check the slot, mark it, insert the booking.

    The slot is re-read `FOR UPDATE` inside the transaction rather than trusted
    from an earlier `check_availability`, which closes the source's
    check-then-create race.
    """
    idempotency_key = f"{conversation_id}:{start_at.isoformat()}"

    existing = (
        await db.execute(select(Booking).where(Booking.idempotency_key == idempotency_key))
    ).scalar_one_or_none()
    if existing is not None:
        log.info("booking.idempotent_replay", booking_id=str(existing.id))
        return BookingReceipt(
            BookingOutcome.ALREADY_BOOKED, existing.id, existing.start_at
        )

    if not customer_name.strip():
        return BookingReceipt(BookingOutcome.INVALID_ARGUMENTS, error="name is empty")
    if not phone_e164.startswith("+"):
        return BookingReceipt(
            BookingOutcome.INVALID_ARGUMENTS, error="phone must be E.164 (validated upstream)"
        )

    slot = (
        await db.execute(
            select(Availability)
            .where(Availability.coach_id == coach_id, Availability.start_at == start_at)
            .with_for_update()
        )
    ).scalar_one_or_none()

    if slot is None:
        return BookingReceipt(
            BookingOutcome.NO_SUCH_SLOT,
            alternatives=await list_free_slots(
                db, coach_id=coach_id, min_lead_minutes=min_lead_minutes, now=now, limit=3
            ),
        )
    if slot.state != SlotStatus.FREE:
        return BookingReceipt(
            BookingOutcome.SLOT_TAKEN,
            alternatives=await list_free_slots(
                db, coach_id=coach_id, min_lead_minutes=min_lead_minutes, now=now, limit=3
            ),
        )

    booking = Booking(
        conversation_id=conversation_id,
        coach_id=coach_id,
        start_at=start_at,
        duration_min=duration_min,
        customer_name=customer_name.strip(),
        phone_e164=phone_e164,
        idempotency_key=idempotency_key,
    )
    slot.state = SlotStatus.BOOKED
    db.add(booking)
    try:
        await db.flush()
    except IntegrityError:
        # Another transaction won the same slot between the FOR UPDATE and the
        # insert. The unique constraint is what makes this survivable.
        await db.rollback()
        log.warning("booking.race_lost", start_at=start_at.isoformat())
        return BookingReceipt(
            BookingOutcome.SLOT_TAKEN,
            alternatives=await list_free_slots(
                db, coach_id=coach_id, min_lead_minutes=min_lead_minutes, now=now, limit=3
            ),
        )

    log.info("booking.committed", booking_id=str(booking.id), start_at=start_at.isoformat())
    return BookingReceipt(BookingOutcome.COMMITTED, booking.id, start_at)


async def seed_availability(
    db: AsyncSession,
    *,
    coach_id: str = DEFAULT_COACH,
    days: int = 14,
    start_hour: int = 9,
    end_hour: int = 21,
    duration_min: int = 30,
    tz: str = "Europe/Berlin",
    now: dt.datetime | None = None,
) -> int:
    """Fill the availability table. Replaces the Google Calendar integration.

    Idempotent: the unique constraint on `(coach_id, start_at)` means re-running
    it adds only the slots that are missing.
    """
    zone = ZoneInfo(tz)
    base = (now or dt.datetime.now(dt.UTC)).astimezone(zone)
    added = 0
    for offset in range(days):
        day = (base + dt.timedelta(days=offset)).date()
        for hour in range(start_hour, end_hour):
            for minute in (0, 30):
                start = dt.datetime.combine(day, dt.time(hour, minute), tzinfo=zone)
                if start <= base:
                    continue
                exists = (
                    await db.execute(
                        select(Availability.id).where(
                            Availability.coach_id == coach_id, Availability.start_at == start
                        )
                    )
                ).scalar_one_or_none()
                if exists is None:
                    db.add(
                        Availability(
                            coach_id=coach_id,
                            start_at=start,
                            duration_min=duration_min,
                            state=SlotStatus.FREE,
                        )
                    )
                    added += 1
    await db.flush()
    return added
