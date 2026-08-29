"""`resolve_datetime` — deterministic first, model only if that fails.

The source made a 7B-class model resolve times from context, with a rule that
has a hole in it: it says to read "2:30" as 14:30 using 09:00-21:00 business
hours, which works only when one reading is in range. A bare "9" is 09:00 and
21:00 and both are valid, and the rule gives no answer. It guesses.

Here `Ambiguous` is a first-class result. Asking which one is correct
behaviour; picking one is the defect `smoke-01` exists to catch.

The parser is deterministic and the model is never asked to do arithmetic —
`list_free_slots` returns slots and the model only phrases them.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from enum import StrEnum
from zoneinfo import ZoneInfo

WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

_WEEKDAY_RE = re.compile(
    r"\b(?P<qualifier>next|this|coming)?\s*(?P<day>" + "|".join(WEEKDAYS) + r")\b", re.I
)
_TODAY_RE = re.compile(r"\b(today|tonight)\b", re.I)
_TOMORROW_RE = re.compile(r"\btomorrow\b", re.I)
_MERIDIEM_RE = re.compile(r"\b(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<mer>am|pm)\b", re.I)
_HHMM_RE = re.compile(r"\b(?P<h>\d{1,2}):(?P<m>\d{2})\b")
_BARE_HOUR_RE = re.compile(r"(?<![\d:])(?P<h>\d{1,2})(?![\d:])")


class Resolution(StrEnum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNPARSEABLE = "unparseable"


@dataclass(frozen=True, slots=True)
class ResolvedDateTime:
    status: Resolution
    value: dt.datetime | None = None
    date: dt.date | None = None
    time: dt.time | None = None
    candidates: list[str] = field(default_factory=list)
    """Populated when AMBIGUOUS: the readings the lead must choose between.
    The clarifying question is built from these, so it names real options."""

    subject: str | None = None
    """`day` or `time_of_day` — what `asks_clarifying_question` asserts about."""

    reason: str | None = None

    @property
    def is_resolved(self) -> bool:
        return self.status is Resolution.RESOLVED


def resolve_datetime(
    expression: str,
    *,
    now: dt.datetime,
    tz: str = "Europe/Berlin",
    business_start: dt.time = dt.time(9, 0),
    business_end: dt.time = dt.time(21, 0),
) -> ResolvedDateTime:
    """Resolve a natural-language day and/or time, or say it cannot.

    `now` is passed in rather than read from the clock so the golden set's
    `context.today` is honoured and the result is reproducible.
    """
    zone = ZoneInfo(tz)
    local_now = now.astimezone(zone)
    text = (expression or "").strip()
    if not text:
        return ResolvedDateTime(Resolution.UNPARSEABLE, reason="empty expression")

    day_result = _resolve_day(text, local_now)
    if day_result.status is Resolution.AMBIGUOUS:
        return day_result

    time_result = _resolve_time(text, business_start, business_end)
    if time_result.status is Resolution.AMBIGUOUS:
        return time_result

    date_part = day_result.date
    time_part = time_result.time
    if date_part is None and time_part is None:
        return ResolvedDateTime(
            Resolution.UNPARSEABLE, reason=f"no day or time found in {text!r}"
        )
    if date_part is None or time_part is None:
        # A partial answer is still progress; the caller asks for the other half.
        return ResolvedDateTime(
            Resolution.RESOLVED, value=None, date=date_part, time=time_part
        )

    return ResolvedDateTime(
        Resolution.RESOLVED,
        value=dt.datetime.combine(date_part, time_part, tzinfo=zone),
        date=date_part,
        time=time_part,
    )


def _resolve_day(text: str, now: dt.datetime) -> ResolvedDateTime:
    if _TODAY_RE.search(text):
        return ResolvedDateTime(Resolution.RESOLVED, date=now.date())
    if _TOMORROW_RE.search(text):
        return ResolvedDateTime(Resolution.RESOLVED, date=now.date() + dt.timedelta(days=1))

    match = _WEEKDAY_RE.search(text)
    if match is None:
        return ResolvedDateTime(Resolution.RESOLVED, date=None)

    target = WEEKDAYS[match.group("day").lower()]
    qualifier = (match.group("qualifier") or "").lower()
    ahead = (target - now.weekday()) % 7

    # "next tuesday" said ON a Tuesday is the case smoke-01 is built around.
    # Both readings are live and nothing in the message chooses between them.
    if qualifier == "next" and ahead == 0:
        this_one = now.date() + dt.timedelta(days=7)
        following = now.date() + dt.timedelta(days=14)
        return ResolvedDateTime(
            Resolution.AMBIGUOUS,
            candidates=[this_one.isoformat(), following.isoformat()],
            subject="day",
            reason=(
                f"'next {match.group('day')}' said on a "
                f"{match.group('day').lower()} — could be either week"
            ),
        )
    if qualifier == "next":
        ahead = ahead + 7 if ahead <= 0 else ahead
        # "next friday" said on a Wednesday is genuinely contested in English:
        # some mean this week's, some mean the following one.
        this_week = now.date() + dt.timedelta(days=ahead)
        next_week = this_week + dt.timedelta(days=7)
        return ResolvedDateTime(
            Resolution.AMBIGUOUS,
            candidates=[this_week.isoformat(), next_week.isoformat()],
            subject="day",
            reason=f"'next {match.group('day')}' is read differently by different people",
        )

    ahead = ahead or 7  # "tuesday" said on a Tuesday means the coming one
    return ResolvedDateTime(Resolution.RESOLVED, date=now.date() + dt.timedelta(days=ahead))


def _resolve_time(
    text: str, business_start: dt.time, business_end: dt.time
) -> ResolvedDateTime:
    if (m := _MERIDIEM_RE.search(text)) is not None:
        hour = int(m.group("h")) % 12
        if m.group("mer").lower() == "pm":
            hour += 12
        return ResolvedDateTime(
            Resolution.RESOLVED, time=dt.time(hour, int(m.group("m") or 0))
        )

    if (m := _HHMM_RE.search(text)) is not None:
        return ResolvedDateTime(
            Resolution.RESOLVED, time=dt.time(int(m.group("h")), int(m.group("m")))
        )

    if (m := _BARE_HOUR_RE.search(text)) is not None:
        hour = int(m.group("h"))
        if not 1 <= hour <= 24:
            return ResolvedDateTime(Resolution.RESOLVED, time=None)
        candidates = {hour % 24, (hour + 12) % 24}
        readings = [h for h in candidates if _in_hours(h, business_start, business_end)]
        if len(readings) > 1:
            # The hole in the source's rule. "9" is 09:00 and 21:00 and both
            # are inside business hours, so there is nothing to disambiguate on.
            return ResolvedDateTime(
                Resolution.AMBIGUOUS,
                candidates=[f"{h:02d}:00" for h in sorted(readings)],
                subject="time_of_day",
                reason=f"'{hour}' could be either of {sorted(readings)} within business hours",
            )
        if len(readings) == 1:
            return ResolvedDateTime(Resolution.RESOLVED, time=dt.time(readings[0], 0))
        return ResolvedDateTime(
            Resolution.UNPARSEABLE, reason=f"'{hour}' is outside business hours either way"
        )

    return ResolvedDateTime(Resolution.RESOLVED, time=None)


def _in_hours(hour: int, start: dt.time, end: dt.time) -> bool:
    return start.hour <= hour <= end.hour
