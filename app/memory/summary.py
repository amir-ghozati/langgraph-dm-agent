"""The typed conversation summary and the booking slots (D3).

Both are JSONB columns on `conversations`, written only through these models.

The reason they are typed rather than free text: a paragraph a model rewrites
each turn is a lossy re-encoding every time, and the thing most likely to be
dropped is the lead's stated goal from turn one — which by turn six is no
longer in the model's immediate context. That is the source system's actual
failure. A record cannot lose a field it has a slot for.

The merge rule follows from the same argument: **add or refine, never delete.**
Dropping a fact requires naming what superseded it, so a summariser cannot
quietly forget.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SlotStatus(StrEnum):
    EMPTY = "empty"
    SUPPLIED = "supplied"
    """The lead said something; it has not been validated yet."""

    INVALID = "invalid"
    VALID = "valid"


class Slot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw: str | None = None
    """What the lead actually typed. Kept for the error message: "0341 9876543
    looks like a landline" is only possible if the raw string survives."""

    value: str | None = None
    """The normalised value — E.164, an ISO date, a 24-hour time. Only ever set
    by a tool, never by a model."""

    status: SlotStatus = SlotStatus.EMPTY
    error: str | None = None
    """The validation failure *kind*, not a sentence: `landline`,
    `missing_country_code`. `phone_corrections_are_distinct` compares these, so
    a generic message would make that assertion unfalsifiable."""

    attempts: int = 0

    @property
    def is_valid(self) -> bool:
        return self.status is SlotStatus.VALID


class Slots(BaseModel):
    """The four fields the consultation booking needs."""

    model_config = ConfigDict(extra="forbid")

    name: Slot = Field(default_factory=Slot)
    phone: Slot = Field(default_factory=Slot)
    day: Slot = Field(default_factory=Slot)
    time: Slot = Field(default_factory=Slot)

    @property
    def all_valid(self) -> bool:
        """The only thing `SLOT_FILLING -> AWAITING_CONFIRMATION` reads.

        Note it requires VALID, not SUPPLIED: a phone number the lead typed but
        that has not passed validation must not advance the funnel, or the
        booking commits against a number nobody can call.
        """
        return all(s.is_valid for s in (self.name, self.phone, self.day, self.time))

    @property
    def missing(self) -> list[str]:
        return [
            field
            for field in ("name", "phone", "day", "time")
            if not getattr(self, field).is_valid
        ]

    def describe(self) -> str:
        """One line for the strategy agent's prompt."""
        parts = [
            f"{f}={getattr(self, f).value or getattr(self, f).raw or '-'}"
            f"({getattr(self, f).status})"
            for f in ("name", "phone", "day", "time")
        ]
        return " ".join(parts)


class Supersession(BaseModel):
    model_config = ConfigDict(extra="forbid")

    removed: str
    superseded_by: str
    at_turn: int


class TypedSummary(BaseModel):
    """What survives once raw exchanges fall out of the window.

    Widening the raw window instead would only delay the loss and multiply
    prompt tokens. This makes the goal unforgettable at constant cost.
    """

    model_config = ConfigDict(extra="forbid")

    stated_goals: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    """Injuries, schedule, diet, budget — anything that bounds what can be
    offered."""

    objections: list[str] = Field(default_factory=list)
    facts_delivered: list[str] = Field(default_factory=list)
    """Tips already given. Without this the agent repeats itself, which the
    source system does."""

    superseded: list[Supersession] = Field(default_factory=list)
    last_updated_turn: int = 0

    def merged_with(self, other: TypedSummary, *, at_turn: int) -> TypedSummary:
        """Add or refine. Never delete.

        Order is preserved and duplicates are dropped, so the lead's first
        stated goal stays first no matter how many turns later this runs.
        """

        def union(a: list[str], b: list[str]) -> list[str]:
            seen: dict[str, None] = {}
            for item in [*a, *b]:
                cleaned = item.strip()
                if cleaned:
                    seen.setdefault(cleaned, None)
            return list(seen)

        return TypedSummary(
            stated_goals=union(self.stated_goals, other.stated_goals),
            constraints=union(self.constraints, other.constraints),
            objections=union(self.objections, other.objections),
            facts_delivered=union(self.facts_delivered, other.facts_delivered),
            superseded=[*self.superseded, *other.superseded],
            last_updated_turn=at_turn,
        )

    def supersede(self, removed: str, superseded_by: str, *, at_turn: int) -> TypedSummary:
        """The only way a fact leaves the summary, and it leaves a record."""
        return TypedSummary(
            stated_goals=[g for g in self.stated_goals if g != removed],
            constraints=[c for c in self.constraints if c != removed],
            objections=[o for o in self.objections if o != removed],
            facts_delivered=[f for f in self.facts_delivered if f != removed],
            superseded=[
                *self.superseded,
                Supersession(removed=removed, superseded_by=superseded_by, at_turn=at_turn),
            ],
            last_updated_turn=at_turn,
        )

    def describe(self) -> str:
        """Rendered into every sub-agent prompt, so the goal is always present
        regardless of what the raw window still holds."""
        lines = []
        for label, values in (
            ("goals", self.stated_goals),
            ("constraints", self.constraints),
            ("objections", self.objections),
            ("already told them", self.facts_delivered),
        ):
            if values:
                lines.append(f"{label}: {'; '.join(values)}")
        return "\n".join(lines) or "(nothing recorded yet)"
