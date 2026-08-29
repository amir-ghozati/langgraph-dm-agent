"""Funnel stages.

The stage is deterministic, persisted state (D2). The LLM proposes a next
stage; the state machine decides. Transitions and guards land in Phase 2 —
this module exists in Phase 1 because `conversations.stage` and
`funnel_transitions` need the vocabulary.
"""

from __future__ import annotations

from enum import StrEnum


class FunnelStage(StrEnum):
    NEW = "NEW"
    """Lead exists, no agent turn sent."""

    OPENER = "OPENER"
    """Greeting + goal question sent."""

    VALUE = "VALUE"
    """Delivering tips."""

    TRANSITION = "TRANSITION"
    """Consultation offered."""

    SLOT_FILLING = "SLOT_FILLING"
    """Collecting name / phone / day / time."""

    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    """All four slots valid; availability check or human approval in flight.

    Reachable by system transition only — never by an LLM proposal.
    """

    BOOKED = "BOOKED"
    """Booking committed. Terminal, success.

    Reachable by system transition only. The confirmation sentence is emitted
    by this transition, not by the model. That is the structural reason the
    system cannot produce a booking confirmation it has not earned.
    """

    ABANDONED = "ABANDONED"
    """Idle timeout reached. Terminal."""

    OPTED_OUT = "OPTED_OUT"
    """Explicit opt-out. Terminal, and NOT a failure — excluded from the
    failure numerator of every funnel metric."""

    HANDED_OFF = "HANDED_OFF"
    """Escalated to a human. Terminal."""


TERMINAL_STAGES: frozenset[FunnelStage] = frozenset(
    {
        FunnelStage.BOOKED,
        FunnelStage.ABANDONED,
        FunnelStage.OPTED_OUT,
        FunnelStage.HANDED_OFF,
    }
)

SYSTEM_ONLY_STAGES: frozenset[FunnelStage] = frozenset(
    {FunnelStage.AWAITING_CONFIRMATION, FunnelStage.BOOKED}
)
"""Stages an LLM proposal can never reach. Enforced in Phase 2's transition
guard and asserted in tests/test_funnel_stages.py."""


def is_terminal(stage: FunnelStage) -> bool:
    return stage in TERMINAL_STAGES
