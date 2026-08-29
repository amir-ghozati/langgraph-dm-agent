"""Human-in-the-loop, via LangGraph `interrupt`.

The source had no such path: every draft was sent. Here a flagged turn stops
*before* delivery, the draft is held in the checkpoint, and a person decides.

Resumption is `Command(resume=...)` with one of three decisions:

* `APPROVE` — send the draft as written
* `EDIT` — send the human's text instead
* `REJECT` — send nothing and hand the conversation off

The payload is typed rather than a free-form dict because a reviewer needs to
know *why* they were asked. "Low confidence" and "the draft claimed to be human"
warrant different answers, and a reviewer who cannot tell them apart will
rubber-stamp both.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.guardrails.checks import GuardVerdict, Violation


class InterruptReason(StrEnum):
    SAFETY_FLAG = "safety_flag"
    LOW_CONFIDENCE = "low_confidence"
    BOOKING_APPROVAL = "booking_approval"
    """`REQUIRE_BOOKING_APPROVAL`. A booking is the one irreversible thing the
    agent does, and it involves a real person's calendar."""

    SLOT_RECOVERY_EXHAUSTED = "slot_recovery_exhausted"


class Decision(StrEnum):
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"


class InterruptPayload(BaseModel):
    """What the reviewer sees. Everything needed to decide without opening a
    database console."""

    reason: InterruptReason
    violations: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    draft: str
    stage: str
    agent_turn_index: int
    conversation_id: str
    confidence: float
    inbound: str = ""

    def summary(self) -> str:
        detail = "; ".join(self.notes) or ", ".join(self.violations) or "no detail"
        return f"[{self.reason}] {detail}"


class HumanDecision(BaseModel):
    decision: Decision
    edited_text: str | None = None
    reviewer: str | None = None

    def resolve(self, draft: str) -> str | None:
        """The text to send, or None when the turn is handed off."""
        match self.decision:
            case Decision.APPROVE:
                return draft
            case Decision.EDIT:
                return (self.edited_text or "").strip() or None
            case Decision.REJECT:
                return None
        return None


def reason_for(verdict: GuardVerdict, *, booking_pending: bool = False) -> InterruptReason | None:
    """Why this turn needs a human, or None if it does not.

    Order matters: a safety flag outranks low confidence, because "the draft
    claimed to be human" is a different conversation from "the model was
    unsure" and the reviewer should see the sharper one.
    """
    if booking_pending:
        return InterruptReason.BOOKING_APPROVAL
    safety = [v for v in verdict.safety_violations if v is not Violation.LOW_CONFIDENCE]
    if safety:
        return InterruptReason.SAFETY_FLAG
    if Violation.LOW_CONFIDENCE in verdict.violations:
        return InterruptReason.LOW_CONFIDENCE
    return None


def build_payload(
    verdict: GuardVerdict,
    *,
    reason: InterruptReason,
    draft: str,
    stage: str,
    agent_turn_index: int,
    conversation_id: str,
    inbound: str = "",
) -> InterruptPayload:
    return InterruptPayload(
        reason=reason,
        violations=[str(v) for v in verdict.violations],
        notes=verdict.notes,
        draft=draft,
        stage=stage,
        agent_turn_index=agent_turn_index,
        conversation_id=conversation_id,
        confidence=verdict.confidence,
        inbound=inbound,
    )


def parse_resume(raw: Any) -> HumanDecision:
    """Interpret whatever `Command(resume=...)` carried.

    Accepts the typed model, a dict, or a bare string decision, because a
    reviewer resuming from a shell should not have to construct a Pydantic
    model. Anything unrecognised is REJECT: failing closed on a turn that was
    already flagged is the only safe default.
    """
    if isinstance(raw, HumanDecision):
        return raw
    if isinstance(raw, dict):
        return HumanDecision.model_validate(raw)
    if isinstance(raw, str):
        try:
            return HumanDecision(decision=Decision(raw.strip().lower()))
        except ValueError:
            return HumanDecision(decision=Decision.REJECT, reviewer="unparseable resume")
    return HumanDecision(decision=Decision.REJECT, reviewer="unparseable resume")
