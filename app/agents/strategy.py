"""The strategy sub-agent.

Runs the AIDA-style qualification funnel — but only as far as *proposing*. The
state machine in `app.funnel.transitions` decides, and the proposal is recorded
either way. That split is D2, and it is what makes `proposal_override_rate` a
real metric rather than a description of what the model felt like doing.

Two things this agent may not do, enforced structurally rather than by prompt:

* It cannot propose `AWAITING_CONFIRMATION` or `BOOKED`. Those are not in
  `ProposableStage` at all, so a proposal naming one fails schema validation
  before the state machine ever sees it — a second lock on top of the guard in
  `decide()`.
* It cannot decide that slots are complete. It reports what the lead supplied;
  validation and the transition are the system's.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.funnel.stages import SYSTEM_ONLY_STAGES, FunnelStage
from app.funnel.transitions import InboundIntent
from app.llm.base import LLMProvider, Message, Role
from app.llm.structured import StructuredOutputFailure, StructuredResult, generate
from app.logging import get_logger

log = get_logger(__name__)

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"


class ProposableStage(StrEnum):
    """The stages an LLM is permitted to name.

    `AWAITING_CONFIRMATION` and `BOOKED` are absent on purpose. The model
    cannot propose them because the schema has no value for them, which means a
    malformed or adversarial proposal fails validation rather than relying on
    the guard to catch it. Belt and braces, and the braces are the guard.
    """

    OPENER = "OPENER"
    VALUE = "VALUE"
    TRANSITION = "TRANSITION"
    SLOT_FILLING = "SLOT_FILLING"
    OPTED_OUT = "OPTED_OUT"
    HANDED_OFF = "HANDED_OFF"
    STAY = "STAY"
    """Explicitly propose no change. Distinct from a failed call, which is why
    the fallback can also be STAY without the two becoming indistinguishable in
    the metrics."""


class SlotReading(BaseModel):
    """What the lead appears to have supplied this turn. Raw, not validated —
    phone validation is `commit_booking`'s job and the day/time resolution is
    `resolve_datetime`'s."""

    name: str | None = None
    phone_raw: str | None = None
    day_expression: str | None = None
    time_expression: str | None = None

    @property
    def any_supplied(self) -> bool:
        return any((self.name, self.phone_raw, self.day_expression, self.time_expression))


class StageProposal(BaseModel):
    """What the strategy agent returns. Every field is a proposal or an
    observation; none of it is a decision."""

    proposed_stage: ProposableStage
    intent: InboundIntent = Field(
        default=InboundIntent.NONE,
        description="Shape of the inbound message: none / direct_question / booking_request.",
    )
    engaged_with_offer: bool = Field(
        default=False,
        description=(
            "True only if the lead affirmatively engaged with the consultation "
            "offer. Ignoring the offer is not engagement."
        ),
    )
    declined: bool = Field(
        default=False, description="True only if the lead declined the offer or the conversation."
    )
    opt_out: bool = Field(
        default=False,
        description=(
            "True if the lead withdrew — including informally, e.g. 'nvm, not "
            "ready for this right now'. No keyword is required."
        ),
    )
    slots: SlotReading = Field(default_factory=SlotReading)
    next_action: str = Field(description="One short sentence on what the reply should do.")
    reason: str = Field(description="Why this stage, in one sentence. Logged against the override.")


@lru_cache(maxsize=1)
def load_playbook(path: Path | None = None) -> str:
    return (path or PROMPTS / "strategy.md").read_text(encoding="utf-8")


_SYSTEM = """\
You track where a sales conversation has reached and propose where it should go
next. You do not decide: a deterministic state machine takes your proposal,
applies its guards, and may override you. Propose what you believe is correct
and say why.

Two things you cannot do, so do not try:

* You cannot move the conversation to a booking-confirmed state. Those stages
  are not available to you. A booking is committed by the system against a real
  calendar row, never by anything you write.
* You cannot decide a phone number or a date is valid. Report what the lead
  said; the system validates it.

Report `opt_out` for any withdrawal, including informal ones with no keyword.
Report `engaged_with_offer` only for affirmative engagement — a reply that
ignores the offer and changes the subject is not engagement.

--- FUNNEL PLAYBOOK ---
{playbook}
--- END PLAYBOOK ---
"""

_USER = """\
Current stage: {stage}
Agent turn about to be produced: {turn_index}
Slots collected so far: {slots}

Recent conversation:
{window}

The lead's latest message(s):
{inbound}
"""


def build_messages(
    stage: FunnelStage,
    turn_index: int,
    inbound: str,
    *,
    window: str = "",
    slots: str = "none",
    playbook: str | None = None,
) -> list[Message]:
    system = _SYSTEM.format(playbook=playbook if playbook is not None else load_playbook())
    user = _USER.format(
        stage=stage,
        turn_index=turn_index,
        slots=slots,
        window=window or "(this is the first exchange)",
        inbound=inbound,
    )
    return [Message(Role.SYSTEM, system), Message(Role.USER, user)]


async def propose(
    provider: LLMProvider,
    stage: FunnelStage,
    turn_index: int,
    inbound: str,
    *,
    window: str = "",
    slots: str = "none",
    settings: Settings | None = None,
    playbook: str | None = None,
    **ladder_kwargs,
) -> StructuredResult:
    settings = settings or get_settings()
    return await generate(
        provider,
        build_messages(
            stage, turn_index, inbound, window=window, slots=slots, playbook=playbook
        ),
        StageProposal,
        settings=settings,
        temperature=0.0,
        **ladder_kwargs,
    )


def stay(reason: str) -> StageProposal:
    """The typed fallback. `STAY` is safe in every stage: it advances nothing,
    so a failed strategy call can never move a lead toward a booking."""
    log.warning("strategy.fallback", reason=reason)
    return StageProposal(
        proposed_stage=ProposableStage.STAY,
        next_action="Reply to what the lead said without advancing the funnel.",
        reason=f"strategy agent unavailable: {reason}",
    )


async def propose_or_stay(
    provider: LLMProvider, stage: FunnelStage, turn_index: int, inbound: str, **kwargs
) -> tuple[StageProposal, StructuredResult | None]:
    try:
        result = await propose(provider, stage, turn_index, inbound, **kwargs)
        return result.value, result  # type: ignore[return-value]
    except StructuredOutputFailure as exc:
        return stay(str(exc)), None


def to_funnel_stage(proposal: ProposableStage, current: FunnelStage) -> FunnelStage | None:
    """Translate a proposal into what `decide()` takes.

    `STAY` becomes `None` — "no proposal" — rather than the current stage, so
    the transition log distinguishes "the model wanted no change" from "the
    model was not asked".
    """
    if proposal is ProposableStage.STAY:
        return None
    stage = FunnelStage(proposal.value)
    assert stage not in SYSTEM_ONLY_STAGES, "ProposableStage must never contain a system-only stage"
    return stage
