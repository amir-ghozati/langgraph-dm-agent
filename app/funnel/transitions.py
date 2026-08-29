"""The funnel state machine (D2).

The LLM proposes a next stage. This module decides. Every call returns a
`Decision` carrying both, plus the reason they differed, which is written to
`funnel_transitions` and aggregated as `proposal_override_rate`.

Two properties this module exists to guarantee:

1. `AWAITING_CONFIRMATION` and `BOOKED` are unreachable by proposal. A model
   cannot talk its way into a booked state, so the booking confirmation
   sentence — emitted by the `BOOKED` transition — cannot be produced without a
   committed row behind it.
2. Once a conversation is `OPTED_OUT`, no path leads back into the funnel.

Deviations from the source system are marked SOURCE-DEVIATION.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.config import Settings
from app.funnel.stages import SYSTEM_ONLY_STAGES, TERMINAL_STAGES, FunnelStage


class InboundIntent(StrEnum):
    """Classification of the *first* inbound message in a conversation.

    SOURCE-DEVIATION (Q1). The source opens every conversation with a greeting
    and a goal question unconditionally. That answers "how much does it cost?"
    with "what's your main fitness goal?", and re-funnels a lead who has
    already asked to book. The port classifies the opening message and routes
    accordingly. The classification is a proposal like any other; the state
    machine decides and logs it.
    """

    NONE = "none"
    """No actionable intent — a greeting, a reaction, an unparseable fragment."""

    DIRECT_QUESTION = "direct_question"
    """The message is shaped as a question.

    This classifies message SHAPE, not corpus coverage. Whether the knowledge
    base can answer it is the knowledge agent's problem, not the router's -- a
    pricing question is out of corpus but still routes here, because answering
    "what's your main fitness goal?" to a direct question is the evasive
    behaviour the conditional opener exists to remove.
    """

    BOOKING_REQUEST = "booking_request"
    """An explicit request to book the consultation call."""


class Trigger(StrEnum):
    SYSTEM = "system"
    LLM_PROPOSAL = "llm_proposal"
    DETECTOR = "detector"
    SCHEDULER = "scheduler"


@dataclass(frozen=True, slots=True)
class FunnelPolicy:
    """The tunable half of the state machine.

    Separated from `Settings` so the transition tests need no environment.
    """

    transition_turn: int = 5
    hard_cap: int = 7
    phone_reask_limit: int = 2
    """SOURCE-DEVIATION (Q3). Re-asks after the first failure, so
    1 + phone_reask_limit attempts in total before handoff."""

    transition_reask_limit: int = 2
    """SOURCE-DEVIATION (Q5). How many times TRANSITION re-asks a lead who
    neither accepts nor declines, before it stops asking and simply replies."""

    @classmethod
    def from_settings(cls, settings: Settings) -> FunnelPolicy:
        return cls(
            transition_turn=settings.funnel_transition_turn,
            hard_cap=settings.funnel_hard_cap,
            phone_reask_limit=settings.phone_reask_limit,
            transition_reask_limit=settings.transition_reask_limit,
        )

    @property
    def phone_attempt_limit(self) -> int:
        return 1 + self.phone_reask_limit


@dataclass(frozen=True, slots=True)
class TransitionContext:
    """Everything the guards read. All of it is observed fact, not model output.

    The exceptions are the fields the supervisor classifies rather than
    observes — `inbound_intent`, `engaged_with_offer` and `user_declined`. Each
    is a proposal like any other: the guard decides and the decision is logged.
    `inbound_intent` is read out of `NEW` (which entry stage) and out of `VALUE`
    (whether the lead asked to book early).
    """

    agent_turn_index: int = 1
    """The 1-based index of the agent turn currently being produced.

    THE CONVENTION, because an off-by-one here changes behaviour in every
    conversation rather than one:

      * The first agent reply of a conversation is `agent_turn_index == 1`.
      * `decide()` runs BEFORE `compose`, so the stage it returns governs the
        reply about to be sent -- not the one already sent.
      * Therefore `FUNNEL_TRANSITION_TURN=5` means "the 5th agent reply is the
        consultation ask", and `FUNNEL_HARD_CAP=7` means "the 7th agent reply
        is the last one that can still be a value turn".
      * `turns.agent_turn_index` stores this same value.
      * `conversations.agent_turn_count` is the number of turns already SENT,
        so during turn N it reads N-1 and `post_turn` sets it to N.

    A conversation with exactly 5 agent turns therefore does reach the forced
    transition, on its final turn.
    """

    inbound_intent: InboundIntent = InboundIntent.NONE
    user_declined: bool = False
    user_responded: bool = True

    engaged_with_offer: bool = False
    """The reply engages affirmatively with the consultation offer.

    SOURCE-DEVIATION (Q5). The guard used to be "did not decline", which read a
    reply that ignored the offer entirely as consent to start collecting a
    phone number. Positive engagement is required instead. Q2 is preserved:
    "yes thursday 3pm" supplies a slot value and goes straight through in one
    turn, with no separate acceptance step.
    """

    slot_value_supplied: bool = False
    """The reply contains at least one of name / phone / day / time."""

    transition_reask_count: int = 0
    """How many times TRANSITION has already re-asked this conversation."""

    slots_all_valid: bool = False
    phone_attempts: int = 0
    booking_committed: bool = False
    slot_unavailable: bool = False

    # Pre-emptive signals, checked before anything else.
    opt_out_detected: bool = False
    guardrail_escalation: bool = False
    idle_expired: bool = False


@dataclass(frozen=True, slots=True)
class Decision:
    from_stage: FunnelStage
    decided_stage: FunnelStage
    proposed_stage: FunnelStage | None
    accepted: bool
    trigger: Trigger
    guard_result: str
    override_reason: str | None = None

    @property
    def changed(self) -> bool:
        return self.from_stage is not self.decided_stage


# Stages a conversation may never re-enter once it has opted out. The funnel
# does not resume; see `decide` and decision D21.
FUNNEL_STAGES_AFTER_OPT_OUT: frozenset[FunnelStage] = frozenset(
    {
        FunnelStage.TRANSITION,
        FunnelStage.SLOT_FILLING,
        FunnelStage.AWAITING_CONFIRMATION,
        FunnelStage.BOOKED,
    }
)

_INTENT_ENTRY_STAGE: dict[InboundIntent, FunnelStage] = {
    InboundIntent.NONE: FunnelStage.OPENER,
    InboundIntent.DIRECT_QUESTION: FunnelStage.VALUE,
    InboundIntent.BOOKING_REQUEST: FunnelStage.TRANSITION,
}


def allows_outbound_initiation(stage: FunnelStage) -> bool:
    """May the business start a new exchange from this stage?

    `OPTED_OUT` blocks business-initiated messaging entirely — no follow-up, no
    re-engagement, ever. A user-initiated message is still answered (see
    `decide`); that asymmetry is the whole point. Other terminal stages block it
    too, because there is nothing left to say.
    """
    return stage not in TERMINAL_STAGES


def _accept(
    current: FunnelStage,
    target: FunnelStage,
    proposal: FunnelStage | None,
    guard: str,
    trigger: Trigger = Trigger.SYSTEM,
) -> Decision:
    return Decision(
        from_stage=current,
        decided_stage=target,
        proposed_stage=proposal,
        accepted=proposal is None or proposal is target,
        trigger=trigger,
        guard_result=guard,
        override_reason=(
            None
            if proposal is None or proposal is target
            else f"proposal {proposal} overridden by guard {guard!r}"
        ),
    )


def decide(
    current: FunnelStage,
    proposal: FunnelStage | None,
    ctx: TransitionContext,
    policy: FunnelPolicy | None = None,
) -> Decision:
    """Return the stage this conversation is actually in after this turn."""
    policy = policy or FunnelPolicy()

    # --- 0. A proposal naming a system-only stage is always rejected --------
    # Checked before everything else so no later branch can accidentally honour
    # it. This is the guarantee behind "never a hallucinated confirmation".
    if proposal is not None and proposal in SYSTEM_ONLY_STAGES:
        proposal_rejected = proposal
        proposal = None
    else:
        proposal_rejected = None

    # --- 1. Pre-emptive terminal signals ------------------------------------
    # These outrank every other rule, including the hard cap, and their order
    # relative to each other is itself load-bearing:
    #
    #   opt-out  >  guardrail escalation  >  idle expiry  >  everything else
    #
    # opt-out before escalation: a lead who withdraws on the same turn a
    # guardrail fires must land in OPTED_OUT, not in a human review queue --
    # otherwise a withdrawn lead gets followed up by a person.
    # Both before the hard cap: a lead who withdraws on turn 7 must not receive
    # the forced consultation ask anyway.
    if current not in TERMINAL_STAGES:
        if ctx.opt_out_detected:
            return _accept(
                current, FunnelStage.OPTED_OUT, proposal, "opt_out_detected", Trigger.DETECTOR
            )
        if ctx.guardrail_escalation:
            return _accept(current, FunnelStage.HANDED_OFF, proposal, "guardrail_escalation")
        if ctx.idle_expired:
            return _accept(
                current, FunnelStage.ABANDONED, proposal, "idle_expired", Trigger.SCHEDULER
            )

    decision = _decide_stage(current, proposal, ctx, policy)

    # Surface the rejected system-only proposal in the log rather than losing it.
    if proposal_rejected is not None:
        return Decision(
            from_stage=decision.from_stage,
            decided_stage=decision.decided_stage,
            proposed_stage=proposal_rejected,
            accepted=False,
            trigger=decision.trigger,
            guard_result=decision.guard_result,
            override_reason=(
                f"{proposal_rejected} is reachable by system transition only; "
                "an LLM proposal can never enter it"
            ),
        )
    return decision


def _decide_stage(
    current: FunnelStage,
    proposal: FunnelStage | None,
    ctx: TransitionContext,
    policy: FunnelPolicy,
) -> Decision:
    match current:
        # --- NEW: three entry paths (Q1) ------------------------------------
        case FunnelStage.NEW:
            target = _INTENT_ENTRY_STAGE[ctx.inbound_intent]
            return _accept(
                current,
                target,
                proposal,
                f"first_inbound_intent={ctx.inbound_intent}",
                Trigger.LLM_PROPOSAL,
            )

        case FunnelStage.OPENER:
            return _accept(current, FunnelStage.VALUE, proposal, "inbound_reply_received")

        case FunnelStage.VALUE:
            # The cap forces the ask regardless of what the model wanted.
            if ctx.agent_turn_index >= policy.hard_cap:
                return _accept(
                    current, FunnelStage.TRANSITION, proposal, "hard_cap_forced_transition"
                )
            wants_to_book = ctx.inbound_intent is InboundIntent.BOOKING_REQUEST
            if ctx.agent_turn_index >= policy.transition_turn or wants_to_book:
                guard = "user_asked_to_book" if wants_to_book else "transition_turn_reached"
                return _accept(current, FunnelStage.TRANSITION, proposal, guard)
            # Below the transition turn the model may only keep delivering value.
            if proposal is FunnelStage.TRANSITION:
                return Decision(
                    from_stage=current,
                    decided_stage=FunnelStage.VALUE,
                    proposed_stage=proposal,
                    accepted=False,
                    trigger=Trigger.LLM_PROPOSAL,
                    guard_result="before_transition_turn",
                    override_reason=(
                        f"proposed TRANSITION at agent turn {ctx.agent_turn_index}, "
                        f"before FUNNEL_TRANSITION_TURN={policy.transition_turn}"
                    ),
                )
            return _accept(
                current, FunnelStage.VALUE, proposal, "value_window", Trigger.LLM_PROPOSAL
            )

        # --- TRANSITION: no acceptance turn (Q2) ----------------------------
        # SOURCE-DEVIATION note: the source's transition message offers the call
        # and asks for the four fields together. There is no separate "user
        # accepts" step, so the guard is "did not decline", not "accepted".
        case FunnelStage.TRANSITION:
            if not ctx.user_responded:
                return _accept(current, FunnelStage.TRANSITION, proposal, "awaiting_response")
            if ctx.user_declined:
                return _accept(current, FunnelStage.TRANSITION, proposal, "offer_declined")
            if ctx.slot_value_supplied or ctx.engaged_with_offer:
                guard = "slot_value_supplied" if ctx.slot_value_supplied else "engaged_with_offer"
                return _accept(
                    current, FunnelStage.SLOT_FILLING, proposal, guard, Trigger.LLM_PROPOSAL
                )
            # Neither accepted nor declined. Hold, and stop asking once the
            # budget is spent: the hard cap forces the ask but does not bound
            # how often it repeats, and asking a disengaged lead on every
            # remaining turn is harassment, not persistence.
            if ctx.transition_reask_count >= policy.transition_reask_limit:
                return _accept(
                    current, FunnelStage.TRANSITION, proposal, "transition_reask_exhausted"
                )
            return _accept(current, FunnelStage.TRANSITION, proposal, "no_engagement_reask")

        case FunnelStage.SLOT_FILLING:
            # Q3: hand off only once the re-ask budget is spent.
            if ctx.phone_attempts > policy.phone_attempt_limit:
                return _accept(
                    current,
                    FunnelStage.HANDED_OFF,
                    proposal,
                    "slot_recovery_exhausted",
                )
            if ctx.slots_all_valid:
                # SYSTEM ONLY. Unreachable by proposal — see decide() step 0.
                return _accept(
                    current, FunnelStage.AWAITING_CONFIRMATION, proposal, "all_slots_valid"
                )
            return _accept(current, FunnelStage.SLOT_FILLING, proposal, "slots_incomplete")

        case FunnelStage.AWAITING_CONFIRMATION:
            if ctx.booking_committed:
                # SYSTEM ONLY, and gated on a committed database row rather than
                # on anything the model said.
                return _accept(current, FunnelStage.BOOKED, proposal, "booking_committed")
            if ctx.slot_unavailable:
                return _accept(current, FunnelStage.SLOT_FILLING, proposal, "slot_taken_re_offer")
            return _accept(
                current, FunnelStage.AWAITING_CONFIRMATION, proposal, "confirmation_in_flight"
            )

        # --- OPTED_OUT: inbound answered, funnel does not resume (Q4) -------
        case FunnelStage.OPTED_OUT:
            override = None
            accepted = proposal is None or proposal is FunnelStage.OPTED_OUT
            if proposal in FUNNEL_STAGES_AFTER_OPT_OUT:
                override = (
                    f"proposed {proposal} after opt-out; a user-initiated message is "
                    "answered minimally but the funnel does not resume"
                )
            return Decision(
                from_stage=current,
                decided_stage=FunnelStage.OPTED_OUT,
                proposed_stage=proposal,
                accepted=accepted,
                trigger=Trigger.SYSTEM,
                guard_result="opted_out_reply_only",
                override_reason=override,
            )

        case _:
            # BOOKED, ABANDONED, HANDED_OFF: nothing further happens here.
            return Decision(
                from_stage=current,
                decided_stage=current,
                proposed_stage=proposal,
                accepted=proposal is None or proposal is current,
                trigger=Trigger.SYSTEM,
                guard_result="terminal",
                override_reason=(
                    None if proposal is None or proposal is current else f"{current} is terminal"
                ),
            )


@dataclass(frozen=True, slots=True)
class TransitionRow:
    """One row of the documented transition table.

    Kept as data so `tests/test_funnel_transitions.py` can assert that every
    documented row is actually reachable, rather than trusting the table and the
    code to agree.
    """

    from_stage: FunnelStage | str
    to_stage: FunnelStage
    guard: str
    trigger: Trigger
    note: str = ""


TRANSITION_TABLE: tuple[TransitionRow, ...] = (
    TransitionRow(
        FunnelStage.NEW,
        FunnelStage.OPENER,
        "first_inbound_intent=none",
        Trigger.LLM_PROPOSAL,
        "default",
    ),
    TransitionRow(
        FunnelStage.NEW,
        FunnelStage.VALUE,
        "first_inbound_intent=direct_question",
        Trigger.LLM_PROPOSAL,
        "Q1",
    ),
    TransitionRow(
        FunnelStage.NEW,
        FunnelStage.TRANSITION,
        "first_inbound_intent=booking_request",
        Trigger.LLM_PROPOSAL,
        "Q1",
    ),
    TransitionRow(
        FunnelStage.OPENER,
        FunnelStage.VALUE,
        "inbound_reply_received",
        Trigger.SYSTEM,
    ),
    TransitionRow(
        FunnelStage.VALUE,
        FunnelStage.VALUE,
        "value_window",
        Trigger.LLM_PROPOSAL,
    ),
    TransitionRow(
        FunnelStage.VALUE,
        FunnelStage.TRANSITION,
        "transition_turn_reached",
        Trigger.SYSTEM,
    ),
    TransitionRow(
        FunnelStage.VALUE,
        FunnelStage.TRANSITION,
        "user_asked_to_book",
        Trigger.SYSTEM,
    ),
    TransitionRow(
        FunnelStage.VALUE,
        FunnelStage.TRANSITION,
        "hard_cap_forced_transition",
        Trigger.SYSTEM,
        "proposal ignored",
    ),
    TransitionRow(
        FunnelStage.TRANSITION,
        FunnelStage.TRANSITION,
        "offer_declined",
        Trigger.SYSTEM,
    ),
    TransitionRow(
        FunnelStage.TRANSITION,
        FunnelStage.TRANSITION,
        "no_engagement_reask",
        Trigger.SYSTEM,
        "Q5",
    ),
    TransitionRow(
        FunnelStage.TRANSITION,
        FunnelStage.TRANSITION,
        "transition_reask_exhausted",
        Trigger.SYSTEM,
        "Q5: stop asking, keep replying",
    ),
    TransitionRow(
        FunnelStage.TRANSITION,
        FunnelStage.SLOT_FILLING,
        "slot_value_supplied",
        Trigger.LLM_PROPOSAL,
        "Q2: no acceptance turn",
    ),
    TransitionRow(
        FunnelStage.TRANSITION,
        FunnelStage.SLOT_FILLING,
        "engaged_with_offer",
        Trigger.LLM_PROPOSAL,
        "Q5",
    ),
    TransitionRow(
        FunnelStage.SLOT_FILLING,
        FunnelStage.SLOT_FILLING,
        "slots_incomplete",
        Trigger.SYSTEM,
    ),
    TransitionRow(
        FunnelStage.SLOT_FILLING,
        FunnelStage.AWAITING_CONFIRMATION,
        "all_slots_valid",
        Trigger.SYSTEM,
        "SYSTEM ONLY",
    ),
    TransitionRow(
        FunnelStage.SLOT_FILLING,
        FunnelStage.HANDED_OFF,
        "slot_recovery_exhausted",
        Trigger.SYSTEM,
        "Q3",
    ),
    TransitionRow(
        FunnelStage.AWAITING_CONFIRMATION,
        FunnelStage.BOOKED,
        "booking_committed",
        Trigger.SYSTEM,
        "SYSTEM ONLY",
    ),
    TransitionRow(
        FunnelStage.AWAITING_CONFIRMATION,
        FunnelStage.SLOT_FILLING,
        "slot_taken_re_offer",
        Trigger.SYSTEM,
    ),
    TransitionRow(
        "any non-terminal",
        FunnelStage.OPTED_OUT,
        "opt_out_detected",
        Trigger.DETECTOR,
    ),
    TransitionRow(
        "any non-terminal",
        FunnelStage.HANDED_OFF,
        "guardrail_escalation",
        Trigger.SYSTEM,
    ),
    TransitionRow(
        "any non-terminal",
        FunnelStage.ABANDONED,
        "idle_expired",
        Trigger.SCHEDULER,
    ),
    TransitionRow(
        FunnelStage.OPTED_OUT,
        FunnelStage.OPTED_OUT,
        "opted_out_reply_only",
        Trigger.SYSTEM,
        "Q4: inbound answered, funnel does not resume",
    ),
)


__all__ = [
    "FUNNEL_STAGES_AFTER_OPT_OUT",
    "TRANSITION_TABLE",
    "Decision",
    "FunnelPolicy",
    "InboundIntent",
    "TransitionContext",
    "TransitionRow",
    "Trigger",
    "allows_outbound_initiation",
    "decide",
]
