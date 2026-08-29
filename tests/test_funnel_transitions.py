"""The funnel state machine.

Every test here corresponds to a row of TRANSITION_TABLE or to one of the two
guarantees the module exists to provide.
"""

from __future__ import annotations

import pytest

from app.funnel.stages import SYSTEM_ONLY_STAGES, TERMINAL_STAGES, FunnelStage
from app.funnel.transitions import (
    TRANSITION_TABLE,
    FunnelPolicy,
    InboundIntent,
    TransitionContext,
    Trigger,
    allows_outbound_initiation,
    decide,
)

POLICY = FunnelPolicy(transition_turn=5, hard_cap=7, phone_reask_limit=2)


def ctx(**kw) -> TransitionContext:
    return TransitionContext(**kw)


# ---------------------------------------------------------------------------
# The two guarantees
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("system_only", sorted(SYSTEM_ONLY_STAGES))
@pytest.mark.parametrize(
    "current",
    [FunnelStage.VALUE, FunnelStage.TRANSITION, FunnelStage.SLOT_FILLING],
)
def test_no_proposal_can_reach_a_system_only_stage(current, system_only):
    """The structural guarantee behind "never a hallucinated confirmation". If
    this test ever fails, the model can talk its way into a booked state."""
    d = decide(current, system_only, ctx(agent_turn_index=3), POLICY)
    assert d.decided_stage is not system_only
    assert not d.accepted
    assert "system transition only" in d.override_reason


def test_a_rejected_system_only_proposal_is_still_recorded():
    """proposal_override_rate is only a real metric if the rejected proposal is
    logged rather than discarded."""
    d = decide(FunnelStage.SLOT_FILLING, FunnelStage.BOOKED, ctx(), POLICY)
    assert d.proposed_stage is FunnelStage.BOOKED
    assert d.decided_stage is FunnelStage.SLOT_FILLING


def test_booked_requires_a_committed_booking_not_a_confident_model():
    waiting = FunnelStage.AWAITING_CONFIRMATION
    assert decide(waiting, None, ctx(booking_committed=False), POLICY).decided_stage is waiting
    committed = decide(waiting, None, ctx(booking_committed=True), POLICY)
    assert committed.decided_stage is FunnelStage.BOOKED


# ---------------------------------------------------------------------------
# Q1 — three entry paths out of NEW
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        (InboundIntent.NONE, FunnelStage.OPENER),
        (InboundIntent.DIRECT_QUESTION, FunnelStage.VALUE),
        (InboundIntent.BOOKING_REQUEST, FunnelStage.TRANSITION),
    ],
)
def test_the_opening_message_routes_by_intent(intent, expected):
    d = decide(FunnelStage.NEW, None, ctx(inbound_intent=intent), POLICY)
    assert d.decided_stage is expected
    assert d.guard_result == f"first_inbound_intent={intent}"


def test_a_lead_who_opens_by_asking_to_book_is_not_re_funnelled():
    """smoke-02 and smoke-03 both open with an explicit booking request. Under
    the source's unconditional opener they would be asked for their fitness
    goal instead."""
    d = decide(
        FunnelStage.NEW, None, ctx(inbound_intent=InboundIntent.BOOKING_REQUEST), POLICY
    )
    assert d.decided_stage is FunnelStage.TRANSITION


def test_a_lead_who_opens_with_a_question_gets_an_answer_stage():
    """smoke-04 opens with a pricing question. Answering it with a goal
    question reads as evasive, which is the opposite of what that conversation
    tests."""
    d = decide(FunnelStage.NEW, None, ctx(inbound_intent=InboundIntent.DIRECT_QUESTION), POLICY)
    assert d.decided_stage is FunnelStage.VALUE


def test_an_out_of_scope_question_still_routes_to_value():
    """DIRECT_QUESTION classifies message SHAPE, not corpus coverage. Price is
    out of corpus; the router must not treat "we cannot answer it" as "it is
    not a question", or smoke-04's opening pricing question gets a goal
    question back -- exactly the behaviour the conditional opener removes.
    Refusing is the knowledge agent's job, downstream of this decision."""
    d = decide(FunnelStage.NEW, None, ctx(inbound_intent=InboundIntent.DIRECT_QUESTION), POLICY)
    assert d.decided_stage is FunnelStage.VALUE
    assert d.decided_stage is not FunnelStage.OPENER


def test_opt_out_outranks_guardrail_escalation():
    """Both can fire on the same turn. If handoff won, a lead who has just
    withdrawn would land in a human queue to be followed up -- the opposite of
    honouring the opt-out."""
    d = decide(
        FunnelStage.SLOT_FILLING,
        None,
        ctx(opt_out_detected=True, guardrail_escalation=True),
        POLICY,
    )
    assert d.decided_stage is FunnelStage.OPTED_OUT
    assert d.trigger is Trigger.DETECTOR


def test_opt_out_outranks_idle_expiry():
    d = decide(
        FunnelStage.VALUE, None, ctx(opt_out_detected=True, idle_expired=True), POLICY
    )
    assert d.decided_stage is FunnelStage.OPTED_OUT


def test_escalation_outranks_idle_expiry():
    d = decide(
        FunnelStage.VALUE, None, ctx(guardrail_escalation=True, idle_expired=True), POLICY
    )
    assert d.decided_stage is FunnelStage.HANDED_OFF


# ---------------------------------------------------------------------------
# The agent_turn_index convention
# ---------------------------------------------------------------------------


def test_turn_index_is_one_based_and_names_the_turn_being_produced():
    """An off-by-one here changes behaviour in every conversation. The 5th
    agent reply is the transition ask, so index 4 must still be VALUE and
    index 5 must not."""
    assert decide(FunnelStage.VALUE, None, ctx(agent_turn_index=4), POLICY).decided_stage is (
        FunnelStage.VALUE
    )
    assert decide(FunnelStage.VALUE, None, ctx(agent_turn_index=5), POLICY).decided_stage is (
        FunnelStage.TRANSITION
    )


def test_a_five_turn_conversation_does_reach_the_forced_transition():
    """smoke-07 has exactly five agent turns. Under this convention the forced
    transition fires on the last one, so `never_stage: [TRANSITION]` would be
    a false assertion."""
    d = decide(FunnelStage.VALUE, None, ctx(agent_turn_index=5), POLICY)
    assert d.decided_stage is FunnelStage.TRANSITION
    assert d.guard_result == "transition_turn_reached"


# ---------------------------------------------------------------------------
# VALUE window and the hard cap
# ---------------------------------------------------------------------------


def test_value_stays_in_the_window_before_the_transition_turn():
    d = decide(FunnelStage.VALUE, FunnelStage.VALUE, ctx(agent_turn_index=3), POLICY)
    assert d.decided_stage is FunnelStage.VALUE
    assert d.accepted


def test_an_early_transition_proposal_is_overridden_and_the_reason_recorded():
    d = decide(FunnelStage.VALUE, FunnelStage.TRANSITION, ctx(agent_turn_index=3), POLICY)
    assert d.decided_stage is FunnelStage.VALUE
    assert not d.accepted
    assert "FUNNEL_TRANSITION_TURN=5" in d.override_reason


def test_the_transition_turn_advances_regardless_of_the_proposal():
    d = decide(FunnelStage.VALUE, FunnelStage.VALUE, ctx(agent_turn_index=5), POLICY)
    assert d.decided_stage is FunnelStage.TRANSITION
    assert d.guard_result == "transition_turn_reached"
    assert not d.accepted


def test_asking_to_book_mid_value_advances_early():
    d = decide(
        FunnelStage.VALUE,
        None,
        ctx(agent_turn_index=2, inbound_intent=InboundIntent.BOOKING_REQUEST),
        POLICY,
    )
    assert d.decided_stage is FunnelStage.TRANSITION
    assert d.guard_result == "user_asked_to_book"


def test_the_hard_cap_forces_the_ask_and_ignores_the_proposal():
    d = decide(FunnelStage.VALUE, FunnelStage.VALUE, ctx(agent_turn_index=7), POLICY)
    assert d.decided_stage is FunnelStage.TRANSITION
    assert d.guard_result == "hard_cap_forced_transition"


# ---------------------------------------------------------------------------
# Q2 — no acceptance turn between the offer and slot collection
# ---------------------------------------------------------------------------


def test_supplying_a_slot_value_begins_slot_filling_in_one_turn():
    """Q2 preserved: the source's transition message offers the call and asks
    for the fields together, so "yes thursday 3pm" needs no separate acceptance
    step."""
    d = decide(FunnelStage.TRANSITION, None, ctx(slot_value_supplied=True), POLICY)
    assert d.decided_stage is FunnelStage.SLOT_FILLING
    assert d.guard_result == "slot_value_supplied"


def test_affirmative_engagement_also_begins_slot_filling():
    d = decide(FunnelStage.TRANSITION, None, ctx(engaged_with_offer=True), POLICY)
    assert d.decided_stage is FunnelStage.SLOT_FILLING
    assert d.guard_result == "engaged_with_offer"


def test_ignoring_the_offer_is_not_consent_to_collect_a_phone_number():
    """Q5. "did not decline" read a reply that ignored the offer entirely as
    consent to start slot collection. smoke-07's final rambling message is
    exactly that: it declines nothing and engages with nothing."""
    d = decide(FunnelStage.TRANSITION, None, ctx(user_declined=False), POLICY)
    assert d.decided_stage is FunnelStage.TRANSITION
    assert d.guard_result == "no_engagement_reask"


def test_the_transition_re_ask_is_bounded():
    """The hard cap forces VALUE -> TRANSITION but does not bound how often
    TRANSITION asks. Without this a permanently disengaged lead receives the
    consultation ask on every remaining turn."""
    within = decide(FunnelStage.TRANSITION, None, ctx(transition_reask_count=1), POLICY)
    assert within.guard_result == "no_engagement_reask"

    spent = decide(FunnelStage.TRANSITION, None, ctx(transition_reask_count=2), POLICY)
    assert spent.decided_stage is FunnelStage.TRANSITION
    assert spent.guard_result == "transition_reask_exhausted"


def test_an_exhausted_re_ask_budget_does_not_create_a_terminal_state():
    """The agent stops asking and keeps replying. Handing off or abandoning a
    lead who is merely chatty would be worse than the repeated ask."""
    d = decide(FunnelStage.TRANSITION, None, ctx(transition_reask_count=9), POLICY)
    assert d.decided_stage is FunnelStage.TRANSITION
    assert d.decided_stage not in TERMINAL_STAGES


def test_engagement_still_wins_after_the_re_ask_budget_is_spent():
    """A lead who goes quiet for three turns and then says "ok thursday" must
    still be able to book."""
    d = decide(
        FunnelStage.TRANSITION,
        None,
        ctx(transition_reask_count=9, slot_value_supplied=True),
        POLICY,
    )
    assert d.decided_stage is FunnelStage.SLOT_FILLING


def test_a_decline_still_outranks_engagement_signals():
    d = decide(
        FunnelStage.TRANSITION,
        None,
        ctx(user_declined=True, engaged_with_offer=True),
        POLICY,
    )
    assert d.decided_stage is FunnelStage.TRANSITION
    assert d.guard_result == "offer_declined"


def test_the_transition_re_ask_budget_is_configurable():
    strict = FunnelPolicy(transition_reask_limit=0)
    d = decide(FunnelStage.TRANSITION, None, ctx(transition_reask_count=0), strict)
    assert d.guard_result == "transition_reask_exhausted"


def test_a_declined_offer_holds_at_transition():
    d = decide(FunnelStage.TRANSITION, None, ctx(user_declined=True), POLICY)
    assert d.decided_stage is FunnelStage.TRANSITION
    assert d.guard_result == "offer_declined"


def test_no_response_yet_holds_at_transition():
    d = decide(FunnelStage.TRANSITION, None, ctx(user_responded=False), POLICY)
    assert d.decided_stage is FunnelStage.TRANSITION


# ---------------------------------------------------------------------------
# Q3 — the phone re-ask budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("attempts", [1, 2, 3])
def test_the_first_three_phone_attempts_stay_in_slot_filling(attempts):
    """smoke-03 supplies three invalid numbers. Under the previous contract —
    one re-ask then handoff — the third was never processed and the
    conversation ended before testing what it was written to test."""
    d = decide(FunnelStage.SLOT_FILLING, None, ctx(phone_attempts=attempts), POLICY)
    assert d.decided_stage is FunnelStage.SLOT_FILLING


def test_a_fourth_phone_attempt_hands_off():
    d = decide(FunnelStage.SLOT_FILLING, None, ctx(phone_attempts=4), POLICY)
    assert d.decided_stage is FunnelStage.HANDED_OFF
    assert d.guard_result == "slot_recovery_exhausted"


def test_the_re_ask_budget_is_configurable():
    strict = FunnelPolicy(phone_reask_limit=0)
    assert decide(FunnelStage.SLOT_FILLING, None, ctx(phone_attempts=1), strict).decided_stage is (
        FunnelStage.SLOT_FILLING
    )
    assert decide(FunnelStage.SLOT_FILLING, None, ctx(phone_attempts=2), strict).decided_stage is (
        FunnelStage.HANDED_OFF
    )


def test_handoff_outranks_complete_slots():
    """Exhausted recovery wins even if the slots later look valid — otherwise a
    conversation could hand off and book in the same turn."""
    d = decide(
        FunnelStage.SLOT_FILLING, None, ctx(phone_attempts=9, slots_all_valid=True), POLICY
    )
    assert d.decided_stage is FunnelStage.HANDED_OFF


def test_all_slots_valid_moves_to_awaiting_confirmation():
    d = decide(FunnelStage.SLOT_FILLING, None, ctx(slots_all_valid=True), POLICY)
    assert d.decided_stage is FunnelStage.AWAITING_CONFIRMATION
    assert d.trigger is Trigger.SYSTEM


def test_an_unavailable_slot_returns_to_slot_filling():
    d = decide(
        FunnelStage.AWAITING_CONFIRMATION, None, ctx(slot_unavailable=True), POLICY
    )
    assert d.decided_stage is FunnelStage.SLOT_FILLING
    assert d.guard_result == "slot_taken_re_offer"


# ---------------------------------------------------------------------------
# Q4 — opt-out: inbound answered, funnel does not resume
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "current",
    [
        FunnelStage.OPENER,
        FunnelStage.VALUE,
        FunnelStage.TRANSITION,
        FunnelStage.SLOT_FILLING,
        FunnelStage.AWAITING_CONFIRMATION,
    ],
)
def test_opt_out_is_detected_from_any_non_terminal_stage(current):
    d = decide(current, None, ctx(opt_out_detected=True), POLICY)
    assert d.decided_stage is FunnelStage.OPTED_OUT
    assert d.trigger is Trigger.DETECTOR


def test_a_user_initiated_message_after_opt_out_is_still_handled():
    """smoke-06: the customer opts out, then asks about shin splints. Refusing
    to respond is not respecting an opt-out, it is ignoring someone."""
    d = decide(FunnelStage.OPTED_OUT, None, ctx(), POLICY)
    assert d.decided_stage is FunnelStage.OPTED_OUT
    assert d.guard_result == "opted_out_reply_only"
    assert d.override_reason is None


@pytest.mark.parametrize(
    "proposal",
    [
        FunnelStage.TRANSITION,
        FunnelStage.SLOT_FILLING,
        FunnelStage.AWAITING_CONFIRMATION,
        FunnelStage.BOOKED,
    ],
)
def test_the_funnel_never_resumes_after_opt_out(proposal):
    d = decide(FunnelStage.OPTED_OUT, proposal, ctx(), POLICY)
    assert d.decided_stage is FunnelStage.OPTED_OUT
    assert not d.accepted


def test_opt_out_blocks_business_initiated_messaging_but_not_replies():
    assert not allows_outbound_initiation(FunnelStage.OPTED_OUT)
    assert allows_outbound_initiation(FunnelStage.SLOT_FILLING)


def test_opt_out_outranks_the_hard_cap():
    """Both fire on the same turn in a plausible conversation. The opt-out
    must win, or a lead who withdraws at turn 7 gets pitched anyway."""
    d = decide(
        FunnelStage.VALUE, None, ctx(agent_turn_index=7, opt_out_detected=True), POLICY
    )
    assert d.decided_stage is FunnelStage.OPTED_OUT


# ---------------------------------------------------------------------------
# Terminal behaviour and table coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("terminal", sorted(TERMINAL_STAGES - {FunnelStage.OPTED_OUT}))
def test_terminal_stages_do_not_move(terminal):
    d = decide(terminal, FunnelStage.VALUE, ctx(agent_turn_index=2), POLICY)
    assert d.decided_stage is terminal
    assert not d.accepted


@pytest.mark.parametrize("terminal", sorted(TERMINAL_STAGES))
def test_pre_emptive_signals_do_not_reopen_a_terminal_conversation(terminal):
    d = decide(terminal, None, ctx(idle_expired=True, guardrail_escalation=True), POLICY)
    assert d.decided_stage is terminal


def test_guardrail_escalation_hands_off():
    d = decide(FunnelStage.VALUE, None, ctx(guardrail_escalation=True), POLICY)
    assert d.decided_stage is FunnelStage.HANDED_OFF


def test_idle_expiry_abandons():
    d = decide(FunnelStage.SLOT_FILLING, None, ctx(idle_expired=True), POLICY)
    assert d.decided_stage is FunnelStage.ABANDONED
    assert d.trigger is Trigger.SCHEDULER


def test_opener_advances_on_any_reply():
    assert decide(FunnelStage.OPENER, None, ctx(), POLICY).decided_stage is FunnelStage.VALUE


def test_every_documented_row_is_reachable():
    """The table in TRANSITION_TABLE is what ARCHITECTURE.md renders. A row that
    no input can produce is documentation that lies."""
    produced: set[tuple[str, str]] = set()
    stages = [s for s in FunnelStage if s not in TERMINAL_STAGES] + [FunnelStage.OPTED_OUT]
    contexts = [
        ctx(inbound_intent=i) for i in InboundIntent
    ] + [
        ctx(agent_turn_index=3),
        ctx(agent_turn_index=5),
        ctx(agent_turn_index=7),
        ctx(agent_turn_index=2, inbound_intent=InboundIntent.BOOKING_REQUEST),
        ctx(user_declined=True),
        ctx(user_responded=False),
        ctx(slot_value_supplied=True),
        ctx(engaged_with_offer=True),
        ctx(transition_reask_count=9),
        ctx(slots_all_valid=True),
        ctx(phone_attempts=4),
        ctx(booking_committed=True),
        ctx(slot_unavailable=True),
        ctx(opt_out_detected=True),
        ctx(guardrail_escalation=True),
        ctx(idle_expired=True),
    ]
    for stage in stages:
        for c in contexts:
            d = decide(stage, None, c, POLICY)
            produced.add((str(d.from_stage), d.guard_result))

    documented = {
        (str(row.from_stage), row.guard)
        for row in TRANSITION_TABLE
        if row.from_stage != "any non-terminal"
    }
    unreachable = documented - produced
    assert not unreachable, f"documented but unreachable: {sorted(unreachable)}"


def test_the_table_covers_every_stage_that_can_move():
    documented_sources = {str(r.from_stage) for r in TRANSITION_TABLE}
    for stage in FunnelStage:
        if stage in TERMINAL_STAGES and stage is not FunnelStage.OPTED_OUT:
            continue
        assert str(stage) in documented_sources, f"{stage} has no documented transition"


def test_policy_is_built_from_settings(settings_kwargs):
    from app.config import Settings

    policy = FunnelPolicy.from_settings(Settings(**settings_kwargs))
    assert policy.transition_turn == 5
    assert policy.hard_cap == 7
    assert policy.phone_attempt_limit == 3
