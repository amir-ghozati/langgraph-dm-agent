from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.funnel.stages import FunnelStage
from evals.schema import Expectation, GoldenConversation
from evals.validate import load_all


def test_every_golden_file_loads():
    """The failure this prevents: writing eight conversations by hand and
    discovering in Phase 3b that the runner cannot read any of them."""
    results = load_all()
    assert len(results) == 8, f"expected 8 golden conversations, found {len(results)}"
    broken = {path.name: r for path, r in results if isinstance(r, str)}
    assert not broken, broken


def test_the_eight_modes_are_each_covered_once():
    modes = sorted(c.mode for _, c in load_all() if isinstance(c, GoldenConversation))
    assert modes == [1, 2, 3, 4, 5, 6, 7, 8]


def test_at_least_a_quarter_probe_ungrounded_questions():
    """Brief v2 section 5: fabrication is the failure that would cost the
    client money, so the set must keep testing for it even as it is edited."""
    convs = [c for _, c in load_all() if isinstance(c, GoldenConversation)]
    ungrounded = [c for c in convs if c.mode in {4, 8}]
    assert len(ungrounded) / len(convs) >= 0.25


def _expect(**overrides):
    base = {
        "final_stage": "SLOT_FILLING",
        "interrupt": {"expected": False},
    }
    return Expectation.model_validate(base | overrides)


def test_worked_example_validates():
    results = dict(load_all())
    conv = next(c for p, c in results.items() if p.name.startswith("smoke-01"))
    assert isinstance(conv, GoldenConversation)
    assert conv.expect is not None
    assert conv.expect.final_stage == "SLOT_FILLING"
    assert "BOOKED" in conv.expect.never_stage
    assert conv.expect.tools.required[0].tool == "resolve_datetime"


def test_scalar_shorthand_binds_to_the_single_parameter():
    """`{phone_error_names: landline}` is what a person actually writes.
    Accepted only where the predicate has exactly one parameter, so it can
    never be ambiguous."""
    from evals.schema import normalise_predicate

    assert normalise_predicate({"phone_error_names": "landline"}) == (
        "phone_error_names",
        {"kind": "landline"},
    )
    assert normalise_predicate("no_price_stated") == ("no_price_stated", {})


def test_shorthand_is_refused_for_a_list_parameter():
    """`set("price")` is five characters, not one topic. Binding a bare string
    to a list parameter fails in a way nobody can read, so it is refused at the
    shorthand."""
    with pytest.raises(ValidationError, match="cannot use the shorthand"):
        _expect(conversation=[{"refuses_without_fabricating": "price"}])


def test_a_list_parameter_given_a_bare_string_is_rejected():
    with pytest.raises(ValidationError, match="must be a list"):
        _expect(conversation=[{"refuses_without_fabricating": {"topics": "price"}}])


def test_an_unknown_phone_error_kind_is_rejected():
    with pytest.raises(ValidationError, match="unknown phone error kind"):
        _expect(turns={1: [{"phone_error_names": "wrong_colour"}]})


def test_never_stage_after_expresses_a_positional_prohibition():
    """smoke-06 visits TRANSITION and SLOT_FILLING legitimately before the
    opt-out. A global never_stage would be false; without the positional form
    the entire Q4 behaviour is unverifiable."""
    e = _expect(
        final_stage="OPTED_OUT",
        stages_visited=["TRANSITION", "SLOT_FILLING", "OPTED_OUT"],
        never_stage_after={"OPTED_OUT": ["TRANSITION", "SLOT_FILLING", "BOOKED"]},
    )
    assert e.never_stage_after[FunnelStage.OPTED_OUT]


def test_a_positional_assertion_with_an_unreachable_pivot_is_rejected():
    """If the pivot is never entered, nothing can violate the rule and the
    assertion always passes -- the same vacuity problem as a turn index past
    the end of the conversation."""
    with pytest.raises(ValidationError, match="vacuous"):
        _expect(never_stage_after={"HANDED_OFF": ["BOOKED"]})


def test_a_positional_assertion_cannot_forbid_its_own_pivot():
    with pytest.raises(ValidationError, match="own pivot"):
        _expect(
            final_stage="OPTED_OUT",
            never_stage_after={"OPTED_OUT": ["OPTED_OUT"]},
        )


def test_final_stage_and_interrupt_are_optional():
    """Three conversations do not turn on either. Omitted means "not
    asserted"."""
    e = Expectation.model_validate({"conversation": ["no_booking_confirmed"]})
    assert e.final_stage is None and e.interrupt is None


def test_judged_predicates_are_separable_from_deterministic_ones():
    """The headline numbers must not all depend on a judge."""
    e = _expect(
        conversation=["no_price_stated", "no_medical_advice_given"],
        turns={1: ["opener_only", "discloses_ai_status"]},
    )
    judged = {p if isinstance(p, str) else next(iter(p)) for p in e.judged_predicates}
    assert judged == {"no_medical_advice_given", "discloses_ai_status"}


def test_a_misspelled_predicate_is_rejected():
    """A predicate name that silently does nothing is a test that always
    passes, which is worse than no test."""
    with pytest.raises(ValidationError, match="unknown predicate"):
        _expect(conversation=["no_price_stateed"])


def test_a_predicate_with_an_unknown_parameter_is_rejected():
    with pytest.raises(ValidationError, match="does not take"):
        _expect(conversation=[{"no_price_stated": {"threshold": 3}}])


def test_an_unknown_ungrounded_topic_is_rejected():
    with pytest.raises(ValidationError, match="ungrounded topic"):
        _expect(conversation=[{"refuses_without_fabricating": {"topics": ["shoe_size"]}}])


def test_a_known_ungrounded_topic_is_accepted():
    e = _expect(conversation=[{"refuses_without_fabricating": {"topics": ["price", "refund"]}}])
    assert e.conversation


def test_final_stage_cannot_also_be_forbidden():
    with pytest.raises(ValidationError, match="never_stage"):
        _expect(never_stage=["SLOT_FILLING"])


def test_a_tool_matcher_needs_exactly_one_key():
    with pytest.raises(ValidationError, match="exactly one key"):
        _expect(
            tools={
                "required": [
                    {"tool": "resolve_datetime", "args": {"tz": {"equals": "UTC", "contains": "U"}}}
                ]
            }
        )


def test_an_assertion_on_a_turn_that_never_happens_is_rejected():
    """A per-turn assertion pointing past the end of the conversation is
    vacuously true and would quietly inflate the pass rate."""
    with pytest.raises(ValidationError, match="only 1 agent turns"):
        GoldenConversation.model_validate(
            {
                "id": "x",
                "mode": 1,
                "title": "t",
                "provenance": "handwritten",
                "context": {"today": "2026-03-10"},
                "turns": [{"inbound": "hi"}, {"agent": "[agent responds]"}],
                "expect": {
                    "final_stage": "OPENER",
                    "interrupt": {"expected": False},
                    "turns": {4: ["opener_only"]},
                },
            }
        )


def test_a_turn_cannot_be_both_sides():
    with pytest.raises(ValidationError, match="exactly one"):
        GoldenConversation.model_validate(
            {
                "id": "x",
                "mode": 1,
                "title": "t",
                "provenance": "handwritten",
                "context": {"today": "2026-03-10"},
                "turns": [{"inbound": "hi", "agent": "hello"}],
            }
        )


# ---------------------------------------------------------------------------
# Burst aggregation against the real golden files
# ---------------------------------------------------------------------------


def test_agent_turns_are_counted_from_bursts_not_from_placeholders():
    """Every file in the set ends on an inbound burst with no `agent:` entry
    after it, so counting placeholders under-counts real agent turns by one in
    all eight — and would reject a correct assertion on the final turn."""
    for _path, conv in load_all():
        assert isinstance(conv, GoldenConversation)
        assert conv.agent_turn_count == len(conv.bursts)
        assert conv.agent_turn_count == conv.scripted_agent_placeholders + 1


def test_the_set_actually_contains_bursts_to_aggregate():
    """If no conversation had a burst, the aggregation rule would be untested
    by the golden set no matter how many unit tests it has."""
    multi = {
        conv.id: [len(b) for b in conv.bursts]
        for _p, conv in load_all()
        if isinstance(conv, GoldenConversation) and any(len(b) > 1 for b in conv.bursts)
    }
    assert len(multi) >= 4, f"expected several conversations with bursts, got {multi}"


def test_no_burst_mixes_an_agent_turn_into_itself():
    for _path, conv in load_all():
        for burst in conv.bursts:
            assert all(t.inbound is not None for t in burst)


# ---------------------------------------------------------------------------
# Severity and coverage
# ---------------------------------------------------------------------------


def test_safety_predicates_span_both_tiers():
    """The point of the severity tag: the most safety-critical assertions are
    judged, so a report that surfaces only deterministic failures in its
    headline hides medical-advice failures."""
    from evals.schema import (
        DETERMINISTIC_PREDICATES,
        JUDGED_PREDICATES,
        SAFETY_PREDICATES,
        severity,
    )

    known = set(DETERMINISTIC_PREDICATES) | set(JUDGED_PREDICATES)
    assert SAFETY_PREDICATES <= known
    assert SAFETY_PREDICATES & set(JUDGED_PREDICATES), "no judged predicate is marked safety"
    assert SAFETY_PREDICATES & set(DETERMINISTIC_PREDICATES)
    assert severity("no_medical_advice_given") == "safety"
    assert severity("single_message") == "quality"


def test_medical_and_fabrication_predicates_are_all_safety_tagged():
    from evals.schema import SAFETY_PREDICATES

    for name in (
        "no_medical_advice_given",
        "defers_to_medical_professional",
        "no_exercise_prescription",
        "no_supplement_recommendation",
        "no_price_stated",
        "refuses_without_fabricating",
        "no_booking_confirmed",
        "no_link_emitted",
    ):
        assert name in SAFETY_PREDICATES, f"{name} is not tagged safety"


def test_coverage_separates_a_decision_from_a_todo():
    """Two failures guarded at once. Omission is invisible without coverage --
    a block asserting almost nothing scores identically to a thorough one. And
    absence alone cannot tell a deliberate omission from an oversight."""
    thin = Expectation.model_validate({"conversation": ["no_booking_confirmed"]})
    cov = thin.coverage()
    assert cov["predicates"] == 1
    assert cov["asserted_fields"] == ["conversation"]
    assert "final_stage" in cov["unexplained"]
    assert cov["deliberately_unasserted"] == []

    explained = Expectation.model_validate(
        {
            "conversation": ["no_booking_confirmed"],
            "unasserted": {
                "interrupt": "guardrail evaluates the draft; a correct interception "
                "would fail expected:false",
            },
        }
    )
    cov = explained.coverage()
    assert cov["deliberately_unasserted"] == ["interrupt"]
    assert "interrupt" not in cov["unexplained"]


def test_an_unasserted_field_must_name_a_real_field():
    with pytest.raises(ValidationError, match="unknown field"):
        _expect(unasserted={"finaal_stage": "typo"})


def test_an_unasserted_field_needs_a_reason():
    with pytest.raises(ValidationError, match="needs a reason"):
        _expect(unasserted={"interrupt": "   "})


def test_a_field_cannot_be_both_asserted_and_deliberately_unasserted():
    """The contradiction survives review because both halves look reasonable
    on their own."""
    with pytest.raises(ValidationError, match="also asserted"):
        _expect(unasserted={"final_stage": "not the point here"})


def test_an_unknown_tool_is_rejected():
    """`forbidden: [commit_bookng]` would otherwise pass forever, since a
    misspelled tool is never called."""
    with pytest.raises(ValidationError, match="unknown tool or sub-agent"):
        _expect(tools={"forbidden": ["commit_bookng"]})


def test_a_sub_agent_is_assertable_like_a_tool():
    """smoke-04 needs to assert the knowledge agent was consulted: refusing is
    the right outcome, but refusing without looking is a different system."""
    e = _expect(tools={"required": [{"tool": "knowledge_agent", "min_calls": 1}]})
    assert e.tools.required[0].tool == "knowledge_agent"


def test_every_predicate_is_in_exactly_one_tier():
    from evals.schema import DETERMINISTIC_PREDICATES, JUDGED_PREDICATES

    overlap = set(DETERMINISTIC_PREDICATES) & set(JUDGED_PREDICATES)
    assert not overlap, f"predicates declared in both tiers: {sorted(overlap)}"
