from __future__ import annotations

from app.funnel.stages import (
    SYSTEM_ONLY_STAGES,
    TERMINAL_STAGES,
    FunnelStage,
    is_terminal,
)


def test_terminal_stages_are_exactly_the_four_endings():
    assert TERMINAL_STAGES == {
        FunnelStage.BOOKED,
        FunnelStage.ABANDONED,
        FunnelStage.OPTED_OUT,
        FunnelStage.HANDED_OFF,
    }
    assert all(is_terminal(s) for s in TERMINAL_STAGES)
    assert not is_terminal(FunnelStage.SLOT_FILLING)


def test_opted_out_is_terminal_but_is_not_a_failure():
    """A funnel metric that counts opt-outs as failures punishes the system for
    behaving correctly. The distinction is asserted here so a later refactor
    that folds OPTED_OUT into ABANDONED fails loudly."""
    assert is_terminal(FunnelStage.OPTED_OUT)
    assert FunnelStage.OPTED_OUT is not FunnelStage.ABANDONED


def test_booking_stages_are_system_only():
    """The structural guarantee behind "never a hallucinated confirmation": an
    LLM proposal cannot reach either of these. Phase 2's transition guard reads
    this set, so this test is what keeps the guarantee from being edited away
    by accident."""
    assert SYSTEM_ONLY_STAGES == {
        FunnelStage.AWAITING_CONFIRMATION,
        FunnelStage.BOOKED,
    }


def test_stage_values_are_stable_strings():
    """These strings are persisted in `conversations.stage` and in three
    columns of `funnel_transitions`, and they appear literally in the partial
    index predicate in migration 0001. Renaming one is a migration, not a
    refactor."""
    assert FunnelStage.NEW == "NEW"
    assert [s.value for s in FunnelStage] == [
        "NEW",
        "OPENER",
        "VALUE",
        "TRANSITION",
        "SLOT_FILLING",
        "AWAITING_CONFIRMATION",
        "BOOKED",
        "ABANDONED",
        "OPTED_OUT",
        "HANDED_OFF",
    ]
