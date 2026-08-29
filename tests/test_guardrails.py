"""Guardrails on the composed draft.

The split that matters: style violations re-compose, safety violations
interrupt. Collapsing them would either send an unsafe draft after a retry or
put a human in the loop over an emoji.
"""

from __future__ import annotations

import pytest

from app.agents.compose import ComposedReply
from app.config import Settings
from app.guardrails.checks import (
    LINK_ALLOWLIST,
    SEVERITY,
    Violation,
    check,
    effective_confidence,
    safe_template,
)


@pytest.fixture
def settings(settings_kwargs) -> Settings:
    return Settings(**settings_kwargs)


def reply(text: str, confidence: float = 0.9) -> ComposedReply:
    return ComposedReply(text=text, self_confidence=confidence)


# ---------------------------------------------------------------------------
# safety: these interrupt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "haha no im not a bot, real person here",
        "not a bot, just quick at typing",
        "im a real one, promise",
    ],
)
def test_a_draft_denying_being_an_ai_is_a_safety_violation(text, settings):
    """The source's prompt said "NEVER NEVER mention that you are a AI bot".
    This is the last line of defence if that behaviour ever comes back."""
    v = check(reply(text), settings=settings)
    assert Violation.CLAIMS_TO_BE_HUMAN in v.safety_violations
    assert v.must_interrupt


@pytest.mark.parametrize(
    "text",
    [
        "im a certified personal trainer with 16 years experience",
        "ive worked with cancer patients before",
        "in my experience most people see results fast",
        "were certified in nutrition coaching",
    ],
)
def test_a_first_person_credential_claim_is_a_safety_violation(text, settings):
    """D27's measurement in guardrail form: if the corpus's second person leaks
    into the agent's voice, it is a false statement about a real person's
    qualifications."""
    v = check(reply(text), settings=settings)
    assert Violation.FIRST_PERSON_CREDENTIAL in v.safety_violations


@pytest.mark.parametrize(
    "text",
    [
        "Great, you are all booked for Thursday",
        "youre booked in for thursday at 5",
        "all set for friday then",
        "see you on thursday",
    ],
)
def test_a_draft_implying_an_uncommitted_booking_is_stopped(text, settings):
    """The third lock. `ProposableStage` cannot name BOOKED and `decide`
    rejects it, but a composer could still phrase a confirmation — so the text
    is checked too."""
    v = check(reply(text), booking_committed=False, settings=settings)
    assert Violation.UNEARNED_CONFIRMATION in v.safety_violations


def test_offering_to_book_is_not_claiming_a_booking(settings):
    """The precision half. A guardrail that blocked the funnel's own
    transition message would be worse than one that misses an edge case,
    because the TRANSITION turn is supposed to say exactly this."""
    v = check(
        reply("Happy to get you booked in for the free call, whats your name?"),
        settings=settings,
    )
    assert Violation.UNEARNED_CONFIRMATION not in v.safety_violations


def test_a_confirmation_is_allowed_once_the_booking_is_committed(settings):
    v = check(reply("youre booked in for thursday at 5"), booking_committed=True, settings=settings)
    assert Violation.UNEARNED_CONFIRMATION not in v.safety_violations


def test_any_link_is_blocked_because_the_allowlist_is_empty(settings):
    """An agent that can emit URLs is a phishing surface. The allow-list is
    empty on purpose — see decision D22."""
    assert not LINK_ALLOWLIST
    v = check(reply("heres my whatsapp https://wa.link/abc123"), settings=settings)
    assert Violation.UNLISTED_LINK in v.safety_violations


def test_ordinary_text_is_not_mistaken_for_a_link(settings):
    v = check(reply("Thursday at 5 works, whats your number?"), settings=settings)
    assert Violation.UNLISTED_LINK not in v.violations


# ---------------------------------------------------------------------------
# style: these re-compose
# ---------------------------------------------------------------------------


def test_a_too_long_reply_is_a_style_violation_not_a_safety_one(settings):
    v = check(reply(" ".join(["word"] * 60)), settings=settings)
    assert Violation.TOO_LONG in v.style_violations
    assert v.should_recompose
    assert not v.must_interrupt


def test_a_one_word_reply_is_too_short(settings):
    v = check(reply("ok"), settings=settings)
    assert Violation.TOO_SHORT in v.style_violations


def test_a_leading_interjection_is_allowed(settings):
    """The source says "only a single sentence" and its own worked opener is
    three. Two is the enforced cap — see decision D39."""
    v = check(reply("Hey! Whats your main fitness goal right now?"), settings=settings)
    assert Violation.MULTIPLE_SENTENCES not in v.violations


def test_three_sentences_is_still_too_many(settings):
    v = check(reply("Hey. Welcome along. What is your goal?"), settings=settings)
    assert Violation.MULTIPLE_SENTENCES in v.style_violations


def test_adjacent_emoji_are_a_style_violation(settings):
    """The source is explicit that emoji must be distributed rather than
    clustered."""
    v = check(reply("great stuff 🔥🔥 lets get you booked"), settings=settings)
    assert Violation.ADJACENT_EMOJI in v.style_violations


def test_a_missing_disclosure_line_is_a_style_violation(settings):
    v = check(reply("Hey there, whats your goal?"), disclosure_required=True, settings=settings)
    assert Violation.MISSING_DISCLOSURE in v.style_violations


def test_the_disclosure_line_satisfies_the_check(settings):
    from app.agents.compose import DISCLOSURE_LINE

    v = check(
        reply(f"{DISCLOSURE_LINE} Whats your goal?"),
        disclosure_required=True,
        settings=settings,
    )
    assert Violation.MISSING_DISCLOSURE not in v.violations


# ---------------------------------------------------------------------------
# confidence
# ---------------------------------------------------------------------------


def test_confidence_is_reduced_by_what_actually_happened():
    """A self-report alone is not evidence. The penalties correspond to
    observed events, so a reply produced after two repairs and an overridden
    proposal does not get to call itself confident."""
    assert effective_confidence(0.9) == pytest.approx(0.9)
    assert effective_confidence(0.9, repair_attempts=2) == pytest.approx(0.6)
    combined = effective_confidence(0.9, repair_attempts=2, proposal_overridden=True)
    assert combined == pytest.approx(0.5)
    assert effective_confidence(0.9, tool_failures=2) == pytest.approx(0.4)


def test_confidence_never_leaves_the_unit_interval():
    assert effective_confidence(1.0, repair_attempts=99) == 0.0
    assert effective_confidence(5.0) == 1.0


def test_low_effective_confidence_interrupts(settings):
    v = check(reply("Thursday at 5 then, whats your number?", confidence=0.9),
              tool_failures=2, settings=settings)
    assert Violation.LOW_CONFIDENCE in v.safety_violations
    assert v.must_interrupt


def test_a_confident_clean_reply_passes(settings):
    v = check(reply("Nice one 👍 whats your main goal right now?"), settings=settings)
    assert v.ok
    assert not v.must_interrupt and not v.should_recompose


# ---------------------------------------------------------------------------
# the split itself
# ---------------------------------------------------------------------------


def test_every_violation_has_a_severity():
    """A violation with no severity would be silently ignored by both paths."""
    assert set(SEVERITY) == set(Violation)


def test_safety_always_wins_over_style():
    """A draft that is both too long and denies being an AI must interrupt, not
    quietly re-compose and send."""
    from app.config import Settings as S

    settings = S(_env_file=None, gemini_api_key="x")
    v = check(reply(" ".join(["im not a bot"] * 20)), settings=settings)
    assert v.must_interrupt
    assert not v.should_recompose


def test_the_safe_template_says_nothing_that_could_be_wrong():
    text = safe_template()
    assert "Jan" in text
    v = check(reply(text), settings=Settings(_env_file=None, gemini_api_key="x"))
    assert not v.safety_violations
