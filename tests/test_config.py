from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import LLMProviderName, Settings


def test_defaults_are_valid(settings_kwargs):
    s = Settings(**settings_kwargs)
    assert s.llm_provider is LLMProviderName.GEMINI
    assert s.funnel_transition_turn == 5
    assert s.funnel_hard_cap == 7


def test_missing_gemini_key_fails_at_startup(settings_kwargs):
    """The failure this prevents: a key that is only discovered missing on the
    first LLM call, which reads as a model failure rather than a setup one."""
    settings_kwargs["gemini_api_key"] = None
    with pytest.raises(ValidationError, match="GEMINI_API_KEY"):
        Settings(**settings_kwargs)


def test_ollama_provider_needs_no_gemini_key(settings_kwargs):
    settings_kwargs["gemini_api_key"] = None
    settings_kwargs["llm_provider"] = "ollama"
    assert Settings(**settings_kwargs).llm_provider is LLMProviderName.OLLAMA


@pytest.mark.parametrize(
    ("value_max", "transition", "cap"),
    [
        (4, 4, 7),  # transition must be strictly after the value window
        (4, 5, 4),  # hard cap must not precede the transition turn
        (9, 5, 7),  # value window must not overrun the transition turn
    ],
)
def test_incoherent_funnel_timings_are_rejected(settings_kwargs, value_max, transition, cap):
    """The source system contradicted itself on funnel timing -- a header saying
    "maximum of 6 messages" over a body that allows 7. Here an incoherent
    configuration cannot start."""
    with pytest.raises(ValidationError, match="funnel timings"):
        Settings(
            **settings_kwargs,
            funnel_value_turn_max=value_max,
            funnel_transition_turn=transition,
            funnel_hard_cap=cap,
        )


def test_business_hours_must_be_ordered(settings_kwargs):
    with pytest.raises(ValidationError, match="business_hours_start"):
        Settings(**settings_kwargs, business_hours_start="21:00", business_hours_end="09:00")


def test_timezone_must_be_a_real_iana_zone(settings_kwargs):
    with pytest.raises(ValidationError):
        Settings(**settings_kwargs, default_timezone="Europe/Nowhere")


def test_settings_are_frozen(settings_kwargs):
    """Configuration read mid-run must be the configuration the run started
    with, otherwise a metric cannot be attributed to a config."""
    s = Settings(**settings_kwargs)
    with pytest.raises(ValidationError):
        s.funnel_hard_cap = 99
