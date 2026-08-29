"""Application settings.

Everything that changes behaviour is here rather than buried in a prompt. That
is deliberate: the source system encoded its funnel timings inside prompt text,
where a change is invisible and untestable.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProviderName(StrEnum):
    GEMINI = "gemini"
    OLLAMA = "ollama"
    SCRIPTED = "scripted"
    """Deterministic, rule-based. Lets the repo run with no API key so the
    quickstart works from a clean clone, and makes the restart demonstration
    reproducible. No published number comes from it — the eval runner refuses
    it. See decision D37."""


class SchemaMode(StrEnum):
    """How structured output is requested from the provider.

    NATIVE  — the provider constrains generation to the schema itself
              (Gemini `responseSchema`, Ollama `format`).
    PROMPT  — the schema is described in the prompt and the reply is parsed and
              repaired. Portable to any provider, including ones with no
              structured-output support at all.

    Both are supported and both are measured. See decision D5.
    """

    NATIVE = "native"
    PROMPT = "prompt"


class EligibilityPolicyName(StrEnum):
    ALLOW_ALL = "allow_all"
    REQUIRE_LEAD_RECORD = "require_lead_record"


class DisclosureMode(StrEnum):
    NONE = "none"
    ON_FIRST_CONTACT = "on_first_contact"
    EVERY_MESSAGE = "every_message"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- database ---------------------------------------------------------
    database_url: str = "postgresql+psycopg://muster:muster@localhost:5432/muster"

    # --- LLM --------------------------------------------------------------
    llm_provider: LLMProviderName = LLMProviderName.GEMINI
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    schema_mode: SchemaMode = SchemaMode.NATIVE

    llm_max_concurrency: int = Field(default=4, ge=1, le=64)
    llm_max_attempts: int = Field(default=5, ge=1, le=10)
    llm_backoff_base_seconds: float = Field(default=1.0, gt=0)
    llm_backoff_max_seconds: float = Field(default=60.0, gt=0)
    llm_timeout_seconds: float = Field(default=120.0, gt=0)

    # --- funnel -----------------------------------------------------------
    funnel_value_turn_min: int = Field(default=2, ge=1)
    funnel_value_turn_max: int = Field(default=4, ge=1)
    funnel_transition_turn: int = Field(default=5, ge=1)
    funnel_hard_cap: int = Field(default=7, ge=1)
    conversation_idle_hours: int = Field(default=24, ge=1)
    # Re-asks allowed after the first invalid phone number, so
    # 1 + phone_reask_limit attempts before human handoff. One re-ask is not
    # a recovery ladder, and handing off on every typo makes the interrupt
    # queue useless for the cases that need a human.
    phone_reask_limit: int = Field(default=2, ge=0, le=5)

    # How many times TRANSITION may re-ask a lead who neither accepts nor
    # declines. The hard cap forces VALUE -> TRANSITION but does not bound
    # how often TRANSITION asks, so without this a permanently disengaged
    # lead gets the consultation ask on every remaining turn.
    transition_reask_limit: int = Field(default=2, ge=0, le=5)

    # --- burst aggregation ------------------------------------------------
    # Consecutive inbound messages with no reply between them are one turn.
    # Live channels debounce on this window; scripted sources (the eval
    # runner) group structurally and ignore it.
    burst_window_seconds: float = Field(default=5.0, ge=0.0, le=120.0)

    # --- booking ----------------------------------------------------------
    default_timezone: str = "Europe/Berlin"
    default_phone_region: str = "DE"
    business_hours_start: dt.time = dt.time(9, 0)
    business_hours_end: dt.time = dt.time(21, 0)
    session_minutes: int = Field(default=30, ge=5)
    min_lead_minutes: int = Field(default=120, ge=0)
    booking_horizon_days: int = Field(default=14, ge=1)

    # --- policy -----------------------------------------------------------
    eligibility_policy: EligibilityPolicyName = EligibilityPolicyName.REQUIRE_LEAD_RECORD
    disclosure_mode: DisclosureMode = DisclosureMode.ON_FIRST_CONTACT
    require_booking_approval: bool = True

    # RequireLeadRecord drops senders with no lead record. In the CLI there
    # is no lead-magnet import to create one, so an operator demo would drop
    # its own first message. Default True keeps the CLI usable; the eval
    # runner and any real channel set it False. Every drop is still logged.
    allow_unknown_first_contact: bool = True

    # --- guardrails -------------------------------------------------------
    # Style rules taken from the source prompt, enforced deterministically
    # rather than trusted to the model.
    reply_min_words: int = Field(default=3, ge=1)
    reply_max_words: int = Field(default=40, ge=5)

    # The source says "only a single sentence", and its own worked opener is
    # "Hey! Welcome. If you were to focus on one main fitness goal, what would
    # it be?" -- three sentences. A leading interjection is normal DM style, so
    # the enforced cap is 2 rather than the stated 1. See decision D39.
    reply_max_sentences: int = Field(default=2, ge=1, le=4)
    guardrail_recompose_limit: int = Field(default=2, ge=0, le=5)

    # Below this effective confidence the turn goes to a human. Effective, not
    # self-reported: see app/guardrails/checks.effective_confidence.
    confidence_threshold: float = Field(default=0.45, ge=0.0, le=1.0)

    # --- memory -----------------------------------------------------------
    k_supervisor: int = Field(default=8, ge=0)
    k_strategy: int = Field(default=6, ge=0)
    k_knowledge: int = Field(default=2, ge=0)

    # --- logging ----------------------------------------------------------
    log_level: str = "INFO"
    log_format: str = "console"

    @field_validator("default_timezone")
    @classmethod
    def _tz_must_be_real(cls, v: str) -> str:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"{v!r} is not an IANA timezone") from exc
        return v

    @model_validator(mode="after")
    def _funnel_timings_are_ordered(self) -> Settings:
        if not (
            self.funnel_value_turn_min
            <= self.funnel_value_turn_max
            < self.funnel_transition_turn
            <= self.funnel_hard_cap
        ):
            raise ValueError(
                "funnel timings must satisfy "
                "value_min <= value_max < transition_turn <= hard_cap; got "
                f"{self.funnel_value_turn_min}, {self.funnel_value_turn_max}, "
                f"{self.funnel_transition_turn}, {self.funnel_hard_cap}"
            )
        return self

    @model_validator(mode="after")
    def _business_hours_are_ordered(self) -> Settings:
        if self.business_hours_start >= self.business_hours_end:
            raise ValueError("business_hours_start must be before business_hours_end")
        return self

    @model_validator(mode="after")
    def _provider_credentials_present(self) -> Settings:
        """Fail at startup, loudly, rather than on the first LLM call.

        A missing key that only surfaces mid-conversation looks like a model
        failure and wastes the time it takes to find out otherwise.
        """
        if self.llm_provider is LLMProviderName.GEMINI and not self.gemini_api_key:
            raise ValueError(
                "LLM_PROVIDER=gemini but GEMINI_API_KEY is not set. "
                "Copy .env.example to .env and fill it in."
            )
        return self

    @property
    def sync_database_url(self) -> str:
        """Alembic runs synchronously; psycopg3 serves both, so the URL is shared."""
        return self.database_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
