"""The golden-set file format.

This is the shape only. The conversations and their assertions are written by
hand; this module exists so a typo in a predicate name fails at
`python -m evals.validate` rather than at 3am in Phase 3b.

The `turns:` half of the format is fixed by the generation prompt and is
reproduced here unchanged. `expect:` is the part added for the runner.

Two design properties worth defending:

* Predicates are split DETERMINISTIC / JUDGED. Deterministic ones are parsing,
  state lookups and database queries -- they cost nothing, never flake, and
  produce the headline numbers. Judged ones need an LLM call and are reported
  separately, because a report whose every metric depends on a judge is
  measuring the judge as much as the system.
* Assertions that cannot fail are rejected at validation time. A per-turn
  assertion pointing past the end of a conversation, or a positional assertion
  whose pivot stage is never reached, is vacuously true and silently inflates
  the pass rate.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.funnel.stages import FunnelStage

# ---------------------------------------------------------------------------
# Predicate vocabulary — closed, on purpose
# ---------------------------------------------------------------------------

DETERMINISTIC_PREDICATES: dict[str, set[str]] = {
    # name -> allowed parameter keys (empty set = takes no parameters)
    # --- shape and pacing
    "opener_only": set(),
    "single_message": set(),
    "max_words": {"n"},
    "asks_clarifying_question": {"about"},
    "mentions_consultation_call": set(),
    "does_not_mention_consultation_call": set(),
    "no_consultation_offer": set(),
    "no_slot_request": set(),
    # --- things that must never appear in the text
    "no_price_stated": set(),
    "no_contract_terms_stated": set(),
    "no_refund_policy_stated": set(),
    "no_capacity_stated": set(),
    "no_booking_confirmed": set(),
    "no_link_emitted": set(),
    "contains_disclosure": set(),
    # --- checkable against tool results or persisted state
    "no_slot_outside_business_hours": set(),
    "offers_alternative_slot": set(),
    "phone_error_names": {"kind"},
    "phone_corrections_are_distinct": {"min_distinct"},
    "stage_not_advanced_past": {"stage"},
}

JUDGED_PREDICATES: dict[str, set[str]] = {
    # --- fabrication
    "refuses_without_fabricating": {"topics"},
    "no_out_of_scope_claim": set(),
    "answers_from_corpus_only": set(),
    "no_hallucinated_context": set(),
    # --- medical scope
    "no_medical_advice_given": set(),
    "defers_to_medical_professional": set(),
    "no_exercise_prescription": set(),
    "no_supplement_recommendation": set(),
    # --- honesty about what it is
    "discloses_ai_status": set(),
    "does_not_claim_to_be_human": set(),
    "no_first_person_credential_claim": set(),
    # --- memory and opt-out
    "answers_from_conversation_history": {"turn"},
    "acknowledges_optout": set(),
    "no_funnel_repitch_after_optout": set(),
    # --- the funnel ask, as behaviour rather than as a keyword
    "asks_for_consultation": set(),
}

ALL_PREDICATES = {**DETERMINISTIC_PREDICATES, **JUDGED_PREDICATES}

# Predicates whose failure is a safety or commercial-harm failure rather than a
# quality one. These surface at the TOP of the report and fail the run,
# regardless of tier.
#
# The reason this exists: the most safety-critical assertions in the set --
# medical scope, fabrication -- are judged rather than parsed. Without a
# severity tag, a report reads "0 deterministic failures" in its headline with
# a medical-advice failure buried in a lower section, which is worse than no
# report. Severity cuts across the deterministic/judged split on purpose.
SAFETY_PREDICATES: frozenset[str] = frozenset(
    {
        # medical harm
        "no_medical_advice_given",
        "defers_to_medical_professional",
        "no_exercise_prescription",
        "no_supplement_recommendation",
        # commercial fabrication -- the failure that would cost the client money
        "no_price_stated",
        "no_contract_terms_stated",
        "no_refund_policy_stated",
        "no_capacity_stated",
        "refuses_without_fabricating",
        "no_out_of_scope_claim",
        # a confirmation the system has not earned
        "no_booking_confirmed",
        # phishing surface
        "no_link_emitted",
        # disclosure and consent
        "discloses_ai_status",
        "does_not_claim_to_be_human",
        # claiming a real person's credentials as your own. Worse than the
        # medical-advice case: that is bad guidance, this is a false statement
        # about a identifiable human being's qualifications and experience.
        "no_first_person_credential_claim",
        "no_funnel_repitch_after_optout",
    }
)


def severity(name: str) -> str:
    return "safety" if name in SAFETY_PREDICATES else "quality"

# Topics the knowledge corpus deliberately cannot answer. A predicate naming a
# topic outside this list is almost certainly a typo, so it is rejected.
UNGROUNDED_TOPICS = {
    "price",
    "programme_length",
    "contract",
    "notice_period",
    "capacity",
    "waiting_list",
    "refund",
    "guarantee",
    "trial",
    "named_medical_condition",
    "muscle_gain_over_40",
    "bulkiness",
    "partner_offering",
    "location",
    "in_person",
    "languages",
    "gym_partnership",
    "response_time",
}

AMBIGUITY_SUBJECTS = {"day", "date", "time_of_day", "hour", "timezone", "name", "phone"}

# Parameters whose value is a list. They are excluded from the scalar shorthand
# and type-checked explicitly: `set("price")` is a set of five characters, not
# a set containing "price", so a bare string here fails in a way that is very
# hard to read in an error message.
LIST_PARAMS = {"topics", "about"}

# Everything the runner can observe being invoked during a turn. Closed for the
# same reason the predicate vocabulary is: `forbidden: [commit_bookng]` would
# otherwise pass forever, since a misspelled tool is never called.
TOOLS = {
    "resolve_datetime",
    "check_availability",
    "list_free_slots",
    "commit_booking",
}

# Sub-agents are not tools -- they are subgraph invocations the supervisor
# requests on the TurnPlan rather than calls the tool_runner dispatches. From
# the eval's point of view the question is identical ("was it invoked, with
# what argument"), so they share one vocabulary and one assertion shape. What
# is actually asserted is the plan field, which is the artifact D1 exists to
# make assertable.
SUBAGENTS = {
    "knowledge_agent",
    "strategy_agent",
}

INVOCABLE = TOOLS | SUBAGENTS

PHONE_ERROR_KINDS = {
    "landline",
    "missing_country_code",
    "non_numeric",
    "too_short",
    "too_long",
    "unparseable",
}


class ArgMatcher(BaseModel):
    """How a single tool argument is checked. Exactly one key must be set."""

    model_config = ConfigDict(extra="forbid")

    equals: Any = None
    contains: str | None = None
    one_of: list[Any] | None = None
    present: bool | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> ArgMatcher:
        set_keys = [
            k for k in ("equals", "contains", "one_of", "present") if getattr(self, k) is not None
        ]
        if len(set_keys) != 1:
            raise ValueError(f"an argument matcher needs exactly one key, got {set_keys or 'none'}")
        return self


class ToolExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    min_calls: int = Field(default=1, ge=1)

    @field_validator("tool")
    @classmethod
    def _known(cls, v: str) -> str:
        if v not in INVOCABLE:
            raise ValueError(f"unknown tool or sub-agent {v!r}; known: {sorted(INVOCABLE)}")
        return v

    max_calls: int | None = Field(default=None, ge=1)
    args: dict[str, ArgMatcher] = Field(default_factory=dict)
    result: dict[str, ArgMatcher] = Field(default_factory=dict)


class ToolsExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required: list[ToolExpectation] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)

    @field_validator("forbidden")
    @classmethod
    def _known(cls, v: list[str]) -> list[str]:
        unknown = sorted(set(v) - INVOCABLE)
        if unknown:
            raise ValueError(f"unknown tool or sub-agent {unknown}; known: {sorted(INVOCABLE)}")
        return v


class InterruptExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected: bool
    reasons: list[str] = Field(default_factory=list)


# A predicate is written as any of:
#     no_price_stated                       -- no parameters
#     {max_words: {n: 40}}                  -- explicit parameters
#     {phone_error_names: landline}         -- shorthand, single-parameter only
Predicate = str | dict[str, Any]


def normalise_predicate(p: Predicate) -> tuple[str, dict[str, Any]]:
    """Return (name, params), accepting the scalar shorthand.

    The shorthand exists because `{phone_error_names: landline}` is what a
    human actually writes, and forcing `{phone_error_names: {kind: landline}}`
    buys nothing. It is only accepted for predicates with exactly one
    parameter, so it can never be ambiguous.
    """
    if isinstance(p, str):
        return p, {}
    if len(p) != 1:
        raise ValueError(f"a predicate mapping must have exactly one key, got {list(p)}")
    name, raw = next(iter(p.items()))
    if raw is None:
        return name, {}
    if isinstance(raw, dict):
        return name, raw
    allowed = ALL_PREDICATES.get(name, set())
    if len(allowed) != 1:
        raise ValueError(
            f"predicate {name!r} takes {len(allowed)} parameters, so it cannot use the "
            f"shorthand form; write it as a mapping"
        )
    only = next(iter(allowed))
    if only in LIST_PARAMS:
        raise ValueError(
            f"predicate {name!r} takes a list for {only!r}, so it cannot use the shorthand "
            f"form; write it as {{{name}: {{{only}: [...]}}}}"
        )
    return name, {only: raw}


def validate_predicate(p: Predicate) -> None:
    name, params = normalise_predicate(p)

    if name not in ALL_PREDICATES:
        close = sorted(k for k in ALL_PREDICATES if k.startswith(name[:5]) or name[:5] in k)
        hint = f"; did you mean {close}?" if close else ""
        raise ValueError(f"unknown predicate {name!r}{hint}")

    allowed = ALL_PREDICATES[name]
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(
            f"predicate {name!r} does not take {sorted(unknown)}; "
            f"allowed: {sorted(allowed) or 'no parameters'}"
        )

    for key in allowed & LIST_PARAMS:
        value = params.get(key)
        if value is not None and not isinstance(value, list):
            raise ValueError(
                f"predicate {name!r} parameter {key!r} must be a list, got {type(value).__name__}"
            )

    if name == "refuses_without_fabricating":
        bad = set(params.get("topics") or []) - UNGROUNDED_TOPICS
        if bad:
            raise ValueError(f"unknown ungrounded topic(s) {sorted(bad)}")
    if name == "asks_clarifying_question":
        bad = set(params.get("about") or []) - AMBIGUITY_SUBJECTS
        if bad:
            raise ValueError(f"unknown ambiguity subject(s) {sorted(bad)}")
    if name == "phone_error_names":
        kind = params.get("kind")
        if kind is not None and kind not in PHONE_ERROR_KINDS:
            raise ValueError(
                f"unknown phone error kind {kind!r}; expected one of {sorted(PHONE_ERROR_KINDS)}"
            )
    if name == "stage_not_advanced_past":
        stage = params.get("stage")
        if stage is not None and stage not in set(FunnelStage):
            raise ValueError(f"{stage!r} is not a funnel stage")


def is_judged(p: Predicate) -> bool:
    return normalise_predicate(p)[0] in JUDGED_PREDICATES


# Fields an `unasserted:` block may name. Anything else is a typo.
ASSERTABLE_FIELDS = frozenset(
    {
        "final_stage",
        "stages_visited",
        "never_stage",
        "never_stage_after",
        "tools.required",
        "tools.forbidden",
        "interrupt",
        "conversation",
        "turns",
    }
)


class Expectation(BaseModel):
    """The `expect:` block."""

    model_config = ConfigDict(extra="forbid")

    final_stage: FunnelStage | None = None
    """Optional. Omit when the conversation's ending stage is not the point;
    omitting it means "not asserted", not "any stage is fine"."""

    stages_visited: list[FunnelStage] = Field(default_factory=list)
    """Ordered subsequence that must appear in the transition log. Not the
    complete path -- stages may repeat and others may appear between these."""

    never_stage: list[FunnelStage] = Field(default_factory=list)
    """Stages that must never be entered at any point."""

    never_stage_after: dict[FunnelStage, list[FunnelStage]] = Field(default_factory=dict)
    """Positional: once the key stage is entered, none of its listed stages may
    be entered again.

    `never_stage` cannot express this. `smoke-06` legitimately visits
    `TRANSITION` and `SLOT_FILLING` *before* the opt-out and must never
    re-enter them *after* it, so a global prohibition would be false and its
    absence would leave the whole Q4 behaviour unverified. The same shape
    applies to `HANDED_OFF`.
    """

    tools: ToolsExpectation = Field(default_factory=ToolsExpectation)
    interrupt: InterruptExpectation | None = None
    """Optional, same meaning as `final_stage`: omitted is "not asserted"."""

    conversation: list[Predicate] = Field(default_factory=list)
    """Checked against every agent reply in the conversation."""

    turns: dict[int, list[Predicate]] = Field(default_factory=dict)
    """Checked against one agent reply, keyed by 1-based agent turn index."""

    unasserted: dict[str, str] = Field(default_factory=dict)
    """Fields deliberately left unasserted, each with the reason why.

    Coverage cannot otherwise distinguish a decision from a TODO. `smoke-08`
    omitting `interrupt` is a decision -- the guardrail evaluates the draft, so
    `expected: false` would penalise a correct interception. `smoke-03`
    omitting `tools.required` was an oversight. Both render as absence.

    With this block, coverage reports asserted / deliberately-unasserted /
    unexplained, and only the last number is a worklist.
    """

    @field_validator("unasserted")
    @classmethod
    def _unasserted_names_real_fields(cls, v: dict[str, str]) -> dict[str, str]:
        unknown = sorted(set(v) - ASSERTABLE_FIELDS)
        if unknown:
            raise ValueError(
                f"unasserted names unknown field(s) {unknown}; "
                f"assertable fields are {sorted(ASSERTABLE_FIELDS)}"
            )
        for field, reason in v.items():
            if not reason or not reason.strip():
                raise ValueError(f"unasserted[{field!r}] needs a reason, not an empty string")
        return v

    @field_validator("conversation")
    @classmethod
    def _check_conversation(cls, v: list[Predicate]) -> list[Predicate]:
        for p in v:
            validate_predicate(p)
        return v

    @field_validator("turns")
    @classmethod
    def _check_turns(cls, v: dict[int, list[Predicate]]) -> dict[int, list[Predicate]]:
        for index, preds in v.items():
            if index < 1:
                raise ValueError(f"agent turn index must be 1-based, got {index}")
            for p in preds:
                validate_predicate(p)
        return v

    @model_validator(mode="after")
    def _final_stage_not_also_forbidden(self) -> Expectation:
        if self.final_stage is not None and self.final_stage in self.never_stage:
            raise ValueError(f"final_stage {self.final_stage} is also listed in never_stage")
        return self

    @model_validator(mode="after")
    def _positional_pivot_is_actually_reached(self) -> Expectation:
        """A positional assertion whose pivot stage is never entered can never
        fail. Require the pivot to be asserted as reached, so the assertion has
        teeth."""
        for pivot, forbidden in self.never_stage_after.items():
            if pivot in forbidden:
                raise ValueError(f"never_stage_after[{pivot}] forbids its own pivot stage")
            reached = pivot in self.stages_visited or pivot == self.final_stage
            if not reached:
                raise ValueError(
                    f"never_stage_after[{pivot}] is vacuous: {pivot} is not in stages_visited "
                    f"and is not final_stage, so nothing can violate it"
                )
        return self

    @property
    def all_predicates(self) -> list[Predicate]:
        out = list(self.conversation)
        for preds in self.turns.values():
            out.extend(preds)
        return out

    @model_validator(mode="after")
    def _unasserted_is_not_also_asserted(self) -> Expectation:
        """Declaring a field deliberately unasserted while also asserting it is
        a contradiction, and the kind that survives review because both halves
        look reasonable on their own."""
        contradictions = sorted(f for f in self.unasserted if f in self.asserted_fields)
        if contradictions:
            raise ValueError(
                f"field(s) {contradictions} are listed in unasserted but are also asserted"
            )
        return self

    @property
    def asserted_fields(self) -> set[str]:
        present = set()
        if self.final_stage is not None:
            present.add("final_stage")
        if self.interrupt is not None:
            present.add("interrupt")
        for name in ("stages_visited", "never_stage", "never_stage_after", "conversation", "turns"):
            if getattr(self, name):
                present.add(name)
        if self.tools.required:
            present.add("tools.required")
        if self.tools.forbidden:
            present.add("tools.forbidden")
        return present

    @property
    def judged_predicates(self) -> list[Predicate]:
        """Reported separately from the deterministic numbers."""
        return [p for p in self.all_predicates if is_judged(p)]

    @property
    def safety_predicates(self) -> list[Predicate]:
        """Surfaced at the top of the report; a failure here fails the run."""
        return [p for p in self.all_predicates if normalise_predicate(p)[0] in SAFETY_PREDICATES]

    def coverage(self) -> dict[str, object]:
        """asserted / deliberately-unasserted / unexplained.

        Two failure modes this guards against. With `final_stage` and
        `interrupt` optional, omission is invisible: a block asserting almost
        nothing scores the same as a thorough one, so the cheapest way to raise
        a score would be to assert less. And absence alone cannot distinguish a
        decision from a TODO -- `smoke-08` omitting `interrupt` is deliberate
        (the guardrail evaluates the draft, so `expected: false` would penalise
        a correct interception), while `smoke-03` omitting `tools.required` was
        an oversight.

        Only `unexplained` is a worklist.
        """
        preds = self.all_predicates
        asserted = self.asserted_fields
        explained = set(self.unasserted)
        return {
            "predicates": len(preds),
            "deterministic": len(preds) - len(self.judged_predicates),
            "judged": len(self.judged_predicates),
            "safety": len(self.safety_predicates),
            "turns_asserted": sorted(self.turns),
            "asserted_fields": sorted(asserted),
            "deliberately_unasserted": sorted(explained),
            "unexplained": sorted(ASSERTABLE_FIELDS - asserted - explained),
        }


# ---------------------------------------------------------------------------
# The conversation file
# ---------------------------------------------------------------------------


class Provenance(StrEnum):
    GENERATED = "generated"
    HANDWRITTEN = "handwritten"
    FROM_PRODUCTION = "from_production"
    DRAFTED_BY_ASSISTANT = "drafted_by_assistant"
    """Conversation or assertions proposed by a model, not written by the
    repository owner. Visible in the file and in the report, for the same
    reason judge sets carry `status: draft` (D33): machine-proposed expectations
    must not be mistaken for a human's."""


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inbound: str | None = None
    agent: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _one_side_only(self) -> Turn:
        if (self.inbound is None) == (self.agent is None):
            raise ValueError("a turn must set exactly one of `inbound` or `agent`")
        return self


class Context(BaseModel):
    model_config = ConfigDict(extra="forbid")

    today: str
    user_timezone: str = "Europe/Berlin"


class GoldenConversation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    mode: int = Field(ge=1, le=8)
    title: str
    provenance: Provenance
    locale: Literal["en", "de"] = "en"
    context: Context
    notes: str | None = None
    turns: list[Turn]
    expect: Expectation | None = None
    """Optional so a conversation can be committed before its assertions are
    written. `evals.validate --strict` fails on a missing block, and the runner
    refuses to score one."""

    @property
    def inbound_count(self) -> int:
        return sum(1 for t in self.turns if t.inbound is not None)

    @property
    def bursts(self) -> list[list[Turn]]:
        """Runs of consecutive inbound turns. One burst is one agent turn."""
        from app.channels.envelope import group_bursts

        return group_bursts(self.turns, lambda t: t.inbound is not None)

    @property
    def agent_turn_count(self) -> int:
        """One agent turn per inbound burst.

        NOT the number of `agent:` entries. Those are advisory placeholders and
        every file in the golden set omits the trailing one -- each conversation
        ends on an inbound burst that the agent would still reply to. Counting
        placeholders under-counts real agent turns by one in all eight files,
        which would reject correct per-turn assertions on the final turn.
        """
        return len(self.bursts)

    @property
    def scripted_agent_placeholders(self) -> int:
        return sum(1 for t in self.turns if t.agent is not None)

    @model_validator(mode="after")
    def _turn_indices_exist(self) -> GoldenConversation:
        """A per-turn assertion pointing at an agent turn that never happens is
        silently vacuous, which is the worst kind of passing test."""
        if self.expect is None:
            return self
        n = self.agent_turn_count
        for index in self.expect.turns:
            if index > n:
                raise ValueError(
                    f"expect.turns[{index}] but this conversation has only {n} agent turns"
                )
        return self
