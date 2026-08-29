"""The supervisor: one planning call that emits a validated `TurnPlan`.

This is D1, and it is the delegation delta the whole port exists to demonstrate.

In the source, both sub-agent inputs were hardcoded to the raw webhook text and
the orchestrator's prompt said to call both every time. It could not choose what
to ask, or whether to ask. Here the supervisor populates `knowledge_query`
itself and may leave it null, and the resulting plan is a single typed artifact
the eval harness can assert against — which a ReAct trace is not.

The plan proposes. Nothing here decides: the funnel stage is settled by
`app.funnel.transitions.decide`, and tool arguments are validated before
execution (Phase 3a).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.funnel.stages import FunnelStage
from app.funnel.transitions import InboundIntent
from app.llm.base import LLMProvider, Message, Role
from app.llm.structured import StructuredOutputFailure, StructuredResult, generate
from app.logging import get_logger
from app.memory.summary import Slots, TypedSummary

log = get_logger(__name__)

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"


class TurnPlan(BaseModel):
    """What to do this turn, decided before anything is done.

    Every field is a *request* or an *observation*. The stage is proposed by
    the strategy agent and settled by the state machine; the reply is written
    by `compose`; tools are executed only after their arguments validate.
    """

    intent: InboundIntent = Field(
        default=InboundIntent.NONE,
        description=(
            "Shape of the inbound burst: none / direct_question / booking_request. "
            "Classify the whole burst, not the first message."
        ),
    )

    knowledge_query: str | None = Field(
        default=None,
        description=(
            "The specific question to put to the knowledge agent, phrased as a "
            "question. Null when the turn needs no factual grounding — a "
            "greeting, an opt-out, or a pure slot-collection turn. Do not pass "
            "the user's raw message through; decide what actually needs asking."
        ),
    )

    needs_strategy: bool = Field(
        default=True,
        description=(
            "Whether the funnel position needs re-evaluating this turn. Almost "
            "always true; false only when the conversation is terminal."
        ),
    )

    user_declined: bool = Field(
        default=False, description="The lead declined the consultation offer."
    )
    engaged_with_offer: bool = Field(
        default=False,
        description=(
            "The lead affirmatively engaged with the consultation offer. "
            "Ignoring the offer and changing the subject is NOT engagement."
        ),
    )

    notes: str = Field(
        default="",
        description="One sentence on what this turn should accomplish. Not shown to the user.",
    )

    @property
    def consults_knowledge(self) -> bool:
        return bool(self.knowledge_query and self.knowledge_query.strip())


_SYSTEM = """\
You plan one turn of a sales conversation for an online fitness coaching
business. You do not write the reply and you do not decide the funnel stage —
you decide what information this turn needs.

Your only job is to fill in the plan:

* `intent` — the shape of what the lead just sent, judged across the WHOLE
  burst rather than the first message. A greeting followed by "hows pricing
  work" is a direct_question.
* `knowledge_query` — if answering well needs a fact about the coach or the
  coaching, write the specific question to look up. If it does not, leave it
  null. Do not copy the lead's message in; work out what actually needs asking.
  A question you already have the answer to does not need looking up.
* `needs_strategy` — whether the funnel position needs re-evaluating.
* `user_declined` / `engaged_with_offer` — only from what the lead actually
  said. A reply that ignores the consultation offer is not engagement.

Be sparing with `knowledge_query`. Looking nothing up on a turn that needs
nothing looked up is correct behaviour, not laziness.
"""

_USER = """\
Funnel stage: {stage}
Agent turn about to be produced: {turn_index}
Slots collected: {slots}

What we know about this lead:
{summary}

Recent exchanges:
{window}

The lead's latest message(s):
{inbound}
"""


def build_messages(
    stage: FunnelStage,
    turn_index: int,
    inbound: str,
    *,
    summary: TypedSummary | None = None,
    slots: Slots | None = None,
    window: str = "",
) -> list[Message]:
    return [
        Message(Role.SYSTEM, _SYSTEM),
        Message(
            Role.USER,
            _USER.format(
                stage=stage,
                turn_index=turn_index,
                slots=(slots or Slots()).describe(),
                summary=(summary or TypedSummary()).describe(),
                window=window or "(this is the first exchange)",
                inbound=inbound,
            ),
        ),
    ]


async def plan(
    provider: LLMProvider,
    stage: FunnelStage,
    turn_index: int,
    inbound: str,
    *,
    summary: TypedSummary | None = None,
    slots: Slots | None = None,
    window: str = "",
    settings: Settings | None = None,
    **ladder_kwargs,
) -> StructuredResult:
    settings = settings or get_settings()
    return await generate(
        provider,
        build_messages(
            stage, turn_index, inbound, summary=summary, slots=slots, window=window
        ),
        TurnPlan,
        settings=settings,
        temperature=0.0,
        **ladder_kwargs,
    )


def minimal_plan(reason: str) -> TurnPlan:
    """The typed fallback when the ladder is exhausted.

    Strategy only, no knowledge lookup, no tools. It cannot advance anything on
    its own, which is the property that matters: a supervisor that failed to
    parse must not be able to move a lead toward a booking.
    """
    log.warning("supervisor.fallback", reason=reason)
    return TurnPlan(
        intent=InboundIntent.NONE,
        knowledge_query=None,
        needs_strategy=True,
        notes=f"planning failed, minimal safe turn: {reason}",
    )


async def plan_or_minimal(
    provider: LLMProvider, stage: FunnelStage, turn_index: int, inbound: str, **kwargs
) -> tuple[TurnPlan, StructuredResult | None]:
    try:
        result = await plan(provider, stage, turn_index, inbound, **kwargs)
        return result.value, result  # type: ignore[return-value]
    except StructuredOutputFailure as exc:
        return minimal_plan(str(exc)), None
