"""Compose the reply the lead actually sees.

Everything upstream produced structured data; this turns it into one DM. It is
the last place a fabrication could enter, so the composer is given the
knowledge agent's *verdict* rather than free rein: when grounding is
`not_in_corpus` it is told to say so, not left to decide.

Style rules come from the source prompt and are enforced deterministically
afterwards rather than trusted here — 3 to 40 words, one sentence, 1 to 3
non-adjacent emoji. The guardrail (Phase 3a) re-checks and re-composes.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.agents.knowledge import Grounding, KnowledgeAnswer
from app.agents.strategy import StageProposal
from app.config import DisclosureMode, Settings, get_settings
from app.funnel.stages import FunnelStage
from app.llm.base import LLMProvider, Message, Role
from app.llm.structured import StructuredOutputFailure, StructuredResult, generate
from app.logging import get_logger
from app.memory.summary import Slots, TypedSummary

log = get_logger(__name__)

DISCLOSURE_LINE = "Quick note — I'm Jan's AI assistant, not Jan himself."


class ComposedReply(BaseModel):
    text: str = Field(description="The message to send. One sentence, 3 to 40 words.")
    self_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How confident you are that this reply is correct and grounded. "
            "Low confidence is useful information, not a failure — it routes "
            "the turn to a human."
        ),
    )
    mentions_call: bool = Field(
        default=False, description="Whether this reply offers or asks about the consultation call."
    )
    asks_for_slot: bool = Field(
        default=False,
        description="Whether this reply asks for a name, phone number, day or time.",
    )


_SYSTEM = """\
You write one Instagram DM on behalf of Coach Jan Mustermann of Musterform
Personal Coaching. You are not Jan; never claim to be, and answer honestly if
asked whether you are an AI.

Style, strictly:
* One sentence. 3 to 40 words.
* Warm and colloquial, like a message from a friend at the gym.
* One to three emoji, spread through the sentence, never adjacent.
* Never send a URL or any external contact detail.

Content rules, in priority order:

1. If the knowledge verdict says NOT IN CORPUS, say plainly that you do not
   have that information and offer the free 30-minute consultation call. Do not
   estimate, approximate, or infer it from anything nearby. This matters most
   on price.
2. If the knowledge verdict says OUT OF SCOPE, decline and point them to a
   doctor or physiotherapist. Give no clinical guidance of any kind — not which
   exercises, not supplements, not whether to train, and do not ask screening
   questions about their symptoms.
3. Never state or imply that a booking is confirmed. You are not able to
   confirm one; the system does that.
4. Never claim Jan's credentials or experience in the first person. "Jan's
   worked with…", never "I've worked with…".
5. Otherwise do what the next action says, using only the grounded answer.
"""

_USER = """\
Funnel stage: {stage}
Next action: {next_action}
Slots still needed: {missing}

Knowledge verdict: {grounding}
Grounded answer: {answer}

What we know about this lead:
{summary}

Recent exchanges:
{window}

The lead's latest message(s):
{inbound}
{disclosure}"""


def build_messages(
    stage: FunnelStage,
    inbound: str,
    *,
    knowledge: KnowledgeAnswer | None = None,
    proposal: StageProposal | None = None,
    summary: TypedSummary | None = None,
    slots: Slots | None = None,
    window: str = "",
    disclose: bool = False,
) -> list[Message]:
    grounding = knowledge.grounding if knowledge else Grounding.NOT_IN_CORPUS
    answer = (knowledge.answer if knowledge else "") or "(nothing looked up this turn)"
    missing = ", ".join((slots or Slots()).missing) or "none"
    disclosure = (
        f'\nBegin the reply with: "{DISCLOSURE_LINE}" then continue in the same message.\n'
        if disclose
        else ""
    )
    return [
        Message(Role.SYSTEM, _SYSTEM),
        Message(
            Role.USER,
            _USER.format(
                stage=stage,
                next_action=(proposal.next_action if proposal else "reply naturally"),
                missing=missing,
                grounding=grounding,
                answer=answer,
                summary=(summary or TypedSummary()).describe(),
                window=window or "(this is the first exchange)",
                inbound=inbound,
                disclosure=disclosure,
            ),
        ),
    ]


def should_disclose(
    stage: FunnelStage, turn_index: int, settings: Settings | None = None
) -> bool:
    """Proactive disclosure only. Answering honestly when asked is not
    configurable and lives in the persona block — see decision D26."""
    settings = settings or get_settings()
    match settings.disclosure_mode:
        case DisclosureMode.NONE:
            return False
        case DisclosureMode.EVERY_MESSAGE:
            return True
        case DisclosureMode.ON_FIRST_CONTACT:
            return turn_index == 1
    return False


async def compose(
    provider: LLMProvider,
    stage: FunnelStage,
    inbound: str,
    *,
    settings: Settings | None = None,
    **kwargs,
) -> StructuredResult:
    settings = settings or get_settings()
    ladder_kwargs = {k: kwargs.pop(k) for k in ("fault_injector",) if k in kwargs}
    return await generate(
        provider,
        build_messages(stage, inbound, **kwargs),
        ComposedReply,
        settings=settings,
        temperature=0.3,
        **ladder_kwargs,
    )


def safe_fallback(stage: FunnelStage, reason: str) -> ComposedReply:
    """The typed fallback. Says nothing that could be wrong.

    Deliberately not a cheerful non-answer: a composer that failed should hand
    the turn to a human, and `self_confidence=0.0` is what routes it there.
    """
    log.warning("compose.fallback", stage=str(stage), reason=reason)
    return ComposedReply(
        text="Sorry — I've got my wires crossed there. Let me get Jan to come back to you 🙏",
        self_confidence=0.0,
        mentions_call=False,
        asks_for_slot=False,
    )


async def compose_or_fallback(
    provider: LLMProvider, stage: FunnelStage, inbound: str, **kwargs
) -> tuple[ComposedReply, StructuredResult | None]:
    try:
        result = await compose(provider, stage, inbound, **kwargs)
        return result.value, result  # type: ignore[return-value]
    except StructuredOutputFailure as exc:
        return safe_fallback(stage, str(exc)), None
