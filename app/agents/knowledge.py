"""The knowledge sub-agent.

Answers a question from the corpus, or says it cannot. The corpus is ~3.8k
tokens and goes in full context — retrieval over 3.5k tokens solves nothing,
and building a retriever here would invite the question "why are you retrieving
over three thousand tokens?" with no good answer.

The delegation delta lives here. In the source, this sub-agent's input was
hardcoded to the raw webhook text and the orchestrator was instructed to call
it every time. Here the supervisor decides *what to ask it* and *whether to ask
it at all*, and the question arrives as a model-populated field on the
`TurnPlan`.

The one behaviour this agent exists to get right: when the corpus has no
answer, say so. The source corpus demonstrated inferring answers from adjacent
context on price, capacity and programme length — see decision D16 — so
`grounded=False` is a first-class outcome, not an error path.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.llm.base import LLMProvider, Message, Role
from app.llm.structured import StructuredOutputFailure, StructuredResult, generate
from app.logging import get_logger

log = get_logger(__name__)

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"


class Grounding(StrEnum):
    GROUNDED = "grounded"
    """Every claim in the answer is supported by the corpus."""

    NOT_IN_CORPUS = "not_in_corpus"
    """The corpus does not answer this. The reply must say so and offer the
    consultation call — never estimate, approximate, or infer."""

    OUT_OF_SCOPE = "out_of_scope"
    """Clinical, legal or financial territory the business does not answer at
    all, regardless of what the corpus contains."""


class KnowledgeAnswer(BaseModel):
    """What the knowledge agent returns to the supervisor.

    Deliberately not free text: `grounding` is what the compose step and the
    guardrail read, and a paragraph that merely *sounds* hedged is not a signal
    either can act on.
    """

    grounding: Grounding
    answer: str = Field(
        description=(
            "The factual answer, in one or two sentences, for the composer to "
            "rephrase. Empty when grounding is not `grounded`."
        )
    )
    cited: list[str] = Field(
        default_factory=list,
        description=(
            "Short verbatim spans from the corpus supporting the answer. "
            "Required when grounding is `grounded` — an answer that cannot "
            "cite the corpus was not taken from it."
        ),
    )
    confidence: float = Field(ge=0.0, le=1.0)

    @property
    def is_answerable(self) -> bool:
        return self.grounding is Grounding.GROUNDED


_SYSTEM = """\
You answer questions about a personal-coaching business from the reference
material below, and only from it.

Rules, in priority order:

1. If the material does not answer the question, set grounding to
   `not_in_corpus`, leave `answer` empty, and cite nothing. Do not estimate, do
   not approximate, do not infer an answer from adjacent context. A refusal is
   a correct answer; a plausible guess is the worst possible one.
2. If the question is clinical, legal or financial — whether to train with a
   diagnosed condition, which exercises to avoid, whether to take a supplement,
   contract or refund terms — set grounding to `out_of_scope` regardless of
   what the material appears to say.
3. Otherwise set grounding to `grounded`, answer in one or two sentences, and
   cite the spans you used. An answer you cannot cite was not taken from the
   material.

Never emit a URL or any external contact detail.

--- REFERENCE MATERIAL ---
{corpus}
--- END REFERENCE MATERIAL ---
"""


@lru_cache(maxsize=1)
def load_corpus(path: Path | None = None) -> str:
    return (path or PROMPTS / "knowledge.md").read_text(encoding="utf-8")


def build_messages(
    question: str, goal: str | None = None, corpus: str | None = None
) -> list[Message]:
    system = _SYSTEM.format(corpus=corpus if corpus is not None else load_corpus())
    user = question if not goal else f"The lead's stated goal is: {goal}\n\nQuestion: {question}"
    return [Message(Role.SYSTEM, system), Message(Role.USER, user)]


async def answer(
    provider: LLMProvider,
    question: str,
    *,
    goal: str | None = None,
    settings: Settings | None = None,
    corpus: str | None = None,
    **ladder_kwargs,
) -> StructuredResult:
    """Ask the corpus. Raises `StructuredOutputFailure` if the ladder is spent.

    The caller's fallback is `unanswerable()` — never a fabricated answer.
    """
    settings = settings or get_settings()
    return await generate(
        provider,
        build_messages(question, goal=goal, corpus=corpus),
        KnowledgeAnswer,
        settings=settings,
        temperature=0.0,
        **ladder_kwargs,
    )


def unanswerable(reason: str) -> KnowledgeAnswer:
    """The typed fallback for a knowledge call that failed outright.

    Returns "we do not have that" rather than nothing, because the composer
    still has to say something, and the only safe something is the refusal.
    """
    log.warning("knowledge.fallback", reason=reason)
    return KnowledgeAnswer(
        grounding=Grounding.NOT_IN_CORPUS, answer="", cited=[], confidence=0.0
    )


async def answer_or_refuse(
    provider: LLMProvider, question: str, **kwargs
) -> tuple[KnowledgeAnswer, StructuredResult | None]:
    """Convenience wrapper: never raises, never fabricates."""
    try:
        result = await answer(provider, question, **kwargs)
        return result.value, result  # type: ignore[return-value]
    except StructuredOutputFailure as exc:
        return unanswerable(str(exc)), None
