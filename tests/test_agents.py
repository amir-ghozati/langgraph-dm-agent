"""The two sub-agents.

No live model. What is tested here is the contract each one is held to: what
its schema permits, what its fallback does, and what it refuses to invent.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents import knowledge, strategy
from app.agents.knowledge import Grounding, KnowledgeAnswer
from app.agents.strategy import ProposableStage, SlotReading, StageProposal
from app.config import SchemaMode, Settings
from app.funnel.stages import SYSTEM_ONLY_STAGES, FunnelStage
from app.funnel.transitions import InboundIntent, TransitionContext, decide
from app.llm.base import LLMResult, Message, Role
from app.llm.structured import StructuredOutputFailure


class Replay:
    name = "fake"
    model = "fake-1"

    def __init__(self, *payloads: str) -> None:
        self._payloads = list(payloads)
        self.calls: list[list[Message]] = []

    def supports_native_schema(self) -> bool:
        return True

    async def complete(self, messages, schema=None, *, schema_mode=None, temperature=None):
        self.calls.append(list(messages))
        text = self._payloads.pop(0) if self._payloads else "{}"
        parsed, error = (None, "exhausted")
        if schema is not None:
            from app.llm.base import parse_into

            parsed, error = parse_into(schema, text)
        return LLMResult(
            text=text,
            parsed=parsed,
            parse_ok=parsed is not None,
            parse_error=error,
            schema_name=getattr(schema, "__name__", None),
            schema_mode=schema_mode or SchemaMode.NATIVE,
            provider="fake",
            model="fake-1",
            latency_ms=1,
            tokens_in=1,
            tokens_out=1,
        )


@pytest.fixture
def settings(settings_kwargs) -> Settings:
    return Settings(**settings_kwargs)


# ---------------------------------------------------------------------------
# knowledge agent
# ---------------------------------------------------------------------------


def test_the_corpus_loads_and_carries_its_framing():
    corpus = knowledge.load_corpus()
    assert "HOW TO READ THIS FILE" in corpus
    assert "NOT IN KNOWLEDGE BASE" in corpus


def test_the_corpus_is_small_enough_that_retrieval_would_be_pointless():
    """The stated reason there is no retriever. If the corpus ever grows past
    roughly a third of the context window, that argument stops holding and this
    test is where it gets noticed."""
    corpus = knowledge.load_corpus()
    approx_tokens = len(corpus) / 4
    assert approx_tokens < 8000, f"corpus is now ~{approx_tokens:.0f} tokens; revisit D-no-RAG"


def test_the_system_prompt_forbids_inferring_an_answer():
    msgs = knowledge.build_messages("what does it cost?")
    system = msgs[0].content
    assert "do not estimate" in system and "do not infer" in system
    assert "A refusal is\n   a correct answer" in system


def test_the_question_is_the_argument_not_the_raw_message():
    """The delegation delta. In the source both sub-agent inputs were hardcoded
    to the raw webhook text; here the supervisor decides what to ask."""
    msgs = knowledge.build_messages("does he coach beginners?", goal="lose 10kg")
    assert "does he coach beginners?" in msgs[1].content
    assert "lose 10kg" in msgs[1].content


def test_not_in_corpus_is_a_first_class_answer_not_an_error():
    a = KnowledgeAnswer(grounding=Grounding.NOT_IN_CORPUS, answer="", confidence=0.0)
    assert not a.is_answerable


def test_a_grounded_answer_is_expected_to_cite():
    a = KnowledgeAnswer(
        grounding=Grounding.GROUNDED,
        answer="He coaches beginners through to competitive athletes.",
        cited=["from absolute beginners to"],
        confidence=0.9,
    )
    assert a.is_answerable and a.cited


async def test_the_knowledge_fallback_refuses_rather_than_invents(settings):
    """Three malformed replies must produce "we do not have that", never a
    plausible sentence. This is the path that would otherwise fabricate a
    price."""
    provider = Replay("nope", "still nope", "nope again")
    answer, result = await knowledge.answer_or_refuse(
        provider, "what does coaching cost?", settings=settings
    )
    assert result is None
    assert answer.grounding is Grounding.NOT_IN_CORPUS
    assert answer.answer == ""
    assert answer.confidence == 0.0


async def test_a_valid_knowledge_reply_is_returned_as_typed_data(settings):
    provider = Replay(
        '{"grounding":"grounded","answer":"He coaches beginners too.",'
        '"cited":["absolute beginners"],"confidence":0.8}'
    )
    answer, result = await knowledge.answer_or_refuse(
        provider, "beginners?", settings=settings
    )
    assert result is not None and result.first_attempt_valid
    assert answer.is_answerable


async def test_knowledge_raises_rather_than_guessing_when_asked_directly(settings):
    with pytest.raises(StructuredOutputFailure):
        await knowledge.answer(Replay("x", "y", "z"), "price?", settings=settings)


# ---------------------------------------------------------------------------
# strategy agent
# ---------------------------------------------------------------------------


def test_the_model_cannot_even_name_a_system_only_stage():
    """A second lock in front of the guard in decide(): the schema has no value
    for these, so an adversarial or malformed proposal fails validation before
    the state machine is reached."""
    proposable = {s.value for s in ProposableStage}
    assert not proposable & {s.value for s in SYSTEM_ONLY_STAGES}
    with pytest.raises(ValidationError):
        StageProposal(proposed_stage="BOOKED", next_action="x", reason="y")


def test_stay_is_distinct_from_a_failed_call():
    """Both end up as "no advance", but only one is a model decision. Collapsing
    them would make proposal_override_rate uninterpretable."""
    explicit = StageProposal(
        proposed_stage=ProposableStage.STAY, next_action="answer them", reason="still gathering"
    )
    fallback = strategy.stay("ladder exhausted")
    assert explicit.proposed_stage is fallback.proposed_stage
    assert "unavailable" in fallback.reason and "unavailable" not in explicit.reason


def test_stay_translates_to_no_proposal_rather_than_the_current_stage():
    """So the transition log can tell "the model wanted no change" from "the
    model was not asked"."""
    assert strategy.to_funnel_stage(ProposableStage.STAY, FunnelStage.VALUE) is None
    assert (
        strategy.to_funnel_stage(ProposableStage.TRANSITION, FunnelStage.VALUE)
        is FunnelStage.TRANSITION
    )


def test_a_slot_reading_is_raw_not_validated():
    """The agent reports what the lead said; validation belongs to the tools.
    A model that could mark a phone number valid could book on a bad one."""
    s = SlotReading(phone_raw="0151 23oh4a78", day_expression="next tuesday")
    assert s.any_supplied
    assert not hasattr(s, "phone_e164")


def test_the_playbook_is_loaded_and_carries_no_impersonation():
    playbook = strategy.load_playbook()
    assert "AIDA" in playbook
    assert "real coach" not in playbook.lower()


async def test_the_strategy_fallback_cannot_advance_the_funnel(settings):
    """A failed strategy call must never move a lead toward a booking. STAY is
    safe in every stage, which is why it is the fallback."""
    proposal, result = await strategy.propose_or_stay(
        Replay("bad", "bad", "bad"), FunnelStage.SLOT_FILLING, 3, "0151 2345678", settings=settings
    )
    assert result is None
    assert proposal.proposed_stage is ProposableStage.STAY

    d = decide(
        FunnelStage.SLOT_FILLING,
        strategy.to_funnel_stage(proposal.proposed_stage, FunnelStage.SLOT_FILLING),
        TransitionContext(),
    )
    assert d.decided_stage is FunnelStage.SLOT_FILLING


async def test_a_proposal_flows_into_the_state_machine_and_can_be_overridden(settings):
    """End to end for D2: the model proposes TRANSITION too early, the guard
    rejects it, and the rejection is recorded with a reason."""
    provider = Replay(
        '{"proposed_stage":"TRANSITION","intent":"none","engaged_with_offer":false,'
        '"declined":false,"opt_out":false,"slots":{},'
        '"next_action":"offer the call","reason":"they seem keen"}'
    )
    proposal, result = await strategy.propose_or_stay(
        provider, FunnelStage.VALUE, 2, "cool thanks", settings=settings
    )
    assert result is not None
    assert proposal.proposed_stage is ProposableStage.TRANSITION

    d = decide(
        FunnelStage.VALUE,
        strategy.to_funnel_stage(proposal.proposed_stage, FunnelStage.VALUE),
        TransitionContext(agent_turn_index=2),
    )
    assert d.decided_stage is FunnelStage.VALUE
    assert not d.accepted
    assert "before FUNNEL_TRANSITION_TURN" in d.override_reason


async def test_an_informal_opt_out_is_reportable_without_a_keyword(settings):
    """smoke-06's opt-out is "actually sorry nvm, dont think im ready for this
    rn" — no stop, no unsubscribe."""
    provider = Replay(
        '{"proposed_stage":"OPTED_OUT","intent":"none","engaged_with_offer":false,'
        '"declined":true,"opt_out":true,"slots":{},'
        '"next_action":"acknowledge and step back","reason":"withdrew"}'
    )
    proposal, _ = await strategy.propose_or_stay(
        provider, FunnelStage.SLOT_FILLING, 3, "actually sorry nvm", settings=settings
    )
    assert proposal.opt_out

    d = decide(
        FunnelStage.SLOT_FILLING,
        strategy.to_funnel_stage(proposal.proposed_stage, FunnelStage.SLOT_FILLING),
        TransitionContext(opt_out_detected=proposal.opt_out),
    )
    assert d.decided_stage is FunnelStage.OPTED_OUT


def test_the_strategy_prompt_states_both_things_the_model_cannot_do():
    msgs = strategy.build_messages(FunnelStage.VALUE, 2, "hi")
    system = msgs[0].content
    assert "cannot move the conversation to a booking-confirmed state" in system
    assert "cannot decide a phone number or a date is valid" in system
    assert "ignores the offer and changes the subject is not engagement" in system


def test_the_strategy_user_turn_carries_the_state_the_guards_read():
    msgs = strategy.build_messages(FunnelStage.TRANSITION, 5, "thursday?", slots="name=dan")
    user = msgs[1].content
    assert "TRANSITION" in user and "5" in user and "name=dan" in user


def test_intent_values_match_the_state_machine_vocabulary():
    """The proposal feeds decide() directly, so a drift here is a silent
    routing bug rather than a validation error."""
    assert {i.value for i in InboundIntent} >= {"none", "direct_question", "booking_request"}
    p = StageProposal(
        proposed_stage=ProposableStage.VALUE,
        intent=InboundIntent.DIRECT_QUESTION,
        next_action="answer",
        reason="asked a question",
    )
    assert p.intent is InboundIntent.DIRECT_QUESTION


def test_role_constants_are_used_rather_than_bare_strings():
    msgs = knowledge.build_messages("q")
    assert msgs[0].role == Role.SYSTEM and msgs[1].role == Role.USER
