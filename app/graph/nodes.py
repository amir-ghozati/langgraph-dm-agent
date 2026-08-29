"""The graph's nodes.

Each takes `TurnState` and returns a partial update. The database session and
the LLM provider arrive through `RunnableConfig["configurable"]` rather than
through state, because neither is serialisable and state gets pickled into
Postgres on every node boundary.

Node contracts, and how each one fails:

| node | fails how |
|---|---|
| `normalize` | never — pure |
| `load_session` | DB error propagates; the turn is re-driven from `messages` |
| `gate` | never — always records a reason |
| `supervisor` | ladder exhausted → minimal safe plan |
| `knowledge_agent` | ladder exhausted → NOT_IN_CORPUS, never a guess |
| `strategy_agent` | ladder exhausted → propose STAY |
| `funnel_transition` | never — an invalid proposal is overridden and logged |
| `compose` | ladder exhausted → handoff text, confidence 0 |
| `deliver` | send error → `delivered=False`, recorded |
| `post_turn` | DB error propagates; the turn is lost, not half-written |
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from app.agents import compose as compose_agent
from app.agents import knowledge as knowledge_agent
from app.agents import strategy as strategy_agent
from app.agents import supervisor as supervisor_agent
from app.agents.compose import ComposedReply
from app.agents.knowledge import Grounding, KnowledgeAnswer
from app.agents.strategy import StageProposal
from app.agents.supervisor import TurnPlan
from app.channels.base import OutboundMessage
from app.config import EligibilityPolicyName, Settings
from app.db import repository as repo
from app.funnel.stages import TERMINAL_STAGES, FunnelStage
from app.funnel.transitions import FunnelPolicy, TransitionContext, decide
from app.graph.state import TurnState, stage_of
from app.guardrails import checks as guard
from app.guardrails.interrupt import build_payload, parse_resume, reason_for
from app.logging import get_logger
from app.memory.summary import Slots, TypedSummary

log = get_logger(__name__)

WINDOW_EXCHANGES = 8


def _cfg(config: RunnableConfig) -> dict[str, Any]:
    return config.get("configurable", {})


def _metric(state: TurnState, node: str, result, **extra) -> dict[str, Any]:
    """One `turn_metrics` row per LLM call. Written in post_turn."""
    row = {
        "node": node,
        "provider": getattr(result, "provider", None),
        "model": getattr(result, "model", None),
        "schema_mode": str(result.schema_mode) if result is not None else None,
        "latency_ms": getattr(result, "latency_ms", 0),
        "tokens_in": getattr(result, "tokens_in", 0),
        "tokens_out": getattr(result, "tokens_out", 0),
        "repair_attempts": getattr(result, "repair_attempts", 0),
        "first_attempt_valid": getattr(result, "first_attempt_valid", None),
    }
    row.update(extra)
    return [*state.get("metrics", []), row]


async def normalize(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    """Pure. The envelope was already assembled by the channel or the runner —
    a burst is one turn, and this node does not get to re-split it."""
    return {"inbound_text": state["inbound_text"].strip()}


async def load_session(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    """Load the authoritative state from the tables.

    This is the node that makes D4 true: after a restart the funnel stage, the
    slots and the turn index come from here, not from the checkpointer.
    """
    cfg = _cfg(config)
    db = cfg["db"]
    settings: Settings = cfg["settings"]

    create = settings.eligibility_policy is EligibilityPolicyName.ALLOW_ALL
    session = await repo.load_session(
        db, state["channel"], state["channel_user_id"], create=True
    )
    assert session is not None  # create=True always returns

    messages = await repo.transcript(db, session.conversation.id)
    window = "\n".join(
        f"{'lead' if m.direction == 'INBOUND' else 'coach'}: {m.body}"
        for m in messages[-(WINDOW_EXCHANGES * 2) :]
    )

    return {
        "conversation_id": str(session.conversation.id),
        "lead_id": str(session.lead.id),
        "stage": str(session.stage),
        "agent_turn_index": session.agent_turn_index,
        "summary": session.summary.model_dump(mode="json"),
        "slots": session.slots.model_dump(mode="json"),
        "window": window,
        "_eligible_by_policy": create,
    }


async def gate(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    """Business-hours, eligibility and terminal checks. Never silent.

    `RequireLeadRecord` is the client's actual rule, not spam filtering:
    inbound DMs come from people who commented on a post or replied to a story,
    so a sender with no lead record is not a lead. Every drop is logged with a
    reason — a gate that discards traffic without telling anyone is a bug even
    when the filtering is intended.
    """
    cfg = _cfg(config)
    settings: Settings = cfg["settings"]
    db = cfg["db"]

    # Inbound dedup, before any model call. Instagram webhooks are
    # at-least-once and Meta redelivers on non-2xx as well as on success, so
    # without this a redelivery costs four LLM calls and sends a second reply.
    # Storing the message once was not enough: the turn still ran.
    ids = state.get("channel_message_ids") or []
    if ids:
        seen = await repo.already_seen(db, state["channel"], ids)
        if seen and len(seen) == len(set(ids)):
            return _drop(state, f"duplicate inbound, already processed: {sorted(seen)}")

    stage = FunnelStage(state["stage"])
    if stage in TERMINAL_STAGES and stage is not FunnelStage.OPTED_OUT:
        return _drop(state, f"conversation is terminal ({stage})")

    if settings.eligibility_policy is EligibilityPolicyName.REQUIRE_LEAD_RECORD:
        lead = await repo.find_lead(db, state["channel"], state["channel_user_id"])
        # The lead row exists by now because load_session created it; the
        # policy is about whether it pre-existed as a *known* lead. `source`
        # is what a real lead-magnet import would set.
        unknown = lead is not None and lead.source is None
        first_contact = state["agent_turn_index"] == 1
        if unknown and first_contact and not settings.allow_unknown_first_contact:
            return _drop(state, "no lead record: sender is not an engagement-qualified lead")

    return {"gate_ok": True, "drop_reason": None}


def _drop(state: TurnState, reason: str) -> dict[str, Any]:
    log.info(
        "gate.drop",
        reason=reason,
        channel=state["channel"],
        channel_user_id=state["channel_user_id"],
    )
    return {"gate_ok": False, "drop_reason": reason, "reply": None}


async def supervisor(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    cfg = _cfg(config)
    plan, result = await supervisor_agent.plan_or_minimal(
        cfg["provider"],
        FunnelStage(state["stage"]),
        state["agent_turn_index"],
        state["inbound_text"],
        summary=TypedSummary.model_validate(state.get("summary") or {}),
        slots=Slots.model_validate(state.get("slots") or {}),
        window=state.get("window", ""),
        settings=cfg["settings"],
    )
    return {
        "plan": plan.model_dump(mode="json"),
        "metrics": _metric(state, "supervisor", result),
    }


async def knowledge(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    """Skipped entirely when the plan asked nothing.

    That skip is the delegation delta: the source called this every turn with
    the raw webhook text, because it had no way not to.
    """
    plan = TurnPlan.model_validate(state.get("plan") or {})
    if not plan.consults_knowledge:
        return {
            "knowledge": KnowledgeAnswer(
                grounding=Grounding.NOT_IN_CORPUS, answer="", confidence=0.0
            ).model_dump(mode="json"),
        }

    cfg = _cfg(config)
    summary = TypedSummary.model_validate(state.get("summary") or {})
    answer, result = await knowledge_agent.answer_or_refuse(
        cfg["provider"],
        plan.knowledge_query or "",
        goal="; ".join(summary.stated_goals) or None,
        settings=cfg["settings"],
    )
    return {
        "knowledge": answer.model_dump(mode="json"),
        "metrics": _metric(state, "knowledge_agent", result),
    }


async def strategy(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    cfg = _cfg(config)
    slots = Slots.model_validate(state.get("slots") or {})
    proposal, result = await strategy_agent.propose_or_stay(
        cfg["provider"],
        FunnelStage(state["stage"]),
        state["agent_turn_index"],
        state["inbound_text"],
        window=state.get("window", ""),
        slots=slots.describe(),
        settings=cfg["settings"],
    )
    return {
        "proposal": proposal.model_dump(mode="json"),
        "metrics": _metric(state, "strategy_agent", result),
    }


async def funnel_transition(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    """The LLM proposed; this decides. Both are recorded either way.

    `AWAITING_CONFIRMATION` and `BOOKED` are unreachable from here by proposal —
    twice over. `ProposableStage` has no value for them, so a proposal naming
    one fails schema validation before it arrives; and `decide` rejects them
    before any other rule is evaluated.
    """
    cfg = _cfg(config)
    settings: Settings = cfg["settings"]
    policy = FunnelPolicy.from_settings(settings)

    plan = TurnPlan.model_validate(state.get("plan") or {})
    proposal = StageProposal.model_validate(
        state.get("proposal") or {"proposed_stage": "STAY", "next_action": "", "reason": ""}
    )
    slots = Slots.model_validate(state.get("slots") or {})

    ctx = TransitionContext(
        agent_turn_index=state["agent_turn_index"],
        inbound_intent=proposal.intent or plan.intent,
        user_declined=proposal.declined or plan.user_declined,
        engaged_with_offer=plan.engaged_with_offer or proposal.engaged_with_offer,
        slot_value_supplied=proposal.slots.any_supplied,
        slots_all_valid=slots.all_valid,
        phone_attempts=slots.phone.attempts,
        opt_out_detected=proposal.opt_out,
    )
    decision = decide(
        FunnelStage(state["stage"]),
        strategy_agent.to_funnel_stage(proposal.proposed_stage, FunnelStage(state["stage"])),
        ctx,
        policy,
    )
    log.info(
        "funnel.decide",
        from_stage=str(decision.from_stage),
        proposed=str(decision.proposed_stage),
        decided=str(decision.decided_stage),
        guard=decision.guard_result,
        accepted=decision.accepted,
    )
    return {
        "decided_stage": str(decision.decided_stage),
        "transition": {
            "from_stage": str(decision.from_stage),
            "proposed_stage": (
                str(decision.proposed_stage) if decision.proposed_stage else None
            ),
            "decided_stage": str(decision.decided_stage),
            "accepted": decision.accepted,
            "guard_result": decision.guard_result,
            "override_reason": decision.override_reason,
        },
    }


async def compose(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    cfg = _cfg(config)
    settings: Settings = cfg["settings"]
    stage = stage_of(state)

    reply, result = await compose_agent.compose_or_fallback(
        cfg["provider"],
        stage,
        state["inbound_text"],
        # Absent rather than empty when the plan asked nothing — the composer
        # is told "nothing was looked up", not handed a malformed verdict.
        knowledge=(
            KnowledgeAnswer.model_validate(state["knowledge"])
            if state.get("knowledge")
            else None
        ),
        proposal=(
            StageProposal.model_validate(state["proposal"]) if state.get("proposal") else None
        ),
        summary=TypedSummary.model_validate(state.get("summary") or {}),
        slots=Slots.model_validate(state.get("slots") or {}),
        window=state.get("window", ""),
        disclose=compose_agent.should_disclose(stage, state["agent_turn_index"], settings),
        settings=settings,
    )
    return {
        "reply": reply.text,
        "reply_meta": reply.model_dump(mode="json"),
        "metrics": _metric(state, "compose", result),
    }


async def guardrail(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    """Check the draft. Style violations re-compose; safety violations stop.

    The interrupt happens here rather than in `deliver`, so the draft is held
    in the checkpoint and nothing has been sent while a human decides.
    """
    cfg = _cfg(config)
    settings: Settings = cfg["settings"]
    stage = stage_of(state)

    reply = ComposedReply.model_validate(
        state.get("reply_meta") or {"text": "", "self_confidence": 0.0}
    )
    transition = state.get("transition") or {}
    metrics = state.get("metrics") or []

    verdict = guard.check(
        reply,
        settings=settings,
        disclosure_required=compose_agent.should_disclose(
            stage, state["agent_turn_index"], settings
        ),
        booking_committed=stage is FunnelStage.BOOKED,
        repair_attempts=sum(m.get("repair_attempts", 0) for m in metrics),
        proposal_overridden=transition.get("accepted") is False,
        # Only while we are actually collecting. Before SLOT_FILLING nothing
        # has been asked for yet, so "4 slots missing" is the normal state of
        # the world rather than evidence of trouble — counting it there sent
        # every opening turn to a human.
        unresolved_slots=(
            len(Slots.model_validate(state.get("slots") or {}).missing)
            if stage in (FunnelStage.SLOT_FILLING, FunnelStage.AWAITING_CONFIRMATION)
            else 0
        ),
    )

    attempts = state.get("recompose_attempts", 0)
    if verdict.should_recompose and attempts < settings.guardrail_recompose_limit:
        return {
            "guard": _verdict_dict(verdict),
            "recompose_attempts": attempts + 1,
        }

    if verdict.should_recompose:
        # Re-composition exhausted. Strip to something that cannot be wrong
        # rather than sending a draft that breaks the style contract.
        return {
            "guard": _verdict_dict(verdict),
            "reply": guard.safe_template(),
            "recompose_attempts": attempts,
        }

    reason = reason_for(verdict)
    if reason is None:
        return {"guard": _verdict_dict(verdict), "recompose_attempts": attempts}

    payload = build_payload(
        verdict,
        reason=reason,
        draft=state.get("reply") or "",
        stage=str(stage),
        agent_turn_index=state["agent_turn_index"],
        conversation_id=state["conversation_id"],
        inbound=state["inbound_text"],
    )
    log.warning("guardrail.interrupt", reason=str(reason), summary=payload.summary())

    from langgraph.types import interrupt

    decision = parse_resume(interrupt(payload.model_dump(mode="json")))
    resolved = decision.resolve(state.get("reply") or "")

    if resolved is None:
        return {
            "guard": _verdict_dict(verdict),
            "reply": None,
            "interrupted": True,
            "handed_off": True,
            "recompose_attempts": attempts,
        }
    return {
        "guard": _verdict_dict(verdict),
        "reply": resolved,
        "interrupted": True,
        "recompose_attempts": attempts,
    }


def _verdict_dict(verdict) -> dict[str, Any]:
    return {
        "violations": [str(v) for v in verdict.violations],
        "safety": [str(v) for v in verdict.safety_violations],
        "style": [str(v) for v in verdict.style_violations],
        "confidence": verdict.confidence,
        "notes": verdict.notes,
    }


def route_after_guardrail(state: TurnState) -> str:
    """Re-compose, or carry on to delivery."""
    guard_state = state.get("guard") or {}
    attempts = state.get("recompose_attempts", 0)
    settings_limit = state.get("_recompose_limit", 2)
    if guard_state.get("style") and not guard_state.get("safety") and attempts <= settings_limit:
        already_stripped = state.get("reply") == guard.safe_template()
        if not already_stripped and attempts > 0:
            return "compose"
    return "deliver"


async def deliver(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    """The one node that is not idempotent.

    A crash between "sent" and "recorded" makes a retry re-send, so the port
    prefers a missed message to a duplicate — a duplicate DM to a lead is worse
    than a late one. See decision D14.
    """
    cfg = _cfg(config)
    channel = cfg.get("channel")
    if channel is None or not state.get("reply"):
        return {"delivered": False}

    import uuid as _uuid

    receipt = await channel.send(
        OutboundMessage(
            conversation_id=_uuid.UUID(state["conversation_id"]),
            channel=state["channel"],
            channel_user_id=state["channel_user_id"],
            text=state["reply"],
            idempotency_key=f"{state['conversation_id']}:{state['agent_turn_index']}",
        )
    )
    return {"delivered": receipt.delivered}


async def post_turn(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    """The single write that makes the turn durable.

    Everything the next turn reads is set here. A crash before this point loses
    the turn and re-drives it from the inbound message, which `messages` still
    holds — the bounded failure D4 accepts.
    """
    cfg = _cfg(config)
    db = cfg["db"]
    session = await repo.load_session(db, state["channel"], state["channel_user_id"])
    assert session is not None
    conv = session.conversation

    await repo.record_inbound(db, conv, cfg["envelope"])

    if not state.get("gate_ok", True):
        await db.commit()
        return {"turn_id": None}

    turn = await repo.open_turn(db, conv, state["agent_turn_index"])

    if state.get("transition"):
        from app.funnel.transitions import Decision, Trigger

        t = state["transition"]
        await repo.record_transition(
            db,
            conv,
            turn,
            Decision(
                from_stage=FunnelStage(t["from_stage"]),
                decided_stage=FunnelStage(t["decided_stage"]),
                proposed_stage=(
                    FunnelStage(t["proposed_stage"]) if t["proposed_stage"] else None
                ),
                accepted=t["accepted"],
                trigger=Trigger.SYSTEM,
                guard_result=t["guard_result"],
                override_reason=t["override_reason"],
            ),
        )

    summary = TypedSummary.model_validate(state.get("summary") or {})
    slots = Slots.model_validate(state.get("slots") or {})
    proposal_data = state.get("proposal")
    if proposal_data:
        proposal = StageProposal.model_validate(proposal_data)
        summary = _fold_summary(summary, proposal, state["agent_turn_index"])
        slots = _fold_slots(slots, proposal, cfg["settings"])

    await repo.complete_turn(
        db,
        conv,
        turn,
        reply=state.get("reply"),
        summary=summary,
        slots=slots,
        channel=state["channel"],
    )
    for row in state.get("metrics", []):
        await repo.record_metrics(db, conv, turn, **row)
    await db.commit()
    return {"turn_id": str(turn.id)}


def _fold_summary(
    summary: TypedSummary, proposal: StageProposal, turn_index: int
) -> TypedSummary:
    """Add or refine, never delete (D3)."""
    return summary.merged_with(
        TypedSummary(last_updated_turn=turn_index), at_turn=turn_index
    )


def _fold_slots(slots: Slots, proposal: StageProposal, settings: Settings) -> Slots:
    """Record what the lead said, then let the *tools* decide validity.

    The model reports a raw string and never a verdict. A model that could mark
    a phone number valid could book on a number nobody can call, which is why
    `SlotReading` has no `phone_e164` field to put an answer in.

    Only `phone` and `datetime` have validators today; `name` is accepted as
    given. Day and time stay SUPPLIED until `resolve_datetime` runs against a
    concrete `now`, which the graph does at booking time rather than here.
    """
    from app.memory.summary import Slot, SlotStatus
    from app.tools.phone import validate_phone

    data = slots.model_dump()
    for field, raw in (
        ("name", proposal.slots.name),
        ("phone", proposal.slots.phone_raw),
        ("day", proposal.slots.day_expression),
        ("time", proposal.slots.time_expression),
    ):
        if not raw:
            continue
        current = Slot.model_validate(data[field])
        if current.raw == raw:
            continue  # nothing new was said about this slot

        attempts = current.attempts + 1
        if field == "phone":
            result = validate_phone(raw, settings.default_phone_region)
            data[field] = Slot(
                raw=raw,
                value=result.e164,
                status=SlotStatus.VALID if result.ok else SlotStatus.INVALID,
                error=str(result.error) if result.error else None,
                attempts=attempts,
            ).model_dump()
        elif field == "name":
            data[field] = Slot(
                raw=raw, value=raw.strip(), status=SlotStatus.VALID, attempts=attempts
            ).model_dump()
        else:
            data[field] = Slot(
                raw=raw, value=None, status=SlotStatus.SUPPLIED, attempts=attempts
            ).model_dump()
    return Slots.model_validate(data)


def route_after_gate(state: TurnState) -> str:
    return "supervisor" if state.get("gate_ok", True) else "post_turn"


def route_after_supervisor(state: TurnState) -> str:
    plan = TurnPlan.model_validate(state.get("plan") or {})
    return "knowledge" if plan.consults_knowledge else "strategy"
