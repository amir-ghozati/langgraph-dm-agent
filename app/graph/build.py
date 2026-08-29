"""Graph construction and the turn driver.

Shape:

    normalize -> load_session -> gate -+-> supervisor -+-> knowledge -> strategy
                                       |               `-> strategy
                                       `-> post_turn        |
                                                            v
                                                    funnel_transition
                                                            |
                                                            v
                              compose <-> guardrail -> deliver -> post_turn
                                             |
                                        [[interrupt]]

The `compose <-> guardrail` loop is bounded by `GUARDRAIL_RECOMPOSE_LIMIT`;
style violations retry, safety violations interrupt instead. The interrupt
fires before `deliver`, so nothing has been sent while a human decides.

The knowledge branch is the delegation delta made structural: the supervisor
decides whether to consult the corpus at all, and the graph skips the node when
it does not. The source called it every turn with the raw webhook text.

Tools, guardrails and the HITL interrupt attach between `funnel_transition` and
`compose` in Phase 3a.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.channels.base import Channel
from app.channels.envelope import InboundEnvelope
from app.config import Settings, get_settings
from app.db import repository as repo
from app.graph import nodes
from app.graph.state import TurnState, new_turn_state, thread_id
from app.llm.base import LLMProvider
from app.logging import get_logger

log = get_logger(__name__)


def build_graph():
    """The compiled graph, without a checkpointer.

    Separated so tests can compile once and attach whatever saver they need —
    and so a checkpointer failure is never mistaken for a graph-shape failure.
    """
    g = StateGraph(TurnState)

    g.add_node("normalize", nodes.normalize)
    g.add_node("load_session", nodes.load_session)
    g.add_node("gate", nodes.gate)
    g.add_node("supervisor", nodes.supervisor)
    g.add_node("knowledge", nodes.knowledge)
    g.add_node("strategy", nodes.strategy)
    g.add_node("funnel_transition", nodes.funnel_transition)
    g.add_node("compose", nodes.compose)
    g.add_node("guardrail", nodes.guardrail)
    g.add_node("deliver", nodes.deliver)
    g.add_node("post_turn", nodes.post_turn)

    g.add_edge(START, "normalize")
    g.add_edge("normalize", "load_session")
    g.add_edge("load_session", "gate")
    g.add_conditional_edges(
        "gate", nodes.route_after_gate, {"supervisor": "supervisor", "post_turn": "post_turn"}
    )
    g.add_conditional_edges(
        "supervisor",
        nodes.route_after_supervisor,
        {"knowledge": "knowledge", "strategy": "strategy"},
    )
    g.add_edge("knowledge", "strategy")
    g.add_edge("strategy", "funnel_transition")
    g.add_edge("funnel_transition", "compose")
    g.add_edge("compose", "guardrail")
    g.add_conditional_edges(
        "guardrail", nodes.route_after_guardrail, {"compose": "compose", "deliver": "deliver"}
    )
    g.add_edge("deliver", "post_turn")
    g.add_edge("post_turn", END)
    return g


@asynccontextmanager
async def checkpointer(settings: Settings | None = None):
    """Postgres checkpointing.

    Postgres rather than in-memory because a portfolio repo that loses its
    conversation on restart cannot demonstrate the one acceptance criterion
    that matters. What it stores is LangGraph's business — the app tables hold
    everything the funnel actually depends on (D4).
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    settings = settings or get_settings()
    # langgraph-checkpoint-postgres speaks psycopg's own URL form, not
    # SQLAlchemy's dialect-qualified one.
    dsn = settings.database_url.replace("postgresql+psycopg://", "postgresql://")
    async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
        await saver.setup()
        yield saver


async def run_turn(
    graph,
    envelope: InboundEnvelope,
    *,
    db,
    provider: LLMProvider,
    settings: Settings,
    channel: Channel | None = None,
) -> TurnState:
    """Drive one turn.

    The conversation is resolved before the graph runs, because `thread_id`
    must be the conversation id and the checkpointer needs it up front.
    """
    session = await repo.load_session(db, envelope.channel, envelope.channel_user_id)
    assert session is not None
    await db.commit()

    state = new_turn_state(
        envelope.channel,
        envelope.channel_user_id,
        envelope.text,
        envelope.channel_message_ids,
    )
    config: dict[str, Any] = {
        "configurable": {
            "thread_id": thread_id(session.conversation.id),
            "db": db,
            "provider": provider,
            "settings": settings,
            "channel": channel,
            "envelope": envelope,
        }
    }
    return await graph.ainvoke(state, config=config)


async def conversation_snapshot(db, channel: str, channel_user_id: str) -> dict[str, Any]:
    """What the restart demonstration prints.

    Deliberately reads through the repository rather than the checkpointer:
    the claim being demonstrated is that the *tables* hold the funnel state.
    """
    session = await repo.load_session(db, channel, channel_user_id, create=False)
    if session is None:
        return {"found": False}
    return {
        "found": True,
        "conversation_id": str(session.conversation.id),
        "stage": str(session.stage),
        "agent_turns_sent": session.conversation.agent_turn_count,
        "next_turn_index": session.agent_turn_index,
        "slots": {
            field: {
                "raw": getattr(session.slots, field).raw,
                "value": getattr(session.slots, field).value,
                "status": str(getattr(session.slots, field).status),
                "error": getattr(session.slots, field).error,
                "attempts": getattr(session.slots, field).attempts,
            }
            for field in ("name", "phone", "day", "time")
        },
        "summary": session.summary.model_dump(mode="json"),
    }


__all__ = [
    "build_graph",
    "checkpointer",
    "conversation_snapshot",
    "run_turn",
    "thread_id",
    "uuid",
]
