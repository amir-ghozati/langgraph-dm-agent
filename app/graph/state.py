"""Graph state.

What lives here versus in the tables is D4: this holds only what a half-finished
turn needs to resume. Anything you would query in a SQL report — the funnel
stage, the slots, the summary, the turn index — is loaded from `conversations`
at the start of every turn and written back at the end.

That split is why `tests/test_restart.py` reads nothing from the checkpointer.
A conversation whose funnel had silently reset to `NEW` would still *look*
fine if the checkpointer were the source of truth, because it restores its own
message list without help.

Everything here must be JSON-serialisable: LangGraph pickles state into
Postgres. The channel is passed through `RunnableConfig["configurable"]`
instead, because a live channel object is not.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, TypedDict

from app.funnel.stages import FunnelStage


def _last(_old: Any, new: Any) -> Any:
    """Last write wins. The graph is sequential, so there is nothing to merge —
    and a reducer that silently merged would hide a node writing twice."""
    return new


class TurnState(TypedDict, total=False):
    """One turn through the graph."""

    # --- set by normalize ---------------------------------------------------
    channel: Annotated[str, _last]
    channel_user_id: Annotated[str, _last]
    inbound_text: Annotated[str, _last]
    """The whole burst, newline-joined. Intent is classified on this, not on
    the first message — see D24."""

    channel_message_ids: Annotated[list[str], _last]

    # --- set by load_session ------------------------------------------------
    conversation_id: Annotated[str, _last]
    """Also the checkpointer's `thread_id`. A UUID as a string, because state
    must serialise."""

    lead_id: Annotated[str, _last]
    stage: Annotated[str, _last]
    agent_turn_index: Annotated[int, _last]
    summary: Annotated[dict[str, Any], _last]
    slots: Annotated[dict[str, Any], _last]
    window: Annotated[str, _last]

    # --- set by gate --------------------------------------------------------
    gate_ok: Annotated[bool, _last]
    drop_reason: Annotated[str | None, _last]

    # --- set by the planning and execution nodes ---------------------------
    plan: Annotated[dict[str, Any], _last]
    knowledge: Annotated[dict[str, Any], _last]
    proposal: Annotated[dict[str, Any], _last]

    # --- set by funnel_transition ------------------------------------------
    decided_stage: Annotated[str, _last]
    transition: Annotated[dict[str, Any], _last]

    # --- set by guardrail ---------------------------------------------------
    guard: Annotated[dict[str, Any], _last]
    recompose_attempts: Annotated[int, _last]
    interrupted: Annotated[bool, _last]
    handed_off: Annotated[bool, _last]

    # --- set by compose / deliver ------------------------------------------
    reply: Annotated[str | None, _last]
    reply_meta: Annotated[dict[str, Any], _last]
    delivered: Annotated[bool, _last]

    # --- accumulated across nodes ------------------------------------------
    metrics: Annotated[list[dict[str, Any]], _last]
    """One row per LLM call, flushed to `turn_metrics` by post_turn."""

    turn_id: Annotated[str | None, _last]


def new_turn_state(
    channel: str, channel_user_id: str, text: str, channel_message_ids: list[str]
) -> TurnState:
    return TurnState(
        channel=channel,
        channel_user_id=channel_user_id,
        inbound_text=text,
        channel_message_ids=channel_message_ids,
        gate_ok=True,
        drop_reason=None,
        metrics=[],
        recompose_attempts=0,
        interrupted=False,
        handed_off=False,
        reply=None,
        delivered=False,
        turn_id=None,
    )


def stage_of(state: TurnState) -> FunnelStage:
    return FunnelStage(state.get("decided_stage") or state.get("stage") or FunnelStage.NEW)


def thread_id(conversation_id: uuid.UUID | str) -> str:
    """`thread_id = conversation_id`, never the channel sender id.

    The source keyed all three of its memory buffers on the Instagram sender
    id, so a lead who booked, finished, and messaged again six weeks later
    resumed the same buffer and the same finished state.
    """
    return str(conversation_id)
