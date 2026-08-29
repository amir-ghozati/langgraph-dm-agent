"""HTTP channel: a FastAPI webhook receiver plus a synchronous reply.

The same graph, the same funnel, the same tables — only the transport changes.
That is the point of the `Channel` protocol, and the acceptance criterion is
that a conversation held over HTTP behaves identically to one held in the CLI.

Two things worth noting about the shape:

* **The channel is not in graph state.** It is passed through
  `RunnableConfig["configurable"]`, because graph state is pickled into
  Postgres and a live channel object is not serialisable.
* **Bursts arrive differently here.** The CLI has an explicit send affordance;
  HTTP callers post a list of messages, and the endpoint treats that list as
  one burst. A real webhook (Instagram) delivers one message per request, which
  is what `BurstBuffer` and `BURST_WINDOW_SECONDS` are for — that debounce is
  documented in `app/channels/instagram.py` and not implemented here.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from app.channels.base import DeliveryReceipt, InboundMessage, OutboundMessage
from app.channels.envelope import InboundEnvelope
from app.logging import get_logger

log = get_logger(__name__)


class InboundRequest(BaseModel):
    """One burst. A list, because a burst is the unit the graph acts on."""

    channel_user_id: str = Field(min_length=1, max_length=128)
    messages: list[str] = Field(min_length=1)
    channel_message_ids: list[str] = Field(default_factory=list)
    """Optional. When supplied, redelivery is deduplicated on
    `(channel, channel_message_id)` exactly as it would be for a real webhook."""


class TurnResponse(BaseModel):
    conversation_id: str | None = None
    reply: str | None = None
    stage: str | None = None
    agent_turn_index: int | None = None
    delivered: bool = False
    dropped: bool = False
    drop_reason: str | None = None
    interrupted: bool = False
    interrupt_reason: str | None = None


class HTTPChannel:
    """Collects the reply for the responding request rather than sending it.

    `send` is called by the `deliver` node; the value is read back by the
    endpoint and returned in the HTTP response. That keeps `deliver` unaware of
    whether it is talking to a terminal, a webhook, or a test.
    """

    name = "http"

    def __init__(self) -> None:
        self.sent: list[OutboundMessage] = []

    def receive(self) -> AsyncIterator[InboundMessage]:
        raise NotImplementedError(
            "HTTPChannel is driven by the FastAPI endpoint, not by a receive loop."
        )

    async def send(self, message: OutboundMessage) -> DeliveryReceipt:
        self.sent.append(message)
        return DeliveryReceipt(delivered=True, channel_message_id=str(uuid.uuid4()))

    @property
    def last_reply(self) -> str | None:
        return self.sent[-1].text if self.sent else None


def build_app(settings=None, provider=None, sessionmaker=None):
    """Construct the FastAPI app.

    Dependencies are injected rather than imported at module scope so tests can
    supply a scripted provider without patching, and so importing this module
    never opens a database connection.
    """
    from fastapi import FastAPI, HTTPException
    from langgraph.checkpoint.memory import InMemorySaver

    from app.config import get_settings
    from app.db.base import build_sessionmaker
    from app.graph.build import build_graph, conversation_snapshot, run_turn
    from app.llm.factory import build_provider

    settings = settings or get_settings()
    provider = provider or build_provider(settings)
    sessionmaker = sessionmaker or build_sessionmaker()

    # An in-memory saver, not the Postgres one. `interrupt()` requires *a*
    # checkpointer to hold the paused turn, and this transport has no resume
    # endpoint: a flagged turn is reported and dropped rather than parked for a
    # human. Conversation-level resumption is unaffected — the funnel state
    # lives in the tables (D4), not the checkpoint — so an HTTP conversation
    # survives a restart exactly as the CLI one does. Only mid-turn resumption
    # is out of scope; see docs/KNOWN_GAPS.md.
    graph = build_graph().compile(checkpointer=InMemorySaver())

    app = FastAPI(
        title="Musterform DM agent",
        summary="HTTP transport over the same graph the CLI uses.",
        version="0.1.0",
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "provider": provider.name, "model": provider.model}

    @app.post("/inbound", response_model=TurnResponse)
    async def inbound(request: InboundRequest) -> TurnResponse:
        envelope = InboundEnvelope.of(
            [
                InboundMessage(
                    channel="http",
                    channel_user_id=request.channel_user_id,
                    text=text,
                    channel_message_id=(
                        request.channel_message_ids[i]
                        if i < len(request.channel_message_ids)
                        else None
                    ),
                )
                for i, text in enumerate(request.messages)
            ]
        )
        channel = HTTPChannel()
        async with sessionmaker() as db:
            state = await run_turn(
                graph,
                envelope,
                db=db,
                provider=provider,
                settings=settings,
                channel=channel,
            )

        interrupts = state.get("__interrupt__") or []
        # A paused turn must not return its draft. `deliver` never ran, but
        # handing the flagged text back in the response body *is* sending it —
        # the caller has no way to know it was withheld.
        reply = None if interrupts else state.get("reply")
        return TurnResponse(
            conversation_id=state.get("conversation_id"),
            reply=reply,
            stage=state.get("decided_stage") or state.get("stage"),
            agent_turn_index=state.get("agent_turn_index"),
            delivered=bool(state.get("delivered")),
            dropped=not state.get("gate_ok", True),
            drop_reason=state.get("drop_reason"),
            interrupted=bool(interrupts) or bool(state.get("interrupted")),
            interrupt_reason=(interrupts[0].value.get("reason") if interrupts else None),
        )

    @app.get("/conversations/{channel_user_id}")
    async def snapshot(channel_user_id: str) -> dict[str, Any]:
        """The persisted funnel state, read from the tables.

        Same data the CLI's `snapshot` command prints, for the same reason: the
        claim being demonstrated is that the tables are authoritative.
        """
        async with sessionmaker() as db:
            data = await conversation_snapshot(db, "http", channel_user_id)
        if not data.get("found"):
            raise HTTPException(status_code=404, detail="no live conversation")
        return data

    return app
