"""The channel abstraction.

The graph must not know which channel it is talking to. A live channel object
is also not checkpoint-serialisable, so it is passed through
`RunnableConfig["configurable"]` and never through graph state — small, but
exactly the kind of detail that decides whether resume-after-crash works.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


@dataclass(frozen=True, slots=True)
class InboundMessage:
    """One message as the channel delivered it.

    Not a turn. Real users send fragments, and the unit the graph acts on is
    the burst — see `app.channels.envelope`.
    """

    channel: str
    channel_user_id: str
    text: str
    channel_message_id: str | None = None
    received_at: dt.datetime = field(default_factory=_now)


@dataclass(frozen=True, slots=True)
class OutboundMessage:
    conversation_id: uuid.UUID
    channel: str
    channel_user_id: str
    text: str
    turn_id: uuid.UUID | None = None
    idempotency_key: str | None = None
    """Written to a `SENDING` delivery intent before the channel is called. On
    resume, an intent still in `SENDING` is not re-sent — we prefer a missed
    message to a duplicate. See decision D14."""


@dataclass(frozen=True, slots=True)
class DeliveryReceipt:
    delivered: bool
    channel_message_id: str | None = None
    delivered_at: dt.datetime = field(default_factory=_now)
    error: str | None = None


@runtime_checkable
class Channel(Protocol):
    name: str

    def receive(self) -> AsyncIterator[InboundMessage]: ...

    async def send(self, message: OutboundMessage) -> DeliveryReceipt: ...
