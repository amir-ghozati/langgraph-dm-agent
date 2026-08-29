"""Burst aggregation: consecutive inbound messages are one turn.

Three reasons, from D24. Real DM users send fragments and the meaning is
distributed across them — the first is frequently a greeting carrying nothing.
Replying per message produces three replies to a three-message burst, which is
the most bot-like behaviour a DM agent can exhibit and exactly what `smoke-05`
is written to catch. And `agent_turn_index` counts agent replies, so bursts do
not inflate the funnel clock: a chatty lead must not hit `FUNNEL_HARD_CAP`
faster than a terse one.

The source diverges here. Each Instagram webhook fired the workflow
independently, so the original replied per message.

Two implementations of one rule:

* **Scripted** (`group_bursts`) — the golden-set format encodes bursts
  structurally, as consecutive `inbound:` entries with no `agent:` between
  them. No timing involved. This is also why a scripted conversation's turn
  count is derived from the grouping and never from counting `agent:`
  placeholders — `GoldenConversation.agent_turn_count` calls straight into
  `group_bursts` for exactly that reason.
* **Live** (`BurstBuffer`) — a debounce. Hold inbound, restart the timer on
  each new message, release when it expires.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from app.channels.base import InboundMessage


@dataclass(frozen=True, slots=True)
class InboundEnvelope:
    """One agent turn's worth of inbound. The unit the graph acts on."""

    channel: str
    channel_user_id: str
    messages: tuple[InboundMessage, ...]

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("an envelope needs at least one message")
        senders = {(m.channel, m.channel_user_id) for m in self.messages}
        if senders != {(self.channel, self.channel_user_id)}:
            raise ValueError(f"envelope mixes senders: {sorted(senders)}")

    @classmethod
    def of(cls, messages: Sequence[InboundMessage]) -> InboundEnvelope:
        first = messages[0]
        return cls(
            channel=first.channel,
            channel_user_id=first.channel_user_id,
            messages=tuple(messages),
        )

    @property
    def text(self) -> str:
        """What intent classification reads.

        The whole burst, not the first message: "heyy loved the free guide" +
        "hows pricing work" is a pricing question, and classifying on the first
        fragment alone would route it to the opener and answer a pricing
        question with a goal question.
        """
        return "\n".join(m.text for m in self.messages)

    @property
    def received_at(self) -> dt.datetime:
        return self.messages[-1].received_at

    @property
    def channel_message_ids(self) -> list[str]:
        """Only the real ones. What the partial unique index on `messages`
        dedups against; channels that issue no id contribute nothing, which is
        why that index is partial."""
        return [m.channel_message_id for m in self.messages if m.channel_message_id is not None]

    def __len__(self) -> int:
        return len(self.messages)


def group_bursts[T](items: Sequence[T], is_inbound: Callable[[T], bool]) -> list[list[T]]:
    """Runs of consecutive inbound items.

    Generic so one rule serves the golden-set YAML and a list of live messages,
    rather than two implementations that can drift apart.
    """
    runs: list[list[T]] = []
    current: list[T] = []
    for item in items:
        if is_inbound(item):
            current.append(item)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


@dataclass(slots=True)
class BurstBuffer:
    """The live debounce, as pure logic so it can be tested without sleeping.

    Hold inbound; restart the timer on each new message; release when
    `window_seconds` have passed since the last one. The async wrapper is a
    thin loop around this, and `CLIChannel` skips it entirely in favour of an
    explicit send affordance — the property worth testing is that N messages
    produce one reply, not that a clock works.
    """

    window_seconds: float = 5.0
    _pending: list[InboundMessage] = field(default_factory=list)
    _last_at: dt.datetime | None = None

    def __post_init__(self) -> None:
        if self.window_seconds < 0:
            raise ValueError("window_seconds must not be negative")

    def add(self, message: InboundMessage, *, now: dt.datetime | None = None) -> None:
        if self._pending:
            first = self._pending[0]
            if (message.channel, message.channel_user_id) != (first.channel, first.channel_user_id):
                raise ValueError("a buffer holds one sender's burst at a time")
        self._pending.append(message)
        self._last_at = now or message.received_at

    def due(self, now: dt.datetime) -> bool:
        """An empty buffer is never due — otherwise the worker would emit
        empty turns forever between conversations."""
        if not self._pending or self._last_at is None:
            return False
        return now >= self._last_at + dt.timedelta(seconds=self.window_seconds)

    def flush(self) -> InboundEnvelope | None:
        """Take everything held, regardless of the timer.

        Called on expiry and on shutdown. Dropping a half-collected burst
        because the process is stopping loses a real user message.
        """
        if not self._pending:
            return None
        envelope = InboundEnvelope.of(self._pending)
        self._pending = []
        self._last_at = None
        return envelope

    def __len__(self) -> int:
        return len(self._pending)
