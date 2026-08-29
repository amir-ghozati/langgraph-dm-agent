"""Interactive terminal channel. The primary development interface.

Bursts are an explicit affordance here rather than a timer: type as many lines
as you like, then submit with a blank line. Real timing would make the
development loop wait five seconds for every turn and would make tests sleep,
which buys nothing — the property under test is that N messages produce one
reply, not that a clock works.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import AsyncIterator

from rich.console import Console

from app.channels.base import DeliveryReceipt, InboundMessage, OutboundMessage

BANNER = """\
[dim]Type your message. A blank line sends it.
Consecutive lines are one burst and get one reply, the way a real DM burst does.
Ctrl-D or /quit to exit.[/dim]
"""


class CLIChannel:
    name = "cli"

    def __init__(
        self,
        channel_user_id: str = "local",
        console: Console | None = None,
    ) -> None:
        self.channel_user_id = channel_user_id
        self.console = console or Console()

    async def receive(self) -> AsyncIterator[InboundMessage]:
        self.console.print(BANNER)
        while True:
            lines = await asyncio.to_thread(self._read_burst)
            if lines is None:
                return
            for line in lines:
                yield InboundMessage(
                    channel=self.name,
                    channel_user_id=self.channel_user_id,
                    text=line,
                    received_at=dt.datetime.now(dt.UTC),
                )

    def _read_burst(self) -> list[str] | None:
        """Blocking read of one burst. Returns None on EOF or /quit."""
        lines: list[str] = []
        while True:
            try:
                line = input("> " if not lines else ". ")
            except EOFError:
                return lines or None
            stripped = line.strip()
            if stripped in {"/quit", "/exit"}:
                return None
            if not stripped:
                if lines:
                    return lines
                continue
            lines.append(stripped)

    async def send(self, message: OutboundMessage) -> DeliveryReceipt:
        self.console.print(f"[bold cyan]coach[/bold cyan]  {message.text}")
        return DeliveryReceipt(
            delivered=True,
            channel_message_id=None,
            delivered_at=dt.datetime.now(dt.UTC),
        )
