"""The CLI conversation loop — the Phase 2 checkpoint.

    python -m app.cli chat            # talk to it
    python -m app.cli snapshot        # print persisted state, no model calls

`snapshot` exists for the restart demonstration: it reads the funnel stage,
the slots and the turn index straight out of the tables, so "it resumed" is
shown from the source of truth rather than inferred from the agent sounding
coherent.
"""

from __future__ import annotations

import uuid

from rich.console import Console

from app.channels.base import InboundMessage
from app.channels.cli import CLIChannel
from app.channels.envelope import InboundEnvelope
from app.config import Settings, get_settings
from app.db.base import build_sessionmaker, configure_event_loop
from app.graph.build import build_graph, checkpointer, conversation_snapshot, run_turn
from app.llm.factory import build_provider
from app.logging import configure_console, configure_logging, get_logger

log = get_logger(__name__)
console = Console()


async def chat(user: str = "local", settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    configure_logging(settings)

    provider = build_provider(settings)
    channel = CLIChannel(channel_user_id=user)
    sessionmaker = build_sessionmaker()
    graph_def = build_graph()

    async with checkpointer(settings) as saver:
        graph = graph_def.compile(checkpointer=saver)
        console.print(
            f"[dim]conversation for {user!r} — "
            f"provider={provider.name} model={provider.model}[/dim]"
        )
        async for message in channel.receive():
            # CLIChannel yields one InboundMessage per line of a burst; the
            # blank line that ended the burst is the send affordance, and the
            # channel has already grouped them.
            envelope = InboundEnvelope.of([message])
            async with sessionmaker() as db:
                state = await run_turn(
                    graph,
                    envelope,
                    db=db,
                    provider=provider,
                    settings=settings,
                    channel=channel,
                )
            if not state.get("gate_ok", True):
                console.print(f"[yellow]dropped:[/yellow] {state.get('drop_reason')}")
            console.print(
                f"[dim]stage={state.get('decided_stage') or state.get('stage')} "
                f"turn={state.get('agent_turn_index')}[/dim]"
            )


async def chat_burst(
    user: str, lines: list[str], settings: Settings | None = None
) -> dict:
    """One burst, one turn, no interactive loop. Used by the restart demo."""
    settings = settings or get_settings()
    provider = build_provider(settings)
    sessionmaker = build_sessionmaker()
    envelope = InboundEnvelope.of(
        [
            InboundMessage(channel="cli", channel_user_id=user, text=line)
            for line in lines
        ]
    )
    async with checkpointer(settings) as saver:
        graph = build_graph().compile(checkpointer=saver)
        async with sessionmaker() as db:
            return await run_turn(
                graph, envelope, db=db, provider=provider, settings=settings
            )


async def snapshot(user: str = "local", settings: Settings | None = None) -> dict:
    """Persisted state, read from the tables. No model calls, no checkpointer."""
    settings = settings or get_settings()
    sessionmaker = build_sessionmaker()
    async with sessionmaker() as db:
        return await conversation_snapshot(db, "cli", user)


def print_snapshot(data: dict) -> None:
    from rich.table import Table

    if not data.get("found"):
        console.print("[yellow]no live conversation for that user[/yellow]")
        return

    console.print(f"[bold]conversation[/bold] {data['conversation_id']}")
    console.print(f"  stage              [bold cyan]{data['stage']}[/bold cyan]")
    console.print(f"  agent turns sent   {data['agent_turns_sent']}")
    console.print(f"  next turn index    {data['next_turn_index']}")

    table = Table(show_header=True, header_style="bold", title="slots")
    for column in ("slot", "raw", "value", "status", "error", "attempts"):
        table.add_column(column)
    for name, slot in data["slots"].items():
        table.add_row(
            name,
            slot["raw"] or "-",
            slot["value"] or "-",
            slot["status"],
            slot["error"] or "-",
            str(slot["attempts"]),
        )
    console.print(table)

    goals = data["summary"].get("stated_goals") or []
    if goals:
        console.print(f"  goals: {'; '.join(goals)}")


def main_chat(user: str = "local") -> None:
    import asyncio

    configure_console()
    configure_event_loop()
    asyncio.run(chat(user))


def main_snapshot(user: str = "local") -> None:
    import asyncio

    configure_console()
    configure_event_loop()
    print_snapshot(asyncio.run(snapshot(user)))


__all__ = ["chat", "chat_burst", "main_chat", "main_snapshot", "print_snapshot", "snapshot", "uuid"]
