"""Operator CLI.

Phase 1 ships `doctor` only. The conversational CLI channel is Phase 2.

`doctor` exists because the three things most likely to be wrong on a fresh
checkout -- an absent API key, an unmigrated database, and a model id that has
been retired since this was written -- are all cheap to check up front and
expensive to diagnose from a failed conversation.
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from app.config import LLMProviderName, SchemaMode, get_settings
from app.db.base import configure_event_loop
from app.logging import configure_console, configure_logging

app = typer.Typer(add_completion=False, help="Musterform DM agent — operator commands")
console = Console()

OK = "[green]ok[/green]"
FAIL = "[red]fail[/red]"
WARN = "[yellow]warn[/yellow]"


@app.command()
def doctor(
    check_llm: bool = typer.Option(
        True, "--llm/--no-llm", help="Make one live provider call (uses quota)."
    ),
) -> None:
    """Verify configuration, database and provider before anything else runs."""
    configure_console()
    configure_event_loop()
    rows: list[tuple[str, str, str]] = []
    ok = True

    try:
        settings = get_settings()
        configure_logging(settings)
        rows.append(("config", OK, f"provider={settings.llm_provider} mode={settings.schema_mode}"))
    except Exception as exc:
        console.print(f"[red]configuration invalid:[/red] {exc}")
        raise typer.Exit(1) from exc

    status, detail = asyncio.run(_check_database(settings.database_url))
    ok &= status == OK
    rows.append(("database", status, detail))

    if check_llm:
        status, detail = asyncio.run(_check_provider())
        ok &= status == OK
        rows.append(("llm provider", status, detail))
    else:
        rows.append(("llm provider", WARN, "skipped (--no-llm)"))

    import os

    tracing = os.environ.get("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"}
    rows.append(
        (
            "langsmith",
            WARN if tracing else OK,
            "ENABLED - traces leave the machine" if tracing else "off",
        )
    )

    table = Table(show_header=True, header_style="bold")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    for name, status, detail in rows:
        table.add_row(name, status, detail)
    console.print(table)
    raise typer.Exit(0 if ok else 1)


async def _check_database(url: str) -> tuple[str, str]:
    from sqlalchemy import text

    from app.db.base import build_engine

    engine = build_engine()
    try:
        async with engine.connect() as conn:
            version = (await conn.execute(text("select version()"))).scalar_one()
            try:
                rev = (
                    await conn.execute(text("select version_num from alembic_version"))
                ).scalar_one_or_none()
            except Exception:  # noqa: BLE001 - table absent before the first migration
                rev = None
        server = str(version).split(",")[0]
        if rev is None:
            return WARN, f"{server} — no migrations applied; run `alembic upgrade head`"
        return OK, f"{server} — at revision {rev}"
    except Exception as exc:  # noqa: BLE001
        return FAIL, f"{type(exc).__name__}: {exc}"
    finally:
        await engine.dispose()


async def _check_provider() -> tuple[str, str]:
    from pydantic import BaseModel

    from app.llm.base import Message, Role
    from app.llm.factory import build_provider

    class Ping(BaseModel):
        ok: bool

    settings = get_settings()
    try:
        provider = build_provider(settings)
        result = await provider.complete(
            [Message(Role.USER, "Reply with {\"ok\": true} and nothing else.")],
            schema=Ping,
            schema_mode=SchemaMode.NATIVE
            if provider.supports_native_schema()
            else SchemaMode.PROMPT,
        )
        if not result.parse_ok:
            return FAIL, f"{provider.model} replied but did not validate: {result.parse_error}"
        return (
            OK,
            f"{provider.model} — {result.latency_ms} ms, "
            f"{result.tokens_in}+{result.tokens_out} tokens",
        )
    except Exception as exc:  # noqa: BLE001
        hint = ""
        if settings.llm_provider is LLMProviderName.GEMINI:
            hint = "  (check GEMINI_API_KEY and GEMINI_MODEL; `models` lists what your key can see)"
        return FAIL, f"{type(exc).__name__}: {exc}{hint}"


@app.command()
def chat(user: str = typer.Option("local", help="Channel user id for this conversation.")) -> None:
    """Hold a conversation in the terminal. The Phase 2 checkpoint."""
    from app.chat import main_chat

    main_chat(user)


@app.command()
def turn(
    text: list[str] = typer.Argument(
        ..., help="One line per message; together they are one burst."
    ),
    user: str = typer.Option("local", help="Channel user id."),
) -> None:
    """Run exactly one turn, then exit.

    Each invocation is a separate process, so a sequence of these is a genuine
    restart between every turn rather than a simulated one. That is how the
    README's restart transcript is produced.
    """
    import asyncio

    from app.chat import chat_burst
    from app.db.base import configure_event_loop

    configure_console()
    configure_event_loop()
    configure_logging(get_settings())
    state = asyncio.run(chat_burst(user, list(text)))
    if not state.get("gate_ok", True):
        console.print(f"[yellow]dropped:[/yellow] {state.get('drop_reason')}")
        raise typer.Exit(0)
    console.print(f"[bold cyan]coach[/bold cyan]  {state.get('reply')}")
    console.print(
        f"[dim]stage={state.get('decided_stage')} turn={state.get('agent_turn_index')}[/dim]"
    )


@app.command()
def snapshot(
    user: str = typer.Option("local", help="Channel user id to inspect."),
) -> None:
    """Print the persisted funnel state, slots and turn index.

    Reads the application tables, not the checkpointer — the point of the
    restart demonstration is that the tables are the source of truth (D4).
    """
    from app.chat import main_snapshot

    main_snapshot(user)


@app.command()
def models() -> None:
    """List the models the configured provider exposes.

    A retired model id is a plausible failure between now and whenever this is
    next run, and it is much clearer to see it here than in a 404 mid-turn.
    """
    from app.llm.factory import build_provider

    provider = build_provider()
    names = asyncio.run(provider.list_models())  # type: ignore[attr-defined]
    for n in sorted(names):
        console.print(n)


if __name__ == "__main__":
    app()
