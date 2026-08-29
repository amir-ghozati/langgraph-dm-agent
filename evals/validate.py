"""Validate the golden set without running it.

    python -m evals.validate            # shape only
    python -m evals.validate --strict   # also require an expect: block

Exists so that eight hand-written conversation files are known to be readable
by the runner before the runner exists. The failure this prevents is writing
all eight and discovering in Phase 3b that none of them load.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from evals.judge_schema import JudgeSet, required_sets
from evals.schema import GoldenConversation

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()

GOLDEN_DIR = Path(__file__).parent / "golden"
JUDGE_DIR = Path(__file__).parent / "judge"


def load_all(directory: Path = GOLDEN_DIR) -> list[tuple[Path, GoldenConversation | str]]:
    """Returns (path, parsed-or-error-string) for every YAML file found."""
    out: list[tuple[Path, GoldenConversation | str]] = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            out.append((path, f"invalid YAML: {exc}"))
            continue
        if raw is None:
            out.append((path, "file is empty"))
            continue
        try:
            out.append((path, GoldenConversation.model_validate(raw)))
        except ValidationError as exc:
            first = exc.errors()[0]
            loc = ".".join(str(p) for p in first["loc"])
            out.append((path, f"{loc}: {first['msg']}"))
    return out


@app.command()
def judge(
    directory: Path = typer.Option(JUDGE_DIR, "--dir", help="Directory of judge-set YAML files."),
) -> None:
    """Validate the judge's own labelled sets.

    Every judged SAFETY predicate needs one before its results in the main
    report mean anything: an unmeasured judge saying "fine" is not evidence.
    """
    found: dict[str, JudgeSet] = {}
    failures = 0
    table = Table(show_header=True, header_style="bold")
    for column in ("file", "status", "cases", "v/c", "borderline", "detail"):
        table.add_column(column)

    for path in sorted(directory.glob("*.yaml")):
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            js = JudgeSet.model_validate(raw)
        except (yaml.YAMLError, ValidationError) as exc:
            failures += 1
            detail = exc.errors()[0]["msg"] if isinstance(exc, ValidationError) else str(exc)
            table.add_row(path.name, "[red]invalid[/red]", "-", "-", "-", detail[:70])
            continue
        s = js.summary()
        # Draft sets load and are readable, but they do not validate a judge.
        if js.is_validated:
            found[js.predicate] = js
        table.add_row(
            path.name,
            "[green]ok[/green]",
            str(s["cases"]),
            f"{s['violation']}/{s['compliant']}",
            str(s["borderline"]),
            (
                "DRAFT - awaiting review, does not validate"
                if not js.is_validated
                else f"reviewed, threshold {s['threshold']}"
            ),
        )

    console.print(table)
    missing = [p for p in required_sets() if p not in found]
    if missing:
        console.print(f"[yellow]no reviewed set yet ({len(missing)}):[/yellow]")
        for name in missing:
            console.print(f"  {name}")
    console.print(
        f"{len(found)}/{len(required_sets())} judged safety predicates have a REVIEWED set"
    )
    raise typer.Exit(1 if failures else 0)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    strict: bool = typer.Option(
        False, "--strict", help="Fail when a conversation has no expect: block."
    ),
    directory: Path = typer.Option(GOLDEN_DIR, "--dir", help="Directory of golden YAML files."),
) -> None:
    """Validate the golden set. `... validate judge` checks the judge sets instead."""
    # Registered as a callback rather than a command so the golden-set check
    # stays the bare `python -m evals.validate [--strict]` it has always been.
    # Adding `judge` as a second command would otherwise have demoted it to
    # `validate main --strict` and silently broken every documented invocation.
    if ctx.invoked_subcommand is not None:
        return

    results = load_all(directory)
    if not results:
        console.print(f"[red]no YAML files found in {directory}[/red]")
        raise typer.Exit(1)

    table = Table(show_header=True, header_style="bold")
    for column in ("file", "status", "turns", "expect", "detail"):
        table.add_column(column)

    failures = 0
    for path, result in results:
        if isinstance(result, str):
            failures += 1
            table.add_row(path.name, "[red]invalid[/red]", "-", "-", result)
            continue
        has_expect = result.expect is not None
        if strict and not has_expect:
            failures += 1
        table.add_row(
            path.name,
            "[green]ok[/green]",
            f"{result.inbound_count} in / {result.agent_turn_count} agent",
            "[green]yes[/green]" if has_expect else "[yellow]missing[/yellow]",
            result.title,
        )

    console.print(table)
    total = len(results)
    console.print(
        f"{total - failures}/{total} valid"
        + (" (strict: expect: block required)" if strict else "")
    )
    raise typer.Exit(1 if failures else 0)


if __name__ == "__main__":
    sys.exit(app())
