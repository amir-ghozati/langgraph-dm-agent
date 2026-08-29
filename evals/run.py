"""The eval runner.

    python -m evals.run --dry-run          # estimated wall-clock, no calls
    python -m evals.run                    # replay the golden set
    python -m evals.run --resume <run-id>  # skip conversations already done
    python -m evals.run --baseline <run-id>

Results are written per conversation as they complete, so an interrupted run
keeps everything it had finished. Each run lands in
`evals/results/<timestamp>/` with one JSON file per conversation and a
`report.md`.

Two refusals worth naming:

* **It will not run against the `scripted` provider.** That provider
  pattern-matches text; a number produced from it would describe the harness
  rather than the system, and would look identical in the README to a real one.
* **Judged predicates report `UNVALIDATED`, never `pass`.** Their judge sets
  exist but are `status: draft`, so nothing has established the judge agrees
  with a human. That is a claim about the evidence, not about the system.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from evals.predicates import Outcome, PredicateResult, TurnFacts, evaluate
from evals.schema import GoldenConversation
from evals.validate import load_all

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()

RESULTS = Path(__file__).parent / "results"

# Rough per-turn cost, used only by --dry-run. Four structured calls per turn
# against a hosted model.
SECONDS_PER_TURN = 6.0


@dataclass
class ConversationResult:
    id: str
    mode: int
    provenance: str
    agent_turns: int
    final_stage: str | None = None
    predicates: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    repair_attempts: int = 0
    first_attempt_valid: list[bool] = field(default_factory=list)
    proposals: list[bool] = field(default_factory=list)
    interrupts: int = 0
    error: str | None = None

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [p for p in self.predicates if p["outcome"] == Outcome.FAIL]

    @property
    def safety_failures(self) -> list[dict[str, Any]]:
        return [p for p in self.failures if p["safety"]]


def _now_id() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


async def replay(
    conversation: GoldenConversation, *, provider, settings, sessionmaker
) -> ConversationResult:
    """Drive one golden conversation through the real graph."""
    from app.channels.base import InboundMessage
    from app.channels.envelope import InboundEnvelope
    from app.graph.build import build_graph, conversation_snapshot, run_turn

    result = ConversationResult(
        id=conversation.id,
        mode=conversation.mode,
        provenance=str(conversation.provenance),
        agent_turns=conversation.agent_turn_count,
    )
    user = f"eval-{conversation.id}-{uuid.uuid4().hex[:8]}"
    graph = build_graph().compile()
    facts: list[TurnFacts] = []

    started = time.perf_counter()
    try:
        for index, burst in enumerate(conversation.bursts, start=1):
            envelope = InboundEnvelope.of(
                [
                    InboundMessage(channel="cli", channel_user_id=user, text=t.inbound)
                    for t in burst
                ]
            )
            async with sessionmaker() as db:
                state = await run_turn(
                    graph, envelope, db=db, provider=provider, settings=settings
                )
                snapshot = await conversation_snapshot(db, "cli", user)

            for metric in state.get("metrics", []):
                result.tokens_in += metric.get("tokens_in", 0)
                result.tokens_out += metric.get("tokens_out", 0)
                result.repair_attempts += metric.get("repair_attempts", 0)
                if metric.get("first_attempt_valid") is not None:
                    result.first_attempt_valid.append(bool(metric["first_attempt_valid"]))

            transition = state.get("transition") or {}
            if transition:
                result.proposals.append(bool(transition.get("accepted")))
            interrupted = bool(state.get("interrupted") or state.get("__interrupt__"))
            if interrupted:
                result.interrupts += 1

            facts.append(
                TurnFacts(
                    # An interrupted turn sent nothing. The first live run
                    # scored the withheld draft and reported two safety
                    # failures for `no_booking_confirmed` -- both of which were
                    # the guardrail catching that exact phrasing and refusing
                    # to send it. HTTPChannel had the same bug; reading a draft
                    # out of the state after an interrupt is how it gets out.
                    reply="" if interrupted else (state.get("reply") or ""),
                    interrupted=interrupted,
                    turn_index=index,
                    stage=state.get("decided_stage") or state.get("stage") or "NEW",
                    slots=snapshot.get("slots", {}),
                    tools_called=state.get("tool_calls", []),
                    booking_committed=(snapshot.get("stage") == "BOOKED"),
                )
            )
            result.final_stage = snapshot.get("stage")
    except Exception as exc:  # noqa: BLE001 - one bad conversation must not end a run
        result.error = f"{type(exc).__name__}: {exc}"

    result.latency_ms = int((time.perf_counter() - started) * 1000)
    result.predicates = [asdict(p) for p in _score(conversation, facts, errored=result.error)]
    result.tool_calls = [c for f in facts for c in f.tools_called]
    return result


def _score(
    conversation: GoldenConversation,
    facts: list[TurnFacts],
    *,
    errored: str | None = None,
) -> list[PredicateResult]:
    expect = conversation.expect
    if expect is None:
        return []

    if errored is not None:
        # The first live run reported "42 passed, 1 failed, 0 safety failures"
        # and exited 0 while all eight conversations had crashed. Every
        # predicate had been scored against an empty transcript, and most of
        # them are absence claims -- `no_link_emitted` over zero replies is
        # vacuously true. A pass that a dead run can produce is not a
        # measurement, so an errored conversation now yields none. Partial
        # credit for the bursts that did complete is deliberately not given:
        # a truncated transcript makes every absence claim easier to satisfy.
        return [
            PredicateResult(str(predicate), Outcome.NOT_RUN, False, False, f"errored: {errored}")
            for predicate in _every_predicate(expect)
        ]

    results: list[PredicateResult] = []
    for predicate in expect.conversation:
        results.append(evaluate(predicate, facts))
    for index, predicates in expect.turns.items():
        scope = next((f for f in facts if f.turn_index == index), None)
        for predicate in predicates:
            if scope is None:
                results.append(
                    PredicateResult(
                        str(predicate), Outcome.NOT_RUN, False, False, f"turn {index} never ran"
                    )
                )
            else:
                results.append(evaluate(predicate, facts, scope))

    if expect.final_stage is not None and facts:
        actual = facts[-1].stage
        results.append(
            PredicateResult(
                "final_stage",
                Outcome.PASS if actual == expect.final_stage else Outcome.FAIL,
                False,
                False,
                f"expected {expect.final_stage}, got {actual}",
            )
        )
    for forbidden in expect.never_stage:
        entered = [f.turn_index for f in facts if f.stage == forbidden]
        results.append(
            PredicateResult(
                f"never_stage:{forbidden}",
                Outcome.FAIL if entered else Outcome.PASS,
                False,
                True,
                f"entered at turns {entered}" if entered else "",
            )
        )
    return results


def _every_predicate(expect) -> list[str]:
    """Every predicate the expect block names, so an errored conversation still
    reports its full coverage rather than shrinking to nothing."""
    names = [str(p) for p in expect.conversation]
    names += [str(p) for ps in expect.turns.values() for p in ps]
    if expect.final_stage is not None:
        names.append("final_stage")
    names += [f"never_stage:{f}" for f in expect.never_stage]
    return names


def aggregate(results: list[ConversationResult]) -> dict[str, Any]:
    valid = [v for r in results for v in r.first_attempt_valid]
    proposals = [p for r in results for p in r.proposals]
    latencies = [r.latency_ms for r in results if not r.error]
    predicates = [p for r in results for p in r.predicates]

    def rate(values: list[bool]) -> float | None:
        return round(sum(values) / len(values), 3) if values else None

    return {
        "conversations": len(results),
        "errored": sum(1 for r in results if r.error),
        "structured_output_validity_rate": rate(valid),
        "structured_output_calls": len(valid),
        "repair_attempts_total": sum(r.repair_attempts for r in results),
        "proposal_override_rate": (
            round(1 - sum(proposals) / len(proposals), 3) if proposals else None
        ),
        "funnel_decisions": len(proposals),
        "interrupt_rate": round(
            sum(r.interrupts for r in results) / max(1, sum(r.agent_turns for r in results)), 3
        ),
        "latency_p50_ms": int(statistics.median(latencies)) if latencies else None,
        "latency_p95_ms": (
            int(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]) if latencies else None
        ),
        "tokens_in": sum(r.tokens_in for r in results),
        "tokens_out": sum(r.tokens_out for r in results),
        "predicates_total": len(predicates),
        "predicates_passed": sum(1 for p in predicates if p["outcome"] == Outcome.PASS),
        "predicates_failed": sum(1 for p in predicates if p["outcome"] == Outcome.FAIL),
        "predicates_unvalidated": sum(
            1 for p in predicates if p["outcome"] == Outcome.UNVALIDATED
        ),
        "predicates_not_run": sum(1 for p in predicates if p["outcome"] == Outcome.NOT_RUN),
        "safety_failures": sum(
            1 for p in predicates if p["outcome"] == Outcome.FAIL and p["safety"]
        ),
    }


def write_report(run_dir: Path, results: list[ConversationResult], meta: dict[str, Any]) -> Path:
    totals = aggregate(results)
    lines: list[str] = [
        "# Evaluation report",
        "",
        f"- run: `{run_dir.name}`",
        f"- provider: `{meta['provider']}` / `{meta['model']}`",
        f"- schema mode: `{meta['schema_mode']}`",
        f"- started: {meta['started']}",
        "",
    ]

    # Before anything else. "Safety failures: None" at the top of a run where
    # every conversation crashed is a true sentence that reads as reassurance.
    errored = [r for r in results if r.error]
    if errored:
        lines += [
            "## This run is not a measurement",
            "",
            f"{len(errored)} of {len(results)} conversations errored. Their predicates "
            "are reported `not_run`; none of them passed, and no metric below "
            "describes the system's behaviour.",
            "",
        ]
        lines += [f"- `{r.id}` — {r.error}" for r in errored]
        lines.append("")

    safety = [
        (r.id, p) for r in results for p in r.safety_failures
    ]
    if safety:
        lines += [
            "## Safety failures",
            "",
            "These surface first regardless of tier, and they fail the run.",
            "",
            "| conversation | predicate | detail |",
            "|---|---|---|",
        ]
        lines += [f"| `{cid}` | `{p['name']}` | {p['detail']} |" for cid, p in safety]
        lines.append("")
    else:
        lines += ["## Safety failures", "", "None.", ""]

    lines += [
        "## Metrics",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for key, value in totals.items():
        lines.append(f"| {key} | {value if value is not None else 'n/a'} |")
    lines += [
        "",
        "## Per conversation",
        "",
        "| id | mode | provenance | final stage | passed | failed | unvalidated | error |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        passed = sum(1 for p in r.predicates if p["outcome"] == Outcome.PASS)
        failed = len(r.failures)
        unval = sum(1 for p in r.predicates if p["outcome"] == Outcome.UNVALIDATED)
        lines.append(
            f"| `{r.id}` | {r.mode} | {r.provenance} | {r.final_stage or '-'} | "
            f"{passed} | {failed} | {unval} | {r.error or '-'} |"
        )

    lines += [
        "",
        "## What UNVALIDATED means",
        "",
        "Judged predicates are evaluated by an LLM. The judge sets are drafted but",
        "not human-reviewed, so all judged predicates report `UNVALIDATED`. An",
        "unvalidated judge is a statement about the evidence, not about the system:",
        "this repository does not claim the agent refuses medical questions, only",
        "that the harness to measure it exists and has not yet been run with",
        "validated ground truth.",
        "",
    ]

    path = run_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate wall-clock, make no calls."),
    resume: str | None = typer.Option(None, "--resume", help="Run id to continue."),
    baseline: str | None = typer.Option(None, "--baseline", help="Run id to diff against."),
    tolerance: float = typer.Option(
        0.05, "--tolerance", help="Allowed drop in pass rate before a regression is reported."
    ),
    only: str | None = typer.Option(None, "--only", help="Run one conversation by id."),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    conversations = [c for _p, c in load_all() if not isinstance(c, str)]
    if only:
        conversations = [c for c in conversations if c.id == only]
    if not conversations:
        console.print("[red]no golden conversations found[/red]")
        raise typer.Exit(1)

    if dry_run:
        _dry_run(conversations)
        raise typer.Exit(0)

    if baseline and not resume:
        _compare(baseline, conversations, tolerance)
        raise typer.Exit(0)

    # Before asyncio.run, not inside it. `_execute` called this too, which
    # looked right and did nothing: by then asyncio.run has already built the
    # ProactorEventLoop, and setting the policy afterwards changes nothing.
    # Every conversation in the first live run died on psycopg refusing it.
    from app.db.base import configure_event_loop

    configure_event_loop()
    asyncio.run(_execute(conversations, resume=resume))


def _dry_run(conversations: list[GoldenConversation]) -> None:
    """Print the cost before spending it.

    Cheap, and it stops the "kick it off and see" failure that turns an
    afternoon into a wasted evening.
    """
    turns = sum(c.agent_turn_count for c in conversations)
    table = Table(show_header=True, header_style="bold", title="dry run")
    for column in ("conversation", "bursts", "predicates", "est. seconds"):
        table.add_column(column)
    for c in conversations:
        preds = c.expect.coverage()["predicates"] if c.expect else 0
        table.add_row(
            c.id,
            str(c.agent_turn_count),
            str(preds),
            f"{c.agent_turn_count * SECONDS_PER_TURN:.0f}",
        )
    console.print(table)
    console.print(
        f"{len(conversations)} conversations, {turns} agent turns, "
        f"~{turns * 4} model calls, estimated ~{turns * SECONDS_PER_TURN / 60:.1f} minutes"
    )


async def _execute(conversations: list[GoldenConversation], *, resume: str | None) -> None:
    from app.config import LLMProviderName, get_settings
    from app.db.base import build_sessionmaker, configure_event_loop
    from app.llm.factory import build_provider
    from app.logging import configure_console

    configure_console()
    configure_event_loop()
    settings = get_settings()

    if settings.llm_provider is LLMProviderName.SCRIPTED:
        # The refusal that keeps the README honest. A scripted provider
        # pattern-matches text; a number from it would describe the harness
        # rather than the system, and would look identical in the README.
        console.print(
            "[red]refusing to run against LLM_PROVIDER=scripted.[/red]\n"
            "It answers by pattern-matching, so any number it produces measures "
            "the harness rather than the system. Set a real provider."
        )
        raise typer.Exit(2)

    provider = build_provider(settings)
    sessionmaker = build_sessionmaker()

    run_id = resume or _now_id()
    run_dir = RESULTS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "provider": provider.name,
        "model": provider.model,
        "schema_mode": str(settings.schema_mode),
        "started": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    results: list[ConversationResult] = []
    for conversation in conversations:
        target = run_dir / f"{conversation.id}.json"
        if resume and target.exists():
            console.print(f"[dim]skip {conversation.id} (already complete)[/dim]")
            results.append(ConversationResult(**json.loads(target.read_text(encoding="utf-8"))))
            continue

        console.print(f"running [bold]{conversation.id}[/bold] ...")
        result = await replay(
            conversation, provider=provider, settings=settings, sessionmaker=sessionmaker
        )
        # Written immediately: an eight-hour run that dies at hour seven with
        # nothing on disk is unacceptable.
        target.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8", newline="\n")
        results.append(result)

    report = write_report(run_dir, results, meta)
    totals = aggregate(results)
    console.print(f"\nreport: [bold]{report}[/bold]")
    console.print(
        f"{totals['predicates_passed']} passed, {totals['predicates_failed']} failed, "
        f"{totals['predicates_unvalidated']} unvalidated, "
        f"{totals['predicates_not_run']} not run, "
        f"{totals['safety_failures']} safety failures"
    )
    if totals["errored"]:
        console.print(
            f"[red]{totals['errored']} of {totals['conversations']} conversations "
            f"errored.[/red] Their predicates are reported NOT_RUN, not passed, "
            "and this run is not a measurement."
        )
        raise typer.Exit(1)
    if totals["safety_failures"]:
        raise typer.Exit(1)


def _compare(baseline_id: str, conversations, tolerance: float) -> None:
    """Diff a completed run against a baseline and fail on regression."""
    base_dir = RESULTS / baseline_id
    if not base_dir.exists():
        console.print(f"[red]no such run: {baseline_id}[/red]")
        raise typer.Exit(1)

    runs = sorted(p for p in RESULTS.iterdir() if p.is_dir() and p.name != baseline_id)
    if not runs:
        console.print("[red]no later run to compare against[/red]")
        raise typer.Exit(1)
    latest = runs[-1]

    def load(directory: Path) -> dict[str, ConversationResult]:
        out = {}
        for path in directory.glob("*.json"):
            if path.name == "meta.json":
                continue
            out[path.stem] = ConversationResult(**json.loads(path.read_text(encoding="utf-8")))
        return out

    before, after = load(base_dir), load(latest)
    table = Table(show_header=True, header_style="bold", title=f"{baseline_id} -> {latest.name}")
    for column in ("conversation", "before", "after", "delta"):
        table.add_column(column)

    regressed: list[str] = []
    for cid in sorted(set(before) | set(after)):
        b, a = before.get(cid), after.get(cid)
        b_rate = _pass_rate(b)
        a_rate = _pass_rate(a)
        delta = None if b_rate is None or a_rate is None else round(a_rate - b_rate, 3)
        if delta is not None and delta < -tolerance:
            regressed.append(cid)
        table.add_row(
            cid,
            "-" if b_rate is None else f"{b_rate:.2f}",
            "-" if a_rate is None else f"{a_rate:.2f}",
            "-" if delta is None else f"{delta:+.2f}",
        )
    console.print(table)

    if regressed:
        console.print(f"[red]regression in {len(regressed)}: {', '.join(regressed)}[/red]")
        raise typer.Exit(1)
    console.print("[green]no regression beyond tolerance[/green]")


def _pass_rate(result: ConversationResult | None) -> float | None:
    if result is None or not result.predicates:
        return None
    scored = [p for p in result.predicates if p["outcome"] in (Outcome.PASS, Outcome.FAIL)]
    if not scored:
        return None
    return sum(1 for p in scored if p["outcome"] == Outcome.PASS) / len(scored)


if __name__ == "__main__":
    app()
