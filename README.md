# Musterform DM Agent

A code-native **LangGraph** port of a production **n8n** multi-agent Instagram DM
agent, with the source's defects fixed rather than reproduced.

The original is 149 nodes of visual workflow, running live for a real German
personal-training client: it answers inbound DMs, qualifies the lead through an
AIDA funnel, and books a free consultation call. This repository rebuilds it as
a typed, tested Python application — and the interesting part is not the
rewrite, it is the list of things the original was quietly getting wrong.

> **Status.** The graph runs, the funnel advances, conversations survive a
> restart, and 456 tests pass. The evaluation harness is complete but **has not
> been run against a live model**, so this README contains no measured numbers.
> See [Evaluation](#evaluation) and `docs/KNOWN_GAPS.md`.

---

## Quickstart

```bash
cp .env.example .env
```

```bash
docker compose up -d && docker compose exec app alembic upgrade head
```

```bash
docker compose exec app python -m app.cli doctor --no-llm
```

**No API key needed to try it.** `LLM_PROVIDER=scripted` is a deterministic
rule-based provider that ships with the repo, so the graph, the funnel and the
guardrails all run from a clean clone:

```bash
LLM_PROVIDER=scripted python -m app.cli chat
```

Set `GEMINI_API_KEY` in `.env` and drop `LLM_PROVIDER` to use the real model.

Postgres publishes on host port **5432** by default. If that is taken — a native
Postgres install is the usual cause, and it silently wins the bind — set
`POSTGRES_HOST_PORT` in `.env` and match it in `DATABASE_URL`.

---

## Architecture

```
inbound burst
   → normalize      channel-agnostic envelope; N messages, one turn
   → load_session   funnel state, slots, summary — from the tables
   → gate           dedup · eligibility · terminal check    ─┐ drop
   → supervisor     one call → a validated TurnPlan          │
        ├─ knowledge_agent   (only if the plan asked)        │
        └─ strategy_agent    (proposes a stage)              │
   → funnel_transition       deterministic; the machine decides
   → compose ⇄ guardrail     style violations re-compose      │
        └─ [[interrupt]]     safety violations stop           │
   → deliver                                                  │
   → post_turn      one write; everything the next turn reads ─┘
```

`thread_id` is the conversation id, never the channel sender id. The full
transition table and the data model are in `ARCHITECTURE.md`.

Markers like *decision D22* throughout this repository refer to a decision log
kept outside it — 47 entries, one per choice, each with the option rejected and
what would change the answer. It is not published. Where a decision carries
weight for reading the code, the reasoning is restated at the point it
matters rather than left as a pointer.

### The delegation delta

The source is described as an orchestrator delegating to two sub-agents. It does
not delegate. Both sub-agent inputs are hardcoded to the raw webhook text:

```
$('Insta Trigger').item.json.body.entry[0].messaging[0].message.text
```

Both carry *identical* boilerplate tool descriptions, and the orchestrator's
prompt says to call both every time. The model never chooses what to ask, or
whether to ask.

Here the supervisor emits a `TurnPlan` with a model-populated `knowledge_query`,
and the graph **skips the knowledge node entirely** when the plan asked nothing.
That skip is asserted in `tests/test_graph.py`.

---

## The two-lock booking guarantee

Two stages — `AWAITING_CONFIRMATION` and `BOOKED` — are reachable by system
transition only, enforced twice:

1. `decide()` rejects any proposal naming them, before any other rule runs, and
   records the rejection.
2. **`ProposableStage`, the enum the strategy agent's schema is built from, has
   no value for either.** A proposal naming one fails schema validation before
   the state machine is ever reached.

The second lock is the one worth the space. "Never confirm a booking you have
not made" stops being an instruction the model is asked to follow, or even a
guard the machine enforces, and becomes **a sentence the model cannot form**.
The confirmation text is emitted by the `BOOKED` transition, which fires only on
a committed database row.

The source's equivalent:

```
your only output "MUST be MUST be MUST be" a raw JSON object
```

---

## Five deviations from the source

Each is a defect found by reading the source against a concrete conversation.

| # | The source | The defect | The fix |
|---|---|---|---|
| 1 | Both sub-agent inputs hardcoded; both called every turn | not delegation, a fan-out with an LLM in front | supervisor emits a `TurnPlan` and populates each input, or skips it — D1 |
| 2 | Message 1 is always a greeting + goal question | answers *"how much does it cost?"* with *"what's your main fitness goal?"*, and re-funnels a lead who already asked to book | opener is conditional on the first burst carrying no actionable intent — D18 |
| 3 | Each webhook fires the workflow independently | a three-message burst gets three replies — the most bot-like thing a DM agent can do | consecutive inbound messages are one turn; the funnel clock counts replies — D24 |
| 4 | Transition proceeds unless the lead declines | a reply that ignores the offer entirely reads as consent to collect a phone number | `SLOT_FILLING` requires positive engagement, and re-asking is bounded — D28 |
| 5 | Knowledge base *demonstrates* inferring price, capacity and programme length — *"not directly mentioned, but inferred from the context: … likely with packages"* | fabricating a price is the failure that would cost the client money | every such answer replaced with an explicit refusal marker — D16 |

Two more worth naming: the source has **no inbound deduplication** despite
at-least-once webhook delivery, and its funnel stage is inferred by a model from
a 4-message memory window under a funnel specified to take six.

---

## Restart

The acceptance criterion is that killing the process mid-conversation and
restarting resumes correctly — and specifically that the **funnel state** comes
back, not just a coherent-sounding transcript. LangGraph's checkpointer would
restore its own message list either way; the claim being tested is D4, that the
application tables are the source of truth.

Each command below is a separate OS process:

```
$ python -m app.cli turn --user lead-1788020583 "hey saw ur reel"
coach  Hey! If you had to pick one fitness goal right now, what would it be? 🙂
stage=OPENER turn=1

$ python -m app.cli turn --user lead-1788020583 "can i book that free call"
coach  Happy to get you booked in for the free call 📅 what's your name?
stage=TRANSITION turn=3

$ python -m app.cli turn --user lead-1788020583 "0341 9876543"
coach  Great 👍 could you send your phone as well?
stage=SLOT_FILLING turn=5

# --- process killed. nothing in memory survives. ---

$ python -m app.cli snapshot --user lead-1788020583
  stage              SLOT_FILLING
  agent turns sent   5
  next turn index    6
┌───────┬──────────────┬───────┬─────────┬──────────┬──────────┐
│ slot  │ raw          │ value │ status  │ error    │ attempts │
├───────┼──────────────┼───────┼─────────┼──────────┼──────────┤
│ name  │ Dan          │ Dan   │ valid   │ -        │ 1        │
│ phone │ 0341 9876543 │ -     │ invalid │ landline │ 1        │
└───────┴──────────────┴───────┴─────────┴──────────┴──────────┘

$ python -m app.cli turn --user lead-1788020583 "sorry its 0151 23456789"
stage=SLOT_FILLING turn=6
# phone: +4915123456789, valid, attempts 2
```

The full transcript is in `docs/restart_transcript.txt`. `snapshot` reads the
tables, never the checkpointer — that is the whole point of the command.

`tests/test_restart.py` asserts the same four properties from a **new engine and
new connection pool**, so a pass cannot be SQLAlchemy's identity map serving
stale objects.

---

## Eight assertions and tests that could not fail

An evaluation harness is only worth its numbers if its assertions can fail.
Eight times during this build, something read as verification and could not.
Each was found by building the thing rather than reviewing it.

| What looked like coverage | What it actually was | Closed in |
|---|---|---|
| `never_stage_after` naming a pivot the conversation never reaches | vacuously true | D25 |
| `forbidden: [commit_bookng]` | a misspelled tool is never called, so it always passed | D25 |
| An empty `conversation: []` block | counted as a block, asserted nothing | D25 |
| A per-turn assertion pointing past the end of a conversation | vacuous, and it inflated the pass rate | D25 |
| `structured_output_validity_rate` under native `responseSchema` | ~100% by construction; the repair ladder never ran | D5 |
| 7 database tests skipping silently in the container | 7 green-looking tests that had never run — the acceptance criterion for D4 | D31 |
| Judge accuracy gated on a point estimate | `0.83 > 0.80` passes a judge that got lucky at every usable n | D35 |
| A judge set labelled by the model being judged | circular; the number measures nothing | D33 |

Two general rules came out of it, both enforced rather than intended:

**Every field accepting an identifier must reject unknown identifiers *and*
accept a known one.** A rejection-only test passes on a field that rejects
everything. `tests/test_schema_falsifiability.py` enumerates the schema rather
than listing cases, so a *new* field cannot reopen the hole — all four schema
instances arrived by being added somewhere new.

**A skip is not a pass.** `456 passed` and `456 passed, 7 skipped` read
identically to anyone scanning. The acceptance run uses `--no-skips`.

```bash
POSTGRES_HOST_PORT=5433 scripts/acceptance.sh
```

That script runs ruff, the suite **twice** (tests that pass once and fail on
rerun are their own class — the restart tests did exactly that), and the whole
thing again inside the container, because a platform-specific fix was once
itself platform-asymmetric: it passed on Windows and broke collection on Linux.

**This list grew after it was written, and it is not closed.** It is the honest
basis for trusting anything else here.

---

## Evaluation

One run, committed in full at
[`evals/results/20260829T174308Z/`](evals/results/20260829T174308Z/report.md).
`gemini-3.5-flash-lite`, native structured output, 8 conversations, 38 agent
turns, 129 model calls.

| metric | value |
|---|---|
| structured-output validity (first attempt) | **1.000** — 129/129 |
| repair attempts | **0** |
| funnel proposal override rate | **0.263** — 10 of 38 turns |
| interrupt rate | **0.026** — 1 of 38 turns |
| predicates passed / failed / unvalidated | **54 / 8 / 18** |
| safety failures | **0** |
| latency per conversation, p50 / p95 | 99.3 s / 116.0 s |
| tokens in / out | 111,624 / 12,174 |

**The number worth arguing about is 0.263.** The deterministic state machine
overrode the model's stage proposal in one turn in four. That is the design
working — the model is a proposer, not a decider — but it is also a quarter of
turns where the model wanted to move the lead somewhere the rules would not
allow. On a booking funnel, three of those overrides are the difference between
a confirmed appointment and a fabricated one.

**Validity 1.000 with zero repairs is a weaker result than it looks.** It says
the three-rung repair ladder never ran, so this figure is evidence about
Gemini's native structured output, not about the repair path. The ladder is
covered by unit tests instead; see the caveat in *Eight assertions that could
not fail*.

### What the eval found

Eight predicate failures, none of them noise:

| failure | what it means |
|---|---|
| `asks_clarifying_question` ×2 (smoke-01) | `resolve_datetime` is written and tested but not dispatched from a node, so "next tuesday" and "can we do 9" pass through unresolved. The gap has a measured cost. |
| `offers_alternative_slot` (smoke-02) | same root cause — `list_free_slots` is not wired in. |
| `phone_error_names` ×2 (smoke-03) | the golden set asserts errors for two numbers that are *valid* under `DEFAULT_PHONE_REGION=DE`. The assertion is wrong, not the validator (D38). |
| `final_stage` HANDED_OFF (smoke-03) | downstream of the above: the number validated, so slot recovery never exhausted. |
| `final_stage` VALUE→TRANSITION (smoke-04) | `transition_turn_reached` fires at turn 5 unconditionally, on a lead who spent four turns asking questions the corpus cannot answer. The forced transition needs a `MIN_VALUE_EXCHANGES` guard. |
| `final_stage` OPTED_OUT (smoke-06) | **the finding.** See below. |

### The finding: opt-out is decided by the model alone

`opt_out_detected` is populated from `StageProposal.opt_out` — an LLM boolean.
There is no deterministic check on the inbound text, despite the transition
being labelled `Trigger.DETECTOR`.

This conversation was run twice against the same model and the same prompt. The
first run reached `OPTED_OUT` (`guard=opt_out_detected, proposed=OPTED_OUT`);
the committed run stayed in `VALUE` and kept selling. The lead said the same
words both times.

Opt-out is the one behaviour here with a compliance flavour, and it is the one
behaviour whose deterministic guard has an LLM as its only input. Everything
else in the funnel has a rule the machine can check for itself. The fix is a
phrase-list detector OR-ed with the model's proposal so the model can only ever
*add* an opt-out, never withhold one; it is written up in
`docs/KNOWN_GAPS.md` §1 and is not implemented, because finding it was this
milestone's job and fixing it is the next one's.

An eval that finds nothing on its first real run is usually an eval that cannot
find anything.

### Three defects in the harness itself

The first live run reported `42 passed, 1 failed, 0 safety failures` and exited
**0** while all eight conversations had crashed on the event loop. The second
reported two `no_booking_confirmed` safety failures that were the guardrail
correctly withholding those exact replies. Both are written up in
decision D46, with the tests that now fail without the fixes.

The general shape is worth stating: **every test in this repository runs
against a scripted provider, and all three defects needed a real model, a real
network and a real quota to appear.** The suite is not a substitute for one
live run, and one live run is not a substitute for the suite.

```bash
python -m evals.run --dry-run        # cost before spending it
python -m evals.run                  # writes evals/results/<ts>/
python -m evals.run --resume <ts>    # continue after a quota wall
python -m evals.run --baseline <ts>  # diff, non-zero on regression
```

**What it measures.** Structured-output validity, repair attempts, funnel
proposal override rate, tool-call correctness, interrupt rate, latency
p50/p95, tokens per turn, plus per-conversation predicate results split into
deterministic and judged.

**The golden set** is 8 synthetic conversations in `evals/golden/`, written from
a documented failure taxonomy: ambiguous date and bare-hour time · unavailable
slot · invalid phone · questions with no grounded answer · bot-detection
probing · informal opt-out · derailment with degenerate input · medical scope
creep. Two of the eight consist mostly of questions the corpus cannot answer,
because that is the only test that catches fabrication. Four have assertions
marked `provenance: drafted_by_assistant` — machine-proposed, visibly not the
owner's.

**The runner refuses `LLM_PROVIDER=scripted`.** That provider pattern-matches
text; a number from it would describe the harness rather than the system, and
would look identical in this README to a real one. It also exits non-zero when
any conversation errors, and reports that conversation's predicates as
`not_run` rather than as passes.

> Judged predicates are evaluated by an LLM. The judge sets are drafted but not
> human-reviewed, so all judged predicates report `UNVALIDATED`. An unvalidated
> judge is a statement about the evidence, not about the system: this repository
> does not claim the agent refuses medical questions, only that the harness to
> measure it exists and has not yet been run with validated ground truth.

18 of the 80 predicate results in the committed run are `UNVALIDATED` for that
reason.

The judge is graded against hand-labelled sets in `evals/judge/`, gated on the
**Wilson 95% lower bound** rather than a point estimate — at n=27 that needs
26/27 correct. Five sets are committed as `status: draft`; a draft counts as
zero, so `python -m evals.validate judge` reports `0/10`.

---

## Anonymisation

The source is real client work. The business, the coach, the persona and all
biographical detail are replaced throughout with **Coach Jan Mustermann** of
**Musterform Personal Coaching** — the German equivalent of *John Doe*, chosen
to be unmistakably a placeholder rather than a plausible invented name that
might collide with a real coach.

Removed: a live WhatsApp contact link, and a biographical passage naming three
specific sports and their competition levels. Individually each detail is
unremarkable; a coach who trains that particular combination is a small enough
set to identify even under a changed name. The details are deliberately not
reproduced here — this file is committed, and a description of a fingerprint
that reprints the fingerprint has not removed it.

`tests/test_anonymisation.py` runs in the acceptance path — 64 checks. A hand
pass missed a lowercase spelling of the client's name in two files and an
entire second biographical passage; the scan found both, which is why the scan
is what ships. Both n8n exports are gitignored.

The scan started as a prompt-file check and had to be widened, because the
prompts were clean and three other things were not:

- `.gitignore` excluded the n8n export **by its literal filename**, and the
  export is named after the client. The one file whose job is to keep the name
  out of git history was the file committing it. It now matches by glob.
- This README and the decision log both justified the anonymisation by reprinting
  the biographical fingerprint they had removed — the leak, in the two files a
  reviewer reads first, produced by documenting the fix too well.
- The working notes taken while reading the source (the port brief, the
  Phase 0 analysis) quote the client's name and an Instagram access token
  throughout. They are gitignored rather than cleaned; a working note is not a
  deliverable.

`test_no_committed_file_carries_a_client_identifier` now walks every file
`.gitignore` would not exclude. It is verified to fail: adding the client's
name to `ARCHITECTURE.md` produces `ARCHITECTURE.md:494: '<name>'` — and this
sentence originally quoted that output literally, so the guard's first catch
after being written was this README. It runs
before `git init`, deliberately — `git check-ignore` would have reported
success on a directory that was not yet a repository.

---

## Known gaps

The full list is `docs/KNOWN_GAPS.md`. The headlines:

- **Opt-out is decided by the model alone** — no deterministic text check
  behind it, and two runs of the same conversation disagreed. The most
  important thing the eval found.
- **Judged predicates report `UNVALIDATED`** — judge sets drafted, not reviewed.
- **Three of four tools are implemented and tested but not yet dispatched from
  the graph.** Phone validation is wired; datetime, availability and booking are
  not. Three of the run's eight predicate failures are this.
- **`smoke-03` asserts a behaviour that would be wrong to implement** — it
  expects `missing_country_code` for a number that is valid under a default
  region. Left failing rather than fixed by weakening the validator; three more
  failures are this one wrong assertion and its consequences.
- **`TypedSummary` is persisted but never populated** by a summariser call.
- Descoped by instruction: German locale, ablations, Instagram, Google Calendar,
  follow-up scheduler, `conversation_analyst`.

---

## Layout

```
app/
  agents/      supervisor (TurnPlan) · knowledge · strategy · compose
  channels/    protocol · cli · http (FastAPI) · instagram (stub) · envelope
  db/          models (8 tables) · repository
  funnel/      stages · transitions (the state machine)
  graph/       state · nodes · build
  guardrails/  checks · interrupt
  llm/         provider protocol · gemini · ollama · scripted · repair ladder
  memory/      typed summary · slots
  tools/       phone · datetime · booking
evals/
  golden/      8 conversations
  judge/       5 draft judge sets
  schema.py    closed predicate vocabulary · run.py · predicates.py
docs/          KNOWN_GAPS.md · OPEN_QUESTIONS.md · restart_transcript.txt
```

`ARCHITECTURE.md` is written to be read before the code: the graph, the funnel
table, the guardrail split, the tools, and the four database constraints that
carry weight.
