# Graph Design — Sign-off

Design approved. It is better than the brief I gave you: the plan-then-execute call, the
system-only transitions, and the checkpointer/table split are all improvements on what I
specified. Two of the eight decisions change, and there is one problem nobody has costed
that outranks all eight.

---

## 0. The problem: the eval harness does not fit on CPU

You promise these ablations: memory policy ×3, parallel sub-agents ×2, RAG vs full-context
×2, locale ×2. Using **your own** throughput figure (8–20 tok/s, mid 12):

| | |
|---|---|
| Per structured LLM call (~350 output tokens) | ~29 s |
| Per turn (supervisor + knowledge + strategy + compose) | ~117 s |
| **One eval run** (40 conversations × 6 turns) | **~7.8 h** |
| Full 24-config matrix | ~187 h (**8 days continuous**) |
| One-factor-at-a-time (7 runs) | ~54 h |

A single run does not fit in a working day. The matrix does not fit in the project.

This is the most likely way this project fails: everything gets built, the eval harness is
the entire point, and it can never actually be run — so the README ships with no numbers
and the differentiator evaporates.

**Required before Phase 1, as part of the design:**

1. **Two golden sets, not one.** `smoke` (8 conversations, ~1.5 h, every ablation runs
   against this) and `full` (40 conversations, one run per major milestone only).
   Ablation conclusions come from `smoke`; headline numbers come from `full`.
2. **One-factor-at-a-time, never the matrix.** Pick a baseline config, vary one axis at a
   time. 7 runs on `smoke` is ~11 h, which is an overnight job. State in the report that
   interactions between factors were not measured — that is an honest limitation, and
   claiming a full factorial you did not run would be worse.
3. **Make the runner resumable and parallel-safe.** Per-conversation results written
   incrementally to `evals/results/<run>/`, `--resume` skips completed conversations. An
   8-hour run that dies at hour 7 with nothing on disk is unacceptable.
4. **Budget the eval in tokens and seconds up front.** `python -m evals.run --dry-run`
   prints estimated wall-clock before starting. Cheap, and it stops the "I'll just kick it
   off and see" failure.
5. **GPU escape hatch.** I have Kaggle (~30 GPU-h/week, P100 or 2×T4). The runner should
   take an `OLLAMA_HOST` so a run can point at a GPU instance. Do not build a
   Kaggle-specific path; just do not hardcode localhost.

Add the measured cost of an eval run to the decision log. "We measured our own harness before
committing to it" is itself a point in the project's favour.

---

## 1–8. Sign-offs

| # | Decision | Verdict |
|---|---|---|
| 1 | D1 plan-then-execute over ReAct | **Approved.** The reasoning holds and the `TurnPlan`-as-assertable-artifact argument is the strongest part. Keep max-1-replan — it covers the case where a tool result invalidates the plan, which is the one real weakness of plan-then-execute. |
| 2 | System-only `→ AWAITING_CONFIRMATION` and `→ BOOKED` | **Approved.** This is the best idea in the document. Turning "never hallucinate a confirmation" from a prompt instruction into a structural impossibility is the single most defensible thing the project will contain. Lead with it in the README. |
| 3 | Forced ask at the hard cap | **Approved.** |
| 4 | Risk a missed message over a duplicate | **Approved**, with a correction — see §9 below. |
| 5 | Postgres queue over APScheduler | **Approved.** `SKIP LOCKED` plus "the schedule is business state we need to query" is the right justification. |
| 6 | `Jan Mustermann` / `Musterform Personal Coaching` | **Approved.** The placeholder look is a feature. Put the one-line README note explaining it. |
| 7 | 16 tables in Phase 1 | **Changed — see below.** |
| 8 | Metrics to Postgres, Langfuse additive | **Approved.** Writing twice is cheap; the eval harness not depending on a tracing backend is correct. |

---

## 7. Changed: stage the migrations by phase

I disagree with shipping all 16 tables in Phase 1. Not because the schema is wrong — each
table does earn its place — but because it front-loads work for features that do not exist
until Phase 5 or 6, and it delays the Phase 2 checkpoint where I first get a working
conversation.

**Phase 1 (core 8):** `leads`, `conversations`, `messages`, `turns`, `funnel_transitions`,
`conversation_summaries`, `slots_collected`, `availability`

**Phase 4:** `bookings`, `tool_calls`, `delivery_intents`

**Phase 5:** `scheduled_followups`, `interrupts`, `gate_drops`, `conversation_outcomes`

**Phase 6:** `turn_metrics`

Same 16 tables, same final schema. Design the whole thing now — put the full ERD in
`ARCHITECTURE.md` at Phase 1 so the target is visible — but migrate incrementally.

Two reasons this is better and not just faster: incremental Alembic migrations against a
database with live data is a real engineering practice and worth demonstrating, whereas one
big-bang migration is not. And a schema written before the code that uses it is a schema
written on assumptions — staging lets each table be shaped by the node that actually
queries it.

---

## 9. Two factual corrections

**Instagram does have read-back.** §10 justifies the delivery trade with "a channel with no
read-back". That is not accurate for Instagram — the Graph API exposes conversations and
messages, so a reconciliation path could confirm whether a message landed. Since
`InstagramChannel` is a stub this changes nothing operationally, but the claim would be
wrong in the decision log and an interviewer with Meta API experience would challenge it.

Restate as: *"for channels where we have not implemented read-back — which is all of them in
this port — we prefer a missed message to a duplicate. Where read-back exists (Instagram
does expose it), reconciliation is the better answer and is noted as future work."* That is
both accurate and a stronger answer, because it shows you know the option exists and chose
against implementing it.

**The 46% figure — accepted, with the itemised form only.** Your census reconciles with
mine and the arithmetic is right. Use the itemised sentence you proposed and never the bare
percentage. The moment "46%" appears without "34 + 34 of 149" beside it, it becomes a number
someone has to trust.

---

## 10. On scope

The design is excellent and it is also considerably larger than it needs to be. I have a
live job application, a separate OCR benchmark, a thesis, and a paper in revision.

So: **the deliverable that matters is a working conversation with numbers behind it.**
If something has to be cut, cut in this order — the RAG/full-context comparison (it is a
negative result, one paragraph will do), the parallel sub-agent comparison (predictable),
the German locale (run English, note the limitation). Do not cut the funnel state machine,
the system-only transitions, the repair loop, or the eval harness itself. Those four are
the project.

Tell me at each checkpoint if you think we are behind, rather than compressing quality to
stay on plan.

---

## Proceed to Phase 1

Deliver: repo skeleton, `docker-compose.yml`, config, the **core-8** migrations, full ERD in
`ARCHITECTURE.md`, the decision log seeded with D1–D4 plus §10 and §11.2 plus the eval budget,
`ruff` + `pytest` wired.

Checkpoint: `docker compose up` and `alembic upgrade head` clean on my machine.

Tell me if the `ollama pull` fails rather than working around it.
