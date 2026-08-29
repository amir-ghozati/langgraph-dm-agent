# Known gaps

Everything here is a deliberate omission or an unresolved defect, recorded
rather than hidden. A gap documented is worth more than a fix that delays the
finish.

## 1. Opt-out is decided by the model, with no deterministic backstop

**Status: found by the first live eval run, understood, not fixed.**

`app/graph/nodes.py` populates the transition context with
`opt_out_detected=proposal.opt_out` — a boolean the LLM returns in its
`StageProposal`. `app/funnel/transitions.py` then treats it as a detector
result and labels the transition `Trigger.DETECTOR`. There is no check on the
inbound text at all.

`smoke-06-optout-midfunnel` was run twice against the same model with the same
prompt and the same inbound words. The first run reached `OPTED_OUT`
(`guard=opt_out_detected, proposed=OPTED_OUT`); the committed run stayed in
`VALUE` and kept selling.

This matters more than the other failures. Every other funnel rule has
something the machine can verify for itself — a turn index, a slot status, a
stage guard. Opt-out has an LLM boolean, and it is the one behaviour in the
funnel with a compliance flavour.

**The fix**, roughly half a day:

```python
opt_out_detected=proposal.opt_out or OPT_OUT_PHRASES.search(inbound_text) is not None
```

OR-ed, deliberately: the model may only ever *add* an opt-out it noticed, never
withhold one the phrase list caught. Then a golden assertion that the phrase
list alone is sufficient, with the model stubbed to `opt_out=False`.

Not done because this milestone's job was to build the harness and run it once.
Finding this is the harness working.

## 2. Judged predicates are UNVALIDATED

All five judge sets are `status: draft` — candidates and labels proposed by a
model, not reviewed by a person. Every judged predicate therefore reports
`UNVALIDATED`, and the report says what that means.

Reviewing them means reading the cases, correcting labels, then setting
`status: reviewed` and `labelled_by` to a person's name. Validation rejects a
reviewed set that still names a model.

Even reviewed, at n=27 the Wilson gate needs 26/27 correct, so `INCONCLUSIVE`
is the likely honest outcome. See decision D35 for what to do if the
judge scores ~0.85 — sharpen the definition, do not add 230 cases.

Five of the ten judged safety predicates have no set at all:
`no_exercise_prescription`, `no_supplement_recommendation`,
`no_out_of_scope_claim`, `no_funnel_repitch_after_optout`,
`refuses_without_fabricating`.

## 3. `smoke-03` asserts a behaviour that would be wrong to implement

Its turn-4 assertion expects `phone_error_names: missing_country_code` for
`0151 23456789`. With `DEFAULT_PHONE_REGION=DE` that number is *valid* — a
default region is a country code, and rejecting the national format would
reject the way a German lead types their own number (D38).

It failed in the committed run exactly as predicted (`phone error was [None],
expected 'missing_country_code'`), along with a second assertion expecting a
`landline` error and the `final_stage: HANDED_OFF` that depended on recovery
being exhausted. Three of the eight failures are this one wrong assertion and
its consequences.

They are left failing rather than "fixed" by weakening the validator, because
the golden set is asserting the wrong behaviour and the honest record of that
is a failing assertion plus this note.

## 4. Tools are implemented but not yet called by the graph

`resolve_datetime`, `check_availability`, `list_free_slots` and `commit_booking`
are complete, transactional and tested (24 tests). Phone validation *is* wired
into the graph — an invalid number records as `INVALID` with its error kind.

The other three are not yet invoked from a node, so `TurnPlan` has no
`tool_calls` field. The tool contracts and their failure paths are the part
worth reviewing; the dispatch wiring is a half-day that was cut to reach a
finished repository.

The committed eval run measures what that costs: three of its eight predicate
failures are here. `smoke-01` never asks which Tuesday or whether nine means
morning or evening, because nothing resolves the ambiguity `resolve_datetime`
was written to detect; `smoke-02` offers no concrete alternative slot, because
`list_free_slots` is never called. The tools are correct and unreachable, which
the golden set states as three failing assertions rather than as a note.

## 5. Mid-turn resume is CLI-only

The HTTP transport compiles the graph with an in-memory checkpointer and has no
resume endpoint: a safety-flagged turn is reported as `interrupted` and the
draft is withheld, but a human cannot approve it over HTTP.

Conversation-level resumption is unaffected on both transports — the funnel
state lives in the tables (D4), not the checkpoint — so an HTTP conversation
survives a restart exactly as the CLI one does.

## 6. Descoped by instruction

Not attempted, and not intended to be:

- **German locale.** English only. The persona, timezone and phone region stay
  German; only the conversation language is fixed.
- **Ablations.** No memory-policy, parallel-sub-agent, or RAG-vs-full-context
  comparison. The no-RAG argument is stated (a 3.8k-token corpus needs no
  retriever) and not measured.
- **Instagram integration.** `InstagramChannel` raises `NotImplementedError`
  with a docstring covering the Graph API v23.0 contract, webhook signature
  verification, the 24-hour messaging window, at-least-once delivery and
  read-back.
- **Google Calendar.** Availability is a Postgres table. The tool contract is
  the point, not the integration.
- **Follow-up scheduler.** No table, no worker, no management command. The
  24-hour re-engagement sequence from the source is not ported.
- **`conversation_analyst`.** The LLM-as-judge conversation classifier is not
  built.

## 7. Smaller things

- **`ProposableStage` has no `AWAITING_CONFIRMATION` or `BOOKED`**, so the
  funnel cannot currently *reach* `BOOKED` at all — the transition exists and is
  tested, but nothing calls `commit_booking` to satisfy its guard (see gap 4).
  The booking path is proven by `tests/test_tools.py`, not end to end.
- **`TypedSummary` is persisted but never populated by a model.** `_fold_summary`
  merges an empty summary each turn, so `stated_goals` stays empty in a live
  conversation. The merge semantics, the supersession rule and the round-trip
  are tested; the summariser call is not built.
- **The `deliver` node is not idempotent** and the `delivery_intents` table from
  the v1 design was cut. The port prefers a missed message to a duplicate; see
  D14.
- **`BurstBuffer` is not wired to a live channel.** The CLI uses an explicit
  send affordance and HTTP takes a list, so the debounce is tested but unused.
- **Availability is not seeded automatically.** `seed_availability` exists and
  is idempotent; nothing calls it at startup.
- **The scripted provider's replies are stage-templated**, so a conversation
  driven by it looks repetitive. It exists to prove the machinery runs without
  a key, not to sound good. It also never emits the disclosure line, so *every*
  turn 1 driven by it exhausts re-composition on `missing_disclosure` and
  delivers `safe_template()`. That is the guardrail working — the observable
  proof that the style contract is enforced against the model rather than
  assumed — but it means a scripted transcript opens with a hand-off.
- **The container stage rebuilds the image every run** (`up -d --build
  --wait`), which costs ~20s on a warm cache. It is there because a stale image
  passed the host stage and failed collection in the container on a missing
  `phonenumbers` — a dependency-drift bug that reads like a platform bug.

## 8. The working notes are gitignored, not cleaned

`LANGGRAPH_PORT_BRIEF*.md`, `PHASE0_SOURCE_ANALYSIS.md`, `PHASE1_GRAPH_DESIGN.md`
and `GOLDEN_SET_FILL_INS.md` are the notes taken while reading the source. They
quote the client's name, an Instagram access token and the biographical detail
throughout, so they are excluded by `.gitignore` and guarded by
`test_no_committed_file_carries_a_client_identifier` (D44).

They are excluded rather than redacted on purpose: cleaning a 400-line analysis
of a client workflow means finding every mention, and the anonymisation pass
already demonstrated that a hand pass misses passages. Deleting them would lose
the reasoning trail. Ignoring them keeps both properties.

If they are ever wanted in the repository, redact them under the scan — the
scan is the arbiter, not a reading.

## 9. Ideas, not commitments

- Wire the three remaining tools and re-run the eval; `smoke-01` and `smoke-02`
  become meaningful only then.
- A `MIN_VALUE_EXCHANGES` guard so the forced transition does not fire cold on
  a lead who never stated a goal (D23). No longer speculative: `smoke-04` spent
  four turns asking questions the corpus cannot answer and was pitched anyway
  at turn 5, because `transition_turn_reached` fires on a turn count alone.
- Reconciliation for `deliver` on channels that expose read-back — Instagram
  does.
- Promote `slots` back out of JSONB if slot analytics ever matter (D8).
