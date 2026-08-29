# Phase 0 — Answers

Good analysis. You caught three errors in the brief; I verified two against the JSON
directly and you are correct on both. Details in §14 at the bottom. Corrections accepted —
the brief is amended accordingly.

---

## Blocking questions

### 1. Model fidelity → **(b), reproduce specified behaviour**

Option (a) is not achievable. The Gemini nodes carry no model name, no temperature, no
top-p, and the hosted model has drifted since the workflow was authored. There is no
artifact to reproduce against.

The golden set asserts against the *specification* — the prompts, the funnel rules, the
tool contracts — not against the original's observed outputs. State this explicitly in
the decision log and in the eval report, because "we could not reproduce the source
because the source was never pinned" is itself a finding worth writing down: it is an
argument for pinned model versions in any system you intend to evaluate.

### 2. The 6-message contradiction → **your proposal, approved**

Value on turns 2–4, transition begins turn 5, hard cap at turn 7. All three
configurable (`FUNNEL_VALUE_TURNS`, `FUNNEL_TRANSITION_TURN`, `FUNNEL_HARD_CAP`).

Document the contradiction itself in the decision log — the source prompt disagreed with
itself, we resolved it deterministically, and the config makes the resolution visible
rather than buried in a prompt. That is a better system than the original regardless of
which number is "right".

### 3. Explicit funnel state → **approved, and promoted to improvement #1**

You are right that this outranks the memory window, and right that the window is the
symptom. Reordering the brief's improvement list: explicit funnel state first, memory
policy second.

Design it as: persisted enum on the conversation row, deterministic transition table,
LLM *proposes* advancement and the state machine *decides*. The LLM's proposal, the
decision, and the reason all get logged per turn. That gives the eval harness something
real to assert and gives Langfuse something real to trace.

This is a deliberate behaviour change from the source. Say so in the README.

### 4. Phone validation → **E.164 via `phonenumbers`, default region DE**

Rules:
- Parse with `phonenumbers.parse(raw, DEFAULT_REGION)`, `DEFAULT_REGION` configurable, default `"DE"`
- Require `is_valid_number()` **and** number type in `{MOBILE, FIXED_LINE_OR_MOBILE}`
- Persist E.164 only; never persist the raw string as the canonical value
- On failure: do not hard-fail the turn. Re-ask once with a specific error ("that
  doesn't look like a mobile number — could you send it with the country code?"), then
  escalate to human handoff via the HITL interrupt

The re-ask path belongs in the golden set as its own scenario. Slot-filling recovery is
a thing interviewers ask about and the source has no answer for it.

### 5. Availability semantics → **diverge, with a source-exact preset**

Defaults for the port:

| Setting | Value | Rationale |
|---|---|---|
| Session length | 30 min | as source |
| Window | 09:00–21:00, 7 days | as source |
| Horizon | rolling 14 days | as source |
| **Minimum lead time** | **120 min** | **new** |
| Inter-session buffer | 0 min | as source |

No lead time is a real defect, not a design choice: the source permits booking a slot
five minutes out, which the coach cannot see or prepare for. Fix it.

Buffer stays at 0 — back-to-back 30-minute consults are plausible for this business and
changing it without cause is scope.

Ship a `SOURCE_EXACT` config preset that restores lead time to 0, so the golden set can
test both and the diff is demonstrable rather than asserted.

### 6. Knowledge agent → **full-context, pluggable retriever, evaluate both**

Your reasoning is right and I want to sharpen why.

At 2,252 words (~3.5k tokens) against a 32k context, retrieval solves nothing. If you
ship RAG here, the first competent interviewer asks "why are you retrieving over three
thousand tokens?" and there is no good answer. That would actively damage the project.

But building the retriever interface and **measuring that it did not help** is a strong
result. `groundedness` and `answer completeness` at full-context vs. top-k retrieval,
with numbers, and the conclusion that retrieval was unnecessary at this corpus size —
that demonstrates you choose architecture from evidence instead of fashion. Put the
comparison table in the README.

One instruction: do **not** stretch this into a RAG showcase. That gap is being closed
by a separate project on a corpus where retrieval is actually necessary. Here, the
honest negative result is worth more than a contrived positive one.

### 7. "Completed conversation" → **approved, with one addition**

A conversation is complete when any of:
- a booking is confirmed, **or**
- `CONVERSATION_IDLE_HOURS` elapse with no inbound message (default 24), **or**
- the user explicitly opts out

Opt-out is a terminal state distinct from abandonment and the analyst must not classify
it as a failure — otherwise your funnel metrics quietly punish correct behaviour.

### 8. The broken analyst input → **PENDING — Amirhossein to confirm**

> *[AMIRHOSSEIN: do you still have access to the live n8n instance or its execution
> logs? Can you confirm whether `$('Input + Output').item.json.body…` resolves at
> runtime, or has been passing `undefined` to the analyst?]*

Proceed on the assumption that it is broken until confirmed otherwise. If it is: port
the analyst's *stated* classification criteria from its prompt, not its observed
behaviour, since the observed behaviour would be classification of empty input.

### 9. The ManyChat gate → **PENDING — Amirhossein to confirm**

> *[AMIRHOSSEIN: was silently dropping senders without a `manychat` row intended —
> i.e. only lead-magnet/paid-ad traffic gets the AI — or is it an accident?]*

Build the gate node as a pluggable `EligibilityPolicy` either way, with `AllowAll` and
`RequireLeadRecord` implementations. Default to `AllowAll` until confirmed. Silent drops
must be logged and counted regardless — a gate that discards traffic without telling
anyone is a bug even when the filtering is intentional.

---

## Non-blocking

### 10. Anonymisation persona → **approved, with a check**

Proposed default is fine. Before committing it, confirm the invented name is not a real
German fitness coach — a synthetic persona that collides with a real person is worse
than no anonymisation. If in doubt, pick something obviously constructed.

### 11. Prompt provenance → **approved, with one correction**

Verbatim `*.original.md` files plus runtime-composed disclosure: yes.

**Correction: the `.original.md` files must contain the anonymised originals, not the
real client prompts.** They encode a real person's persona, business knowledge, and
sales methodology — that is the client's IP and it does not go in a public repo, even as
a "provenance" artifact. Anonymise first, then treat the anonymised version as the
canonical original for the diff.

### 12. Language → **(c), but invert the default**

Configurable locale, evaluated in both — agreed. Change: make **English the primary eval
locale and German the fidelity locale**, and report both in the eval report.

The cross-locale comparison is the most portable finding this project will produce.
Structured-output validity rate, tool-call correctness, and groundedness measured on the
same graph and the same 7B model across two languages tells you what degrades when you
move a local model off English. Give it its own section in `report.md` with a table.

### 13. The 10-second delay → **configurable, default 0, documented**

Approved as proposed. It exists to sell the human impersonation; with disclosure enabled
it has no function. Document it as deliberately disabled and say why.

---

## 14. On your three disagreements

**Sheets count — you are right.** Verified: 48 `googleSheets` nodes, of which 35 are
appends to the `Error` sheet. Only 13 touch real data (11 `Reminder`, 2 `manychat`)
across 2 sheets. The brief's "48-node spreadsheet database" was wrong. Your reframe is
better and more interesting: a large fraction of this workflow exists solely because n8n
has no exception handling, and the port deletes all of it.

One correction to your correction: I can verify 35 error-append nodes = 23% of the 149
total. Your figure of 46% presumably counts other error-path nodes (branches, no-ops,
routing). **Show the working for that number or use 23%.** An unverifiable statistic in
a portfolio README has exactly the failure mode we are trying to avoid.

**Delegation — you are right.** Verified: both `agentTool` inputs are hardcoded to
`$('Insta Trigger').item.json.body.entry[0].messaging[0].message.text`. The model never
populates the argument. Additional evidence you can use: both tools carry *identical*
boilerplate `toolDescription` values ("a specialized tool for processing the user's
request"), so even a model that could choose has no basis for choosing. It is a
hardcoded fan-out.

Frame the port exactly as you proposed: replacing a hardcoded fan-out with a real
supervisor, where the orchestrator populates each sub-agent's input and decides which to
call. **The delta is the story.** Put a before/after diagram in `ARCHITECTURE.md`.

**Improvement ordering — you are right.** Funnel state first, memory second. Amended.

---

## Proceed to Phase 1

Questions 8 and 9 are with Amirhossein. Neither blocks Phase 1 — build the eligibility
gate as a pluggable policy and the analyst against its stated criteria, and we will
revisit when he answers.

Before you build: bring me the graph design for approval, per §4 of the brief.
