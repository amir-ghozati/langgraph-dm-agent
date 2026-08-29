# Open questions

All four resolved. Kept as a record of what was asked and how it was answered;
the decisions themselves live in decision D18–D21 and the behaviour lives
in `app/funnel/transitions.py`.

| # | Question | Resolution | Recorded |
|---|---|---|---|
| Q1 | The opener-only rule collides with three golden conversations whose first message already carries intent | Conditional opener: three guarded transitions out of `NEW`, classified by the supervisor and decided by the state machine — not a prompt hedge | D18 |
| Q2 | Is there an acceptance turn between the offer and slot collection? | No. `TRANSITION → SLOT_FILLING` is guarded on "did not decline", following the source. Coupled to Q1: fixing one alone would not have fixed the turn budget | D19 |
| Q3 | The phone failure path handed off after one re-ask, so `smoke-03`'s third input was never processed | Three attempts total, `PHONE_REASK_LIMIT=2`, configurable | D20 |
| Q4 | `OPTED_OUT` is terminal, but `smoke-06` expects a reply to a user-initiated message after opt-out | Outbound initiation blocked; inbound answered minimally; funnel never resumes | D21 |

Q1 and Q3 changed what is assertable in `smoke-02`, `smoke-03` and `smoke-04`.
All four were settled before those `expect:` blocks were written.

## Not a question, recorded for accuracy

`GOLDEN_SET_FILL_INS.md` §2 locates the 6-versus-7 message contradiction between
prompts 2 and 3. Reading prompt 3 directly, it is also *inside* prompt 3: the
heading says "AIDA Model in a Maximum of 6 Messages" while the body specifies
the opener at 1, value through 4, transition beginning at 5 and running "2 or 3
messages" — which reaches 7.

This does not change the resolution; it strengthens it. `FUNNEL_HARD_CAP=7` now
matches the source's own message arithmetic rather than only matching prompt 2.
See decision D16.
