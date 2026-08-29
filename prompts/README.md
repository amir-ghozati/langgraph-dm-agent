# Prompts

Populated in Phase 2. Each prompt ships as a pair:

- `<name>.original.md` — the source prompt, **anonymised**, kept verbatim
  otherwise so the diff against our version stays visible
- `<name>.md` — the version actually used

The `.original.md` files contain anonymised text only. The real client prompts
encode a real person's persona, business knowledge and sales methodology; that
is the client's IP and it does not go in this repository, even as a provenance
artifact.

Three changes are already known to be required when the corpus is built here:

1. Every "not directly mentioned, but inferred from the context…" parenthetical
   on price, capacity and programme length is replaced with an explicit
   out-of-scope marker instructing refusal plus an offer of the consultation
   call. See decision D16.
2. Biographical specifics that identify the real coach even under a changed name
   are generalised.
3. The live WhatsApp contact link is removed. An agent that can emit URLs is a
   phishing surface, so the port allow-lists any link it is permitted to send.

This directory is committed with only this file so the Docker build's
`COPY prompts/` succeeds on a clean checkout.
