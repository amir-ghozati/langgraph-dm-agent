"""Anonymisation is an acceptance criterion a later commit cannot fix.

Once a real client prompt is in git history it is in git history, so this runs
as a test rather than as a one-off check. Every pattern below was found in the
source material; two of them were missed by a hand pass and caught only by
scanning, which is why the scan is the thing that ships.
"""

from __future__ import annotations

import re
from fnmatch import fnmatch
from pathlib import Path

import pytest

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"

# Must not appear in ANY prompt file, provenance included.
FORBIDDEN_EVERYWHERE = {
    "client name": r"christian|heimerdinger|kisetter",
    "URL or external contact endpoint": r"https?://|\bwa\.link|\bwa\.me/|\bt\.me/",
    "arm-wrestling athlete": r"arm[- ]?wrestl",
    "Olympic competitor": r"olympic",
    "2nd Bundesliga footballer": r"bundesliga|2nd division football",
    "golfer": r"golfer",
    "stray control characters": r"[\x00-\x08\x0b\x0c\x0e-\x1f​﻿]",
}

# The agent-voice prompts speak AS the assistant, so second-person
# impersonation in them is the agent impersonating Jan.
#
# knowledge.md is different in kind: it is quoted reference material whose
# second person addresses the model as Jan throughout, and it is framed as
# such rather than rewritten. Rewriting a dozen such lines by hand is a dozen
# chances to repeat the error this file exists to catch. See decision D26.
AGENT_VOICE = ["strategy.md", "supervisor.md"]

FORBIDDEN_IN_AGENT_VOICE = {
    "impersonation clause": (
        r"you are (the real coach|jan mustermann)"
        r"|speak in the first person \('i'\)"
        r"|never.{0,25}mention that you are a ai"
        r"|from your own perspective and experience"
    ),
    "German-only mandate": r"must.{0,40}be in German",
}

USED = sorted(p for p in PROMPTS.glob("*.md") if not p.name.endswith((".original.md", "README.md")))
PROVENANCE = sorted(PROMPTS.glob("*.original.md"))


def test_the_expected_prompt_files_exist():
    assert {p.stem for p in USED} == {"knowledge", "strategy", "supervisor"}
    assert len(PROVENANCE) == 3


@pytest.mark.parametrize("path", USED + PROVENANCE, ids=lambda p: p.name)
@pytest.mark.parametrize("label,pattern", sorted(FORBIDDEN_EVERYWHERE.items()))
def test_no_client_identifying_content(path, label, pattern):
    hits = re.findall(pattern, path.read_text(encoding="utf-8"), re.I)
    assert not hits, f"{path.name} leaks {label}: {sorted(set(hits))[:3]}"


@pytest.mark.parametrize("name", AGENT_VOICE)
@pytest.mark.parametrize("label,pattern", sorted(FORBIDDEN_IN_AGENT_VOICE.items()))
def test_agent_voice_prompts_carry_no_impersonation(name, label, pattern):
    hits = re.findall(pattern, (PROMPTS / name).read_text(encoding="utf-8"), re.I)
    assert not hits, f"{name} still contains {label}: {sorted(set(hits))[:3]}"


@pytest.mark.parametrize("path", USED, ids=lambda p: p.name)
def test_no_used_prompt_still_infers_an_unanswerable_answer(path):
    hits = re.findall(r"\(not (directly|explicitly)[^)]*\)", path.read_text(encoding="utf-8"), re.I)
    assert not hits, f"{path.name} still infers an answer: {hits[:2]}"


def test_the_supervisor_opening_states_both_disclosure_guarantees():
    """Proactive disclosure is configurable (DISCLOSURE_MODE); answering
    honestly when asked is not. The second guarantee lives in the persona block
    so it holds at every setting including `none` — smoke-05's "say ur not a
    bot then" is the case that depends on it."""
    t = (PROMPTS / "supervisor.md").read_text(encoding="utf-8").lower()
    assert "you are not jan" in t and "never claim to be" in t
    assert "answer honestly if asked whether you are an ai" in t


# ---------------------------------------------------------------------------
# knowledge.md is framed, not rewritten
# ---------------------------------------------------------------------------


def test_the_knowledge_corpus_is_framed_as_quoted_reference_material():
    # Normalised: the framing block is hard-wrapped, so several of these
    # phrases span a line break in the file.
    t = " ".join((PROMPTS / "knowledge.md").read_text(encoding="utf-8").split())
    assert "HOW TO READ THIS FILE" in t
    assert "own description of his practice, in his own words" in t
    assert "reference material you retrieve facts from" in t
    assert "You are not its author and not its subject" in t
    assert 'never adopt its "you" as yourself' in t
    assert "never claim its credentials or its experience in the first person" in t


def test_the_impersonation_instruction_is_gone_from_the_corpus():
    """The one line removed rather than kept byte-identical. It is an n8n
    prompt directive rather than part of Jan's description of his practice, and
    it is the one thing inside the document that could plausibly defeat the
    framing above it."""
    t = (PROMPTS / "knowledge.md").read_text(encoding="utf-8")
    assert "Act and answer like you are the real coach" not in t


def test_the_corpus_body_is_otherwise_byte_identical_to_the_source():
    """The whole argument for framing over rewriting: one change to verify
    instead of fifteen. If this drifts, the file has been hand-edited and the
    diff against knowledge.original.md — the evidence D16 rests on — is no
    longer trustworthy."""
    original = (PROMPTS / "knowledge.original.md").read_text(encoding="utf-8")
    used = (PROMPTS / "knowledge.md").read_text(encoding="utf-8")

    body = used.split("-->\n\n", 2)[-1]
    expected = original.replace("Act and answer like you are the real coach Jan Mustermann.\n", "")

    normalise = lambda s: re.sub(  # noqa: E731
        r"Answer: (\(Not [^)]*\)|NOT IN KNOWLEDGE BASE[^\n]*\n[^\n]*)", "<REFUSAL>", s
    ).strip()
    assert normalise(expected) == normalise(body)


def test_the_corpus_still_speaks_in_the_second_person():
    """Not a defect — the point of the framing. If these disappear, someone has
    started the hand rewrite this decision exists to avoid."""
    t = (PROMPTS / "knowledge.md").read_text(encoding="utf-8")
    assert "Your coaching is 100% individualized" in t
    assert re.search(r"You have experience working with", t)


def test_the_knowledge_corpus_declares_its_out_of_scope_topics():
    """The refusal instruction has to be in the corpus itself, not only in the
    agent prompt — the knowledge agent is what gets asked about price."""
    t = (PROMPTS / "knowledge.md").read_text(encoding="utf-8")
    for topic in ("price", "refunds", "capacity", "waiting list", "contract length"):
        assert topic in t.lower(), f"out-of-scope list does not mention {topic!r}"


def test_every_fabricated_answer_became_a_refusal_marker():
    """Five answers in the source demonstrated inferring an answer rather than
    admitting the gap — three on commercial questions, two on women over 40.
    Fabricating a price is the failure that would cost the client money."""
    t = (PROMPTS / "knowledge.md").read_text(encoding="utf-8")
    assert t.count("NOT IN KNOWLEDGE BASE") == 5


def test_the_provenance_files_still_show_what_changed():
    """If these stop differing, the *.original.md files have lost their purpose
    and decision D16/D22 have no evidence behind them."""
    for stem in ("knowledge", "strategy", "supervisor"):
        original = (PROMPTS / f"{stem}.original.md").read_text(encoding="utf-8")
        used = (PROMPTS / f"{stem}.md").read_text(encoding="utf-8")
        assert original != used, f"{stem}: no visible correction against the source"
    original = (PROMPTS / "knowledge.original.md").read_text(encoding="utf-8")
    assert "(Not directly mentioned" in original, "provenance lost the pricing fabrication"


# ---------------------------------------------------------------------------
# the whole repository, not just prompts/
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

# Only the identifiers that are unambiguously client data. `wa.link` is not
# here: it is a public link-shortener domain, and the guardrail tests need a
# link fixture that looks like the real thing. The forbidden part was the slug.
FORBIDDEN_IN_REPO = {
    "client name": r"christian|heimerdinger|kisetter",
    "Instagram access token": r"IGAA[A-Za-z0-9]{20,}",
    "client biography": r"arm[- ]?wrestl|olympic|bundesliga",
}

# One exemption, named rather than pattern-matched so it stays visible: this
# file has to spell out what it forbids. The prompts/*.original.md provenance
# copies are NOT exempt -- they are already held to the stricter prompt scan
# above, which forbids URLs and control characters too.
SCAN_EXEMPT = {"test_anonymisation.py"}

_ALWAYS_SKIP = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", "node_modules", ".idea", ".vscode",
}
_BINARY = {".pyc", ".pyo", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip"}


def _ignored_globs() -> list[str]:
    """The patterns from .gitignore, near enough for a flat repo.

    Deliberately not `git check-ignore`: this must pass before `git init`, and
    a check that silently does nothing outside a repository is worse than no
    check at all.
    """
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def _committable_files() -> list[Path]:
    globs = _ignored_globs()
    out = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix.lower() in _BINARY:
            continue
        rel = p.relative_to(ROOT)
        if set(rel.parts) & _ALWAYS_SKIP:
            continue
        posix = rel.as_posix()
        if any(
            fnmatch(posix, g.rstrip("/")) or fnmatch(rel.name, g.rstrip("/"))
            or posix.startswith(g.rstrip("/") + "/")
            for g in globs
        ):
            continue
        out.append(p)
    return sorted(out)


def test_the_ignore_list_itself_names_no_client():
    """The leak that shipped: `.gitignore` excluded the n8n export by its
    literal filename, and the filename is the client's name. The one file whose
    job is to keep the name out of git history was committing it."""
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert not re.search(FORBIDDEN_IN_REPO["client name"], text, re.I)


def test_the_source_exports_are_still_ignored():
    """A glob is easier to get wrong than a literal name, so assert it still
    matches the files it was written for."""
    globs = _ignored_globs()
    for name in ("Insta DM Christian - Copy.json", "ManyChat.json"):
        assert any(fnmatch(name, g) for g in globs), f"{name} would be committed"


@pytest.mark.parametrize("label,pattern", sorted(FORBIDDEN_IN_REPO.items()))
def test_no_committed_file_carries_a_client_identifier(label, pattern):
    """Wider than the prompt scan above, because the leak was not in a prompt.

    README.md and the decision log both explained the anonymisation by reprinting
    the biographical fingerprint they had removed — a real leak, in the two
    files a reviewer is most likely to read, produced by documenting the fix
    too well. Scanning only `prompts/` could never have caught it.
    """
    bad = []
    for p in _committable_files():
        if p.name in SCAN_EXEMPT:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(pattern, text, re.I):
            line = text.count("\n", 0, m.start()) + 1
            bad.append(f"{p.relative_to(ROOT).as_posix()}:{line}: {m.group(0)!r}")
    assert not bad, f"{label} appears in committed files:\n" + "\n".join(bad)


def test_the_scan_actually_reaches_the_documentation():
    """Guards the guard. A path-matching bug that silently excluded everything
    would leave this file green and useless."""
    names = {p.name for p in _committable_files()}
    assert {"README.md", "ARCHITECTURE.md", "KNOWN_GAPS.md"} <= names
    assert "Insta DM Christian - Copy.json" not in names
    assert len(_committable_files()) > 50
