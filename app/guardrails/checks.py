"""Guardrails on the composed draft, before anything is sent.

Two remediation paths, deliberately different:

* **Deterministic style violations re-compose.** Length, sentence count, emoji
  placement, a missing disclosure line. These are mechanical and a second
  attempt usually fixes them, bounded at `GUARDRAIL_RECOMPOSE_LIMIT`.
* **Safety flags interrupt.** Impersonation, medical scope, an unlisted link,
  low confidence. A human decides; the draft is not sent while they do.

Confidence is not the model's self-report alone. A 7B model's self-assessment
is not trustworthy and a frontier model's is over-confident in exactly the
cases that matter, so the composed `self_confidence` is reduced by *observed*
failure signals — repairs used, tools failed, proposal overridden, slots
unresolved. Combining a claim with evidence is the point.

The link check is an allow-list rather than a blocklist. An agent that can emit
URLs is a phishing surface, and the allow-list is currently empty (D22).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from app.agents.compose import DISCLOSURE_LINE, ComposedReply
from app.config import Settings, get_settings
from app.logging import get_logger

log = get_logger(__name__)

URL = re.compile(r"https?://\S+|\bwww\.\S+|\b\w+\.(?:com|de|link|me|io|co)\b/?\S*", re.I)
EMOJI = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff✀-➿️]"
)
SENTENCE_END = re.compile(r"[.!?]+(?:\s|$)")

# Currently empty, on purpose. Every link the agent may send goes here first.
LINK_ALLOWLIST: frozenset[str] = frozenset()

IMPERSONATION = re.compile(
    r"\bi'?m not (?:a |an )?(?:bot|ai|robot)\b"
    r"|\bi'?m (?:a )?real (?:person|human|one)\b"
    r"|\bnot a bot\b"
    r"|\bi'?m human\b",
    re.I,
)

CREDENTIAL_CLAIM = re.compile(
    r"\bi'?(?:m| am) (?:a |an )?(?:certified|qualified|licen[cs]ed)\b"
    r"|\bi'?ve (?:been )?(?:coach|train)(?:ed|ing)\b"
    r"|\bi'?ve worked with\b"
    r"|\bin my experience\b"
    r"|\bwe'?re certified\b",
    re.I,
)

CONFIRMATION = re.compile(
    r"\b(?:you'?re|you are|your|you'?ve been|i'?ve)\s+(?:all\s+|now\s+)?"
    r"(?:booked|scheduled|confirmed|set)\b"
    r"|\bbooking (?:is |has been )?confirmed\b"
    r"|\bconfirmed for\b"
    r"|\ball set for\b"
    r"|\bsee you (?:on|at|then)\b"
    r"|\block(?:ed)? (?:it |that |you )?in\b",
    re.I,
)


class Severity(StrEnum):
    STYLE = "style"
    """Re-compose. Mechanical, bounded."""

    SAFETY = "safety"
    """Interrupt. A human decides."""


class Violation(StrEnum):
    TOO_LONG = "too_long"
    TOO_SHORT = "too_short"
    MULTIPLE_SENTENCES = "multiple_sentences"
    TOO_MANY_EMOJI = "too_many_emoji"
    ADJACENT_EMOJI = "adjacent_emoji"
    MISSING_DISCLOSURE = "missing_disclosure"

    CLAIMS_TO_BE_HUMAN = "claims_to_be_human"
    FIRST_PERSON_CREDENTIAL = "first_person_credential"
    UNLISTED_LINK = "unlisted_link"
    UNEARNED_CONFIRMATION = "unearned_confirmation"
    LOW_CONFIDENCE = "low_confidence"


SEVERITY: dict[Violation, Severity] = {
    Violation.TOO_LONG: Severity.STYLE,
    Violation.TOO_SHORT: Severity.STYLE,
    Violation.MULTIPLE_SENTENCES: Severity.STYLE,
    Violation.TOO_MANY_EMOJI: Severity.STYLE,
    Violation.ADJACENT_EMOJI: Severity.STYLE,
    Violation.MISSING_DISCLOSURE: Severity.STYLE,
    Violation.CLAIMS_TO_BE_HUMAN: Severity.SAFETY,
    Violation.FIRST_PERSON_CREDENTIAL: Severity.SAFETY,
    Violation.UNLISTED_LINK: Severity.SAFETY,
    Violation.UNEARNED_CONFIRMATION: Severity.SAFETY,
    Violation.LOW_CONFIDENCE: Severity.SAFETY,
}


@dataclass(frozen=True, slots=True)
class GuardVerdict:
    violations: list[Violation] = field(default_factory=list)
    confidence: float = 1.0
    notes: list[str] = field(default_factory=list)

    @property
    def style_violations(self) -> list[Violation]:
        return [v for v in self.violations if SEVERITY[v] is Severity.STYLE]

    @property
    def safety_violations(self) -> list[Violation]:
        return [v for v in self.violations if SEVERITY[v] is Severity.SAFETY]

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def must_interrupt(self) -> bool:
        return bool(self.safety_violations)

    @property
    def should_recompose(self) -> bool:
        return bool(self.style_violations) and not self.safety_violations


def effective_confidence(
    self_confidence: float,
    *,
    repair_attempts: int = 0,
    tool_failures: int = 0,
    proposal_overridden: bool = False,
    unresolved_slots: int = 0,
) -> float:
    """The model's claim, reduced by what actually happened.

    A self-report alone is not evidence. Each penalty corresponds to something
    observed during the turn, so a confident reply produced after two repairs
    and an overridden proposal does not get to call itself confident.
    """
    penalty = (
        0.15 * repair_attempts
        + 0.25 * tool_failures
        + (0.10 if proposal_overridden else 0.0)
        + 0.05 * min(unresolved_slots, 4)
    )
    return max(0.0, min(1.0, self_confidence) - penalty)


def check(
    reply: ComposedReply,
    *,
    settings: Settings | None = None,
    disclosure_required: bool = False,
    booking_committed: bool = False,
    repair_attempts: int = 0,
    tool_failures: int = 0,
    proposal_overridden: bool = False,
    unresolved_slots: int = 0,
) -> GuardVerdict:
    settings = settings or get_settings()
    text = reply.text or ""
    violations: list[Violation] = []
    notes: list[str] = []

    words = len(text.split())
    if words > settings.reply_max_words:
        violations.append(Violation.TOO_LONG)
    if words < settings.reply_min_words:
        violations.append(Violation.TOO_SHORT)

    # A fragment with no letters or digits is not a sentence. Without this a
    # trailing emoji after the final "?" counts as one, so every correctly
    # styled reply ending in an emoji — which the style rules *require* —
    # would be flagged and pointlessly re-composed.
    sentences = [s for s in SENTENCE_END.split(text) if re.search(r"\w", s)]
    if len(sentences) > settings.reply_max_sentences:
        violations.append(Violation.MULTIPLE_SENTENCES)

    emoji = EMOJI.findall(text)
    if len(emoji) > 3:
        violations.append(Violation.TOO_MANY_EMOJI)
    if re.search(f"(?:{EMOJI.pattern})\\s*(?:{EMOJI.pattern})", text):
        violations.append(Violation.ADJACENT_EMOJI)

    if disclosure_required and DISCLOSURE_LINE.split("—")[-1].strip()[:20] not in text:
        violations.append(Violation.MISSING_DISCLOSURE)

    # --- safety ------------------------------------------------------------
    if IMPERSONATION.search(text):
        violations.append(Violation.CLAIMS_TO_BE_HUMAN)
        notes.append("draft denies being an AI")

    if CREDENTIAL_CLAIM.search(text):
        violations.append(Violation.FIRST_PERSON_CREDENTIAL)
        notes.append("draft claims Jan's credentials in the first person")

    for url in URL.findall(text):
        if url not in LINK_ALLOWLIST:
            violations.append(Violation.UNLISTED_LINK)
            notes.append(f"draft contains a link not on the allow-list: {url}")
            break

    if CONFIRMATION.search(text) and not booking_committed:
        # Belt, braces and a third lock. The BOOKED transition is the only
        # thing that emits a confirmation, and this catches a draft that
        # phrases one anyway.
        violations.append(Violation.UNEARNED_CONFIRMATION)
        notes.append("draft implies a booking that has not been committed")

    confidence = effective_confidence(
        reply.self_confidence,
        repair_attempts=repair_attempts,
        tool_failures=tool_failures,
        proposal_overridden=proposal_overridden,
        unresolved_slots=unresolved_slots,
    )
    if confidence < settings.confidence_threshold:
        violations.append(Violation.LOW_CONFIDENCE)
        notes.append(f"effective confidence {confidence:.2f} below threshold")

    if violations:
        log.info("guardrail.violations", violations=[str(v) for v in violations])
    return GuardVerdict(violations=violations, confidence=confidence, notes=notes)


def safe_template(stage_hint: str = "") -> str:
    """What is sent when re-composition is exhausted.

    Says nothing that could be wrong, and hands the turn to a person.
    """
    return "Let me get Jan to pick this one up with you directly 🙏"
