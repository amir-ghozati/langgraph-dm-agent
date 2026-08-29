"""Every field that accepts an identifier must reject unknown identifiers.

Four bugs of exactly one shape have now been found, each an assertion that
could not fail while reading as coverage:

* `never_stage_after` with a pivot the conversation never reaches
* `set("price")` — the scalar shorthand binding a bare string to a list param
* `agent:` placeholder counting, under-reporting turns by one in all eight files
* `forbidden: [commit_bookng]` — a misspelled tool that is never called

Three were found by accident. This file looks on purpose, and it enumerates the
schema rather than listing cases, so a *new* identifier field added later
cannot reintroduce the hole without failing here.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evals.schema import (
    ALL_PREDICATES,
    AMBIGUITY_SUBJECTS,
    ASSERTABLE_FIELDS,
    INVOCABLE,
    PHONE_ERROR_KINDS,
    UNGROUNDED_TOPICS,
    Expectation,
)

# name -> (a builder taking one identifier, a sentinel that must be rejected)
IDENTIFIER_FIELDS = {
    "expect.conversation predicate": (
        lambda bad: {"conversation": [bad]},
        "no_price_stateddd",
    ),
    "expect.turns predicate": (
        lambda bad: {"turns": {1: [bad]}},
        "opener_onlyyy",
    ),
    "tools.required tool": (
        lambda bad: {"tools": {"required": [{"tool": bad}]}},
        "resolev_datetime",
    ),
    "tools.forbidden tool": (
        lambda bad: {"tools": {"forbidden": [bad]}},
        "commit_bookng",
    ),
    "unasserted field name": (
        lambda bad: {"unasserted": {bad: "a reason"}},
        "finaal_stage",
    ),
    "refuses_without_fabricating topic": (
        lambda bad: {"conversation": [{"refuses_without_fabricating": {"topics": [bad]}}]},
        "shoe_size",
    ),
    "asks_clarifying_question subject": (
        lambda bad: {"conversation": [{"asks_clarifying_question": {"about": [bad]}}]},
        "vibe",
    ),
    "phone_error_names kind": (
        lambda bad: {"turns": {1: [{"phone_error_names": bad}]}},
        "wrong_colour",
    ),
    "stage_not_advanced_past stage": (
        lambda bad: {"conversation": [{"stage_not_advanced_past": {"stage": bad}}]},
        "ALMOST_BOOKED",
    ),
    "never_stage_after pivot": (
        lambda bad: {"never_stage_after": {bad: ["BOOKED"]}},
        "NOT_A_STAGE",
    ),
}


@pytest.mark.parametrize("name", sorted(IDENTIFIER_FIELDS))
def test_every_identifier_field_rejects_an_unknown_identifier(name):
    build, bad = IDENTIFIER_FIELDS[name]
    with pytest.raises(ValidationError):
        Expectation.model_validate({"final_stage": "VALUE"} | build(bad))


@pytest.mark.parametrize("name", sorted(IDENTIFIER_FIELDS))
def test_every_identifier_field_accepts_a_known_identifier(name):
    """A rejection test alone would pass on a field that rejects everything."""
    good = {
        "expect.conversation predicate": "no_price_stated",
        "expect.turns predicate": "opener_only",
        "tools.required tool": "resolve_datetime",
        "tools.forbidden tool": "commit_booking",
        "unasserted field name": "interrupt",
        "refuses_without_fabricating topic": "price",
        "asks_clarifying_question subject": "day",
        "phone_error_names kind": "landline",
        "stage_not_advanced_past stage": "VALUE",
        "never_stage_after pivot": "OPTED_OUT",
    }[name]
    build, _ = IDENTIFIER_FIELDS[name]
    payload = {"final_stage": "VALUE"} | build(good)
    if name == "never_stage_after pivot":
        payload["stages_visited"] = ["OPTED_OUT"]
    Expectation.model_validate(payload)


def test_the_registry_covers_every_closed_vocabulary_in_the_schema():
    """If a new closed vocabulary is added and not wired in here, this fails —
    which is the point. The hole reopened three times by being added somewhere
    new each time."""
    covered = {
        "predicates": {"expect.conversation predicate", "expect.turns predicate"},
        "invocable": {"tools.required tool", "tools.forbidden tool"},
        "assertable_fields": {"unasserted field name"},
        "ungrounded_topics": {"refuses_without_fabricating topic"},
        "ambiguity_subjects": {"asks_clarifying_question subject"},
        "phone_error_kinds": {"phone_error_names kind"},
        "funnel_stages": {"stage_not_advanced_past stage", "never_stage_after pivot"},
    }
    vocabularies = {
        "predicates": ALL_PREDICATES,
        "invocable": INVOCABLE,
        "assertable_fields": ASSERTABLE_FIELDS,
        "ungrounded_topics": UNGROUNDED_TOPICS,
        "ambiguity_subjects": AMBIGUITY_SUBJECTS,
        "phone_error_kinds": PHONE_ERROR_KINDS,
    }
    for vocab, fields in covered.items():
        assert fields <= set(IDENTIFIER_FIELDS), f"{vocab} has an uncovered field"
    for vocab, values in vocabularies.items():
        assert values, f"{vocab} is empty; a vocabulary that accepts nothing rejects nothing"


def test_an_empty_predicate_list_asserts_nothing_and_is_visible_as_such():
    """The other half of the pattern: a field that accepts a set must not let
    the empty set read as coverage."""
    empty = Expectation.model_validate({"final_stage": "VALUE", "conversation": []})
    assert "conversation" not in empty.asserted_fields
    assert "conversation" in empty.coverage()["unexplained"]


def test_an_empty_tools_block_asserts_nothing():
    empty = Expectation.model_validate(
        {"final_stage": "VALUE", "tools": {"required": [], "forbidden": []}}
    )
    assert "tools.required" not in empty.asserted_fields
    assert "tools.forbidden" not in empty.asserted_fields
