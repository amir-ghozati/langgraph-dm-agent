from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.llm.base import Message, Role, parse_into, split_system, unwrap_json_text


class Plan(BaseModel):
    action: str
    confidence: float


def test_plain_json_parses():
    obj, err = parse_into(Plan, '{"action": "ask", "confidence": 0.8}')
    assert err is None
    assert obj.action == "ask"


def test_fenced_json_is_unwrapped():
    """Models fence their output constantly. Removing a fence that delimits the
    whole payload is a deterministic unwrap, not a search through prose."""
    text = '```json\n{"action": "ask", "confidence": 0.8}\n```'
    assert unwrap_json_text(text) == '{"action": "ask", "confidence": 0.8}'
    obj, err = parse_into(Plan, text)
    assert err is None and obj.confidence == 0.8


def test_unfenced_bare_backticks_also_unwrap():
    obj, err = parse_into(Plan, '```\n{"action": "ask", "confidence": 0.1}\n```')
    assert err is None and obj.action == "ask"


def test_json_buried_in_prose_is_a_failure_not_a_rescue():
    """This is the deliberate line. The source system used a regex to fish JSON
    out of surrounding text; recreating that would hide exactly the failure the
    structured-output validity metric is supposed to count. A model that
    editorialises has failed the call, and the repair ladder handles it."""
    obj, err = parse_into(Plan, 'Sure! Here you go: {"action": "ask", "confidence": 0.8}')
    assert obj is None
    assert err is not None and "not valid JSON" in err


def test_schema_violation_returns_a_usable_error():
    """The error text is fed back to the model on the repair attempt, so it has
    to describe the problem rather than merely signal one."""
    obj, err = parse_into(Plan, '{"action": "ask"}')
    assert obj is None
    assert err is not None and "confidence" in err


def test_empty_output_is_a_failure():
    obj, err = parse_into(Plan, "")
    assert obj is None and err is not None


def test_split_system_pulls_system_messages_out_of_band():
    system, rest = split_system(
        [
            Message(Role.SYSTEM, "you are a coach"),
            Message(Role.USER, "hi"),
            Message(Role.ASSISTANT, "hello"),
            Message(Role.SYSTEM, "be brief"),
        ]
    )
    assert system == "you are a coach\n\nbe brief"
    assert [m.role for m in rest] == [Role.USER, Role.ASSISTANT]


def test_split_system_with_no_system_message():
    system, rest = split_system([Message(Role.USER, "hi")])
    assert system is None
    assert len(rest) == 1


@pytest.mark.parametrize("text", ["null", "[]", '"a string"', "42"])
def test_valid_json_of_the_wrong_shape_is_rejected(text):
    obj, err = parse_into(Plan, text)
    assert obj is None and err is not None
