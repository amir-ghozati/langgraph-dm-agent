"""Burst aggregation and the channel abstraction.

The property under test is not "grouping works" — it is that N inbound
messages produce exactly one agent turn, because replying per message is the
most bot-like behaviour a DM agent can exhibit and `smoke-05` exists to catch
that class of tell.
"""

from __future__ import annotations

import datetime as dt

import pytest

from app.channels.base import Channel, DeliveryReceipt, InboundMessage, OutboundMessage
from app.channels.cli import CLIChannel
from app.channels.envelope import BurstBuffer, InboundEnvelope, group_bursts
from app.channels.instagram import InstagramChannel
from app.funnel.stages import FunnelStage
from app.funnel.transitions import FunnelPolicy, TransitionContext, decide

NOW = dt.datetime(2026, 3, 10, 12, 0, tzinfo=dt.UTC)


def msg(text: str, user: str = "u1", channel: str = "cli", at: dt.datetime = NOW) -> InboundMessage:
    return InboundMessage(channel=channel, channel_user_id=user, text=text, received_at=at)


# ---------------------------------------------------------------------------
# The property that matters
# ---------------------------------------------------------------------------


def test_a_three_message_burst_is_one_turn_and_advances_the_index_by_one():
    """The rule, stated as the thing it protects: a chatty lead must not get
    three replies, and must not reach FUNNEL_HARD_CAP faster than a terse one."""
    burst = [msg("hey"), msg("saw ur reel"), msg("the one about knees")]
    envelopes = [InboundEnvelope.of(b) for b in group_bursts(burst, lambda m: True)]

    assert len(envelopes) == 1, "three messages must produce one turn, not three"
    assert len(envelopes[0]) == 3

    policy = FunnelPolicy()
    before = 3
    after = before + len(envelopes)
    assert after == 4, "the funnel clock advances once per agent reply, not per message"
    assert decide(
        FunnelStage.VALUE, None, TransitionContext(agent_turn_index=after), policy
    ).decided_stage is FunnelStage.VALUE


def test_intent_is_classified_on_the_whole_burst_not_the_first_message():
    """smoke-04 opens "heyy loved the free guide" then "hows pricing work". The
    first fragment carries nothing; classifying on it alone would route to
    OPENER and answer a pricing question with a goal question."""
    envelope = InboundEnvelope.of([msg("heyy loved the free guide"), msg("hows pricing work")])
    assert "pricing" in envelope.text
    assert envelope.text.startswith("heyy")


def test_burst_text_is_newline_joined_not_run_together():
    envelope = InboundEnvelope.of([msg("ugh fine"), msg("0151 23oh4a78")])
    assert envelope.text == "ugh fine\n0151 23oh4a78"


# ---------------------------------------------------------------------------
# group_bursts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kinds", "expected"),
    [
        (["in"], [1]),
        (["in", "out", "in"], [1, 1]),
        (["in", "in", "out", "in"], [2, 1]),
        (["in", "out", "in", "in", "in"], [1, 3]),
        (["out", "in"], [1]),
        (["out"], []),
        ([], []),
    ],
)
def test_group_bursts_splits_on_agent_replies(kinds, expected):
    runs = group_bursts(kinds, lambda k: k == "in")
    assert [len(r) for r in runs] == expected


def test_a_trailing_burst_is_not_dropped():
    """Every golden conversation ends on an inbound burst with no `agent:`
    placeholder after it. Dropping it would lose the final agent turn."""
    runs = group_bursts(["in", "out", "in", "in"], lambda k: k == "in")
    assert [len(r) for r in runs] == [1, 2]


# ---------------------------------------------------------------------------
# Envelope invariants
# ---------------------------------------------------------------------------


def test_an_envelope_needs_at_least_one_message():
    with pytest.raises(ValueError, match="at least one message"):
        InboundEnvelope(channel="cli", channel_user_id="u1", messages=())


def test_an_envelope_cannot_mix_senders():
    """Two leads' messages in one turn would cross conversations, which the
    partial unique index on `conversations` cannot catch because it is one row
    per lead, not per turn."""
    with pytest.raises(ValueError, match="mixes senders"):
        InboundEnvelope.of([msg("hi", user="u1"), msg("hi", user="u2")])


def test_envelope_carries_only_real_channel_message_ids():
    e = InboundEnvelope.of(
        [
            InboundMessage("ig", "u1", "a", channel_message_id="m1"),
            InboundMessage("ig", "u1", "b", channel_message_id=None),
        ]
    )
    assert e.channel_message_ids == ["m1"]


def test_envelope_timestamp_is_the_last_message():
    early, late = NOW, NOW + dt.timedelta(seconds=4)
    e = InboundEnvelope.of([msg("a", at=early), msg("b", at=late)])
    assert e.received_at == late


# ---------------------------------------------------------------------------
# BurstBuffer — the live debounce, tested without sleeping
# ---------------------------------------------------------------------------


def test_each_new_message_restarts_the_window():
    buf = BurstBuffer(window_seconds=5)
    buf.add(msg("hey"), now=NOW)
    assert buf.due(NOW + dt.timedelta(seconds=4)) is False
    buf.add(msg("wait"), now=NOW + dt.timedelta(seconds=4))
    # Without the restart this would already be due at t+5.
    assert buf.due(NOW + dt.timedelta(seconds=6)) is False
    assert buf.due(NOW + dt.timedelta(seconds=9)) is True


def test_an_empty_buffer_is_never_due():
    assert BurstBuffer(window_seconds=5).due(NOW + dt.timedelta(hours=1)) is False


def test_flush_returns_one_envelope_and_empties_the_buffer():
    buf = BurstBuffer(window_seconds=5)
    buf.add(msg("a"), now=NOW)
    buf.add(msg("b"), now=NOW)
    envelope = buf.flush()
    assert envelope is not None and len(envelope) == 2
    assert len(buf) == 0 and buf.flush() is None


def test_flush_works_before_the_window_expires():
    """Called on shutdown as well as on expiry. Dropping a half-collected burst
    because the process is stopping loses a real user message."""
    buf = BurstBuffer(window_seconds=30)
    buf.add(msg("half a thought"), now=NOW)
    assert buf.due(NOW) is False
    assert buf.flush() is not None


def test_a_buffer_holds_one_sender():
    buf = BurstBuffer(window_seconds=5)
    buf.add(msg("hi", user="u1"), now=NOW)
    with pytest.raises(ValueError, match="one sender"):
        buf.add(msg("hi", user="u2"), now=NOW)


def test_a_zero_window_is_allowed_and_makes_every_message_its_own_turn():
    """SOURCE_EXACT-style configuration: reproduces the original's
    reply-per-webhook behaviour for comparison."""
    buf = BurstBuffer(window_seconds=0)
    buf.add(msg("hey"), now=NOW)
    assert buf.due(NOW) is True


def test_a_negative_window_is_rejected():
    with pytest.raises(ValueError, match="must not be negative"):
        BurstBuffer(window_seconds=-1)


# ---------------------------------------------------------------------------
# Channel implementations
# ---------------------------------------------------------------------------


def test_cli_channel_satisfies_the_protocol():
    assert isinstance(CLIChannel(), Channel)


def test_instagram_channel_satisfies_the_protocol_but_refuses_to_run():
    channel = InstagramChannel()
    assert isinstance(channel, Channel)
    with pytest.raises(NotImplementedError, match="documented stub"):
        channel.receive()


async def test_instagram_send_also_refuses():
    with pytest.raises(NotImplementedError):
        await InstagramChannel().send(
            OutboundMessage(
                conversation_id=__import__("uuid").uuid4(),
                channel="instagram",
                channel_user_id="u1",
                text="hi",
                idempotency_key="k",
            )
        )


def test_cli_burst_reading_ends_on_a_blank_line(monkeypatch):
    """The explicit affordance that replaces a timer in development: type
    lines, blank line submits them as one burst."""
    lines = iter(["hey", "saw ur reel", "", "next"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(lines))
    assert CLIChannel()._read_burst() == ["hey", "saw ur reel"]


def test_cli_quit_ends_the_session(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt="": "/quit")
    assert CLIChannel()._read_burst() is None


def test_cli_eof_ends_the_session(monkeypatch):
    def _raise(_prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise)
    assert CLIChannel()._read_burst() is None


def test_cli_blank_lines_before_any_input_do_not_submit_an_empty_burst(monkeypatch):
    lines = iter(["", "  ", "hello", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(lines))
    assert CLIChannel()._read_burst() == ["hello"]


async def test_cli_send_reports_delivery():
    receipt = await CLIChannel().send(
        OutboundMessage(
            conversation_id=__import__("uuid").uuid4(),
            channel="cli",
            channel_user_id="local",
            text="hey!",
            idempotency_key="k1",
        )
    )
    assert isinstance(receipt, DeliveryReceipt)
    assert receipt.delivered and receipt.delivered_at is not None
