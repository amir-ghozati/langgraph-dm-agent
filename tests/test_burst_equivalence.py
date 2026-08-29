"""The CLI and the eval runner must produce the same envelope.

Two burst implementations is the bug just removed from `envelope.py`, one layer
up. The runner derives bursts structurally from the YAML; a terminal user needs
some way to say "these three lines are one turn". If those two paths ever
disagree, every golden-set result is measured against a turn structure the CLI
never produces, and the eval stops describing the system.

`CLIChannel` uses an explicit affordance — type lines, blank line sends —
rather than the live 5-second debounce. Real timing would make the development
loop wait five seconds per turn and make these tests sleep, and the property
under test is that N messages produce one envelope, not that a clock works.
`BurstBuffer` is the debounce, exercised separately in `test_channels.py`.
"""

from __future__ import annotations

import datetime as dt

from app.channels.base import InboundMessage
from app.channels.envelope import BurstBuffer, InboundEnvelope, group_bursts
from evals.schema import GoldenConversation

NOW = dt.datetime(2026, 3, 10, 12, 0, tzinfo=dt.UTC)
LINES = ["hey saw ur reel", "the one about knees", "anyway"]


def _cli_envelope(lines: list[str]) -> InboundEnvelope:
    """What `CLIChannel` yields for one burst: the lines it read before the
    blank line, grouped by the same function the runner uses."""
    messages = [
        InboundMessage(channel="cli", channel_user_id="local", text=line, received_at=NOW)
        for line in lines
    ]
    groups = group_bursts(messages, lambda _m: True)
    return InboundEnvelope.of(groups[0])


def _runner_envelope(lines: list[str]) -> InboundEnvelope:
    """What the eval runner yields for the same three messages, taking the
    structural route through a golden-set document."""
    conv = GoldenConversation.model_validate(
        {
            "id": "equivalence",
            "mode": 1,
            "title": "burst equivalence",
            "provenance": "handwritten",
            "context": {"today": "2026-03-10"},
            "turns": [{"inbound": line} for line in lines] + [{"agent": "[agent responds]"}],
        }
    )
    burst = conv.bursts[0]
    messages = [
        InboundMessage(channel="cli", channel_user_id="local", text=t.inbound, received_at=NOW)
        for t in burst
    ]
    return InboundEnvelope.of(messages)


def test_the_cli_and_the_runner_produce_the_same_envelope():
    cli, runner = _cli_envelope(LINES), _runner_envelope(LINES)
    assert len(cli) == len(runner) == 3
    assert cli.text == runner.text
    assert cli.channel == runner.channel
    assert cli.channel_user_id == runner.channel_user_id


def test_both_paths_agree_that_a_burst_is_one_turn():
    """The property, stated as the thing it protects: a chatty lead gets one
    reply, not three, on either path."""
    assert len(_cli_envelope(LINES).messages) == 3
    assert len(_runner_envelope(LINES).messages) == 3


def test_the_debounce_agrees_with_both():
    """The third path — live channels with a real timer — has to land in the
    same place."""
    buf = BurstBuffer(window_seconds=5)
    for line in LINES:
        buf.add(
            InboundMessage(channel="cli", channel_user_id="local", text=line, received_at=NOW),
            now=NOW,
        )
    assert buf.due(NOW + dt.timedelta(seconds=5))
    debounced = buf.flush()
    assert debounced is not None
    assert debounced.text == _runner_envelope(LINES).text


def test_a_single_message_is_still_a_burst_of_one():
    """The degenerate case both paths must agree on, since most turns are one
    message and a disagreement here would be invisible in aggregate."""
    assert len(_cli_envelope(["just one"])) == 1
    assert _cli_envelope(["just one"]).text == _runner_envelope(["just one"]).text


def test_the_runner_groups_a_real_golden_conversation_the_same_way():
    """Against a committed file rather than a constructed one, so a change to
    the YAML format shows up here."""
    from evals.validate import load_all

    conv = next(c for _p, c in load_all() if c.id.startswith("smoke-01"))
    first = conv.bursts[0]
    assert len(first) == 2, "smoke-01 opens with a two-message burst"

    messages = [
        InboundMessage(channel="cli", channel_user_id="local", text=t.inbound) for t in first
    ]
    assert InboundEnvelope.of(messages).text == "\n".join(t.inbound for t in first)
