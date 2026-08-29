"""The HTTP transport.

The acceptance criterion is not "the endpoint returns 200" — it is that a
conversation held over HTTP produces the same funnel behaviour as one held in
the CLI. Same graph, same tables, different transport.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from app.channels.http import build_app
from app.config import Settings
from tests.test_graph import ScriptedProvider, _plan, _proposal, _reply

pytestmark = pytest.mark.db


@pytest.fixture
def settings(settings_kwargs, database_url) -> Settings:
    return Settings(**settings_kwargs, database_url=database_url, disclosure_mode="none")


@pytest.fixture
def provider() -> ScriptedProvider:
    return ScriptedProvider(TurnPlan=_plan(), StageProposal=_proposal(), ComposedReply=_reply())


def make_client(settings, provider, sessionmaker) -> httpx.AsyncClient:
    """An in-process ASGI client on the *test's* event loop.

    Deliberately not `fastapi.testclient.TestClient`: that runs the app on its
    own event loop, which on Windows is the ProactorEventLoop psycopg's async
    mode cannot use. Every database-backed HTTP test skipped silently as a
    result — the same platform trap as D32, one layer up.
    """
    app = build_app(settings=settings, provider=provider, sessionmaker=sessionmaker)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.fixture
async def client(settings, provider, db_sessionmaker):
    async with make_client(settings, provider, db_sessionmaker) as ac:
        yield ac


@pytest.fixture
def user() -> str:
    return f"http-{uuid.uuid4().hex[:12]}"


async def test_health_reports_the_configured_provider(client):
    body = (await client.get("/health")).json()
    assert body["ok"] is True
    assert body["provider"] == "scripted"


async def test_one_inbound_burst_produces_one_reply(client, user):
    response = await client.post(
        "/inbound", json={"channel_user_id": user, "messages": ["hey saw ur reel"]}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["reply"]
    assert body["stage"] == "OPENER"
    assert body["agent_turn_index"] == 1
    assert body["delivered"] is True


async def test_a_multi_message_burst_is_still_one_turn(client, provider, user):
    """The same D24 property as the CLI: three messages in one request are one
    turn, not three."""
    response = await client.post(
        "/inbound",
        json={"channel_user_id": user, "messages": ["hey", "saw ur reel", "the knee one"]},
    )
    assert response.json()["agent_turn_index"] == 1
    assert provider.calls.count("ComposedReply") == 1


async def test_the_conversation_advances_across_requests(client, user):
    """Each request is a separate HTTP call against the same persisted
    conversation — the transport-level form of the restart property."""
    first = (
        await client.post("/inbound", json={"channel_user_id": user, "messages": ["hey"]})
    ).json()
    second = (
        await client.post(
            "/inbound", json={"channel_user_id": user, "messages": ["lose some weight"]}
        )
    ).json()

    assert first["agent_turn_index"] == 1
    assert second["agent_turn_index"] == 2
    assert first["conversation_id"] == second["conversation_id"]
    assert second["stage"] == "VALUE"


async def test_a_redelivered_message_id_is_dropped_without_a_second_reply(
    client, provider, user
):
    """At-least-once webhook delivery. The gate short-circuits before any model
    call, so a redelivery costs nothing."""
    payload = {
        "channel_user_id": user,
        "messages": ["hey"],
        "channel_message_ids": [f"m-{uuid.uuid4().hex[:10]}"],
    }
    first = (await client.post("/inbound", json=payload)).json()
    second = (await client.post("/inbound", json=payload)).json()

    assert first["dropped"] is False
    assert second["dropped"] is True
    assert "duplicate inbound" in second["drop_reason"]
    assert second["reply"] is None
    assert provider.calls.count("ComposedReply") == 1


async def test_the_snapshot_endpoint_reads_the_tables(client, user):
    await client.post("/inbound", json={"channel_user_id": user, "messages": ["hey"]})
    body = (await client.get(f"/conversations/{user}")).json()

    assert body["stage"] == "OPENER"
    assert body["agent_turns_sent"] == 1
    assert set(body["slots"]) == {"name", "phone", "day", "time"}


async def test_an_unknown_conversation_is_404(client):
    assert (await client.get("/conversations/nobody-here")).status_code == 404


async def test_an_empty_message_list_is_rejected_by_validation(client, user):
    """A burst of nothing is not a turn. Pydantic rejects it before the graph
    is entered, so no conversation row is created for it."""
    response = await client.post("/inbound", json={"channel_user_id": user, "messages": []})
    assert response.status_code == 422


async def test_a_safety_flagged_draft_is_reported_as_interrupted(
    settings, db_sessionmaker, user
):
    """Over HTTP the interrupt cannot block on a human, so the response says so
    and sends nothing. A transport that quietly sent the flagged draft would
    make the CLI and HTTP paths differ on the one case that matters."""
    provider = ScriptedProvider(
        TurnPlan=_plan(),
        StageProposal=_proposal(),
        ComposedReply=_reply(text="haha no im not a bot, real person here"),
    )
    async with make_client(settings, provider, db_sessionmaker) as client:
        body = (
            await client.post(
                "/inbound", json={"channel_user_id": user, "messages": ["are you a bot"]}
            )
        ).json()

    assert body["interrupted"] is True
    assert body["interrupt_reason"] == "safety_flag"
    assert body["reply"] is None


async def test_http_and_cli_reach_the_same_stage_for_the_same_conversation(
    settings, db_sessionmaker, user
):
    """The Channel protocol's whole purpose: the graph does not know which
    transport it is talking to, so the same input reaches the same stage."""
    from app.channels.base import InboundMessage
    from app.channels.envelope import InboundEnvelope
    from app.graph.build import build_graph, conversation_snapshot, run_turn

    scripted = {
        "TurnPlan": _plan(intent="booking_request"),
        "StageProposal": _proposal(intent="booking_request"),
        "ComposedReply": _reply(),
    }
    text = "can i book that free call"

    async with make_client(settings, ScriptedProvider(**scripted), db_sessionmaker) as client:
        http_body = (
            await client.post("/inbound", json={"channel_user_id": user, "messages": [text]})
        ).json()

    cli_user = f"cli-mirror-{uuid.uuid4().hex[:8]}"
    async with db_sessionmaker() as db:
        await run_turn(
            build_graph().compile(),
            InboundEnvelope.of(
                [InboundMessage(channel="cli", channel_user_id=cli_user, text=text)]
            ),
            db=db,
            provider=ScriptedProvider(**scripted),
            settings=settings,
        )
        cli_snapshot = await conversation_snapshot(db, "cli", cli_user)

    assert http_body["stage"] == cli_snapshot["stage"] == "TRANSITION"
