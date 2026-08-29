from __future__ import annotations

import asyncio
import os
import sys

import pytest

# Every setting this project reads. Cleared for the whole test session so a
# developer's real .env or shell cannot change what a test asserts.
_APP_ENV_PREFIXES = (
    "DATABASE_URL",
    "LLM_",
    "GEMINI_",
    "OLLAMA_",
    "SCHEMA_MODE",
    "FUNNEL_",
    "CONVERSATION_",
    "DEFAULT_",
    "BUSINESS_",
    "SESSION_",
    "MIN_LEAD",
    "BOOKING_",
    "ELIGIBILITY_",
    "DISCLOSURE_",
    "REQUIRE_",
    "K_",
    "LOG_",
)


@pytest.fixture(autouse=True, scope="session")
def _isolate_env() -> None:
    for key in list(os.environ):
        if key.startswith(_APP_ENV_PREFIXES):
            del os.environ[key]


@pytest.fixture
def settings_kwargs() -> dict[str, object]:
    """Minimum viable settings. `_env_file=None` stops pydantic-settings
    reading the developer's real .env during a unit test."""
    return {"_env_file": None, "gemini_api_key": "test-key"}


# ---------------------------------------------------------------------------
# Database fixtures. Marked `db`; skipped when no Postgres is reachable.
# ---------------------------------------------------------------------------

POSTGRES_HOST_PORT = os.environ.get("POSTGRES_HOST_PORT", "5432")
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    f"postgresql+psycopg://muster:muster@localhost:{POSTGRES_HOST_PORT}/muster",
)


# ---------------------------------------------------------------------------
# Acceptance mode: a skip is not a pass
# ---------------------------------------------------------------------------


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--no-skips",
        action="store_true",
        default=False,
        help=(
            "Treat any skipped test as a failure. Use for the acceptance run: "
            "`319 passed` and `319 passed, 7 skipped` read identically to "
            "anyone scanning, and the second one is missing an acceptance "
            "criterion."
        ),
    )


@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_runtest_makereport(item, call):
    """A skipped test in acceptance mode is a failure.

    The sixth instance of a recurring shape in this repo: something that reads
    as verification and cannot fail. The database tests skipped silently in the
    container for a whole session — seven green-looking tests that had never
    run, and they happen to be the acceptance criterion for D4.

    Skipping stays available for local convenience. It is forbidden where the
    number gets quoted.
    """
    report = yield
    if report.skipped and item.config.getoption("--no-skips"):
        report.outcome = "failed"
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else report.longrepr
        report.longrepr = (
            f"SKIPPED IN ACCEPTANCE MODE (--no-skips): {reason}\n"
            "A skipped test is not a passing test. Make the resource available "
            "or remove the test."
        )
    return report


def pytest_asyncio_loop_factories(config, item):
    """psycopg's async mode cannot run on Windows' default ProactorEventLoop.

    Without this every database test skips with a connection error on Windows
    and passes on Linux — the worst of both: green locally, green in CI, and
    nobody ever runs the assertions that matter.

    Uses the hook rather than overriding the `event_loop_policy` fixture, which
    pytest-asyncio 1.4 deprecates.

    Once this hook is registered at all it must return a non-empty mapping on
    every platform — returning `None` to mean "no opinion" is a UsageError, so
    the non-Windows branch names the default factory explicitly rather than
    declining to answer. Returning `None` here passed on Windows and broke
    collection on Linux, which is the same platform-asymmetry trap the hook
    exists to close.
    """
    if sys.platform == "win32":
        return {"selector": asyncio.SelectorEventLoop}
    return {"default": asyncio.new_event_loop}


@pytest.fixture(scope="session")
def database_url() -> str:
    return TEST_DATABASE_URL


@pytest.fixture
async def db_engine(database_url):
    """A live engine, or skip. No testcontainers: the compose Postgres is
    already there and one fewer dependency is one fewer thing to explain."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    # connect_timeout, and a wall-clock timeout on the probe itself. Without
    # both, an unreachable-but-not-refusing Postgres (a stopped Docker VM is
    # the usual cause) makes the suite hang instead of skipping — which is
    # worse than either failing or skipping, because nothing reports anything.
    engine = create_async_engine(
        database_url, poolclass=None, connect_args={"connect_timeout": 5}
    )
    try:
        async with asyncio.timeout(10):
            async with engine.connect() as conn:
                await conn.execute(text("select 1"))
    except (Exception, TimeoutError) as exc:  # noqa: BLE001 - any failure is a skip
        await engine.dispose()
        pytest.skip(f"no database at {database_url}: {type(exc).__name__}")
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_sessionmaker(db_engine):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    return async_sessionmaker(db_engine, expire_on_commit=False)


@pytest.fixture
async def fresh_engine(database_url):
    """A session from a *new* engine and pool, so a reload cannot be served by
    identity map or connection state left over from the writer. Without this,
    "it survived a restart" would only prove SQLAlchemy caches well."""
    from contextlib import asynccontextmanager

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engines = []

    @asynccontextmanager
    async def _make():
        engine = create_async_engine(database_url)
        engines.append(engine)
        maker = async_sessionmaker(engine, expire_on_commit=False)
        async with maker() as session:
            yield session

    yield _make
    for engine in engines:
        await engine.dispose()
