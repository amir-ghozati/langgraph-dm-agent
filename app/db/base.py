"""Engine, session factory, and the declarative base."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import Settings, get_settings


def configure_event_loop() -> None:
    """Windows only: psycopg's async mode cannot use the default
    ProactorEventLoop.

    Called from the CLI entrypoints rather than at import time, because a
    library that reconfigures the event loop policy on import is a nasty
    surprise for anything embedding it. The documented path is Docker, where
    this is a no-op — but the CLI channel is the primary dev interface and
    people will run it natively.
    """
    import asyncio
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class Base(DeclarativeBase):
    pass


def build_engine(settings: Settings | None = None):
    settings = settings or get_settings()
    # psycopg3 serves both the sync (Alembic) and async (app) paths from one
    # URL, which is why there is no second driver in the dependency list.
    return create_async_engine(settings.database_url, pool_pre_ping=True, future=True)


def build_sessionmaker(engine=None) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine or build_engine(), expire_on_commit=False)


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession] | None = None,
) -> AsyncIterator[AsyncSession]:
    factory = factory or build_sessionmaker()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
