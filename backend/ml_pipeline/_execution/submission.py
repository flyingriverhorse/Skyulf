"""Database reservations covering job lookup, version allocation, and creation."""

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from weakref import WeakKeyDictionary

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, AsyncSession
from sqlalchemy.pool import StaticPool

_memory_locks: WeakKeyDictionary[AsyncEngine, asyncio.Lock] = WeakKeyDictionary()


def _advisory_key(key: tuple[str, str, int]) -> int:
    """Produce a stable signed PostgreSQL lock key without delimiter ambiguity."""
    payload = json.dumps(["skyulf-job-submission", *key], ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=True)


@asynccontextmanager
async def _submission_connection(session: AsyncSession) -> AsyncIterator[AsyncConnection]:
    """Borrow the request's connection, serializing in-memory SQLite reservations."""
    bind = session.bind
    if bind is None:
        raise ValueError("Submission requires a bound database session")
    engine = bind.engine if isinstance(bind, AsyncConnection) else bind
    if engine.dialect.name == "sqlite" and isinstance(engine.pool, StaticPool):
        lock = _memory_locks.setdefault(engine, asyncio.Lock())
        async with lock:
            yield await session.connection()
    else:
        yield await session.connection()


async def _validate_request_transaction(session: AsyncSession, connection: AsyncConnection) -> None:
    """Reject unrelated pending writes without committing or rolling them back."""
    if session.new or session.dirty or session.deleted:
        raise ValueError("Submission requires a request session without pending writes")
    if connection.dialect.name == "sqlite":
        raw = await connection.get_raw_connection()
        driver = raw.driver_connection
        if driver is None:
            raise ValueError("Submission requires an open SQLite connection")
        if driver.in_transaction:
            raise ValueError("Submission requires SQLite reads outside an explicit transaction")


async def _acquire_reservation(connection: AsyncConnection, key: tuple[str, str, int]) -> None:
    """Lock before reading, including when no job row exists yet."""
    if connection.dialect.name == "sqlite":
        # The application enables driver autocommit. Explicit BEGIN also
        # ensures releasing a session SAVEPOINT cannot commit the reservation.
        await connection.exec_driver_sql("BEGIN IMMEDIATE")
    elif connection.dialect.name == "postgresql":
        await connection.execute(
            text("SELECT pg_advisory_xact_lock(:key)"), {"key": _advisory_key(key)}
        )
    else:
        raise ValueError(f"Unsupported submission database: {connection.dialect.name}")


@asynccontextmanager
async def submission_session(
    session: AsyncSession, key: tuple[str, str, int]
) -> AsyncIterator[AsyncSession]:
    """Commit one reservation after all internal manager commits have finished.

    Existing managers commit version allocation and job creation separately.
    Joining them through savepoints keeps those commits, and allocator retry
    rollbacks, inside one connection-owned transaction. Exceptions and task
    cancellation roll it back. Run and retry only read through their request
    session before reserving; SQLite uses driver autocommit for those reads.
    PostgreSQL uses the engine's default READ COMMITTED isolation so the
    duplicate lookup sees jobs committed while waiting for the advisory lock.
    """
    async with _submission_connection(session) as connection:
        await _validate_request_transaction(session, connection)
        try:
            await _acquire_reservation(connection, key)
            async with AsyncSession(
                bind=connection, expire_on_commit=False, join_transaction_mode="create_savepoint"
            ) as reserved:
                yield reserved
            await session.commit()
        except BaseException:
            await session.rollback()
            raise
