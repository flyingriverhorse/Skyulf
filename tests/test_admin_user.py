import sys
from pathlib import Path

import pytest
import pytest_asyncio

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import select

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from core.database.models import Base
    from core.database.repository import get_user_repository
    from core.auth.auth_core import get_password_hash, verify_password
except ImportError as exc:  # pragma: no cover - allow tests without install
    if "passlib" in str(exc).lower():
        pytest.skip("passlib not installed", allow_module_level=True)
    raise


@pytest_asyncio.fixture()
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_maker = async_sessionmaker(engine, expire_on_commit=False)
    async with session_maker() as session:
        yield session

    await engine.dispose()


@pytest.mark.asyncio
async def test_insert_admin_user(db_session):
    username = "test_admin"
    password = "secret123"
    email = "test_admin@example.local"

    repo = get_user_repository(db_session)

    # ensure no leftover
    existing = await repo.get_by_username(username)
    if existing:
        await repo.delete(existing.id)

    admin_data = {
        "username": username,
        "email": email,
        "password_hash": get_password_hash(password),
        "full_name": "Test Administrator",
        "is_active": True,
        "is_admin": True,
        "is_verified": True,
    }

    created = await repo.create(admin_data)

    assert created.username == username
    assert created.email == email
    assert created.is_admin is True
    assert verify_password(password, created.password_hash)

    # verify via query
    users = (await db_session.execute(select(created.__class__))).scalars().all()
    assert any(u.username == username for u in users)

    # cleanup
    await repo.delete(created.id)
