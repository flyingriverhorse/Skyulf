"""Authentication service utilities."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, TYPE_CHECKING, cast

from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.models import User as DBUser, get_database_session

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .auth_core import UserInDB

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _session_scope(session: Optional[AsyncSession] = None):
    """Yield an async session, creating one when missing."""
    if session is not None:
        yield session
        return

    async with get_database_session() as generated_session:
        yield generated_session


def _map_user(db_user: DBUser) -> "UserInDB":
    """Convert a SQLAlchemy ``User`` model to auth ``UserInDB``."""
    from .auth_core import UserInDB, AD_GROUPS

    user_id = cast(Optional[int], getattr(db_user, "id", None))
    username_value = cast(str, getattr(db_user, "username", ""))
    email_value = str(getattr(db_user, "email", ""))
    hashed_password_value = cast(str, getattr(db_user, "password_hash", ""))

    full_name_value = cast(Optional[str], getattr(db_user, "full_name", None))
    display_name = full_name_value or username_value

    ad_groups = [AD_GROUPS["User"]]
    if db_user.is_admin:
        # Ensure admins keep standard user capabilities too
        if AD_GROUPS["Admin"] not in ad_groups:
            ad_groups.append(AD_GROUPS["Admin"])

    created_at_raw = cast(Optional[datetime], getattr(db_user, "created_at", None))
    created_at = created_at_raw or datetime.utcnow()
    last_login_value = cast(Optional[datetime], getattr(db_user, "last_login", None))
    is_active_value = bool(getattr(db_user, "is_active", False))
    is_admin_value = bool(getattr(db_user, "is_admin", False))

    return UserInDB(
        id=user_id,
        username=username_value,
        email=email_value,
        display_name=display_name,
        hashed_password=hashed_password_value,
        ad_groups=ad_groups,
        is_active=is_active_value,
        is_admin=is_admin_value,
        created_date=created_at,
        last_login=last_login_value,
        failed_attempts=0,
        account_locked=False,
        locked_until=None,
    )


async def get_user_by_username(
    username: str,
    session: Optional[AsyncSession] = None,
) -> Optional["UserInDB"]:
    """Fetch a user by username and map to ``UserInDB``."""
    async with _session_scope(session) as scoped_session:
        result = await scoped_session.execute(
            select(DBUser).where(DBUser.username == username)
        )
        db_user = result.scalar_one_or_none()

        if not db_user:
            return None

        return _map_user(db_user)


async def get_user_by_id(
    user_id: int,
    session: Optional[AsyncSession] = None,
) -> Optional["UserInDB"]:
    """Fetch a user by database ID."""
    async with _session_scope(session) as scoped_session:
        result = await scoped_session.execute(
            select(DBUser).where(DBUser.id == user_id)
        )
        db_user = result.scalar_one_or_none()

        if not db_user:
            return None

        return _map_user(db_user)


async def authenticate_credentials(
    username: str,
    password: str,
    session: Optional[AsyncSession] = None,
) -> Optional["UserInDB"]:
    """Validate username/password against the database."""
    from .auth_core import verify_password

    user = await get_user_by_username(username=username, session=session)
    if not user:
        return None

    if not verify_password(password, user.hashed_password):
        return None

    if not user.is_active:
        return None

    # Update login metrics asynchronously, but don't block on failures
    if user.id is not None:
        try:
            await record_successful_login(user_id=user.id, session=session)
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.warning("Failed to update login metadata for %s: %s", username, exc)

    return user


async def record_successful_login(
    user_id: int,
    session: Optional[AsyncSession] = None,
) -> None:
    """Update ``last_login`` and ``login_count`` after successful authentication."""
    async with _session_scope(session) as scoped_session:
        await scoped_session.execute(
            update(DBUser)
            .where(DBUser.id == user_id)
            .values(
                last_login=datetime.utcnow(),
                login_count=DBUser.login_count + 1,
            )
        )


async def record_failed_login(
    user_id: int,
    session: Optional[AsyncSession] = None,
) -> None:
    """Placeholder hook for tracking failed logins on real users."""
    try:
        async with _session_scope(session) as scoped_session:
            await scoped_session.execute(
                update(DBUser)
                .where(DBUser.id == user_id)
                .values(last_login=datetime.utcnow())
            )
    except SQLAlchemyError as exc:  # pragma: no cover - telemetry only
        logger.debug("Failed login audit skip for user %s: %s", user_id, exc)
