"""
Repository Pattern for Database Operations

Provides async CRUD operations that mirror the existing Flask db/crud.py functionality.
Uses the repository pattern to separate database operations from business logic.
"""

import logging
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, cast

from sqlalchemy import delete, func, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.utils.datetime import utcnow
from .engine import Base
from .models import User, DataSource

logger = logging.getLogger(__name__)

# Generic type for repository operations
ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """
    Generic repository class for common database operations.
    Provides async equivalents of Flask CRUD operations.
    """

    def __init__(self, session: AsyncSession, model: Type[ModelType]):
        self.session = session
        self.model = model

    async def create(self, obj_in: Dict[str, Any]) -> ModelType:
        """
        Create a new record.

        Args:
            obj_in: Dictionary of field values

        Returns:
            ModelType: Created model instance
        """
        db_obj = self.model(**obj_in)
        self.session.add(db_obj)
        await self.session.commit()
        await self.session.refresh(db_obj)

        logger.debug(f"Created {self.model.__name__} with id {db_obj.id}")
        return db_obj

    async def get(self, id: int) -> Optional[ModelType]:
        """
        Get a record by ID.

        Args:
            id: Record ID

        Returns:
            Optional[ModelType]: Model instance or None
        """
        result = await self.session.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalar_one_or_none()

    async def get_multi(
        self,
        *,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None
    ) -> List[ModelType]:
        """
        Get multiple records with pagination and filtering.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            filters: Dictionary of field filters
            order_by: Field name to order by

        Returns:
            List[ModelType]: List of model instances
        """
        query = select(self.model)

        # Apply filters
        if filters:
            for field, value in filters.items():
                if hasattr(self.model, field):
                    query = query.where(getattr(self.model, field) == value)

        # Apply ordering
        if order_by and hasattr(self.model, order_by):
            query = query.order_by(getattr(self.model, order_by))

        # Apply pagination
        query = query.offset(skip).limit(limit)

        result = await self.session.execute(query)
        records = result.scalars().all()
        return list(records)

    async def update(self, id: int, obj_in: Dict[str, Any]) -> Optional[ModelType]:
        """
        Update a record by ID.

        Args:
            id: Record ID
            obj_in: Dictionary of field values to update

        Returns:
            Optional[ModelType]: Updated model instance or None
        """
        # First check if record exists
        db_obj = await self.get(id)
        if not db_obj:
            return None

        # Update fields
        for field, value in obj_in.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)

        await self.session.commit()
        await self.session.refresh(db_obj)

        logger.debug(f"Updated {self.model.__name__} with id {id}")
        return db_obj

    async def delete(self, id: int) -> bool:
        """
        Delete a record by ID.

        Args:
            id: Record ID

        Returns:
            bool: True if deleted, False if not found
        """
        result = await self.session.execute(
            delete(self.model).where(self.model.id == id)
        )

        cursor = cast(CursorResult[Any], result)
        affected = cursor.rowcount or 0
        if affected > 0:
            await self.session.commit()
            logger.debug(f"Deleted {self.model.__name__} with id {id}")
            return True

        return False

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count records with optional filtering.

        Args:
            filters: Dictionary of field filters

        Returns:
            int: Number of records
        """
        query = select(func.count(self.model.id))

        if filters:
            for field, value in filters.items():
                if hasattr(self.model, field):
                    query = query.where(getattr(self.model, field) == value)

        result = await self.session.execute(query)
        count_value = result.scalar_one()
        return int(count_value)

    async def exists(self, id: int) -> bool:
        """
        Check if a record exists.

        Args:
            id: Record ID

        Returns:
            bool: True if exists, False otherwise
        """
        result = await self.session.execute(
            select(func.count(self.model.id)).where(self.model.id == id)
        )
        count_value = result.scalar_one()
        return int(count_value) > 0


class UserRepository(BaseRepository[User]):
    """Repository for User model operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, User)

    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        result = await self.session.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()

    async def get_active_users(self) -> List[User]:
        """Get all active users."""
        result = await self.session.execute(
            select(User).where(User.is_active.is_(True))
        )
        return list(result.scalars().all())

    async def update_last_login(self, user_id: int) -> bool:
        """Update user's last login timestamp."""
        result = await self.session.execute(
            update(User)
            .where(User.id == user_id)
            .values(
                last_login=utcnow(),
                login_count=User.login_count + 1
            )
        )

        cursor = cast(CursorResult[Any], result)
        affected = cursor.rowcount or 0
        if affected > 0:
            await self.session.commit()
            return True
        return False


class DataSourceRepository(BaseRepository[DataSource]):
    """Repository for DataSource model operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, DataSource)

    async def get_by_name(self, name: str) -> Optional[DataSource]:
        """Get data source by name."""
        result = await self.session.execute(
            select(DataSource).where(DataSource.name == name)
        )
        return result.scalar_one_or_none()

    async def get_by_type(self, source_type: str) -> List[DataSource]:
        """Get data sources by type."""
        result = await self.session.execute(
            select(DataSource).where(DataSource.type == source_type)
        )
        return list(result.scalars().all())

    async def get_active_sources(self) -> List[DataSource]:
        """Get all active data sources."""
        result = await self.session.execute(
            select(DataSource).where(DataSource.is_active.is_(True))
        )
        return list(result.scalars().all())

    async def get_by_file_hash(self, file_hash: str) -> Optional[DataSource]:
        """Fetch a data source whose JSON config stores the specified file hash."""
        if not file_hash:
            return None

        try:
            file_hash_field = DataSource.config["file_hash"].as_string()
            stmt = (
                select(DataSource)
                .options(selectinload(DataSource.creator))
                .where(file_hash_field == file_hash)
            )
            result = await self.session.execute(stmt)
            return result.scalars().first()
        except Exception as exc:
            logger.warning("Failed querying DataSource by file hash: %s", exc)
            return None


# Convenience functions for repository creation
def get_user_repository(session: AsyncSession) -> UserRepository:
    """Get user repository instance."""
    return UserRepository(session)


def get_data_source_repository(session: AsyncSession) -> DataSourceRepository:
    """Get data source repository instance."""
    return DataSourceRepository(session)


# Removed unused repositories: DataIngestionJobRepository, SystemLogRepository
# These were associated with unused tables and caused import errors
