"""Async data ingestion service for FastAPI."""

import asyncio
import hashlib
import json
import logging
import re
import unicodedata
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set, cast, Union

import aiofiles
import pandas as pd
from fastapi import UploadFile
from sqlalchemy import select, text, func, String
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.database.models import User, DataSource
from core.utils.file_utils import (
    safe_delete_path, extract_file_path_from_source,
    cleanup_empty_directories
)
from core.utils.logging_utils import log_data_action
from config import get_settings
from .models import DataSourceCreate, DataSourceUpdate, SourceType
from .exceptions import (
    DataIngestionException,
    DataSourceNotFoundError,
    FileUploadError,
    DataProcessingError,
)

# Import new migrated components
from .serialization import AsyncJSONSafeSerializer, DataTypeConverter

logger = logging.getLogger(__name__)


class DataIngestionService:
    """Async data ingestion service with migrated core components."""

    def __init__(self, session: AsyncSession, upload_dir: str = "uploads/data"):
        self.session = session
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        self.settings = get_settings()
        self.allowed_extensions: Set[str] = {
            ext.lower() for ext in getattr(self.settings, "ALLOWED_EXTENSIONS", [])
        }
        self.csv_like_extensions: Set[str] = {".csv", ".tsv", ".txt"}

        # Initialize migrated components
        self.serializer = AsyncJSONSafeSerializer()
        self.type_converter = DataTypeConverter()

        logger.info("DataIngestionService initialized with core components")

    async def _count_user_data_sources(self, user_id: int) -> int:
        """Return how many data sources the given user has created."""
        try:
            result = await self.session.execute(
                select(func.count(DataSource.id)).where(DataSource.created_by == user_id)
            )
            count = result.scalar_one_or_none()
            return int(count or 0)
        except Exception as exc:
            logger.error("Failed counting data sources for user %s: %s", user_id, exc)
            return 0

    async def _get_user_id_from_username(self, username: str) -> Optional[int]:
        """Get user ID from username."""
        try:
            result = await self.session.execute(
                select(User.id).where(User.username == username)
            )
            user_id = result.scalar_one_or_none()
            return user_id
        except Exception as e:
            logger.error(f"Error looking up user {username}: {e}")
            return None

    async def create_data_source(
        self,
        data: DataSourceCreate,
        created_by: str
    ) -> DataSource:
        """Create a new data source."""
        try:
            # Get user ID from username
            user_id = (
                await self._get_user_id_from_username(created_by)
                if created_by
                else None
            )

            # Check user data source limit
            settings = get_settings()
            if user_id and settings.USER_MAX_DATA_SOURCES > 0:
                user_source_count = await self._count_user_data_sources(user_id)
                if user_source_count >= settings.USER_MAX_DATA_SOURCES:
                    raise ValueError(
                        "User has reached the maximum limit of "
                        f"{settings.USER_MAX_DATA_SOURCES} data sources"
                    )

            # Generate unique source ID
            source_type_enum = (
                data.source_type
                if hasattr(data.source_type, "value")
                else SourceType.FILE
            )
            unique_source_id = await self._generate_source_id()  # Use the simpler version

            # Create new data source using DataSource model
            new_source = DataSource(
                name=data.name,
                source_id=unique_source_id,  # Set the string source_id
                type=(
                    source_type_enum.value
                    if hasattr(source_type_enum, "value")
                    else str(source_type_enum)
                ),
                config=data.connection_info or {},
                source_metadata=(
                    data.metadata if hasattr(data, "metadata") else None
                ),  # Set source metadata if available
                description=f"Data source: {data.name}",
                is_active=True,
                test_status="untested",
                created_by=user_id
            )

            # Add to session and commit
            self.session.add(new_source)
            await self.session.commit()
            await self.session.refresh(new_source)

            logger.info(
                "Created data source: %s - %s by %s",
                new_source.id,
                new_source.name,
                created_by,
            )

            return new_source

        except Exception as e:
            logger.error(f"Error creating data source: {e}")
            await self.session.rollback()
            if isinstance(e, DataIngestionException):
                raise
            raise DataIngestionException(f"Failed to create data source: {str(e)}")

    async def get_data_source(self, source_id: int) -> DataSource:
        """Get a data source by ID."""
        try:
            stmt = select(DataSource).where(DataSource.id == source_id)
            result = await self.session.execute(stmt)
            row = result.scalar_one_or_none()

            if not row:
                raise DataSourceNotFoundError(str(source_id))

            return row

        except DataSourceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error getting data source {source_id}: {e}")
            raise DataIngestionException(f"Failed to get data source: {str(e)}")

    async def get_data_source_by_id(self, source_id: int) -> Optional[DataSource]:
        """Get a data source by ID, returning None if not found."""
        try:
            return await self.get_data_source(source_id)
        except DataSourceNotFoundError:
            return None

    async def get_data_source_by_source_id(self, source_id: str) -> Optional[DataSource]:
        """Get a data source by source_id (string identifier), returning None if not found."""
        try:
            logger.info(f"Looking for data source with source_id: '{source_id}'")
            stmt = select(DataSource).where(DataSource.source_id == source_id)
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting data source by source_id {source_id}: {e}")
            return None

    async def list_data_sources(
        self,
        created_by: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[DataSource]:
        """List data sources with optional filtering by creator."""
        try:
            # Use selectinload to eager load the creator relationship
            stmt = select(DataSource).options(selectinload(DataSource.creator))

            if created_by:
                # Filter by username - need to join with User table
                user_id = await self._get_user_id_from_username(created_by)
                if user_id:
                    stmt = stmt.where(DataSource.created_by == user_id)
                else:
                    # If user not found, return empty list
                    return []

            stmt = stmt.limit(limit).offset(offset)
            result = await self.session.execute(stmt)
            sources = list(result.scalars().all())

            return sources

        except Exception as e:
            logger.error(f"Error listing data sources: {e}")
            raise DataIngestionException(f"Failed to list data sources: {str(e)}")

    async def search_data_sources(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        source_type: Optional[str] = None,
        limit: int = 50,
        order_by: str = "created_desc"
    ) -> List[DataSource]:
        """
        Search data sources with filtering and ordering.

        Args:
            query: Search query to match against name, description, and metadata
            category: Filter by category (file, database, api)
            source_type: Filter by specific source type
            limit: Maximum number of results
            order_by: Order by field (created_desc, created_asc, name_asc, name_desc)

        Returns:
            List of matching DataSource objects
        """
        try:
            from sqlalchemy import func, or_

            logger.info(
                "Starting search with query='%s', category='%s', source_type='%s'",
                query,
                category,
                source_type,
            )

            # Base query with eager loading of creator relationship
            stmt = select(DataSource).options(selectinload(DataSource.creator))

            # Text search across multiple fields
            if query and query.strip():
                search_term = f"%{query.strip()}%"
                logger.debug(f"Adding text search for term: {search_term}")
                stmt = stmt.where(
                    or_(
                        DataSource.name.ilike(search_term),
                        DataSource.description.ilike(search_term),
                        func.cast(DataSource.config, String).ilike(search_term),
                        func.cast(DataSource.source_metadata, String).ilike(search_term)
                    )
                )

            # Get the type column using table metadata to avoid name collision
            type_col = DataSource.__table__.c.type
            logger.debug(f"Using type column: {type_col}")

            # Category filter (map categories to source types) - use simple string matching
            if category:
                logger.debug(f"Adding category filter: {category}")
                if category.lower() == "file":
                    file_types = "','".join(['csv', 'excel', 'json', 'txt', 'parquet', 'xml', 'xlsx', 'xls', 'doc'])
                    stmt = stmt.where(text(f"type IN ('{file_types}')"))
                elif category.lower() == "database":
                    db_types = "','".join(['postgres', 'mysql', 'sqlite', 'postgresql'])
                    stmt = stmt.where(text(f"type IN ('{db_types}')"))
                elif category.lower() == "api":
                    api_types = "','".join(['api', 'rest', 'graphql', 'http'])
                    stmt = stmt.where(text(f"type IN ('{api_types}')"))

            # Source type filter - use simple string comparison
            if source_type:
                logger.debug(f"Adding source type filter: {source_type}")
                stmt = stmt.where(text(f"type = '{source_type.lower()}'"))

            # Ordering
            if order_by == "created_desc":
                stmt = stmt.order_by(DataSource.created_at.desc())
            elif order_by == "created_asc":
                stmt = stmt.order_by(DataSource.created_at.asc())
            elif order_by == "name_asc":
                stmt = stmt.order_by(DataSource.name.asc())
            elif order_by == "name_desc":
                stmt = stmt.order_by(DataSource.name.desc())
            else:
                # Default fallback
                stmt = stmt.order_by(DataSource.created_at.desc())

            # Apply limit
            stmt = stmt.limit(limit)

            logger.debug(f"Executing query: {stmt}")
            result = await self.session.execute(stmt)
            sources = list(result.scalars().all())

            logger.info(
                "Search found %s data sources (query='%s', category='%s', source_type='%s')",
                len(sources),
                query,
                category,
                source_type,
            )
            return sources

        except Exception as e:
            logger.error(f"Error searching data sources: {e}")
            logger.error(f"Exception type: {type(e)}")
            logger.error(f"Exception args: {e.args}")

            # If there's still an issue with type column, fall back to a simple query
            try:
                logger.warning("Falling back to simple query without type filtering")
                fallback_stmt = select(DataSource).options(selectinload(DataSource.creator))

                # Only add text search if provided
                if query and query.strip():
                    search_term = f"%{query.strip()}%"
                    fallback_stmt = fallback_stmt.where(DataSource.name.ilike(search_term))

                fallback_stmt = fallback_stmt.order_by(DataSource.created_at.desc()).limit(limit)
                result = await self.session.execute(fallback_stmt)
                sources = list(result.scalars().all())

                logger.info(f"Fallback query returned {len(sources)} sources")
                return sources

            except Exception as fallback_error:
                logger.error(f"Even fallback query failed: {fallback_error}")
                raise DataIngestionException(f"Failed to search data sources: {str(e)}")

    async def search_data_sources_with_formatting(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        source_type: Optional[str] = None,
        limit: int = 50,
        order_by: str = "created_desc",
        current_user=None,
        include_user_scoping: bool = True,
    ) -> Dict[str, Any]:
        """
        Enhanced search method that includes formatting, permissions, and complete response structure.

        This method encapsulates all the business logic that was previously in the route handler,
        including permission filtering, type display mapping, and response formatting.
        """
        try:
            # Get raw search results
            db_sources = await self.search_data_sources(
                query=query,
                category=category,
                source_type=source_type,
                limit=limit,
                order_by=order_by
            )

            # Permission filtering
            filtered_sources = db_sources
            is_admin = False
            scope = "admin_full"

            if current_user and include_user_scoping:
                is_admin = current_user.has_permission("admin")
                scope = "admin_full" if is_admin else "user_scoped"

                if not is_admin:
                    # Filter to only show sources created by current user
                    user_sources = []
                    for source in db_sources:
                        if hasattr(source, 'creator') and source.creator:
                            if source.creator.username == current_user.username:
                                user_sources.append(source)
                        elif hasattr(source, 'created_by') and source.created_by:
                            # Fallback: check created_by field if creator relationship not loaded
                            user_id = await self._get_user_id_from_username(current_user.username)
                            if source.created_by == user_id:
                                user_sources.append(source)
                    filtered_sources = user_sources

            # Convert database sources to API format with enhanced response structure
            sources = []
            type_display_map = {
                "csv": "CSV File",
                "json": "JSON File",
                "xlsx": "Excel File",
                "postgresql": "PostgreSQL Database",
                "mysql": "MySQL Database",
                "sqlite": "SQLite Database"
            }

            for source in filtered_sources:
                source_dict = source.to_dict()
                source_type_raw = source_dict.get("type", "unknown")
                display_type = type_display_map.get(source_type_raw, source_type_raw.upper())

                sources.append({
                    "id": source_dict.get("id"),
                    "source_id": source_dict.get("source_id") or str(source_dict.get("id", "")),
                    "name": source_dict.get("name", "Unnamed Source"),
                    "display_name": source_dict.get("name", "Unnamed Source"),
                    "type": source_type_raw,
                    "source_type": display_type,
                    "category": (
                        "Database"
                        if source_type_raw in ["postgresql", "mysql", "sqlite"]
                        else "File"
                    ),
                    "status": source_dict.get("test_status", "unknown"),
                    "is_active": source_dict.get("is_active", False),
                    "created_at": source_dict.get("created_at"),
                    "updated_at": source_dict.get("updated_at"),
                    "last_tested": source_dict.get("last_tested"),
                    "description": source_dict.get("description", ""),
                    "metadata": {
                        "created_by": source_dict.get("created_by", "Unknown"),
                        "connector_type": source_dict.get("type", "unknown"),
                        "test_status": source_dict.get("test_status", "unknown")
                    }
                })

            return {
                "success": True,
                "results": sources,
                "sources": sources,  # Keep both for compatibility
                "total": len(sources),
                "count": len(sources),
                "query": query or "",
                "filters": {"category": category, "source_type": source_type},
                "order": order_by,
                "limit": limit,
                "scope": scope,
                "message": (
                    f"Found {len(sources)} matching data source(s)"
                    if sources
                    else "No data sources found matching your search criteria."
                ),
            }

        except Exception as e:
            logger.error(f"Error in enhanced search data sources: {e}")
            return {
                "success": False,
                "results": [],
                "sources": [],
                "total": 0,
                "count": 0,
                "error": str(e),
                "message": "Error searching data sources"
            }

    async def get_data_ingestion_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for admin dashboard."""
        try:
            from sqlalchemy import func, case

            # Get basic counts
            result = await self.session.execute(
                select(
                    func.count(DataSource.id).label('total_sources'),
                    func.count(case((DataSource.is_active.is_(True), 1))).label('active_sources'),
                    func.count(case((DataSource.test_status == 'success', 1))).label('tested_sources'),
                    func.count(case((DataSource.test_status == 'failed', 1))).label('failed_sources')
                )
            )
            stats_row = result.one_or_none()

            total_sources = int(stats_row.total_sources or 0) if stats_row else 0
            active_sources = int(stats_row.active_sources or 0) if stats_row else 0
            tested_sources = int(stats_row.tested_sources or 0) if stats_row else 0
            failed_sources = int(stats_row.failed_sources or 0) if stats_row else 0

            # Get source type distribution using table metadata to avoid column name issues
            type_col = DataSource.__table__.c.type.label('source_type')
            type_result = await self.session.execute(
                select(
                    type_col,
                    func.count(DataSource.id).label('type_count')
                ).group_by(type_col)
            )
            sources_by_type: Dict[str, int] = {}
            for row in type_result:
                row_mapping = row._mapping
                source_type_value = cast(Optional[str], row_mapping.get('source_type'))
                if not source_type_value:
                    continue
                type_count_value = row_mapping.get('type_count')
                sources_by_type[source_type_value] = int(type_count_value or 0)

            # Create category distribution (map types to categories)
            sources_by_category: Dict[str, int] = {}
            for source_type, count in sources_by_type.items():
                if source_type in ['csv', 'excel', 'json', 'txt', 'parquet', 'xml', 'xlsx', 'xls', 'doc']:
                    category = 'file'
                elif source_type in ['postgres', 'mysql', 'sqlite', 'postgresql']:
                    category = 'database'
                elif source_type in ['api', 'rest', 'graphql', 'http']:
                    category = 'api'
                else:
                    category = source_type

                if category in sources_by_category:
                    sources_by_category[category] += count
                else:
                    sources_by_category[category] = count

            # Get recent data source
            recent_result = await self.session.execute(
                select(DataSource.created_at)
                .order_by(DataSource.created_at.desc())
                .limit(1)
            )
            latest_created = recent_result.scalar_one_or_none()

            # Calculate actual data volume and records from configs and metadata
            all_sources = await self.session.execute(
                select(DataSource.config, type_col, DataSource.source_metadata)
            )

            total_size_bytes = 0
            total_records = 0

            for source_row in all_sources:
                row_mapping = source_row._mapping
                config = cast(Dict[str, Any], row_mapping.get('config') or {})
                metadata = cast(Dict[str, Any], row_mapping.get('source_metadata') or {})
                source_type = cast(Optional[str], row_mapping.get('source_type')) or 'unknown'

                # Extract size and row information from config (old format) or metadata (new format)
                file_size = config.get('file_size_bytes', 0) or metadata.get('file_size', 0)
                estimated_rows = config.get('estimated_rows', 0)

                # For new format, check quality_metrics in metadata
                if estimated_rows == 0 and metadata.get('quality_metrics'):
                    quality_metrics = metadata['quality_metrics']
                    estimated_rows = quality_metrics.get('estimated_rows', 0)

                # If no row count in config/metadata, estimate based on type and size
                if estimated_rows == 0 and file_size > 0:
                    if source_type == 'csv':
                        # Better estimate: typical CSV row is ~100 bytes
                        estimated_rows = max(1, file_size // 100)
                    elif source_type in ['excel', 'json']:
                        # Excel/JSON tend to be more verbose
                        estimated_rows = max(1, file_size // 150)
                    elif source_type in ['txt']:
                        # Text files: estimate lines
                        estimated_rows = max(1, file_size // 50)

                # If still no estimate, use minimal default for active sources
                if estimated_rows == 0 and config.get('is_active', True):
                    estimated_rows = 50  # More conservative default

                total_size_bytes += file_size
                total_records += estimated_rows

            # Format data size
            if total_size_bytes == 0:
                data_volume = "0 MB"
                total_size_formatted = "0 B"
            elif total_size_bytes < 1024:
                data_volume = f"{total_size_bytes} B"
                total_size_formatted = f"{total_size_bytes} B"
            elif total_size_bytes < 1024 * 1024:
                kb = total_size_bytes / 1024
                data_volume = f"{kb:.1f} KB"
                total_size_formatted = f"{kb:.1f} KB"
            elif total_size_bytes < 1024 * 1024 * 1024:
                mb = total_size_bytes / (1024 * 1024)
                data_volume = f"{mb:.1f} MB"
                total_size_formatted = f"{mb:.1f} MB"
            else:
                gb = total_size_bytes / (1024 * 1024 * 1024)
                data_volume = f"{gb:.1f} GB"
                total_size_formatted = f"{gb:.1f} GB"

            return {
                "total_sources": total_sources,
                "active_sources": active_sources,
                "total_ingestions": tested_sources,
                "successful_ingestions": tested_sources,
                "failed_ingestions": failed_sources,
                "last_ingestion": latest_created.isoformat() if latest_created else None,
                "data_volume": data_volume,
                "total_size_formatted": total_size_formatted,
                "total_records": total_records,
                "total_rows": total_records,  # Add this for frontend compatibility
                "sources_by_type": sources_by_type,
                "sources_by_category": sources_by_category,  # Now properly calculated
                "message": (
                    "Statistics calculated from current data sources"
                    if total_sources > 0
                    else (
                        "No data sources configured yet. Add your first data source to see "
                        "statistics and charts here."
                    )
                )
            }

        except Exception as e:
            logger.error(f"Error calculating data ingestion stats: {e}")
            # Return safe default stats on error
            return {
                "total_sources": 0,
                "active_sources": 0,
                "total_ingestions": 0,
                "successful_ingestions": 0,
                "failed_ingestions": 0,
                "last_ingestion": None,
                "data_volume": "0 MB",
                "total_size_formatted": "0 B",
                "total_records": 0,
                "total_rows": 0,
                "sources_by_type": {},
                "sources_by_category": {},
                "message": "Error calculating statistics. Please check the logs."
            }

    async def get_data_sample(self, source_id: Union[int, str], limit: int = 5) -> List[Dict[str, Any]]:
        """Get a sample of data from the source."""
        # Resolve source_id to int if possible, or use string lookup
        try:
            source_id_int = int(source_id)
            source = await self.get_data_source(source_id_int)
        except (ValueError, DataSourceNotFoundError):
            # Try string lookup if int lookup fails or if source_id is not an int
            source = await self.get_data_source_by_source_id(str(source_id))
            if not source:
                raise DataSourceNotFoundError(str(source_id))

        # Extract file path
        source_dict = source.to_dict()
        file_path = extract_file_path_from_source(source_dict)
        
        if not file_path or not file_path.exists():
            raise DataProcessingError(f"File not found for source {source_id}")

        # Load sample
        data, _, _, _ = await self._load_file_data(str(file_path), limit, 0)
        return data

    async def update_data_source(
        self,
        source_id: int,  # Changed to int to match DataSource model
        data: DataSourceUpdate,
        username: str
    ) -> DataSource:
        """Update a data source."""
        try:
            # Get existing source
            existing = await self.get_data_source(source_id)

            # Update fields
            if data.name is not None:
                setattr(existing, 'name', data.name)

            if hasattr(data, 'description') and data.description is not None:
                setattr(existing, 'description', data.description)

            # Update config if provided
            if hasattr(data, 'config') and data.config is not None:
                setattr(existing, 'config', data.config)

            await self.session.commit()
            await self.session.refresh(existing)

            logger.info(f"Updated data source: {source_id} by {username}")

            return existing

        except DataSourceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error updating data source {source_id}: {e}")
            await self.session.rollback()
            raise DataIngestionException(f"Failed to update data source: {str(e)}")

    async def delete_data_source(self, source_id: str, username: str, is_admin: bool = False) -> bool:
        """Delete a data source by source_id (string identifier)."""
        settings = get_settings()

        try:
            # Get existing source
            existing = await self.get_data_source_by_source_id(source_id)
            if existing is None:
                raise DataSourceNotFoundError(source_id)

            # Check permissions - users can only delete their own sources, admins can delete any
            if not is_admin:
                user_id = await self._get_user_id_from_username(username)
                if existing.created_by != user_id:
                    logger.warning(
                        "User %s tried to delete source %s owned by %s",
                        username,
                        source_id,
                        existing.created_by,
                    )
                    return False

            # Extract file path before deleting from database
            file_deleted = False
            if settings.DELETE_FILES_ON_SOURCE_REMOVAL:
                try:
                    # Convert SQLAlchemy model to dict for the utility function
                    # Note: DataSource model uses 'config' field, not 'connection_info'
                    source_dict = {
                        'file_path': getattr(existing, 'file_path', None),
                        'path': getattr(existing, 'path', None),
                        'source_path': getattr(existing, 'source_path', None),
                        'location': getattr(existing, 'location', None),
                        'file_location': getattr(existing, 'file_location', None),
                        'source_name': existing.name,
                        'connection_info': getattr(existing, 'config', {}),  # Use 'config' field from DataSource model
                    }

                    file_path = extract_file_path_from_source(source_dict)
                    if file_path:
                        file_deleted = safe_delete_path(
                            file_path,
                            force_delete=True,
                            files_only=settings.FILES_ONLY_DELETION
                        )

                        if file_deleted:
                            logger.info("Deleted file/folder: %s", file_path)
                            # Clean up empty parent directories
                            try:
                                cleanup_empty_directories(file_path.parent)
                            except Exception as cleanup_error:
                                logger.warning(
                                    "Failed to cleanup empty directories: %s",
                                    cleanup_error,
                                )
                        else:
                            logger.warning("Failed to delete file/folder: %s", file_path)
                    else:
                        logger.info("No file path found for data source %s", source_id)

                except Exception as file_error:
                    logger.error(
                        "Error deleting file for source %s: %s",
                        source_id,
                        file_error,
                    )
                    # Continue with database deletion even if file deletion fails

            # Delete from database
            await self.session.delete(existing)
            await self.session.commit()

            success_msg = f"Deleted data source: {source_id} by {username}"
            if settings.DELETE_FILES_ON_SOURCE_REMOVAL:
                success_msg += f" [File deletion: {'✓' if file_deleted else '✗'}]"
            logger.info(success_msg)

            return True

        except DataSourceNotFoundError:
            logger.warning("Attempted to delete non-existent source: %s", source_id)
            return False
        except Exception as e:
            logger.error("Error deleting data source %s: %s", source_id, e)
            await self.session.rollback()
            raise DataIngestionException(f"Failed to delete data source: {str(e)}")

    async def delete_data_source_by_id(self, source_id: int) -> bool:
        """Delete a data source by integer ID (simplified version for admin/API use)."""
        settings = get_settings()

        try:
            # Get existing source directly with same session to ensure it's attached
            from sqlalchemy import select
            stmt = select(DataSource).where(DataSource.id == source_id)
            result = await self.session.execute(stmt)
            source = result.scalar_one_or_none()

            if not source:
                logger.warning("Data source with ID %s not found", source_id)
                return False

            source_name = source.name
            logger.info(
                "Attempting to delete data source: %s (ID: %s)",
                source_name,
                source_id,
            )

            # DEBUG: Log the source data
            logger.info("DEBUG - Source config: %s", source.config)

            # Extract file path before deleting from database
            file_deleted = False
            if settings.DELETE_FILES_ON_SOURCE_REMOVAL:
                logger.info("DEBUG - File deletion is ENABLED")
                try:
                    # Convert SQLAlchemy model to dict for the utility function
                    # Note: DataSource model uses 'config' field, not 'connection_info'
                    source_dict = {
                        'file_path': getattr(source, 'file_path', None),
                        'path': getattr(source, 'path', None),
                        'source_path': getattr(source, 'source_path', None),
                        'location': getattr(source, 'location', None),
                        'file_location': getattr(source, 'file_location', None),
                        'source_name': source.name,
                        'connection_info': getattr(source, 'config', {}),  # Use 'config' field from DataSource model
                    }

                    logger.info("DEBUG - Source dict: %s", source_dict)

                    file_path = extract_file_path_from_source(source_dict)
                    logger.info("🎯 EXTRACTED FILE PATH FOR DELETION: %s", file_path)

                    if file_path:
                        logger.info("📁 PATH ANALYSIS:")
                        logger.info("   - Exists: %s", file_path.exists())
                        logger.info("   - Is file: %s", file_path.is_file())
                        logger.info("   - Is directory: %s", file_path.is_dir())
                        logger.info("   - Parent directory: %s", file_path.parent)

                        # EXTRA SAFETY CHECK: Only proceed if it's actually a file
                        if not file_path.is_file():
                            logger.warning(
                                "❌ PATH IS NOT A FILE, SKIPPING DELETION: %s",
                                file_path,
                            )
                            path_type = "directory" if file_path.is_dir() else "other/unknown"
                            logger.warning("   - Path type: %s", path_type)
                            file_deleted = False
                        else:
                            logger.info("✅ PATH IS A FILE, PROCEEDING WITH DELETION")

                            logger.info(
                                "🗑️  CALLING safe_delete_path with files_only=%s",
                                settings.FILES_ONLY_DELETION,
                            )
                            file_deleted = safe_delete_path(
                                file_path,
                                force_delete=True,
                                files_only=settings.FILES_ONLY_DELETION  # Use config setting
                            )
                            logger.info("🔄 DELETION RESULT: %s", file_deleted)

                        if file_deleted:
                            logger.info("SUCCESS - Deleted file/folder: %s", file_path)
                            # Clean up empty parent directories
                            try:
                                cleanup_empty_directories(file_path.parent)
                            except Exception as cleanup_error:
                                logger.warning(
                                    "Failed to cleanup empty directories: %s",
                                    cleanup_error,
                                )
                        else:
                            logger.error("FAILED - Could not delete file/folder: %s", file_path)
                    else:
                        logger.warning(
                            "DEBUG - No file path found for data source %s (ID: %s)",
                            source_name,
                            source_id,
                        )

                except Exception as file_error:
                    logger.error(
                        "ERROR - Exception deleting file for source %s: %s",
                        source_id,
                        file_error,
                    )
                    logger.error("ERROR - Traceback: %s", traceback.format_exc())
                    # Continue with database deletion even if file deletion fails
            else:
                logger.info("DEBUG - File deletion is DISABLED")

            # Delete from database using SQLAlchemy async session
            await self.session.delete(source)  # Use await for async delete
            await self.session.commit()  # This commits the transaction

            success_msg = f"Successfully deleted data source: {source_name} (ID: {source_id})"
            if settings.DELETE_FILES_ON_SOURCE_REMOVAL:
                success_msg += f" [File deletion: {'✓' if file_deleted else '✗'}]"
            logger.info(success_msg)

            return True

        except Exception as e:
            logger.error("Error deleting data source ID %s: %s", source_id, e)
            logger.error("Full traceback: %s", traceback.format_exc())
        await self.session.rollback()  # Rollback on error
        return False  # Return False instead of raising exception

    # File operations
    async def _load_file_data(
        self,
        file_path: str,
        limit: int,
        offset: int
    ) -> Tuple[List[Dict], List[str], int, Dict[str, str]]:
        """Load data from file with pagination."""
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.csv':
            # For CSV, we need to count total rows first
            df_count = pd.read_csv(file_path, usecols=[0])
            total_rows = len(df_count)

            # Load requested slice
            df = pd.read_csv(file_path, skiprows=range(1, offset + 1), nrows=limit)
        elif file_ext in ['.xlsx', '.xls']:
            df_full = pd.read_excel(file_path)
            total_rows = len(df_full)
            df = df_full.iloc[offset:offset + limit]
        elif file_ext == '.json':
            df_full = pd.read_json(file_path, lines=True)
            total_rows = len(df_full)
            df = df_full.iloc[offset:offset + limit]
        else:
            raise DataProcessingError(f"Unsupported file type: {file_ext}")

        # Clean data for JSON serialization
        df_clean = df.fillna("")
        data = df_clean.to_dict('records')
        columns = list(df.columns)
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

        return data, columns, total_rows, dtypes

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

    def _utc_now_iso(self) -> str:
        """Return a UTC ISO8601 timestamp with Z suffix."""
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    async def _generate_source_id(self) -> str:
        """Generate a unique 8-character source ID."""
        import uuid
        return str(uuid.uuid4())[:8]

    async def _persist_upload_file(
        self,
        upload: UploadFile,
        destination: Path,
        extension: str
    ) -> Tuple[int, str, int]:
        """Stream an UploadFile to disk enforcing size limits and returning metadata."""
        chunk_size = max(524288, int(getattr(self.settings, "DATA_CHUNK_SIZE", 0) or 524288))
        total_bytes = 0
        newline_count = 0
        digest = hashlib.sha256()

        await upload.seek(0)

        try:
            async with aiofiles.open(destination, "wb") as buffer:
                while True:
                    chunk = await upload.read(chunk_size)
                    if not chunk:
                        break

                    total_bytes += len(chunk)
                    max_allowed = getattr(self.settings, "MAX_UPLOAD_SIZE", None)
                    if max_allowed and total_bytes > max_allowed:
                        raise FileUploadError(
                            f"File exceeds maximum allowed size of {self._format_file_size(max_allowed)}"
                        )

                    await buffer.write(chunk)
                    digest.update(chunk)

                    if extension in self.csv_like_extensions:
                        newline_count += chunk.count(b"\n")

            return total_bytes, digest.hexdigest(), newline_count

        except Exception:
            if destination.exists():
                safe_delete_path(destination, files_only=True)
            raise
        finally:
            try:
                await upload.close()
            except Exception:
                pass

    def _sanitize_filename(self, filename: str) -> str:
        """Generate a filesystem-safe filename while preserving extension."""
        if not filename:
            return "upload"

        filename = unicodedata.normalize("NFKD", filename)
        filename = filename.replace("\u0000", "")

        parts = filename.replace("\\", "/").split("/")
        basename = parts[-1] if parts else filename
        path_obj = Path(basename)

        stem = path_obj.stem
        suffix = path_obj.suffix.lower()

        cleaned_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem)
        cleaned_stem = cleaned_stem.strip("._-") or "upload"

        sanitized = cleaned_stem[:120]
        if suffix:
            valid_suffix = re.sub(r"[^A-Za-z0-9.]+", "", suffix)
            sanitized = f"{sanitized}{valid_suffix}"

        return sanitized or "upload"

    def _validate_extension(self, extension: str) -> None:
        """Ensure the file extension is allowed by configuration."""
        normalized = extension.lower()
        allowed = self.allowed_extensions

        if allowed and normalized not in allowed and normalized.lstrip('.') not in allowed:
            raise FileUploadError(f"File type '{extension}' is not permitted")

    async def _estimate_rows_from_file(
        self,
        file_path: Path,
        extension: str,
        newline_count: int,
        file_size_bytes: int
    ) -> int:
        """Estimate the number of rows for the uploaded file."""
        normalized = extension.lstrip('.')

        if extension in self.csv_like_extensions:
            rows = newline_count
            if rows == 0 and file_size_bytes > 0:
                rows = 1
            return max(0, rows - 1 if rows > 0 else 0)

        if normalized in {"json"}:
            return await self._estimate_json_rows(file_path, file_size_bytes)

        if normalized in {"xlsx", "xls"}:
            return 100

        if normalized in {"parquet"}:
            return 1000

        return 0

    async def _estimate_json_rows(self, file_path: Path, file_size_bytes: int) -> int:
        """Best-effort estimation of rows in a JSON file."""
        if file_size_bytes > 2 * 1024 * 1024:  # Skip large files to avoid blocking
            return 0

        def _load() -> int:
            try:
                with file_path.open("r", encoding="utf-8", errors="ignore") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    return len(data)
                if isinstance(data, dict):
                    for value in data.values():
                        if isinstance(value, list):
                            return len(value)
                    return len(data)
            except Exception:
                return 0
            return 0

        return await asyncio.to_thread(_load)

    async def _find_existing_source_by_hash(self, file_hash: str) -> Optional[DataSource]:
        """Check for an existing data source with the same file hash."""
        if not file_hash:
            return None

        try:
            file_hash_field = DataSource.config["file_hash"].as_string()
            stmt = (
                select(DataSource)
                .options(selectinload(DataSource.creator))
                .where(file_hash_field == file_hash)
                .limit(1)
            )
            result = await self.session.execute(stmt)
            existing = result.scalars().first()
            if existing:
                return existing
        except Exception as exc:
            logger.warning("Duplicate lookup query failed; falling back to scan: %s", exc)

        # Fallback path for databases without JSON operator support
        try:
            stmt = select(DataSource).options(selectinload(DataSource.creator))
            result = await self.session.execute(stmt)
            for source in result.scalars().all():
                config = cast(Dict[str, Any], source.config or {})
                if config.get("file_hash") == file_hash:
                    return source
        except Exception as exc:
            logger.error("Duplicate lookup fallback failed: %s", exc)

        return None

    async def handle_file_upload(self, file: UploadFile, current_user: Optional[User] = None) -> Dict[str, Any]:
        """
        Handle complete file upload process including file saving and database record creation.

        Returns:
            Dict containing file_info, source_id, and other upload details
        """
        try:
            uploader = current_user.username if current_user else 'anonymous'
            original_filename = file.filename or "upload"
            logger.info("File upload attempt by %s: %s", uploader, original_filename)

            sanitized_filename = self._sanitize_filename(original_filename)
            extension = Path(sanitized_filename).suffix.lower()

            if not extension:
                fallback_extension = Path(original_filename).suffix.lower()
                if fallback_extension and not sanitized_filename.lower().endswith(fallback_extension):
                    sanitized_filename = f"{sanitized_filename}{fallback_extension}"
                extension = fallback_extension

            if not extension:
                mime_map = {
                    "text/csv": ".csv",
                    "application/json": ".json",
                    "text/plain": ".txt",
                    "application/vnd.ms-excel": ".xls",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx"
                }
                mapped_extension = mime_map.get((file.content_type or "").lower())
                if mapped_extension:
                    extension = mapped_extension
                    if not sanitized_filename.lower().endswith(mapped_extension):
                        sanitized_filename = f"{sanitized_filename}{mapped_extension}"

            if not extension:
                raise FileUploadError("Unable to determine file type. Please upload a file with a valid extension.")

            normalized_extension = extension if extension.startswith('.') else f'.{extension}'
            self._validate_extension(normalized_extension)

            source_id = await self._generate_source_id()
            file_path = self.upload_dir / f"{source_id}_{sanitized_filename}"

            file_content_type = file.content_type
            file_size_bytes, file_hash, newline_count = await self._persist_upload_file(
                file,
                file_path,
                normalized_extension
            )

            estimated_rows = await self._estimate_rows_from_file(
                file_path,
                normalized_extension,
                newline_count,
                file_size_bytes
            )

            duplicate_source = await self._find_existing_source_by_hash(file_hash)
            if duplicate_source:
                safe_delete_path(file_path, files_only=True)

                existing_config: Dict[str, Any] = cast(Dict[str, Any], duplicate_source.config or {})
                existing_credentials: Dict[str, Any] = cast(Dict[str, Any], duplicate_source.credentials or {})
                resolved_rows = existing_config.get("estimated_rows")
                if isinstance(resolved_rows, str) and resolved_rows.isdigit():
                    resolved_rows = int(resolved_rows)
                if not isinstance(resolved_rows, int):
                    resolved_rows = estimated_rows

                preview_rows = resolved_rows if isinstance(resolved_rows, int) else 0
                existing_size = existing_config.get("file_size_bytes", file_size_bytes)

                file_info = {
                    "source_id": duplicate_source.source_id,
                    "original_filename": existing_credentials.get("original_filename", duplicate_source.name),
                    "display_name": duplicate_source.name,
                    "file_size_bytes": existing_size,
                    "file_size": self._format_file_size(existing_size),
                    "file_type": (existing_config.get("extension") or normalized_extension.lstrip('.')).upper(),
                    "estimated_rows": resolved_rows,
                    "preview_rows": min(preview_rows, 100) if isinstance(preview_rows, int) else 0,
                    "row_count": resolved_rows,
                    "file_path": existing_config.get("file_path"),
                    "upload_timestamp": existing_config.get("upload_timestamp", self._utc_now_iso()),
                    "status": "duplicate_reused"
                }

                try:
                    log_data_action(
                        "UPLOAD_FILE_DUPLICATE",
                        True,
                        f"Duplicate upload of {file_info['original_filename']} by {uploader}"
                    )
                except Exception:
                    logger.warning("Failed to write duplicate upload audit log")

                return {
                    "success": True,
                    "source_id": duplicate_source.source_id,
                    "file_info": file_info,
                    "is_duplicate": True,
                    "duplicate_of": duplicate_source.source_id,
                    "message": f"File '{original_filename}' already exists. Using existing data source."
                }

            upload_timestamp = self._utc_now_iso()
            created_by_id = current_user.id if current_user else None

            db_source = DataSource(
                source_id=source_id,
                name=original_filename,
                type=normalized_extension.lstrip('.') or "file",
                config={
                    "file_path": str(file_path),
                    "stored_filename": sanitized_filename,
                    "file_hash": file_hash,
                    "file_size_bytes": file_size_bytes,
                    "estimated_rows": estimated_rows,
                    "extension": normalized_extension.lstrip('.'),
                    "mime_type": file_content_type,
                    "upload_timestamp": upload_timestamp
                },
                credentials={
                    "upload_method": "web_interface",
                    "original_filename": original_filename,
                    "uploaded_by": uploader
                },
                description=(
                    f"Uploaded file: {original_filename} "
                    f"({self._format_file_size(file_size_bytes)}, ~{estimated_rows} rows)"
                ),
                is_active=True,
                test_status="uploaded",
                created_by=created_by_id,
                source_metadata={
                    "upload": {
                        "hash": file_hash,
                        "size_bytes": file_size_bytes,
                        "timestamp": upload_timestamp,
                        "uploader": uploader
                    }
                }
            )

            self.session.add(db_source)
            await self.session.commit()
            await self.session.refresh(db_source)

            logger.info("Added uploaded file to database: %s (ID: %s)", original_filename, db_source.id)

            try:
                log_data_action(
                    "UPLOAD_FILE",
                    True,
                    f"Uploaded {original_filename} as {source_id} by {uploader}"
                )
            except Exception:
                logger.warning("Failed to write upload audit log")

            preview_rows = min(estimated_rows, 100) if isinstance(estimated_rows, int) else 0

            return {
                "success": True,
                "source_id": source_id,
                "file_info": {
                    "source_id": source_id,
                    "original_filename": original_filename,
                    "display_name": original_filename,
                    "file_size_bytes": file_size_bytes,
                    "file_size": self._format_file_size(file_size_bytes),
                    "file_type": normalized_extension.lstrip('.').upper(),
                    "estimated_rows": estimated_rows,
                    "preview_rows": preview_rows,
                    "row_count": estimated_rows,
                    "file_path": str(file_path),
                    "upload_timestamp": upload_timestamp,
                    "status": "uploaded"
                },
                "message": f"File '{original_filename}' uploaded successfully",
                "is_duplicate": False,
                "duplicate_of": None
            }

        except FileUploadError:
            raise
        except Exception as e:
            logger.error(f"Error in handle_file_upload: {e}")
            try:
                if 'file_path' in locals() and file_path.exists():
                    safe_delete_path(file_path, files_only=True)
            except Exception:
                logger.warning("Failed to remove temporary upload after error")
            raise FileUploadError(f"Failed to upload file: {str(e)}")

    def _get_type_display_mapping(self) -> Dict[str, str]:
        """Get mapping of data source types to human-readable display names."""
        return {
            "snowflake": "Snowflake",
            "cloud_sheets": "Google Sheets",
            "database": "Database",
            "file_upload": "File Upload",
            "txt": "Text File",
            "pdf": "PDF File",
            "doc": "Word Document",
            "docx": "Word Document",
            "parquet": "Parquet File",
            "csv": "CSV File",
            "json": "JSON File",
            "xlsx": "Excel File",
            "xml": "XML File",
            "xls": "Excel File",
            "postgresql": "PostgreSQL Database",
            "mysql": "MySQL Database",
            "sqlite": "SQLite Database"
        }

    def _convert_source_to_api_format(self, source, type_display_map: Dict[str, str]) -> Dict[str, Any]:
        """Convert a database source object to API response format."""
        source_dict = source.to_dict() if hasattr(source, "to_dict") else {}
        source_type = source_dict.get("type", getattr(source, "type", "unknown"))
        display_type = type_display_map.get(source_type, source_type.upper())
        
        config = source_dict.get("config") or getattr(source, "config", {}) or {}
        file_size_bytes = config.get("file_size_bytes", 0)

        return {
            "id": source_dict.get("id") or getattr(source, "id", None),
            "source_id": source_dict.get("source_id") or str(source_dict.get("id", getattr(source, "id", ""))),
            "name": source_dict.get("name") or getattr(source, "name", "Unnamed Source"),
            "display_name": source_dict.get("name") or getattr(source, "name", "Unnamed Source"),
            "type": source_type,
            "source_type": display_type,
            "category": "Database" if source_type in ["postgresql", "mysql", "sqlite"] else "File",
            "status": source_dict.get("test_status", getattr(source, "test_status", "unknown")),
            "is_active": source_dict.get("is_active", getattr(source, "is_active", False)),
            "created_at": source_dict.get("created_at", getattr(source, "created_at", None)),
            "updated_at": source_dict.get("updated_at", getattr(source, "updated_at", None)),
            "last_tested": source_dict.get("last_tested", getattr(source, "last_tested", None)),
            "description": source_dict.get("description", getattr(source, "description", "")),
            "row_count": config.get("estimated_rows", 0),
            "rows": config.get("estimated_rows", 0),
            "column_count": config.get("column_count", 0),
            "columns": config.get("column_count", 0),
            "file_size": self._format_file_size(file_size_bytes) if file_size_bytes else "Unknown",
            "file_size_bytes": file_size_bytes,
            "size_bytes": file_size_bytes,
            "metadata": {
                "created_by": source_dict.get("created_by", getattr(source, "created_by", "Unknown")),
                "connector_type": source_dict.get("type", source_type),
                "test_status": source_dict.get("test_status", getattr(source, "test_status", "unknown"))
            }
        }

    def _sanitize_sensitive_data(self, sources: List[Dict[str, Any]], is_admin: bool) -> None:
        """Remove sensitive connection info for non-admin users."""
        if not is_admin:
            for source in sources:
                conn = source.get("connection_info")
                if isinstance(conn, dict):
                    source["connection_info"] = {
                        k: v for k, v in conn.items()
                        if k not in ["password", "secret", "key", "token"]
                    }

    async def get_formatted_data_sources_list(self, current_user: Optional[User] = None) -> Dict[str, Any]:
        """
        Get formatted list of data sources for API response.

        Returns:
            Dict containing sources list, total count, and scope information
        """
        try:
            username = current_user.username if current_user else "anonymous"
            logger.info(f"Getting formatted data sources list for user: {username}")

            # Determine scope based on user permissions
            is_admin = True
            if current_user:
                is_admin = current_user.has_permission("admin")
            
            scope = "admin_full" if is_admin else "user_scoped"

            # Use existing service method to get raw data sources
            created_by: Optional[str] = None
            if not is_admin:
                username_value = getattr(current_user, 'username', None)
                if isinstance(username_value, str):
                    created_by = username_value
                elif username_value is not None:
                    created_by = str(username_value)

            db_sources = await self.list_data_sources(
                created_by=created_by,
                limit=1000
            )

            # Get type display mapping
            type_display_map = self._get_type_display_mapping()

            # Convert sources to API format
            sources = []
            for source in db_sources:
                formatted_source = self._convert_source_to_api_format(source, type_display_map)
                sources.append(formatted_source)

            # Sanitize sensitive data for non-admin users
            self._sanitize_sensitive_data(sources, is_admin)

            return {
                "success": True,
                "sources": sources,
                "total_count": len(sources),
                "scope": scope
            }

        except Exception as e:
            logger.error(f"Error getting formatted data sources list: {e}")
            raise DataIngestionException(f"Failed to get formatted data sources list: {str(e)}")


# Global service instance
data_ingestion_service: Optional[DataIngestionService] = None


def get_data_ingestion_service(session: AsyncSession) -> DataIngestionService:
    """Get or create data ingestion service instance."""
    # For now, create a new instance each time since we're using sessions
    # In production, you might want to implement proper dependency injection
    return DataIngestionService(session)
