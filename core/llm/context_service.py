"""
Async Data Context Service for FastAPI

Service for gathering comprehensive data context from database and files.
Migrated from Flask with async patterns.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import aiosqlite
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class AsyncDataContextService:
    """Async service for gathering comprehensive data context from the application"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or self._get_db_path()
        self.api_base = "/data/api/sources"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _get_db_path(self) -> str:
        """Get the database path"""
        # Try to find the main database
        possible_paths = [
            "mlops_database.db",
            "test_database.db",
            os.path.join(os.getcwd(), "mlops_database.db"),
            os.path.join(os.path.dirname(os.getcwd()), "mlops_database.db"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return possible_paths[0]  # Default fallback

    async def get_data_context(self, source_id: str) -> Dict[str, Any]:
        """
        Gather comprehensive data context for a given source ID

        Args:
            source_id: The data source identifier

        Returns:
            Dictionary with data context information
        """
        context = {
            "source_id": source_id,
            "api_data": await self._get_api_data(source_id)
        }

        # Only try database methods if we have tables (avoid error spam)
        try:
            # Quick check if we have any tables
            async with aiosqlite.connect(self.db_path) as conn:
                async with conn.execute("SELECT name FROM sqlite_master WHERE type='table'") as cursor:
                    tables = await cursor.fetchall()

                if tables:
                    context.update({
                        "database_stats": await self._get_database_stats(),
                        "source_metadata": await self._get_source_metadata(source_id),
                        "data_preview": await self._get_data_preview(source_id),
                        "schema_info": await self._get_schema_info(source_id),
                    })
                else:
                    self.logger.info("No database tables found, skipping database context")

        except Exception as e:
            self.logger.warning(f"Could not gather database context: {e}")
            context["database_error"] = str(e)

        return context

    async def _get_api_data(self, source_id: str) -> Dict[str, Any]:
        """Get data from API endpoints (mock for now)"""
        try:
            # Mock API data - in real implementation, this would make HTTP calls
            # to the data ingestion API endpoints
            api_data = {
                "source_exists": False,
                "api_accessible": True,
                "last_checked": "2024-09-16T00:00:00Z"
            }
            
            self.logger.debug(f"Retrieved API data for source {source_id}")
            return api_data

        except Exception as e:
            self.logger.error(f"Error getting API data for {source_id}: {e}")
            return {"error": str(e), "api_accessible": False}

    async def _get_database_stats(self) -> Dict[str, Any]:
        """Get general database statistics"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                stats = {}

                # Get table count
                async with conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                ) as cursor:
                    table_count = await cursor.fetchone()
                    stats["table_count"] = table_count[0] if table_count else 0

                # Get database size
                stats["database_size_mb"] = round(os.path.getsize(self.db_path) / (1024 * 1024), 2)

                return stats

        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}

    async def _get_source_metadata(self, source_id: str) -> Dict[str, Any]:
        """Get metadata for a specific data source"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                # Try to find source in data_sources table
                query = """
                    SELECT id, name, type, status, connection_string, 
                           created_at, updated_at, created_by
                    FROM data_sources 
                    WHERE id = ?
                """
                
                async with conn.execute(query, (source_id,)) as cursor:
                    row = await cursor.fetchone()
                    
                    if row:
                        columns = [description[0] for description in cursor.description]
                        return dict(zip(columns, row))
                    else:
                        return {"error": "Source not found", "source_id": source_id}

        except Exception as e:
            self.logger.error(f"Error getting source metadata for {source_id}: {e}")
            return {"error": str(e)}

    async def _get_data_preview(self, source_id: str, limit: int = 5) -> Dict[str, Any]:
        """Get a preview of the actual data"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                # This would need to be adapted based on your actual data storage structure
                # For now, just return a placeholder
                return {
                    "preview_available": False,
                    "reason": "Data preview not implemented for async version",
                    "limit": limit
                }

        except Exception as e:
            self.logger.error(f"Error getting data preview for {source_id}: {e}")
            return {"error": str(e)}

    async def _get_schema_info(self, source_id: str) -> Dict[str, Any]:
        """Get schema information for the data source"""
        try:
            # Mock schema info - would need actual implementation based on data structure
            return {
                "schema_available": False,
                "reason": "Schema detection not implemented for async version"
            }

        except Exception as e:
            self.logger.error(f"Error getting schema info for {source_id}: {e}")
            return {"error": str(e)}

    async def format_data_context(
        self, 
        source_id: str, 
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format data context into a readable string for LLM consumption
        
        Args:
            source_id: The data source identifier
            user_context: Additional context from user's current view
            
        Returns:
            Formatted context string
        """
        try:
            context = await self.get_data_context(source_id)
            
            # Build formatted context string
            formatted_parts = []
            
            # Source information
            if "source_metadata" in context and context["source_metadata"]:
                metadata = context["source_metadata"]
                formatted_parts.append(f"**Data Source: {source_id}**")
                if metadata.get("name"):
                    formatted_parts.append(f"Name: {metadata['name']}")
                if metadata.get("type"):
                    formatted_parts.append(f"Type: {metadata['type']}")
                if metadata.get("status"):
                    formatted_parts.append(f"Status: {metadata['status']}")
                formatted_parts.append("")

            # Database stats
            if "database_stats" in context:
                stats = context["database_stats"]
                formatted_parts.append("**Database Information:**")
                if stats.get("table_count"):
                    formatted_parts.append(f"Tables: {stats['table_count']}")
                if stats.get("database_size_mb"):
                    formatted_parts.append(f"Size: {stats['database_size_mb']} MB")
                formatted_parts.append("")

            # User context
            if user_context:
                formatted_parts.append("**Current View Context:**")
                for key, value in user_context.items():
                    formatted_parts.append(f"{key}: {value}")
                formatted_parts.append("")

            # Data preview (if available)
            if "data_preview" in context and not context["data_preview"].get("error"):
                formatted_parts.append("**Data Preview:**")
                formatted_parts.append("(Data preview functionality to be implemented)")
                formatted_parts.append("")

            result = "\n".join(formatted_parts)
            self.logger.debug(f"Formatted context for {source_id} ({len(result)} characters)")
            
            return result

        except Exception as e:
            self.logger.error(f"Error formatting context for {source_id}: {e}")
            return f"Error gathering context for {source_id}: {str(e)}"