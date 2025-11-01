"""
FastAPI Admin Services Module (Simplified)

Modern async admin services for system management, migrated from Flask.
Provides data ingestion management, user management, and system monitoring.
This simplified version doesn't depend on database imports to avoid circular dependencies.
"""

import logging
import os
import platform
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

from core.database.engine import get_async_session, health_check
from config import get_settings
from typing import Dict, List, Any, Optional

from core.database.engine import get_async_session
from core.data_ingestion.service import DataIngestionService

logger = logging.getLogger(__name__)


def get_postgres_connection():
    """
    Unified PostgreSQL connection function using environment variables.
    Returns connection object or None if PostgreSQL not configured.
    """
    try:
        import psycopg2
        from config import get_settings
        
        settings = get_settings()
        
        # Check if PostgreSQL is configured
        if not all([settings.DB_USER, settings.DB_PASSWORD, settings.DB_HOST, settings.DB_NAME]):
            logger.warning("PostgreSQL not fully configured - missing credentials in environment")
            return None
        
        conn = psycopg2.connect(
            host=settings.DB_HOST,
            database=settings.DB_NAME,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
            port=settings.DB_PORT or 5432,
            sslmode=settings.DB_SSLMODE or 'require',
            connect_timeout=10
        )
        return conn
        
    except ImportError:
        logger.error("psycopg2 not installed - PostgreSQL connections not available")
        return None
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        return None


def format_file_size(file_size_bytes: int) -> str:
    """Format file size in human readable format - shared utility function"""
    if file_size_bytes > 0:
        if file_size_bytes < 1024:
            return f"{file_size_bytes} B"
        elif file_size_bytes < 1024 * 1024:
            return f"{file_size_bytes / 1024:.1f} KB"
        elif file_size_bytes < 1024 * 1024 * 1024:
            return f"{file_size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{file_size_bytes / (1024 * 1024 * 1024):.1f} GB"
    return "N/A"


class AsyncAdminDataService:
    """Async service for data ingestion management operations"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def get_data_ingestion_stats(self) -> Dict[str, Any]:
        """Get comprehensive data ingestion statistics"""
        try:
            async for session in get_async_session():
                service = DataIngestionService(session)
                # Use the comprehensive stats method from DataIngestionService
                stats = await service.get_data_ingestion_stats()
                self.logger.info("Retrieved comprehensive data ingestion statistics")
                return stats
                
        except Exception as e:
            self.logger.error(f"Error getting data ingestion stats: {e}")
            # Return safe default stats on error
            return {
                "total_sources": 0,
                "active_sources": 0,
                "failed_sources": 0,
                "total_jobs": 0,
                "successful_jobs": 0,
                "failed_jobs": 0, 
                "last_24h_jobs": 0,
                "data_volume": "0 B",
                "total_size_formatted": "0 B",
                "total_records": 0,
                "total_rows": 0,
                "sources_by_type": {},
                "sources_by_category": {},
                "message": "Error calculating statistics. Please check the logs."
            }

    async def get_data_sources_detailed(self, limit: int = 1000, offset: int = 0) -> List[Dict[str, Any]]:
        """Get detailed information about data sources with admin formatting"""
        try:
            async for session in get_async_session():
                service = DataIngestionService(session)
                sources = await service.list_data_sources(limit=limit, offset=offset)
                
                sources_list = []
                for source in sources:
                    source_dict = source.to_dict() if hasattr(source, 'to_dict') else dict(source)

                    # Map type to category
                    category = source_dict.get('type', 'unknown')
                    if category in ['csv', 'excel', 'json', 'txt', 'parquet', 'xml', 'xlsx', 'xls', 'doc']:
                        mapped_category = 'file'
                    elif category in ['postgres', 'mysql', 'sqlite']:
                        mapped_category = 'database'
                    elif category in ['api', 'rest', 'graphql']:
                        mapped_category = 'api'
                    else:
                        mapped_category = 'other'

                    config = source_dict.get('config', {}) or {}
                    file_size_bytes = config.get('file_size_bytes', 0) or 0
                    estimated_rows = config.get('estimated_rows', 0) or 0

                    # Try to get file info from filesystem if missing
                    if file_size_bytes == 0 and source_dict.get('type') in ['csv', 'excel', 'json', 'txt']:
                        try:
                            import os
                            file_path = config.get('file_path') or config.get('uploaded_file_path')
                            if file_path and os.path.exists(file_path):
                                file_size_bytes = os.path.getsize(file_path)

                                if estimated_rows == 0 and source_dict.get('type') == 'csv':
                                    try:
                                        with open(file_path, 'r', encoding='utf-8') as f:
                                            line_count = sum(1 for _ in f)
                                            estimated_rows = max(0, line_count - 1)
                                    except Exception:
                                        pass
                        except Exception as e:
                            self.logger.debug(f"Could not get file info for {source_dict.get('name')}: {e}")

                    # Quality score heuristics
                    quality_score = self._calculate_quality_score(
                        source_dict.get('type'), 
                        source_dict.get('test_status'),
                        file_size_bytes, 
                        estimated_rows
                    )

                    # Format file size
                    file_size_formatted = format_file_size(file_size_bytes)

                    sources_list.append({
                        "id": source_dict.get('id', 'N/A'),
                        "source_id": source_dict.get('source_id', 'N/A'),
                        "name": source_dict.get('name', 'Unnamed'),
                        "type": source_dict.get('type', 'unknown'),
                        "category": mapped_category,
                        "status": source_dict.get('test_status', 'unknown'),
                        "created_by": source_dict.get('created_by', 'Unknown'),
                        "created_at": source_dict.get('created_at'),
                        "last_tested": source_dict.get('last_tested'),
                        "description": source_dict.get('description', ''),
                        "is_active": source_dict.get('is_active', False),
                        "file_size_formatted": file_size_formatted,
                        "file_size_bytes": file_size_bytes,
                        "estimated_rows": estimated_rows if estimated_rows > 0 else None,
                        "quality_score": quality_score
                    })

                self.logger.info(f"Retrieved {len(sources_list)} detailed data sources")
                return sources_list
            
        except Exception as e:
            self.logger.error(f"Error getting data sources: {e}")
            raise

    async def delete_data_source(self, source_id: str) -> bool:
        """Delete a data source by ID"""
        try:
            from ..data_ingestion.service import DataIngestionService
            data_service = DataIngestionService()
            result = await data_service.delete_data_source(source_id)
            self.logger.info(f"Delete result for data source {source_id}: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error deleting data source {source_id}: {e}")
            raise

    async def bulk_delete_data_sources(self, source_ids: List[str]) -> Dict[str, Any]:
        """Bulk delete multiple data sources"""
        try:
            deleted = 0
            errors = []
            
            for source_id in source_ids:
                try:
                    success = await self.delete_data_source(source_id)
                    if success:
                        deleted += 1
                except Exception as e:
                    errors.append(f"Error deleting {source_id}: {str(e)}")
            
            result = {
                "deleted": deleted,
                "total_requested": len(source_ids),
                "errors": errors
            }
            
            self.logger.info(f"Bulk delete completed: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in bulk delete operation: {e}")
            raise

    def _calculate_quality_score(self, source_type: str, test_status: str, file_size_bytes: int, estimated_rows: int) -> Optional[float]:
        """Calculate quality score based on source type and available data"""
        if source_type in ['csv', 'excel', 'json', 'txt']:
            if file_size_bytes > 0 and estimated_rows > 0:
                return 0.85
            elif file_size_bytes > 0:
                return 0.70
            elif test_status == 'success':
                return 0.60
            else:
                return 0.40
        elif source_type in ['postgres', 'mysql', 'sqlite']:
            if test_status == 'success':
                return 0.90
            elif test_status == 'failed':
                return 0.30
            else:
                return 0.50
        elif source_type in ['api', 'rest', 'graphql']:
            if test_status == 'success':
                return 0.80
            elif test_status == 'failed':
                return 0.25
            else:
                return 0.45
        return None


class AsyncAdminUserService:
    """Async service for user management operations"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def get_user_stats(self) -> Dict[str, Any]:
        """Get user statistics and activity"""
        try:
            async for db in get_async_session():
                from sqlalchemy import select, func
                from core.database.models import User, DataSource
                
                # Get user counts
                total_users_result = await db.execute(select(func.count(User.id)))
                total_users = total_users_result.scalar() or 0
                
                active_users_result = await db.execute(
                    select(func.count(User.id)).where(User.is_active == True)
                )
                active_users = active_users_result.scalar() or 0
                
                admin_users_result = await db.execute(
                    select(func.count(User.id)).where(User.is_admin == True)
                )
                admin_users = admin_users_result.scalar() or 0
                
                # Get users who have logged in recently (last 24 hours)
                from datetime import datetime, timedelta
                yesterday = datetime.utcnow() - timedelta(hours=24)
                recent_logins_result = await db.execute(
                    select(func.count(User.id)).where(User.last_login >= yesterday)
                )
                last_24h_logins = recent_logins_result.scalar() or 0
                
                # Get users created this month
                from datetime import datetime
                first_day_of_month = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                new_users_result = await db.execute(
                    select(func.count(User.id)).where(User.created_at >= first_day_of_month)
                )
                new_users_this_month = new_users_result.scalar() or 0
                
                # Get data sources by user
                data_sources_by_user = {}
                if total_users > 0:
                    # Get data sources grouped by created_by user
                    sources_query = select(
                        User.username,
                        func.count(DataSource.id).label('source_count')
                    ).select_from(
                        User
                    ).outerjoin(
                        DataSource, User.id == DataSource.created_by
                    ).group_by(User.id, User.username)
                    
                    sources_result = await db.execute(sources_query)
                    sources_rows = sources_result.fetchall()
                    
                    for row in sources_rows:
                        if row.source_count > 0:  # Only include users with data sources
                            data_sources_by_user[row.username] = row.source_count
                
                stats = {
                    "total_users": total_users,
                    "active_users": active_users,
                    "admin_users": admin_users,
                    "last_24h_logins": last_24h_logins,
                    "new_users_this_month": new_users_this_month,
                    "data_sources_by_user": data_sources_by_user,
                    "users_by_role": {
                        "admin": admin_users,
                        "user": total_users - admin_users
                    },
                    "online_users": last_24h_logins  # Using recent logins as proxy for online users
                }
                
                self.logger.info(f"Retrieved user statistics: {total_users} total users, {len(data_sources_by_user)} with data sources")
                return stats
            
        except Exception as e:
            self.logger.error(f"Error getting user stats: {e}")
            # Return default stats in case of error
            return {
                "total_users": 0,
                "active_users": 0,
                "admin_users": 0,
                "last_24h_logins": 0,
                "new_users_this_month": 0,
                "data_sources_by_user": {},
                "users_by_role": {"admin": 0, "user": 0},
                "online_users": 0
            }

    async def get_user_stats_detailed(self) -> Dict[str, Any]:
        """Get detailed user statistics including per-user data source counts, sizes and user table data"""
        try:
            # Get base stats first
            base_stats = await self.get_user_stats()
            
            async for db in get_async_session():
                from core.database.models import User as DBUser, DataSource
                from sqlalchemy import select, func, text
                from datetime import datetime, timedelta
                from pathlib import Path
                
                enhanced_data_sources_by_user = {}
                user_table_data = []
                
                # Query per-user aggregates with data source info
                user_query = select(
                    DBUser.id,
                    DBUser.username,
                    DBUser.email,
                    DBUser.is_admin,
                    DBUser.is_active,
                    DBUser.last_login,
                    func.count(DataSource.id).label('data_source_count'),
                    func.max(DataSource.updated_at).label('last_data_activity')
                ).select_from(DBUser).outerjoin(
                    DataSource, DBUser.id == DataSource.created_by
                ).group_by(
                    DBUser.id, DBUser.username, DBUser.email, DBUser.is_admin, DBUser.is_active, DBUser.last_login
                )

                result = await db.execute(user_query)
                user_rows = result.fetchall()

                for row in user_rows:
                    total_size_bytes = 0
                    if getattr(row, 'data_source_count', 0) > 0:
                        size_query = select(DataSource).where(DataSource.created_by == row.id)
                        size_result = await db.execute(size_query)
                        user_sources = size_result.scalars().all()

                        for source in user_sources:
                            try:
                                config = source.config or {}
                                file_size = config.get('file_size_bytes', 0)
                                if file_size == 0 and config.get('file_path'):
                                    file_path = Path(config['file_path'])
                                    if file_path.exists():
                                        file_size = file_path.stat().st_size
                                total_size_bytes += file_size
                            except Exception as e:
                                self.logger.warning(f"Error calculating size for source {getattr(source,'id',None)}: {e}")

                        # Format size using shared utility function
                        size_formatted = format_file_size(total_size_bytes)

                        enhanced_data_sources_by_user[row.username] = {
                            'count': row.data_source_count,
                            'size': size_formatted,
                            'last_activity': row.last_data_activity.strftime('%Y-%m-%d') if row.last_data_activity else 'Never'
                        }

                    user_table_data.append({
                        "username": row.username,
                        "role": "admin" if row.is_admin else "user",
                        "email": row.email,
                        "data_sources": getattr(row, 'data_source_count', 0),
                        "last_login": row.last_login.strftime("%Y-%m-%d %H:%M:%S") if row.last_login else "Never",
                        "status": "active" if row.is_active else "inactive"
                    })

                # Calculate online users (last 30 minutes)
                thirty_minutes_ago = datetime.utcnow() - timedelta(minutes=30)
                online_users_query = select(func.count(DBUser.id)).where(
                    DBUser.is_active == True,
                    DBUser.last_login >= thirty_minutes_ago
                )
                online_result = await db.execute(online_users_query)
                online_count = online_result.scalar() or 0

                # Compose enhanced stats
                total_users = base_stats.get('total_users', 0)
                data_sources_by_user = base_stats.get('data_sources_by_user', {})

                enhanced_stats = {
                    **base_stats,
                    "user_table": user_table_data,
                    "inactive_users": total_users - base_stats.get('active_users', 0),
                    "data_sources_by_user_detailed": enhanced_data_sources_by_user,
                    "online_users": online_count,
                    "data_sources_summary": {
                        "total_sources": sum(data_sources_by_user.values()) if isinstance(data_sources_by_user, dict) else 0,
                        "average_per_user": round(sum(data_sources_by_user.values()) / total_users, 1) if total_users > 0 and isinstance(data_sources_by_user, dict) else 0,
                        "by_user": data_sources_by_user
                    }
                }
                
                self.logger.info(f"Retrieved detailed user statistics with {len(user_table_data)} users and {online_count} online")
                return enhanced_stats
                
        except Exception as e:
            self.logger.error(f"Error getting detailed user stats: {e}")
            # Return base stats with error handling
            return {
                **base_stats,
                "user_table": [],
                "inactive_users": 0,
                "data_sources_by_user_detailed": {},
                "online_users": 0,
                "data_sources_summary": {
                    "total_sources": 0,
                    "average_per_user": 0,
                    "by_user": {}
                }
            }

    async def get_users(
        self, 
        skip: int = 0, 
        limit: int = 100, 
        search: Optional[str] = None,
        is_active: Optional[bool] = None,
        is_admin: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Get paginated list of users with filtering"""
        try:
            async for db in get_async_session():
                from sqlalchemy import select, func, or_
                from core.database.models import User
                
                # Build query with filters
                query = select(User)
                
                if search:
                    search_term = f"%{search}%"
                    query = query.where(or_(
                        User.username.ilike(search_term),
                        User.email.ilike(search_term),
                        User.full_name.ilike(search_term)
                    ))
                
                if is_active is not None:
                    query = query.where(User.is_active == is_active)
                
                if is_admin is not None:
                    query = query.where(User.is_admin == is_admin)
                
                # Get total count for pagination
                count_query = select(func.count(User.id))
                if search:
                    count_query = count_query.where(or_(
                        User.username.ilike(search_term),
                        User.email.ilike(search_term),
                        User.full_name.ilike(search_term)
                    ))
                if is_active is not None:
                    count_query = count_query.where(User.is_active == is_active)
                if is_admin is not None:
                    count_query = count_query.where(User.is_admin == is_admin)
                
                total_result = await db.execute(count_query)
                total = total_result.scalar() or 0
                
                # Apply pagination and ordering
                query = query.order_by(User.created_at.desc()).offset(skip).limit(limit)
                
                result = await db.execute(query)
                users = result.scalars().all()
                
                # Convert to dict format
                users_data = []
                for user in users:
                    users_data.append({
                        "id": user.id,
                        "username": user.username,
                        "email": user.email,
                        "full_name": user.full_name,
                        "is_active": user.is_active,
                        "is_admin": user.is_admin,
                        "is_verified": user.is_verified,
                        "last_login": user.last_login.isoformat() if user.last_login else None,
                        "created_at": user.created_at.isoformat() if user.created_at else None
                    })
                
                self.logger.info(f"Retrieved {len(users_data)} users (total: {total})")
                return {
                    "users": users_data,
                    "total": total,
                    "skip": skip,
                    "limit": limit
                }
                
        except Exception as e:
            self.logger.error(f"Error getting users: {e}")
            raise

    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        full_name: Optional[str] = None,
        is_admin: bool = False,
        is_verified: bool = False
    ) -> Dict[str, Any]:
        """Create a new user"""
        try:
            async for db in get_async_session():
                from core.database.models import User
                from core.auth.auth_core import get_password_hash
                from sqlalchemy import select
                from sqlalchemy.exc import IntegrityError
                
                # Check if username already exists
                existing_username = await db.execute(
                    select(User).where(User.username == username)
                )
                if existing_username.scalar_one_or_none():
                    raise ValueError(f"Username '{username}' already exists")
                
                # Check if email already exists
                existing_email = await db.execute(
                    select(User).where(User.email == email)
                )
                if existing_email.scalar_one_or_none():
                    raise ValueError(f"Email '{email}' already exists")
                
                # Hash password and create user
                hashed_password = get_password_hash(password)
                new_user = User(
                    username=username,
                    email=email,
                    password_hash=hashed_password,
                    full_name=full_name,
                    is_admin=is_admin,
                    is_active=True,
                    is_verified=is_verified
                )
                
                try:
                    db.add(new_user)
                    await db.commit()
                    await db.refresh(new_user)
                except IntegrityError as e:
                    await db.rollback()
                    if "username" in str(e).lower():
                        raise ValueError(f"Username '{username}' already exists")
                    elif "email" in str(e).lower():
                        raise ValueError(f"Email '{email}' already exists")
                    else:
                        raise ValueError(f"Database constraint error: {str(e)}")
                
                self.logger.info(f"Created new user: {username}")
                return {
                    "id": new_user.id,
                    "username": new_user.username,
                    "email": new_user.email,
                    "full_name": new_user.full_name,
                    "is_active": new_user.is_active,
                    "is_admin": new_user.is_admin,
                    "is_verified": new_user.is_verified,
                    "created_at": new_user.created_at.isoformat() if new_user.created_at else None
                }
                
        except ValueError:
            # Re-raise ValueError as-is (these are our custom validation errors)
            raise
        except Exception as e:
            self.logger.error(f"Error creating user: {e}")
            raise ValueError(f"Failed to create user: {str(e)}")

    async def update_user(self, user_id: int, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user details"""
        try:
            async for db in get_async_session():
                from core.database.models import User
                from sqlalchemy import select
                
                # Get user
                result = await db.execute(select(User).where(User.id == user_id))
                user = result.scalar_one_or_none()
                
                if not user:
                    raise ValueError(f"User with ID {user_id} not found")
                
                # Check for username/email conflicts if they're being updated
                if 'username' in update_data and update_data['username'] != user.username:
                    existing_user = await db.execute(
                        select(User).where(
                            (User.username == update_data['username']) & (User.id != user_id)
                        )
                    )
                    if existing_user.scalar_one_or_none():
                        raise ValueError("Username already exists")
                
                if 'email' in update_data and update_data['email'] != user.email:
                    existing_user = await db.execute(
                        select(User).where(
                            (User.email == update_data['email']) & (User.id != user_id)
                        )
                    )
                    if existing_user.scalar_one_or_none():
                        raise ValueError("Email already exists")
                
                # Update user attributes
                for field, value in update_data.items():
                    if hasattr(user, field):
                        setattr(user, field, value)
                
                await db.commit()
                await db.refresh(user)
                
                self.logger.info(f"Updated user {user_id}: {list(update_data.keys())}")
                return {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.full_name,
                    "is_active": user.is_active,
                    "is_admin": user.is_admin,
                    "is_verified": user.is_verified,
                    "last_login": user.last_login.isoformat() if user.last_login else None,
                    "created_at": user.created_at.isoformat() if user.created_at else None
                }
                
        except Exception as e:
            self.logger.error(f"Error updating user {user_id}: {e}")
            raise

    async def delete_user(self, user_id: int) -> Dict[str, Any]:
        """Delete a user by ID and their associated data sources"""
        try:
            async for db in get_async_session():
                from core.database.models import User, DataSource
                from sqlalchemy import select
                
                # Get user
                result = await db.execute(select(User).where(User.id == user_id))
                user = result.scalar_one_or_none()
                
                if not user:
                    raise ValueError(f"User with ID {user_id} not found")
                
                username = user.username
                
                # Count and delete associated data sources
                data_sources_result = await db.execute(
                    select(DataSource).where(DataSource.created_by == user_id)
                )
                data_sources = data_sources_result.scalars().all()
                deleted_data_sources = len(data_sources)
                
                # Delete data sources first (foreign key constraint)
                for data_source in data_sources:
                    await db.delete(data_source)
                
                # Delete the user
                await db.delete(user)
                await db.commit()
                
                self.logger.info(f"Deleted user {user_id}: {username} and {deleted_data_sources} data sources")
                
                return {
                    "message": f"User '{username}' deleted successfully",
                    "deleted_data_sources": deleted_data_sources,
                    "username": username
                }
                
        except Exception as e:
            self.logger.error(f"Error deleting user {user_id}: {e}")
            raise


class AsyncAdminSystemService:
    """Async service for system management and monitoring"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        try:
            # Use user-provided psutil-based implementation for richer system info
            try:
                import psutil
                import sys

                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/') if os.name != 'nt' else psutil.disk_usage('C:\\')
                boot_time = psutil.boot_time()

                def _get_dir_size_and_count(path: str) -> Dict[str, Any]:
                    """Return total size in bytes and file count for a directory."""
                    total_size = 0
                    total_files = 0
                    try:
                        if not os.path.exists(path):
                            return {"size": 0, "count": 0}

                        for root, dirs, files in os.walk(path):
                            for f in files:
                                try:
                                    fp = os.path.join(root, f)
                                    if os.path.islink(fp):
                                        continue
                                    total_size += os.path.getsize(fp)
                                    total_files += 1
                                except Exception:
                                    # Skip files that can't be accessed
                                    continue
                    except Exception:
                        return {"size": 0, "count": 0}

                    return {"size": total_size, "count": total_files}

                def _format_size(num_bytes: int) -> str:
                    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                        if num_bytes < 1024.0:
                            return f"{num_bytes:.1f} {unit}"
                        num_bytes /= 1024.0
                    return f"{num_bytes:.1f} PB"

                # Compute directory sizes
                uploads_info = _get_dir_size_and_count(os.path.join('uploads'))
                cache_info = _get_dir_size_and_count(os.path.join('temp'))
                logs_info = _get_dir_size_and_count(os.path.join('logs'))
                exports_info = _get_dir_size_and_count(os.path.join('exports'))

                info = {
                    "memory": {
                        "percent": memory.percent,
                        "used_formatted": f"{memory.used / (1024**3):.1f} GB",
                        "total_formatted": f"{memory.total / (1024**3):.1f} GB"
                    },
                    "cpu": {
                        "percent": cpu_percent,
                        "count": psutil.cpu_count()
                    },
                    "disk": {
                        "percent": disk.percent,
                        "used_formatted": f"{disk.used / (1024**3):.1f} GB",
                        "total_formatted": f"{disk.total / (1024**3):.1f} GB"
                    },
                    "uptime": {
                        "uptime_formatted": "System Online",
                        "boot_time": datetime.fromtimestamp(boot_time).isoformat()
                    },
                    "system": {
                        "platform": platform.system(),
                        "platform_version": platform.version(),
                        "python_version": sys.version.split()[0],
                        "server_start_time": datetime.now().isoformat()
                    },
                    "storage": {
                        "uploads": {"size_bytes": uploads_info['size'], "size_formatted": _format_size(uploads_info['size'])},
                        "cache": {"size_bytes": cache_info['size'], "size_formatted": _format_size(cache_info['size'])},
                        "logs": {"size_bytes": logs_info['size'], "size_formatted": _format_size(logs_info['size'])},
                        "exports": {"size_bytes": exports_info['size'], "size_formatted": _format_size(exports_info['size'])},
                        "total_files": uploads_info['count'] + cache_info['count'] + logs_info['count'] + exports_info['count']
                    }
                }

                # Try to add some basic platform/top-level info
                info["platform"] = platform.system()
                info["platform_version"] = platform.version()
                info["python_version"] = platform.python_version()

                self.logger.info("Retrieved comprehensive system information (psutil)")
                return info
            except ImportError:
                # Fallback if psutil is not available
                self.logger.warning("psutil not available; returning minimal system info")
                minimal = {
                    "platform": platform.system(),
                    "platform_version": platform.version(),
                    "python_version": platform.python_version(),
                    "message": "psutil not installed, install psutil for richer system info"
                }
                return minimal
            
        except Exception as e:
            self.logger.error(f"Error getting system info: {e}")
            raise

    async def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""
        try:
            # Check database connectivity
            db_health = await health_check()
            
            report = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "checks": {},
                "warnings": [],
                "errors": []
            }
            
            # Database connectivity check
            if db_health:
                report["checks"]["database"] = {"status": "healthy", "message": "Database connection successful"}
            else:
                report["checks"]["database"] = {"status": "error", "message": "Database connection failed"}
                report["status"] = "unhealthy"
                report["errors"].append("Database connectivity issues detected")
            
            # File system check
            try:
                from pathlib import Path
                test_file = Path("temp/health_check.tmp")
                test_file.parent.mkdir(exist_ok=True)
                test_file.write_text("health check")
                test_file.unlink()
                report["checks"]["filesystem"] = {"status": "healthy", "message": "File system read/write successful"}
            except Exception as e:
                report["checks"]["filesystem"] = {"status": "error", "message": f"File system check failed: {e}"}
                report["status"] = "unhealthy"
                report["errors"].append(f"File system issues: {e}")
            
            # Memory and system checks with psutil if available
            try:
                import psutil
                
                # Memory check
                memory = psutil.virtual_memory()
                if memory.percent > 90:
                    report["checks"]["memory"] = {"status": "warning", "message": f"High memory usage: {memory.percent:.1f}%"}
                    report["warnings"].append(f"High memory usage: {memory.percent:.1f}%")
                    if report["status"] == "healthy":
                        report["status"] = "warning"
                else:
                    report["checks"]["memory"] = {"status": "healthy", "message": f"Memory usage: {memory.percent:.1f}%"}
                
                # CPU check
                cpu_percent = psutil.cpu_percent(interval=1)
                if cpu_percent > 85:
                    report["checks"]["cpu"] = {"status": "warning", "message": f"High CPU usage: {cpu_percent:.1f}%"}
                    report["warnings"].append(f"High CPU usage: {cpu_percent:.1f}%")
                    if report["status"] == "healthy":
                        report["status"] = "warning"
                else:
                    report["checks"]["cpu"] = {"status": "healthy", "message": f"CPU usage: {cpu_percent:.1f}%"}
                
                # Disk check
                disk = psutil.disk_usage('.')
                disk_percent = disk.used / disk.total * 100
                if disk_percent > 90:
                    report["checks"]["disk"] = {"status": "warning", "message": f"High disk usage: {disk_percent:.1f}%"}
                    report["warnings"].append(f"High disk usage: {disk_percent:.1f}%")
                    if report["status"] == "healthy":
                        report["status"] = "warning"
                else:
                    report["checks"]["disk"] = {"status": "healthy", "message": f"Disk usage: {disk_percent:.1f}%"}
                    
            except ImportError:
                report["checks"]["system_monitoring"] = {
                    "status": "info", 
                    "message": "psutil not available - install for detailed system monitoring"
                }
            except Exception as e:
                report["checks"]["system_monitoring"] = {
                    "status": "warning", 
                    "message": f"System monitoring error: {e}"
                }
                report["warnings"].append(f"System monitoring issues: {e}")
            
            self.logger.info(f"Generated system health report - Status: {report['status']}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating health report: {e}")
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "checks": {},
                "warnings": [],
                "errors": [f"Health check failed: {e}"]
            }

    async def get_database_status(self) -> Dict[str, Any]:
        """Get database connection status for both SQLite and PostgreSQL"""
        status = {
            "sqlite": {"connected": False, "count": 0, "error": None},
            "postgres": {"connected": False, "count": 0, "error": None},
            "current_database": "sqlite",
            "primary_db": "sqlite",
            "auto_init": {"performed": False, "results": None}
        }
        
        # First, ensure database schemas exist and are synchronized
        try:
            init_results = await self.ensure_database_schemas()
            status["auto_init"] = {
                "performed": True,
                "results": init_results
            }
            self.logger.info("Database auto-initialization completed")
        except Exception as e:
            self.logger.warning(f"Database auto-initialization failed: {e}")
            status["auto_init"] = {
                "performed": False,
                "results": {"error": str(e)}
            }
        
        # Check SQLite database
        try:
            import sqlite3
            import os
            
            db_path = 'mlops_database.db'
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM data_sources")
                sqlite_count = cursor.fetchone()[0]
                conn.close()
                
                status["sqlite"] = {
                    "connected": True,
                    "count": sqlite_count,
                    "error": None
                }
                self.logger.info(f"SQLite connected with {sqlite_count} records")
            else:
                status["sqlite"] = {
                    "connected": False,
                    "count": 0,
                    "error": f"Database file not found: {db_path}"
                }
                self.logger.warning(f"SQLite database file not found: {db_path}")
                
        except Exception as e:
            error_msg = f"SQLite error: {str(e)}"
            status["sqlite"] = {
                "connected": False,
                "count": 0,
                "error": error_msg
            }
            self.logger.error(error_msg)
        
        # Check PostgreSQL database
        try:
            conn = get_postgres_connection()
            if conn is None:
                status["postgres"] = {
                    "connected": False,
                    "count": 0,
                    "error": "PostgreSQL not configured (missing DB_USER, DB_PASSWORD, DB_HOST, or DB_NAME in environment)"
                }
                return status
            
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM data_sources")
            postgres_count = cursor.fetchone()[0]
            conn.close()
            
            status["postgres"] = {
                "connected": True,
                "count": postgres_count,
                "error": None
            }
            self.logger.info(f"PostgreSQL connected with {postgres_count} records")
            
        except Exception as e:
            error_msg = f"PostgreSQL error: {str(e)}"
            status["postgres"] = {
                "connected": False,
                "count": 0,
                "error": error_msg
            }
            self.logger.error(error_msg)
        
        self.logger.info(f"Database status check completed - SQLite: {status['sqlite']['count']} records, PostgreSQL: {status['postgres']['count']} records")
        return status

    async def ensure_database_schemas(self) -> Dict[str, Any]:
        """
        Ensure both SQLite and PostgreSQL databases have correct schemas.
        Creates databases and tables if they don't exist, syncs schemas if they differ.
        """
        result = {
            "sqlite": {"initialized": False, "tables_created": [], "errors": []},
            "postgres": {"initialized": False, "tables_created": [], "errors": []},
            "schema_sync": {"performed": False, "changes": [], "errors": []}
        }
        
        # Initialize SQLite
        sqlite_result = await self._ensure_sqlite_schema()
        result["sqlite"] = sqlite_result
        
        # Initialize PostgreSQL
        postgres_result = await self._ensure_postgres_schema()
        result["postgres"] = postgres_result
        
        # Sync schemas if both are successful
        if sqlite_result["initialized"] and postgres_result["initialized"]:
            sync_result = await self._sync_database_schemas()
            result["schema_sync"] = sync_result
        
        self.logger.info(f"Database schema initialization completed")
        return result

    async def _ensure_sqlite_schema(self) -> Dict[str, Any]:
        """Ensure SQLite database and tables exist with correct schema."""
        result = {"initialized": False, "tables_created": [], "errors": []}
        
        try:
            import sqlite3
            import os
            
            db_path = 'mlops_database.db'
            db_existed = os.path.exists(db_path)
            
            # Connect to SQLite (creates file if doesn't exist)
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            if not db_existed:
                self.logger.info("SQLite database file didn't exist, created new one")
            
            # Define the correct schema for data_sources table
            create_data_sources_sql = """
            CREATE TABLE IF NOT EXISTS data_sources (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                type VARCHAR(50) NOT NULL,
                config JSON NOT NULL,
                credentials JSON,
                is_active BOOLEAN NOT NULL DEFAULT 1,
                last_tested DATETIME,
                test_status VARCHAR(20) NOT NULL DEFAULT 'untested',
                created_by INTEGER,
                description TEXT,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            # Create other necessary tables
            create_users_sql = """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username VARCHAR(80) UNIQUE NOT NULL,
                email VARCHAR(120) UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT 1,
                is_admin BOOLEAN NOT NULL DEFAULT 0,
                last_login DATETIME,
                login_count INTEGER DEFAULT 0,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            create_jobs_sql = """
            CREATE TABLE IF NOT EXISTS data_ingestion_jobs (
                id INTEGER PRIMARY KEY,
                source_id INTEGER NOT NULL,
                status VARCHAR(20) NOT NULL DEFAULT 'pending',
                started_at DATETIME,
                completed_at DATETIME,
                error_message TEXT,
                records_processed INTEGER DEFAULT 0,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_id) REFERENCES data_sources (id)
            )
            """
            
            # Execute table creation
            tables = [
                ("data_sources", create_data_sources_sql),
                ("users", create_users_sql),
                ("data_ingestion_jobs", create_jobs_sql)
            ]
            
            for table_name, sql in tables:
                cursor.execute(sql)
                result["tables_created"].append(table_name)
                self.logger.info(f"Ensured SQLite table exists: {table_name}")
            
            # Create indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS ix_data_sources_name ON data_sources(name)",
                "CREATE INDEX IF NOT EXISTS ix_data_sources_id ON data_sources(id)",
                "CREATE INDEX IF NOT EXISTS ix_users_username ON users(username)",
                "CREATE INDEX IF NOT EXISTS ix_users_id ON users(id)"
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            conn.commit()
            conn.close()
            
            result["initialized"] = True
            self.logger.info("SQLite schema initialization completed successfully")
            
        except Exception as e:
            error_msg = f"SQLite schema initialization failed: {str(e)}"
            result["errors"].append(error_msg)
            self.logger.error(error_msg)
        
        return result

    async def _ensure_postgres_schema(self) -> Dict[str, Any]:
        """Ensure PostgreSQL database and tables exist with correct schema matching SQLite."""
        result = {"initialized": False, "tables_created": [], "errors": []}
        
        try:
            conn = get_postgres_connection()
            if conn is None:
                return {"success": False, "error": "PostgreSQL not configured in environment variables"}
            
            cursor = conn.cursor()
            
            # Define PostgreSQL schema that matches SQLite exactly
            create_data_sources_sql = """
            CREATE TABLE IF NOT EXISTS data_sources (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                type VARCHAR(50) NOT NULL,
                config JSONB NOT NULL,
                credentials JSONB,
                is_active BOOLEAN NOT NULL DEFAULT true,
                last_tested TIMESTAMP,
                test_status VARCHAR(20) NOT NULL DEFAULT 'untested',
                created_by INTEGER,
                description TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            create_users_sql = """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(80) UNIQUE NOT NULL,
                email VARCHAR(120) UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT true,
                is_admin BOOLEAN NOT NULL DEFAULT false,
                last_login TIMESTAMP,
                login_count INTEGER DEFAULT 0,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            create_jobs_sql = """
            CREATE TABLE IF NOT EXISTS data_ingestion_jobs (
                id SERIAL PRIMARY KEY,
                source_id INTEGER NOT NULL,
                status VARCHAR(20) NOT NULL DEFAULT 'pending',
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT,
                records_processed INTEGER DEFAULT 0,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_id) REFERENCES data_sources (id)
            )
            """
            
            # Execute table creation
            tables = [
                ("data_sources", create_data_sources_sql),
                ("users", create_users_sql),
            ]
            
            for table_name, sql in tables:
                cursor.execute(sql)
                result["tables_created"].append(table_name)
                self.logger.info(f"Ensured PostgreSQL table exists: {table_name}")
            
            # Create indexes that match SQLite
            indexes = [
                "CREATE INDEX IF NOT EXISTS ix_data_sources_name ON data_sources(name)",
                "CREATE INDEX IF NOT EXISTS ix_data_sources_id ON data_sources(id)",
                "CREATE INDEX IF NOT EXISTS ix_users_username ON users(username)",
                "CREATE INDEX IF NOT EXISTS ix_users_id ON users(id)"
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            conn.commit()
            conn.close()
            
            result["initialized"] = True
            self.logger.info("PostgreSQL schema initialization completed successfully")
            
        except Exception as e:
            error_msg = f"PostgreSQL schema initialization failed: {str(e)}"
            result["errors"].append(error_msg)
            self.logger.error(error_msg)
        
        return result

    async def _sync_database_schemas(self) -> Dict[str, Any]:
        """Compare and sync schemas between SQLite and PostgreSQL."""
        result = {"performed": False, "changes": [], "errors": []}
        
        try:
            # Get schema info from both databases
            sqlite_schema = await self._get_sqlite_schema_info()
            postgres_schema = await self._get_postgres_schema_info()
            
            # Compare schemas and identify differences
            differences = self._compare_schemas(sqlite_schema, postgres_schema)
            
            if differences:
                result["changes"] = differences
                self.logger.info(f"Schema differences detected: {len(differences)} differences")
                
                # For now, just log the differences
                # In the future, we could implement automatic schema migration
                for diff in differences:
                    self.logger.warning(f"Schema difference: {diff}")
            else:
                self.logger.info("Database schemas are synchronized")
            
            result["performed"] = True
            
        except Exception as e:
            error_msg = f"Schema synchronization failed: {str(e)}"
            result["errors"].append(error_msg)
            self.logger.error(error_msg)
        
        return result

    async def _get_sqlite_schema_info(self) -> Dict[str, Any]:
        """Get schema information from SQLite database."""
        try:
            import sqlite3
            conn = sqlite3.connect('mlops_database.db')
            cursor = conn.cursor()
            
            # Get table information
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            schema_info = {}
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                schema_info[table] = {
                    "columns": [(col[1], col[2]) for col in columns]  # (name, type)
                }
            
            conn.close()
            return schema_info
            
        except Exception as e:
            self.logger.error(f"Failed to get SQLite schema info: {e}")
            return {}

    async def _get_postgres_schema_info(self) -> Dict[str, Any]:
        """Get schema information from PostgreSQL database."""
        try:
            conn = get_postgres_connection()
            if conn is None:
                return {"error": "PostgreSQL not configured in environment variables"}
            
            cursor = conn.cursor()
            
            # Get table information
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            schema_info = {}
            for table in tables:
                cursor.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = %s AND table_schema = 'public'
                    ORDER BY ordinal_position
                """, (table,))
                columns = cursor.fetchall()
                schema_info[table] = {
                    "columns": [(col[0], col[1]) for col in columns]  # (name, type)
                }
            
            conn.close()
            return schema_info
            
        except Exception as e:
            self.logger.error(f"Failed to get PostgreSQL schema info: {e}")
            return {}

    def _compare_schemas(self, sqlite_schema: Dict, postgres_schema: Dict) -> List[str]:
        """Compare schemas and return list of differences."""
        differences = []
        
        # Check for missing tables
        sqlite_tables = set(sqlite_schema.keys())
        postgres_tables = set(postgres_schema.keys())
        
        missing_in_postgres = sqlite_tables - postgres_tables
        missing_in_sqlite = postgres_tables - sqlite_tables
        
        for table in missing_in_postgres:
            differences.append(f"Table '{table}' exists in SQLite but not in PostgreSQL")
        
        for table in missing_in_sqlite:
            differences.append(f"Table '{table}' exists in PostgreSQL but not in SQLite")
        
        # Check for column differences in common tables
        common_tables = sqlite_tables & postgres_tables
        
        for table in common_tables:
            sqlite_cols = {col[0]: col[1] for col in sqlite_schema[table]["columns"]}
            postgres_cols = {col[0]: col[1] for col in postgres_schema[table]["columns"]}
            
            sqlite_col_names = set(sqlite_cols.keys())
            postgres_col_names = set(postgres_cols.keys())
            
            missing_in_postgres_cols = sqlite_col_names - postgres_col_names
            missing_in_sqlite_cols = postgres_col_names - sqlite_col_names
            
            for col in missing_in_postgres_cols:
                differences.append(f"Column '{table}.{col}' exists in SQLite but not in PostgreSQL")
            
            for col in missing_in_sqlite_cols:
                differences.append(f"Column '{table}.{col}' exists in PostgreSQL but not in SQLite")
        
        return differences

    async def migrate_database(self, from_db: str, to_db: str) -> Dict[str, Any]:
        """Migrate data from one database to another"""
        try:
            # This is a placeholder for database migration logic
            # In a real implementation, you would:
            # 1. Connect to source database
            # 2. Export data from all tables
            # 3. Connect to target database
            # 4. Import data to target database
            
            result = {
                "status": "success",
                "from_database": from_db,
                "to_database": to_db,
                "migrated_records": 0,
                "errors": [],
                "warnings": ["Migration functionality not fully implemented"],
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Mock migration from {from_db} to {to_db}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during migration: {e}")
            raise

    async def get_database_contents(self, database_type: str = "current", limit: int = 100) -> Dict[str, Any]:
        """Get database contents for display"""
        try:
            from ..data_ingestion.service import DataIngestionService
            
            # Determine which database to connect to based on database_type
            if database_type.lower() in ["postgres", "postgresql"]:
                # For PostgreSQL, we need to create a direct connection
                import json
                
                conn = get_postgres_connection()
                if conn is None:
                    return {"error": "PostgreSQL not configured in environment variables", "contents": []}
                
                cursor = conn.cursor()
                
                # Get record count
                cursor.execute("SELECT COUNT(*) FROM data_sources")
                count = cursor.fetchone()[0]
                
                # Get actual records
                cursor.execute(f"SELECT id, name, type, created_by, created_at FROM data_sources LIMIT {limit}")
                records = cursor.fetchall()
                
                # Convert to expected format
                formatted_records = []
                for record in records:
                    formatted_records.append({
                        "id": record[0],
                        "name": record[1], 
                        "type": record[2],
                        "created_by": record[3],
                        "created_at": record[4].isoformat() if record[4] else None
                    })
                
                conn.close()
                
                contents = {
                    "database_type": "postgresql",
                    "tables": {
                        "data_sources": {
                            "count": count,
                            "records": formatted_records
                        }
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
            else:
                # For SQLite (current/default), use the existing session
                async for session in get_async_session():
                    service = DataIngestionService(session)
                    sources = await service.list_data_sources(limit=limit)
                    
                    contents = {
                        "database_type": "sqlite",
                        "tables": {
                            "data_sources": {
                                "count": len(sources),
                                "records": [
                                    {
                                        "id": source.id,
                                        "name": source.name,
                                        "type": source.type,
                                        "created_by": source.created_by,
                                        "created_at": source.created_at.isoformat() if source.created_at else None
                                    } for source in sources
                                ]
                            }
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                    break
                
            self.logger.info(f"Retrieved {database_type} database contents: {contents['tables']['data_sources']['count']} records")
            return contents
                
        except Exception as e:
            self.logger.error(f"Error getting database contents: {e}")
            raise