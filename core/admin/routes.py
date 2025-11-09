"""
FastAPI Admin Routes

Modern async admin endpoints for system management, migrated from Flask.
Provides comprehensive admin functionality with proper authentication and logging.
"""

import logging
import os
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

from ..auth.dependencies import get_current_admin_user
from ..auth.auth_core import User
from .services import AsyncAdminDataService, AsyncAdminUserService, AsyncAdminSystemService
from ..utils.maintenance import SystemMaintenanceService

logger = logging.getLogger(__name__)

# Create the admin router
admin_router = APIRouter(prefix="/admin", tags=["admin"])


# Pydantic models for request/response
class BulkDeleteRequest(BaseModel):
    """Request model for bulk delete operations"""
    source_ids: List[str] = Field(..., description="List of source IDs to delete")


class UserCreateRequest(BaseModel):
    """Request model for creating a new user"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None
    is_admin: bool = False
    is_verified: bool = False


class UserUpdateRequest(BaseModel):
    """Request model for updating user details"""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[str] = Field(None, pattern=r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
    full_name: Optional[str] = None
    is_admin: Optional[bool] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None


class AdminActionLog(BaseModel):
    """Model for admin action logging"""
    action: str
    success: bool = True
    details: Optional[str] = None
    user: Optional[str] = None
    ip_address: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


def log_admin_action(
    request: Request,
    user: User,
    action: str,
    success: bool = True,
    details: Optional[str] = None,
    **kwargs
):
    """Security logging for admin actions"""
    try:
        client_ip = request.client.host if request.client else "unknown"
        user_info = f"user:{user.username}" if user else "user:anonymous"
        status_text = "SUCCESS" if success else "FAILED"

        log_msg = f"ADMIN_ACTION [{status_text}] {action} | {user_info} | IP:{client_ip}"
        if details:
            log_msg += f" | {details}"
        if kwargs:
            log_msg += f" | {kwargs}"

        if success:
            logger.info(log_msg)
        else:
            logger.warning(log_msg)
    except Exception as e:
        logger.error(f"Error logging admin action: {e}")


# Initialize services
data_service = AsyncAdminDataService()
user_service = AsyncAdminUserService()
system_service = AsyncAdminSystemService()
maintenance_service = SystemMaintenanceService()


# =============================================================================
# Data Ingestion Management APIs
# =============================================================================

@admin_router.get("/api/data-ingestion/stats")
async def get_data_ingestion_stats(
    request: Request,
    current_user: User = Depends(get_current_admin_user)
):
    """Get comprehensive data ingestion statistics"""
    try:
        stats = await data_service.get_data_ingestion_stats()
        log_admin_action(
            request,
            current_user,
            "DATA_INGESTION_STATS_VIEW",
            details=f"sources:{stats.get('total_sources', 0)}"
        )
        return {"success": True, "stats": stats}
    except Exception as e:
        logger.error(f"Error getting data ingestion stats: {e}")
        log_admin_action(
            request,
            current_user,
            "DATA_INGESTION_STATS_VIEW",
            success=False,
            details=f"error:{str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get statistics"
        )


@admin_router.get("/api/data-ingestion/sources")
async def get_data_ingestion_sources(
    request: Request,
    limit: int = 1000,
    offset: int = 0,
    current_user: User = Depends(get_current_admin_user)
):
    """Get detailed list of all data sources"""
    try:
        sources = await data_service.get_data_sources_detailed(limit=limit, offset=offset)
        log_admin_action(
            request,
            current_user,
            "DATA_INGESTION_SOURCES_VIEW",
            details=f"count:{len(sources)}"
        )
        return {"success": True, "sources": sources, "total": len(sources)}
    except Exception as e:
        logger.error(f"Error getting data ingestion sources: {e}")
        log_admin_action(
            request,
            current_user,
            "DATA_INGESTION_SOURCES_VIEW",
            success=False,
            details=f"error:{str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get data sources"
        )


@admin_router.delete("/api/data-ingestion/source/{source_id}")
async def delete_data_source(
    source_id: str,
    request: Request,
    current_user: User = Depends(get_current_admin_user)
):
    """Delete a data source by ID"""
    try:
        # Validate source_id as integer
        try:
            source_id_int = int(source_id)
        except ValueError:
            log_admin_action(
                request,
                current_user,
                "DATA_SOURCE_DELETE",
                success=False,
                details=f"invalid_source_id:{source_id}"
            )
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid source ID")

        from core.database.engine import get_async_session
        from core.data_ingestion.service import DataIngestionService

        async for session in get_async_session():
            service = DataIngestionService(session)

            try:
                source = await service.get_data_source_by_id(source_id_int)
                if not source:
                    log_admin_action(
                        request,
                        current_user,
                        "DATA_SOURCE_DELETE",
                        success=False,
                        details=f"source_id:{source_id} not_found"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Data source not found",
                    )

                source_name = getattr(source, 'name', str(source_id))

                deletion_result = await service.delete_data_source_by_id(source_id_int)
                if deletion_result:
                    log_admin_action(
                        request,
                        current_user,
                        "DATA_SOURCE_DELETE",
                        details=f"source_id:{source_id},source_name:{source_name}"
                    )
                    return {"success": True, "message": f"Data source '{source_name}' deleted successfully"}
                else:
                    log_admin_action(
                        request,
                        current_user,
                        "DATA_SOURCE_DELETE",
                        success=False,
                        details=f"source_id:{source_id} deletion_failed"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to delete data source",
                    )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error deleting data source {source_id} from admin: {e}")
                log_admin_action(
                    request,
                    current_user,
                    "DATA_SOURCE_DELETE",
                    success=False,
                    details=f"source_id:{source_id} error:{str(e)}"
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e),
                )

        # If no session loop, return error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete data source",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete_data_source handler: {e}")
        log_admin_action(
            request,
            current_user,
            "DATA_SOURCE_DELETE",
            success=False,
            details=f"source_id:{source_id} error:{str(e)}"
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete data source")


@admin_router.post("/api/data-ingestion/sources/bulk-delete")
async def bulk_delete_data_sources(
    bulk_request: BulkDeleteRequest,
    request: Request,
    current_user: User = Depends(get_current_admin_user)
):
    """Bulk delete multiple data sources"""
    try:
        if not bulk_request.source_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No source IDs provided"
            )

        result = await data_service.bulk_delete_data_sources(bulk_request.source_ids)

        log_admin_action(
            request,
            current_user,
            "DATA_SOURCES_BULK_DELETE",
            details=f"requested:{result['total_requested']} deleted:{result['deleted']}"
        )

        return {
            "success": True,
            "deleted": result["deleted"],
            "total_requested": result["total_requested"],
            "errors": result["errors"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk delete: {e}")
        log_admin_action(
            request,
            current_user,
            "DATA_SOURCES_BULK_DELETE",
            success=False,
            details=f"error:{str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform bulk delete"
        )


# =============================================================================
# User Management APIs
# =============================================================================

@admin_router.get("/api/users/stats")
async def get_user_stats(
    request: Request,
    current_user: User = Depends(get_current_admin_user)
):
    """Get user statistics and activity"""
    try:
        stats = await user_service.get_user_stats_detailed()

        log_admin_action(
            request,
            current_user,
            "USER_STATS_VIEW",
            details=f"users:{stats.get('total_users', 0)}"
        )

        return {"success": True, "stats": stats}

    except Exception as e:
        logger.error(f"Error in admin_users_stats: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "stats": {
                "total_users": 0,
                "active_users": 0,
                "inactive_users": 0,
                "users_by_role": {"admin": 0, "user": 0},
                "data_sources_by_user": {},
                "user_table": [],
                "data_sources_summary": {
                    "total_sources": 0,
                    "average_per_user": 0,
                    "by_user": {}
                }
            }
        }


@admin_router.get("/api/users")
async def get_users(
    request: Request,
    skip: int = 0,
    limit: int = 100,
    search: Optional[str] = None,
    is_active: Optional[bool] = None,
    is_admin: Optional[bool] = None,
    current_user: User = Depends(get_current_admin_user)
):
    """Get paginated list of users with optional filtering"""
    try:
        result = await user_service.get_users(
            skip=skip,
            limit=limit,
            search=search,
            is_active=is_active,
            is_admin=is_admin
        )

        log_admin_action(
            request,
            current_user,
            "USERS_LIST_VIEW",
            details=f"count:{len(result.get('users', []))}"
        )

        return {"success": True, **result}

    except Exception as e:
        logger.error(f"Error getting users: {e}")
        log_admin_action(
            request,
            current_user,
            "USERS_LIST_VIEW",
            success=False,
            details=f"error:{str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get users"
        )


@admin_router.post("/api/users")
async def create_user(
    request: Request,
    user_data: UserCreateRequest,
    current_user: User = Depends(get_current_admin_user)
):
    """Create a new user"""
    try:
        result = await user_service.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name,
            is_admin=user_data.is_admin,
            is_verified=user_data.is_verified
        )

        log_admin_action(
            request,
            current_user,
            "USER_CREATE",
            details=f"username:{user_data.username},admin:{user_data.is_admin}"
        )

        return {"success": True, "user": result}

    except ValueError as e:
        logger.warning(f"Validation error creating user: {e}")
        log_admin_action(
            request,
            current_user,
            "USER_CREATE",
            success=False,
            details=f"validation_error:{str(e)},username:{getattr(user_data, 'username', None)}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error creating user: {e}")
        log_admin_action(
            request,
            current_user,
            "USER_CREATE",
            success=False,
            details=f"error:{str(e)},username:{getattr(user_data, 'username', None)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}",
        )


@admin_router.patch("/api/users/{user_id}")
async def update_user(
    user_id: int,
    request: Request,
    current_user: User = Depends(get_current_admin_user)
):
    """PATCH /admin/api/users/{user_id} - Update user (used for pause/activate)"""
    try:
        body = await request.json()

        result = await user_service.update_user(user_id, body)

        action = "updated"
        if 'is_active' in body:
            action = "activated" if body['is_active'] else "paused"

        log_admin_action(
            request,
            current_user,
            "USER_UPDATE",
            details=f"user_id:{user_id},action:{action}"
        )

        return {
            "success": True,
            "message": f"User {result.get('username', 'user')} has been {action}",
            "user": result
        }

    except ValueError as e:
        logger.warning(f"User not found for update: {e}")
        log_admin_action(
            request,
            current_user,
            "USER_UPDATE",
            success=False,
            details=f"user_id:{user_id} not_found:{str(e)}"
        )
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        log_admin_action(
            request,
            current_user,
            "USER_UPDATE",
            success=False,
            details=f"user_id:{user_id} error:{str(e)}"
        )
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.delete("/api/users/{user_id}")
async def delete_user(
    user_id: int,
    request: Request,
    current_user: User = Depends(get_current_admin_user)
):
    """Delete a user by ID"""
    try:
        # Prevent self-deletion
        if hasattr(current_user, 'id') and current_user.id == user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own account"
            )

        result = await user_service.delete_user(user_id)

        log_admin_action(
            request,
            current_user,
            "USER_DELETE",
            details=f"user_id:{user_id},deleted_data_sources:{result.get('deleted_data_sources', 0)}"
        )

        return {
            "success": True,
            "message": result.get('message', 'User deleted successfully'),
            "deleted_data_sources": result.get('deleted_data_sources', 0)
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {e}")
        log_admin_action(
            request,
            current_user,
            "USER_DELETE",
            success=False,
            details=f"error:{str(e)},user_id:{user_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user: {str(e)}"
        )


# =============================================================================
# System Management APIs
# =============================================================================

@admin_router.get("/api/system/info")
async def get_system_info(
    request: Request,
    current_user: User = Depends(get_current_admin_user)
):
    """Get system information and status"""
    try:
        info = await system_service.get_system_info()
        log_admin_action(
            request,
            current_user,
            "SYSTEM_INFO_VIEW"
        )
        # Return 'info' to match admin_dashboard.js which expects result.info
        return {"success": True, "info": info}
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        log_admin_action(
            request,
            current_user,
            "SYSTEM_INFO_VIEW",
            success=False,
            details=f"error:{str(e)}"
        )
        # Return a JSON response with an 'error' field so the frontend can display it
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": "Failed to get system information"}
        )


@admin_router.get("/api/system/health")
async def get_system_health_report(
    request: Request,
    current_user: User = Depends(get_current_admin_user)
):
    """Get comprehensive system health report"""
    try:
        report = await system_service.get_system_health_report()
        log_admin_action(
            request,
            current_user,
            "SYSTEM_HEALTH_CHECK",
            details=f"status:{report.get('status', 'unknown')}"
        )
        return {"success": True, "health_report": report}
    except Exception as e:
        logger.error(f"Error getting system health report: {e}")
        log_admin_action(
            request,
            current_user,
            "SYSTEM_HEALTH_CHECK",
            success=False,
            details=f"error:{str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system health report"
        )


# ---------------------------
# Application log endpoints
# ---------------------------


@admin_router.get('/api/logs/tail')
async def admin_get_logs_tail(
    request: Request,
    lines: int = 100,
    current_user: User = Depends(get_current_admin_user)
):
    """Return the last N lines of the application log file (fastapi_app.log)."""
    try:
        from pathlib import Path

        log_file_path = Path('logs') / 'fastapi_app.log'

        if not log_file_path.exists():
            log_admin_action(
                request,
                current_user,
                'APP_LOGS_TAIL',
                details='no_log_file',
            )
            return {
                "success": True,
                "tail": "No application log file found.",
                "file_info": {
                    "path": str(log_file_path),
                    "exists": False,
                },
            }

        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            all_lines = f.readlines()

        recent = all_lines[-lines:] if len(all_lines) > lines else all_lines
        tail_text = ''.join(recent).strip() or 'Log file is empty or contains no readable entries.'

        log_admin_action(
            request,
            current_user,
            'APP_LOGS_TAIL',
            details=f'lines_returned:{len(recent)}',
        )

        return {
            "success": True,
            "tail": tail_text,
            "file_info": {
                "path": str(log_file_path),
                "size": f"{log_file_path.stat().st_size} bytes",
                "total_lines": len(all_lines),
                "showing_last": min(lines, len(all_lines))
            }
        }

    except Exception as e:
        logger.error(f"Error reading application log tail: {e}")
        log_admin_action(
            request,
            current_user,
            'APP_LOGS_TAIL',
            success=False,
            details=f'error:{e}',
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read application log",
        )


@admin_router.get('/api/logs/download')
async def admin_download_log(
    request: Request,
    current_user: User = Depends(get_current_admin_user)
):
    """Download the application log file as an attachment."""
    try:
        from fastapi.responses import FileResponse
        from pathlib import Path

        log_file_path = Path('logs') / 'fastapi_app.log'
        if not log_file_path.exists():
            log_admin_action(
                request,
                current_user,
                'APP_LOGS_DOWNLOAD',
                details='no_log_file',
            )
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "success": False,
                    "error": "Log file not found",
                },
            )

        log_admin_action(request, current_user, 'APP_LOGS_DOWNLOAD')
        return FileResponse(
            path=str(log_file_path),
            filename='fastapi_app.log',
            media_type='text/plain',
        )

    except Exception as e:
        logger.error(f"Error preparing log download: {e}")
        log_admin_action(
            request,
            current_user,
            'APP_LOGS_DOWNLOAD',
            success=False,
            details=f'error:{e}',
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to prepare log download",
        )

# =============================================================================
# Debug and Monitoring Endpoints
# =============================================================================


@admin_router.get("/debug/logs/tail")
async def get_recent_logs(
    request: Request,
    lines: int = 100,
    current_user: User = Depends(get_current_admin_user)
):
    """Get recent application logs for debugging"""
    try:
        log_file_path = os.path.join("logs", "app.log")

        if not os.path.exists(log_file_path):
            return {"success": True, "logs": [], "message": "No log file found"}

        # Read last N lines from log file
        with open(log_file_path, "r", encoding="utf-8") as f:
            log_lines = f.readlines()
            recent_logs = log_lines[-lines:] if len(log_lines) > lines else log_lines

        log_admin_action(
            request,
            current_user,
            "DEBUG_LOGS_VIEW",
            details=f"lines:{len(recent_logs)}"
        )

        return {
            "success": True,
            "logs": [line.strip() for line in recent_logs],
            "total_lines": len(recent_logs)
        }
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        log_admin_action(
            request,
            current_user,
            "DEBUG_LOGS_VIEW",
            success=False,
            details=f"error:{str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read application logs"
        )


@admin_router.get("/ping")
async def admin_ping(
    request: Request,
    current_user: User = Depends(get_current_admin_user)
):
    """Simple admin-only ping to verify admin routes are working"""
    try:
        log_admin_action(request, current_user, "ADMIN_PING")
        return {
            "success": True,
            "message": "Admin routes registered and working",
            "timestamp": datetime.now().isoformat(),
            "user": current_user.username
        }
    except Exception as e:
        logger.error(f"Admin ping failed: {e}")
        log_admin_action(
            request,
            current_user,
            "ADMIN_PING",
            success=False,
            details=f"error:{str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin ping failed"
        )


# Database Management APIs
@admin_router.get("/api/database/status")
async def get_database_status(
    request: Request,
    current_user: User = Depends(get_current_admin_user)
):
    """Get database connection status for both SQLite and PostgreSQL"""
    try:
        status = await system_service.get_database_status()

        log_admin_action(
            request,
            current_user,
            "DATABASE_STATUS_CHECK"
        )

        return {"success": True, "status": status}

    except Exception as e:
        logger.error(f"Error getting database status: {e}")
        log_admin_action(
            request,
            current_user,
            "DATABASE_STATUS_CHECK",
            success=False,
            details=f"error:{str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get database status"
        )


@admin_router.post("/api/database/migrate")
async def migrate_database(
    request: Request,
    from_db: str,
    to_db: str,
    current_user: User = Depends(get_current_admin_user)
):
    """Migrate data from one database to another"""
    try:
        result = await system_service.migrate_database(from_db, to_db)

        log_admin_action(
            request,
            current_user,
            "DATABASE_MIGRATION",
            details=f"from:{from_db} to:{to_db}"
        )

        return {"success": True, "migration_result": result}

    except Exception as e:
        logger.error(f"Error during database migration: {e}")
        log_admin_action(
            request,
            current_user,
            "DATABASE_MIGRATION",
            success=False,
            details=f"error:{str(e)} from:{from_db} to:{to_db}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to migrate database"
        )


@admin_router.get("/api/database/contents")
async def get_database_contents(
    request: Request,
    database_type: str = "current",
    limit: int = 100,
    current_user: User = Depends(get_current_admin_user)
):
    """Get database contents for display"""
    try:
        contents = await system_service.get_database_contents(database_type, limit)

        log_admin_action(
            request,
            current_user,
            "DATABASE_CONTENTS_VIEW",
            details=f"database:{database_type} limit:{limit}"
        )

        return {"success": True, "database_contents": contents}

    except Exception as e:
        logger.error(f"Error getting database contents: {e}")
        log_admin_action(
            request,
            current_user,
            "DATABASE_CONTENTS_VIEW",
            success=False,
            details=f"error:{str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get database contents"
        )

# =============================================================================
# System Maintenance APIs
# =============================================================================


@admin_router.get("/api/maintenance/status")
async def get_maintenance_status(
    request: Request,
    current_user: User = Depends(get_current_admin_user)
):
    """Get system maintenance status and statistics"""
    try:
        status = maintenance_service.get_stats()

        log_admin_action(
            request,
            current_user,
            "MAINTENANCE_STATUS_VIEW",
            details=f"last_cleanup:{status.get('last_cleanup_time', 'never')}"
        )

        return {"success": True, "status": status}

    except Exception as e:
        logger.error(f"Error getting maintenance status: {e}")
        log_admin_action(
            request,
            current_user,
            "MAINTENANCE_STATUS_VIEW",
            success=False,
            details=f"error:{str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get maintenance status"
        )


@admin_router.post("/api/maintenance/run")
async def run_maintenance(
    request: Request,
    current_user: User = Depends(get_current_admin_user)
):
    """Manually trigger system maintenance and cleanup"""
    try:
        results = await maintenance_service.run_full_maintenance()

        # Extract summary info for logging
        total_deleted = sum(result.files_removed for result in results.values() if result.success)
        successful_operations = sum(1 for result in results.values() if result.success)

        log_admin_action(
            request,
            current_user,
            "MAINTENANCE_RUN",
            details=f"files_deleted:{total_deleted},successful_operations:{successful_operations}"
        )

        return {"success": True, "results": results}

    except Exception as e:
        logger.error(f"Error running maintenance: {e}")
        log_admin_action(
            request,
            current_user,
            "MAINTENANCE_RUN",
            success=False,
            details=f"error:{str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run maintenance: {str(e)}"
        )

# === HTML TEMPLATE ROUTES ===


@admin_router.get("/users", response_class=HTMLResponse)
async def admin_users_page(request: Request):
    """Admin users page with configurable authentication"""
    from core.auth.page_security import create_page_response

    # Use the centralized page response helper
    return create_page_response(
        request=request,
        template_name="admin/dashboard.html",
        extra_context={
            "page_title": "Admin Users Management"
        }
    )


@admin_router.get("/dashboard", response_class=HTMLResponse)
@admin_router.get("", response_class=HTMLResponse)  # Handle /admin directly
async def admin_dashboard_page(request: Request):
    """Admin dashboard page with configurable authentication"""
    from core.auth.page_security import create_page_response
    from config import get_settings

    # Get database configuration for the dashboard
    settings = get_settings()
    db_config = settings.get_database_config_info()

    # Use the centralized page response helper
    return create_page_response(
        request=request,
        template_name="admin/dashboard.html",
        extra_context={
            "page_title": "Admin Dashboard",
            "db_config": db_config
        }
    )
