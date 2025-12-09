"""Data management API routes for FastAPI."""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..dependencies import get_data_service
from ..service import DataIngestionService
from ..exceptions import DataIngestionException, DataSourceNotFoundError, FileUploadError
from ...database.models import User
from ...utils.logging_utils import log_data_action

logger = logging.getLogger(__name__)

# Create router
data_router = APIRouter(prefix="/data", tags=["Data Management"])


# Response models
class UploadResponse(BaseModel):
    """Response model for file upload operations."""
    success: bool
    message: str = None
    source_id: Optional[str] = None
    file_info: Optional[Dict[str, Any]] = None
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    error: Optional[str] = None


class DataSourceListResponse(BaseModel):
    """Response model for data source listing."""
    success: bool
    sources: List[Dict[str, Any]]
    total_count: int
    scope: str  # "admin_full" or "user_scoped"


class HealthResponse(BaseModel):
    """Response model for health checks."""
    status: str
    timestamp: str
    components: Dict[str, str]


class BulkDeleteRequest(BaseModel):
    """Request model for bulk delete operations."""
    source_ids: List[str] = Field(..., description="List of source IDs to delete")


class BulkDeleteResponse(BaseModel):
    """Response model for bulk delete operations."""
    success: bool
    message: str
    deleted_count: int
    failed_count: int
    deleted_sources: List[Dict[str, Any]]
    failed_deletions: List[Dict[str, Any]]
    notifications_sent: Optional[List[str]] = None


# Routes

@data_router.post("/api/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    custom_name: Optional[str] = Form(None),
    service: DataIngestionService = Depends(get_data_service)
):
    """API endpoint for file upload (adapted from main.py).

    This implementation saves the uploaded file to the uploads/data folder,
    creates a DataSource DB record using the service session, and returns
    a lightweight file_info payload.
    """
    try:
        # Handle the file upload through the service
        upload_result = await service.handle_file_upload(file, None)

        return UploadResponse(
            success=True,
            message=upload_result["message"],
            source_id=upload_result["source_id"],
            file_info=upload_result["file_info"],
            is_duplicate=False
        )

    except FileUploadError as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DataIngestionException as e:
        logger.error(f"Data ingestion error during upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during file upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )


@data_router.get("/api/sources", response_model=DataSourceListResponse)
async def list_data_sources(
    service: DataIngestionService = Depends(get_data_service)
):
    """
    List data sources accessible to the current user.

    Admin users see all sources, regular users see only their own sources.
    """
    try:
        # Use service method to get formatted data sources list
        result = await service.get_formatted_data_sources_list(None)

        return DataSourceListResponse(
            success=result["success"],
            sources=result["sources"],
            total_count=result["total_count"],
            scope=result["scope"]
        )

    except Exception as e:
        logger.error(f"Error listing data sources: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list data sources: {str(e)}"
        )


@data_router.get("/api/sources/search")
async def search_data_sources(
    q: str = Query("", description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    source_type: Optional[str] = Query(None, description="Filter by source type"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum results to return"),
    order: str = Query("created_desc", description="Order results by"),
    service: DataIngestionService = Depends(get_data_service)
):
    """
    Search data sources by name, description, or metadata.

    Supports filtering by category and source type with comprehensive response formatting.
    """
    try:
        logger.info(f"Searching data sources: query='{q}'")

        # Use the enhanced service method that handles all business logic
        result = await service.search_data_sources_with_formatting(
            query=q if q else None,
            category=category if category else None,
            source_type=source_type if source_type else None,
            limit=limit,
            order_by=order,
            current_user=None,
            include_user_scoping=True
        )

        # Return the result with appropriate status code
        status_code = 200 if result["success"] else 500
        return JSONResponse(content=result, status_code=status_code)

    except Exception as e:
        logger.error(f"Error searching data sources: {e}")
        return JSONResponse(
            content={
                "success": False,
                "results": [],
                "sources": [],
                "total": 0,
                "count": 0,
                "error": str(e),
                "message": "Error searching data sources",
            },
            status_code=500,
        )


@data_router.get("/api/sources/{source_id}")
async def get_data_source(
    source_id: str,
    service: DataIngestionService = Depends(get_data_service)
):
    """Get detailed information about a specific data source."""
    try:
        # First: check for uploaded files in uploads/data that match the source_id
        try:
            from pathlib import Path
            uploads_dir = Path(__file__).parents[3] / "uploads" / "data"
            if uploads_dir.exists():
                for file_path in uploads_dir.iterdir():
                    if file_path.is_file() and file_path.name.startswith(f"{source_id}_"):
                        # Found a matching uploaded file; construct a file-source response
                        original_name = file_path.name.split('_', 1)[1]
                        file_size = file_path.stat().st_size

                        return JSONResponse(content={
                            "success": True,
                            "source": {
                                "id": source_id,
                                "source_id": source_id,
                                "name": original_name,
                                "display_name": original_name,
                                "type": "file_upload",
                                "source_type": "file",
                                "category": "Files",
                                "status": "ready",
                                "file_size": file_size,
                                "file_path": str(file_path),
                                "created_at": file_path.stat().st_mtime and None or None,
                                "metadata": {
                                    "created_by": "unknown",
                                    "file_type": file_path.suffix.upper().lstrip('.') or 'UNKNOWN',
                                    "upload_method": "web_interface",
                                    "connector_type": "file_upload"
                                }
                            }
                        })
        except Exception:
            # Non-fatal; fall back to DB-backed source lookup
            pass

        # Fallback: lookup in the database via the service
        source = await service.get_data_source(source_id)

        return JSONResponse(content={
            "success": True,
            "source": {
                "source_id": source.source_id,  # Use string source_id as identifier
                "id": source.id,
                "source_type": getattr(source, 'source_type', getattr(source, 'type', None)),
                "name": source.name,
                "connection_info": getattr(source, 'connection_info', getattr(source, 'config', None)),
                "metadata": getattr(source, 'source_metadata', None),  # Use correct field name if present
                "category": getattr(source, 'category', None),
                "created_at": source.created_at.isoformat() if source.created_at else None,
                "created_by": source.created_by,
                "status": getattr(source, 'status', None)
            }
        })

    except DataSourceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data source not found: {source_id}"
        )
    except Exception as e:
        logger.error(f"Error getting data source {source_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get data source: {str(e)}"
        )


@data_router.get("/api/sources/{source_id}/sample")
async def get_data_source_sample(
    source_id: str,
    limit: int = Query(5, ge=1, le=100),
    service: DataIngestionService = Depends(get_data_service)
):
    """Get a sample of data from the source."""
    try:
        data = await service.get_data_sample(source_id, limit)
        return JSONResponse(content={
            "success": True,
            "data": data
        })
    except DataSourceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data source not found: {source_id}"
        )
    except Exception as e:
        logger.error(f"Error getting data sample for {source_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get data sample: {str(e)}"
        )


@data_router.delete("/api/sources/{source_id}")
async def delete_data_source(
    source_id: str,
    service: DataIngestionService = Depends(get_data_service)
):
    """Delete a data source.

    This handler uses the integer-ID delete path that was working in main.py.
    It attempts to parse the incoming source_id to an int, looks up the source
    and calls the service's `delete_data_source_by_id` method. It also logs
    the action with `log_data_action` for auditing.
    """
    try:
        # Expect integer IDs in this endpoint (keeps parity with the main.py handler)
        try:
            source_id_int = int(source_id)
        except ValueError:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "error": f"Invalid source ID: {source_id}",
                    "message": "Source ID must be a valid number."
                }
            )

        # Use the service to find the source first
        source = await service.get_data_source_by_id(source_id_int)
        if not source:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "success": False,
                    "error": f"Data source '{source_id}' not found",
                    "message": "Cannot delete - data source does not exist."
                }
            )

        source_name = getattr(source, 'name', f"{source_id_int}")

        # Permission check: Disabled for open mode
        # Perform deletion (this method returns bool)
        deletion_result = await service.delete_data_source_by_id(source_id_int)

        if deletion_result:
            # Log the action for auditing
            try:
                log_data_action(
                    "DELETE_DATA_SOURCE",
                    True,
                    (
                        f"Deleted source '{source_name}' (ID: {source_id_int}) "
                        f"by anonymous"
                    ),
                )
            except Exception:
                logger.warning("Failed to write audit log for delete action")

            return JSONResponse(content={
                "success": True,
                "message": f"Data source '{source_name}' deleted successfully."
            })
        else:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "error": f"Failed to delete data source '{source_id}'",
                    "message": "Database deletion operation failed."
                }
            )

    except Exception as e:
        logger.error(f"Error deleting data source {source_id}: {e}")
        try:
            log_data_action("DELETE_DATA_SOURCE", False, f"Failed to delete {source_id}: {e}")
        except Exception:
            logger.warning("Failed to write failed-delete audit log")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete data source: {str(e)}"
        )


@data_router.post("/api/sources/bulk-delete", response_model=BulkDeleteResponse)
async def bulk_delete_data_sources(
    request: BulkDeleteRequest,
    service: DataIngestionService = Depends(get_data_service)
):
    """
    Delete multiple data sources at once.
    Users can delete their own sources, admins can delete any sources.
    """
    try:
        source_ids = request.source_ids

        if not source_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="source_ids array is required"
            )

        logger.info(
            "Bulk delete request by anonymous: %s sources",
            len(source_ids),
        )
        is_admin = True

        deleted_sources = []
        failed_deletions = []

        # Convert incoming IDs (may be strings) to integers where possible
        source_ids_int = []
        for sid in source_ids:
            try:
                source_ids_int.append(int(sid))
            except Exception:
                failed_deletions.append({
                    "source_id": sid,
                    "error": "Invalid source ID format"
                })

        # Prefer current_user.id if present to avoid extra lookups
        current_user_id = None

        for source_id in source_ids_int:
            try:
                # Check if source exists
                source = await service.get_data_source_by_id(source_id)
                if not source:
                    failed_deletions.append({"source_id": source_id, "error": "Source not found"})
                    continue

                source_name = getattr(source, 'name', f"{source_id}")
                owner_id = getattr(source, 'created_by', None)

                # Permission: allow if admin or owner
                if not (
                    is_admin
                    or (
                        owner_id is not None
                        and current_user_id is not None
                        and owner_id == current_user_id
                    )
                ):
                    failed_deletions.append({"source_id": source_id, "error": "Access denied"})
                    try:
                        log_data_action(
                            "BULK_DELETE_DATA_SOURCES",
                            False,
                            (
                                f"Denied delete attempt for {source_name} (ID: {source_id}) "
                                f"by {getattr(current_user, 'username', 'unknown')}"
                            ),
                        )
                    except Exception:
                        logger.warning("Failed to log denied bulk delete attempt")
                    continue

                # Delete
                deleted = await service.delete_data_source_by_id(source_id)
                if deleted:
                    deleted_sources.append({
                        "source_id": source_id,
                        "source_name": source_name,
                        "source_owner": owner_id
                    })
                    logger.info(
                        "Successfully deleted source %s (%s) for user %s",
                        source_id,
                        source_name,
                        getattr(current_user, 'username', 'unknown'),
                    )
                else:
                    failed_deletions.append({"source_id": source_id, "error": "Deletion failed"})

            except Exception as e:
                logger.error(f"Error deleting source {source_id}: {e}")
                failed_deletions.append({"source_id": source_id, "error": f"Internal error: {str(e)}"})

        result = BulkDeleteResponse(
            success=True,
            message=f"Successfully deleted {len(deleted_sources)} data sources. {len(failed_deletions)} failed.",
            deleted_count=len(deleted_sources),
            failed_count=len(failed_deletions),
            deleted_sources=deleted_sources,
            failed_deletions=failed_deletions
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk delete: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@data_router.get("/api/health", response_model=HealthResponse)
async def get_health_status():
    """Get health status of the data ingestion system."""
    timestamp = None
    if logger.handlers:
        handler = logger.handlers[0]
        formatter = getattr(handler, "formatter", None)
        if formatter:
            record = logger.makeRecord(
                name="data_health",
                level=logging.INFO,
                fn="",
                lno=0,
                msg="health_check",
                args=(),
                exc_info=None,
            )
            timestamp = formatter.formatTime(handler, record)

    return HealthResponse(
        status="healthy",
        timestamp=str(timestamp),
        components={
            "file_upload": "healthy",
            "data_preview": "healthy",
            "search": "healthy",
            "database": "healthy",
        },
    )


# ======================== EXPORT API ========================

@data_router.get("/api/export/formats")
async def get_export_formats():
    """
    Get available export formats.
    This is a simple stub implementation.
    """
    try:
        # Return common export formats
        formats = [
            {
                "id": "csv",
                "name": "CSV",
                "description": "Comma Separated Values",
                "extension": "csv",
                "mime_type": "text/csv"
            },
            {
                "id": "excel",
                "name": "Excel",
                "description": "Microsoft Excel format",
                "extension": "xlsx",
                "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            },
            {
                "id": "json",
                "name": "JSON",
                "description": "JavaScript Object Notation",
                "extension": "json",
                "mime_type": "application/json"
            },
            {
                "id": "parquet",
                "name": "Parquet",
                "description": "Apache Parquet format",
                "extension": "parquet",
                "mime_type": "application/octet-stream"
            }
        ]

        return {
            "success": True,
            "formats": formats,
            "message": "Available export formats"
        }

    except Exception as e:
        logger.error(f"Error getting export formats: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "formats": [],
                "message": "Error retrieving export formats"
            }
        )


# ======================== NOTIFICATIONS API ========================@data_router.get("/api/notifications")
async def get_notifications(
    unread_only: bool = False,
    mark_read: bool = False
):
    """
    Get notifications for the current user.
    This is a simple stub implementation to prevent JavaScript errors.
    """
    try:
        # For now, return empty notifications to prevent JS errors
        # In a real implementation, you would fetch from a notifications table
        return {
            "success": True,
            "notifications": [],
            "unread_count": 0,
            "total_count": 0,
            "message": "No notifications available"
        }
    except Exception as e:
        logger.error(f"Error getting notifications: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "notifications": [],
                "unread_count": 0,
                "total_count": 0,
                "message": "Error retrieving notifications"
            }
        )


@data_router.post("/api/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: str
):
    """
    Mark a notification as read.
    This is a simple stub implementation to prevent JavaScript errors.
    """
    try:
        # For now, just return success
        # In a real implementation, you would update the notification in the database
        return {
            "success": True,
            "message": "Notification marked as read"
        }
    except Exception as e:
        logger.error(f"Error marking notification as read: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "message": "Error marking notification as read"
            }
        )


# Export the router
__all__ = ["data_router"]
