"""
System Maintenance Service

Provides comprehensive file cleanup and system maintenance operations
with configurable scheduling and detailed logging.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from core.utils.file_utils import cleanup_old_files, cleanup_uploads_directory
from config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class CleanupResult:
    """Result of a cleanup operation."""
    operation: str
    success: bool
    files_removed: int
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class SystemMaintenanceService:
    """
    Comprehensive system maintenance service.

    Handles:
    - Log file cleanup
    - Temporary file cleanup
    - Export file cleanup
    - Model cache cleanup
    - Upload directory cleanup
    - Configurable scheduling
    """

    def __init__(self):
        self.settings = get_settings()
        self.last_cleanup = None
        self.cleanup_stats = {
            "total_runs": 0,
            "total_files_removed": 0,
            "last_run": None,
            "operations": {}
        }

    async def run_full_maintenance(self) -> Dict[str, CleanupResult]:
        """
        Run all enabled maintenance operations.

        Returns:
            Dict mapping operation names to their results
        """
        if not self.settings.SYSTEM_CLEANUP_ENABLED:
            logger.info("System cleanup is disabled in configuration")
            return {}

        logger.info("🧹 Starting comprehensive system maintenance...")

        results = {}
        total_files_removed = 0

        # Run all cleanup operations
        operations = [
            self._cleanup_logs,
            self._cleanup_temp_files,
            self._cleanup_exports,
            self._cleanup_models,
            self._cleanup_uploads,
            self._cleanup_data_files
        ]

        for operation in operations:
            try:
                result = await operation()
                results[result.operation] = result

                if result.success:
                    total_files_removed += result.files_removed
                    logger.info(f"✅ {result.operation}: Removed {result.files_removed} files")
                else:
                    logger.warning(f"❌ {result.operation}: Failed - {result.error_message}")

            except Exception as e:
                operation_name = operation.__name__.replace('_cleanup_', '').replace('_', ' ').title()
                logger.error(f"❌ {operation_name} cleanup failed: {e}")
                results[operation_name] = CleanupResult(
                    operation=operation_name,
                    success=False,
                    files_removed=0,
                    error_message=str(e)
                )

        # Update statistics
        self._update_stats(total_files_removed)

        logger.info(f"🎯 System maintenance completed. Total files removed: {total_files_removed}")
        return results

    async def _cleanup_logs(self) -> CleanupResult:
        """Clean up old log files."""
        if not self.settings.LOG_CLEANUP_ENABLED:
            return CleanupResult(
                operation="Log Cleanup",
                success=True,
                files_removed=0,
                error_message="Disabled in configuration"
            )

        logs_dir = Path("logs")
        if not logs_dir.exists():
            return CleanupResult(
                operation="Log Cleanup",
                success=True,
                files_removed=0,
                error_message="Logs directory does not exist"
            )

        try:
            result = cleanup_old_files(
                directory=logs_dir,
                max_files=self.settings.LOG_CLEANUP_MAX_FILES,
                max_age_days=self.settings.LOG_CLEANUP_MAX_AGE_DAYS,
                file_pattern=self.settings.LOG_FILES_PATTERN
            )

            return CleanupResult(
                operation="Log Cleanup",
                success=result["status"] == "success",
                files_removed=result.get("files_removed", 0),
                details=result
            )

        except Exception as e:
            return CleanupResult(
                operation="Log Cleanup",
                success=False,
                files_removed=0,
                error_message=str(e)
            )

    async def _cleanup_temp_files(self) -> CleanupResult:
        """Clean up temporary processing files."""
        if not self.settings.TEMP_CLEANUP_ENABLED:
            return CleanupResult(
                operation="Temp Cleanup",
                success=True,
                files_removed=0,
                error_message="Disabled in configuration"
            )

        temp_dir = Path(self.settings.TEMP_DIR)
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = cleanup_old_files(
                directory=temp_dir,
                max_files=self.settings.TEMP_CLEANUP_MAX_FILES,
                max_age_days=self.settings.TEMP_CLEANUP_MAX_AGE_DAYS,
                file_pattern=self.settings.TEMP_FILES_PATTERN
            )

            return CleanupResult(
                operation="Temp Cleanup",
                success=result["status"] == "success",
                files_removed=result.get("files_removed", 0),
                details=result
            )

        except Exception as e:
            return CleanupResult(
                operation="Temp Cleanup",
                success=False,
                files_removed=0,
                error_message=str(e)
            )

    async def _cleanup_exports(self) -> CleanupResult:
        """Clean up old export files."""
        if not self.settings.EXPORT_CLEANUP_ENABLED:
            return CleanupResult(
                operation="Export Cleanup",
                success=True,
                files_removed=0,
                error_message="Disabled in configuration"
            )

        export_dir = Path(self.settings.EXPORT_DIR)
        export_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = cleanup_old_files(
                directory=export_dir,
                max_files=self.settings.EXPORT_CLEANUP_MAX_FILES,
                max_age_days=self.settings.EXPORT_CLEANUP_MAX_AGE_DAYS,
                file_pattern=self.settings.EXPORT_FILES_PATTERN
            )

            return CleanupResult(
                operation="Export Cleanup",
                success=result["status"] == "success",
                files_removed=result.get("files_removed", 0),
                details=result
            )

        except Exception as e:
            return CleanupResult(
                operation="Export Cleanup",
                success=False,
                files_removed=0,
                error_message=str(e)
            )

    async def _cleanup_models(self) -> CleanupResult:
        """Clean up old model files."""
        if not self.settings.MODEL_CLEANUP_ENABLED:
            return CleanupResult(
                operation="Model Cleanup",
                success=True,
                files_removed=0,
                error_message="Disabled in configuration"
            )

        models_dir = Path(self.settings.MODELS_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = cleanup_old_files(
                directory=models_dir,
                max_files=self.settings.MODEL_CLEANUP_MAX_FILES,
                max_age_days=self.settings.MODEL_CLEANUP_MAX_AGE_DAYS,
                file_pattern=self.settings.MODEL_FILES_PATTERN
            )

            return CleanupResult(
                operation="Model Cleanup",
                success=result["status"] == "success",
                files_removed=result.get("files_removed", 0),
                details=result
            )

        except Exception as e:
            return CleanupResult(
                operation="Model Cleanup",
                success=False,
                files_removed=0,
                error_message=str(e)
            )

    async def _cleanup_uploads(self) -> CleanupResult:
        """Clean up old upload files."""
        if not self.settings.UPLOAD_CLEANUP_ENABLED:
            return CleanupResult(
                operation="Upload Cleanup",
                success=True,
                files_removed=0,
                error_message="Disabled in configuration"
            )

        upload_dir = Path(self.settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = cleanup_uploads_directory(
                uploads_dir=upload_dir,
                max_files=self.settings.UPLOAD_CLEANUP_MAX_FILES,
                max_age_days=self.settings.UPLOAD_CLEANUP_MAX_AGE_DAYS,
                file_extensions=None  # Use defaults from file_utils
            )

            return CleanupResult(
                operation="Upload Cleanup",
                success=result["status"] == "success",
                files_removed=result.get("total_files_removed", 0),
                details=result
            )

        except Exception as e:
            return CleanupResult(
                operation="Upload Cleanup",
                success=False,
                files_removed=0,
                error_message=str(e)
            )

    async def _cleanup_data_files(self) -> CleanupResult:
        """Clean up data files and remove associated data sources from database."""
        if not self.settings.DATA_CLEANUP_ENABLED:
            return CleanupResult(
                operation="Data Cleanup",
                success=True,
                files_removed=0,
                details={"status": "disabled"}
            )

        logger.info("🗂️ Starting data files cleanup...")

        # Data directory path
        data_dir = Path("uploads/data")
        if not data_dir.exists():
            return CleanupResult(
                operation="Data Cleanup",
                success=True,
                files_removed=0,
                details={"status": "success", "reason": "no_data_directory"}
            )

        try:
            result = cleanup_old_files(
                directory=data_dir,
                max_files=self.settings.DATA_CLEANUP_MAX_FILES,
                max_age_days=self.settings.DATA_CLEANUP_MAX_AGE_DAYS,
                file_pattern=self.settings.DATA_FILES_PATTERN
            )

            return CleanupResult(
                operation="Data Cleanup",
                success=result["status"] == "success",
                files_removed=result.get("files_removed", 0),
                details={
                    **result,
                    "criteria": {
                        "max_files": self.settings.DATA_CLEANUP_MAX_FILES,
                        "max_age_days": self.settings.DATA_CLEANUP_MAX_AGE_DAYS,
                        "pattern": self.settings.DATA_FILES_PATTERN
                    }
                }
            )

        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
            return CleanupResult(
                operation="Data Cleanup",
                success=False,
                files_removed=0,
                error_message=str(e)
            )

    def _update_stats(self, files_removed: int):
        """Update cleanup statistics."""
        self.cleanup_stats["total_runs"] += 1
        self.cleanup_stats["total_files_removed"] += files_removed
        self.cleanup_stats["last_run"] = datetime.now().isoformat()
        self.last_cleanup = datetime.now()

    def get_stats(self) -> Dict[str, Any]:
        """Get cleanup statistics."""
        return {
            **self.cleanup_stats,
            "next_scheduled": self._get_next_scheduled_time(),
            "scheduler_enabled": self.settings.CLEANUP_SCHEDULER_ENABLED,
            "configuration": {
                "system_cleanup_enabled": self.settings.SYSTEM_CLEANUP_ENABLED,
                "cleanup_on_startup": self.settings.CLEANUP_ON_STARTUP,
                "schedule_hours": self.settings.CLEANUP_SCHEDULE_HOURS,
                "operations_enabled": {
                    "logs": self.settings.LOG_CLEANUP_ENABLED,
                    "temp": self.settings.TEMP_CLEANUP_ENABLED,
                    "exports": self.settings.EXPORT_CLEANUP_ENABLED,
                    "models": self.settings.MODEL_CLEANUP_ENABLED,
                    "uploads": self.settings.UPLOAD_CLEANUP_ENABLED,
                    "data": self.settings.DATA_CLEANUP_ENABLED,
                },
                "operation_details": {
                    "logs": {
                        "enabled": self.settings.LOG_CLEANUP_ENABLED,
                        "max_files": self.settings.LOG_CLEANUP_MAX_FILES,
                        "max_age_days": self.settings.LOG_CLEANUP_MAX_AGE_DAYS,
                        "pattern": self.settings.LOG_FILES_PATTERN,
                        "description": "Clean up old log files"
                    },
                    "temp": {
                        "enabled": self.settings.TEMP_CLEANUP_ENABLED,
                        "max_files": self.settings.TEMP_CLEANUP_MAX_FILES,
                        "max_age_days": self.settings.TEMP_CLEANUP_MAX_AGE_DAYS,
                        "pattern": self.settings.TEMP_FILES_PATTERN,
                        "description": "Clean up temporary processing files"
                    },
                    "exports": {
                        "enabled": self.settings.EXPORT_CLEANUP_ENABLED,
                        "max_files": self.settings.EXPORT_CLEANUP_MAX_FILES,
                        "max_age_days": self.settings.EXPORT_CLEANUP_MAX_AGE_DAYS,
                        "pattern": self.settings.EXPORT_FILES_PATTERN,
                        "description": "Clean up old export files"
                    },
                    "models": {
                        "enabled": self.settings.MODEL_CLEANUP_ENABLED,
                        "max_files": self.settings.MODEL_CLEANUP_MAX_FILES,
                        "max_age_days": self.settings.MODEL_CLEANUP_MAX_AGE_DAYS,
                        "pattern": self.settings.MODEL_FILES_PATTERN,
                        "description": "Clean up cached model files"
                    },
                    "uploads": {
                        "enabled": self.settings.UPLOAD_CLEANUP_ENABLED,
                        "max_files": self.settings.UPLOAD_CLEANUP_MAX_FILES,
                        "max_age_days": self.settings.UPLOAD_CLEANUP_MAX_AGE_DAYS,
                        "description": "Clean up upload directories comprehensively"
                    },
                    "data": {
                        "enabled": self.settings.DATA_CLEANUP_ENABLED,
                        "max_files": self.settings.DATA_CLEANUP_MAX_FILES,
                        "max_age_days": self.settings.DATA_CLEANUP_MAX_AGE_DAYS,
                        "pattern": self.settings.DATA_FILES_PATTERN,
                        "remove_data_sources": self.settings.DATA_CLEANUP_REMOVE_DATA_SOURCES,
                        "description": "Clean up data files and remove associated database records"
                    }
                }
            }
        }

    def _get_next_scheduled_time(self) -> Optional[str]:
        """Get the next scheduled cleanup time."""
        if not self.settings.CLEANUP_SCHEDULER_ENABLED or not self.last_cleanup:
            return None

        next_time = self.last_cleanup + timedelta(hours=self.settings.CLEANUP_SCHEDULE_HOURS)
        return next_time.isoformat()

    def should_run_cleanup(self) -> bool:
        """Check if cleanup should run based on schedule."""
        if not self.settings.CLEANUP_SCHEDULER_ENABLED:
            return False

        if not self.last_cleanup:
            return True

        time_since_last = datetime.now() - self.last_cleanup
        return time_since_last.total_seconds() >= (self.settings.CLEANUP_SCHEDULE_HOURS * 3600)


# Global maintenance service instance
maintenance_service = SystemMaintenanceService()


async def run_startup_maintenance():
    """Run maintenance on application startup if enabled."""
    settings = get_settings()

    if settings.CLEANUP_ON_STARTUP and settings.SYSTEM_CLEANUP_ENABLED:
        logger.info("🚀 Running startup maintenance...")
        try:
            results = await maintenance_service.run_full_maintenance()
            total_files = sum(r.files_removed for r in results.values())
            logger.info(f"✅ Startup maintenance completed. Removed {total_files} files total")
        except Exception as e:
            logger.error(f"❌ Startup maintenance failed: {e}")
    else:
        logger.info("⏭️ Startup maintenance disabled")


async def run_scheduled_maintenance():
    """Run scheduled maintenance if it's time."""
    if maintenance_service.should_run_cleanup():
        logger.info("⏰ Running scheduled maintenance...")
        try:
            results = await maintenance_service.run_full_maintenance()
            total_files = sum(r.files_removed for r in results.values())
            logger.info(f"✅ Scheduled maintenance completed. Removed {total_files} files total")
        except Exception as e:
            logger.error(f"❌ Scheduled maintenance failed: {e}")


def get_maintenance_service() -> SystemMaintenanceService:
    """Get the global maintenance service instance."""
    return maintenance_service
