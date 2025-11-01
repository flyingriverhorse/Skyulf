"""
File System Utilities for Data Source Management

Provides safe file and folder deletion functionality with backup options,
plus file cleanup and maintenance utilities.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def safe_delete_path(
    path: Union[str, Path], 
    force_delete: bool = True,
    files_only: bool = True
) -> bool:
    """
    Safely delete a file or directory.
    
    Args:
        path: Path to the file or directory to delete
        force_delete: If True, delete immediately (default True, backup removed)
        files_only: If True, only delete files, not directories
        
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    path = Path(path)
    
    if not path.exists():
        logger.warning(f"Path does not exist: {path}")
        return False
    
    # SAFETY: Log what type of path we're dealing with
    logger.info(f"safe_delete_path called with: {path}")
    logger.info(f"  - Exists: {path.exists()}")
    logger.info(f"  - Is file: {path.is_file()}")
    logger.info(f"  - Is directory: {path.is_dir()}")
    logger.info(f"  - files_only: {files_only}")
    
    # If files_only is True and this is a directory, skip it
    if files_only and path.is_dir():
        logger.warning(f"SKIPPING directory deletion (files_only=True): {path}")
        return False
        
    try:
        return _delete_immediately(path, files_only=files_only)
    except Exception as e:
        logger.error(f"Failed to delete {path}: {e}")
        return False

def _delete_immediately(path: Path, files_only: bool = True) -> bool:
    """Immediately delete file or directory."""
    try:
        if path.is_file():
            path.unlink()
            logger.info(f"✓ FILE DELETED SUCCESSFULLY: {path}")
            return True
        elif path.is_dir():
            if files_only:
                logger.warning(f"⚠ DIRECTORY DELETION BLOCKED (files_only=True): {path}")
                return False
            else:
                logger.warning(f"⚠ DELETING DIRECTORY (files_only=False): {path}")
                import shutil
                shutil.rmtree(path)
                logger.info(f"✓ DIRECTORY DELETED SUCCESSFULLY: {path}")
                return True
        else:
            logger.warning(f"⚠ Unknown path type: {path}")
            return False
    except Exception as e:
        logger.error(f"❌ FAILED TO DELETE {path}: {e}")
        return False


def cleanup_empty_directories(base_path: Union[str, Path]) -> int:
    """
    Remove empty directories recursively from base_path.
    
    Args:
        base_path: Base directory to start cleanup from
        
    Returns:
        int: Number of directories removed
    """
    base_path = Path(base_path)
    removed_count = 0
    
    if not base_path.exists() or not base_path.is_dir():
        return removed_count
    
    try:
        # Walk bottom-up to remove empty directories
        for dirpath, dirnames, filenames in os.walk(base_path, topdown=False):
            dir_path = Path(dirpath)
            
            # Skip the base directory itself
            if dir_path == base_path:
                continue
                
            # Check if directory is empty
            if not any(dir_path.iterdir()):
                try:
                    dir_path.rmdir()
                    logger.info(f"Removed empty directory: {dir_path}")
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Failed to remove empty directory {dir_path}: {e}")
                    
    except Exception as e:
        logger.error(f"Error during directory cleanup: {e}")
    
    return removed_count


def extract_file_path_from_source(source_data: dict) -> Optional[Path]:
    """
    Extract the file path from a data source record.
    
    Args:
        source_data: Data source dictionary from database
        
    Returns:
        Optional[Path]: Path to the file if it exists
    """
    # Check various possible path fields
    path_candidates = [
        source_data.get('file_path'),
        source_data.get('path'),
        source_data.get('source_path'),
        source_data.get('location'),
        source_data.get('file_location'),
    ]
    
    # Check connection_info for file path (this is where uploaded files store their paths)
    connection_info = source_data.get('connection_info', {})
    if isinstance(connection_info, dict):
        path_candidates.extend([
            connection_info.get('file_path'),  # This is the main one for uploaded files
            connection_info.get('filepath'),
            connection_info.get('path'),
            connection_info.get('upload_path'),
        ])
    
    # Also check config field (legacy data sources)
    config = source_data.get('config', {})
    if isinstance(config, dict):
        path_candidates.extend([
            config.get('file_path'),
            config.get('filepath'),
            config.get('path'),
            config.get('upload_path'),
        ])
    
    for candidate in path_candidates:
        if candidate and isinstance(candidate, (str, Path)):
            path = Path(candidate)
            if path.exists() and path.is_file():  # Only return if it's actually a file
                logger.debug(f"Found file path: {path}")
                return path
    
    # Also check if source_name contains a path
    source_name = source_data.get('source_name', '')
    if source_name and ('/' in source_name or '\\' in source_name):
        path = Path(source_name)
        if path.exists():
            return path
    
    return None


def cleanup_old_files(
    directory: Union[str, Path],
    max_files: int = 10,
    max_age_days: int = 7,
    file_pattern: str = "*"
) -> dict:
    """
    Clean up old files in a directory based on count and age limits.
    
    Args:
        directory: Directory to clean up
        max_files: Maximum number of files to keep (newest are preserved)
        max_age_days: Remove files older than this many days
        file_pattern: File pattern to match (default: all files)
        
    Returns:
        dict: Summary of cleanup operation
    """
    directory = Path(directory)
    if not directory.exists():
        return {"status": "skipped", "reason": "directory_not_found", "files_removed": 0}
    
    try:
        # Get all files matching the pattern
        all_files = list(directory.glob(file_pattern))
        files_only = [f for f in all_files if f.is_file()]
        
        if not files_only:
            return {"status": "success", "files_removed": 0, "reason": "no_files_found"}
        
        removed_count = 0
        removed_files = []
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # Sort files by modification time (newest first)
        files_by_mtime = sorted(files_only, key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Remove files older than max_age_days
        for file_path in files_only:
            file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_mtime < cutoff_date:
                try:
                    file_path.unlink()
                    removed_files.append(str(file_path))
                    removed_count += 1
                    logger.info(f"Removed old file (age): {file_path}")
                except Exception as e:
                    logger.error(f"Failed to remove old file {file_path}: {e}")
        
        # Remove excess files beyond max_files limit
        remaining_files = [f for f in files_by_mtime if f.exists()]
        if len(remaining_files) > max_files:
            excess_files = remaining_files[max_files:]
            for file_path in excess_files:
                try:
                    file_path.unlink()
                    removed_files.append(str(file_path))
                    removed_count += 1
                    logger.info(f"Removed excess file (count): {file_path}")
                except Exception as e:
                    logger.error(f"Failed to remove excess file {file_path}: {e}")
        
        return {
            "status": "success",
            "files_removed": removed_count,
            "removed_files": removed_files,
            "remaining_files": len([f for f in files_only if f.exists()]) - removed_count
        }
        
    except Exception as e:
        logger.error(f"Error during file cleanup in {directory}: {e}")
        return {"status": "error", "error": str(e), "files_removed": 0}


def cleanup_uploads_directory(
    uploads_dir: Union[str, Path],
    max_files: int = 10,
    max_age_days: int = 7,
    file_extensions: Optional[List[str]] = None
) -> dict:
    """
    Clean up uploaded files based on settings.
    
    Args:
        uploads_dir: Uploads directory path
        max_files: Maximum files to keep
        max_age_days: Maximum age in days
        file_extensions: List of extensions to clean (default: common data files)
        
    Returns:
        dict: Cleanup summary
    """
    if file_extensions is None:
        file_extensions = ['.csv', '.xlsx', '.xls', '.json', '.parquet', '.txt', '.pkl', '.feather']
    
    uploads_dir = Path(uploads_dir)
    total_removed = 0
    results = {}
    
    # Clean up files by extension
    for ext in file_extensions:
        pattern = f"*{ext}"
        result = cleanup_old_files(uploads_dir, max_files, max_age_days, pattern)
        results[ext] = result
        total_removed += result.get('files_removed', 0)
    
    # Also clean up any temporary files
    temp_result = cleanup_old_files(uploads_dir, max_files=5, max_age_days=1, file_pattern="*.tmp")
    results['temp_files'] = temp_result
    total_removed += temp_result.get('files_removed', 0)
    
    return {
        "status": "success",
        "total_files_removed": total_removed,
        "results_by_extension": results,
        "directory": str(uploads_dir)
    }