#!/usr/bin/env python3
"""
Comprehensive Database Migration Script - Add source_id column
Supports both SQLite and PostgreSQL databases
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def migrate_sqlite(db_path: Path) -> bool:
    """Migrate SQLite database to add source_id column."""
    import sqlite3
    import json
    
    try:
        if not db_path.exists():
            logger.error(f"SQLite database file not found: {db_path}")
            return False
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check if column already exists
        cursor.execute("PRAGMA table_info(data_sources)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'source_id' in columns:
            logger.info("SQLite: source_id column already exists")
        else:
            # Add the column
            logger.info("SQLite: Adding source_id column to data_sources table")
            cursor.execute("ALTER TABLE data_sources ADD COLUMN source_id VARCHAR(50)")
            logger.info("SQLite: Column added successfully")
        
        # Create index for the new column
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS ix_data_sources_source_id ON data_sources (source_id)")
            logger.info("SQLite: Index created for source_id column")
        except sqlite3.Error as e:
            logger.warning(f"SQLite: Could not create index: {e}")
        
        # Populate source_id from config for records that have it
        cursor.execute("""
            SELECT id, config 
            FROM data_sources 
            WHERE source_id IS NULL 
            AND config IS NOT NULL
        """)
        
        rows_to_update = cursor.fetchall()
        logger.info(f"SQLite: Found {len(rows_to_update)} rows to populate with source_id")
        
        updated_count = 0
        for row_id, config_json in rows_to_update:
            if config_json:
                try:
                    config = json.loads(config_json)
                    if isinstance(config, dict) and 'source_id' in config:
                        source_id = config['source_id']
                        cursor.execute(
                            "UPDATE data_sources SET source_id = ? WHERE id = ?",
                            (source_id, row_id)
                        )
                        updated_count += 1
                        logger.info(f"SQLite: Updated row {row_id} with source_id: {source_id}")
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"SQLite: Could not parse config for row {row_id}: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"SQLite migration completed successfully. Updated {updated_count} records.")
        return True
        
    except Exception as e:
        logger.error(f"SQLite migration failed: {e}")
        return False


async def migrate_postgresql(database_url: str) -> bool:
    """Migrate PostgreSQL database to add source_id column."""
    try:
        import asyncpg
        import json
        
        # Parse connection details from URL
        # Format: postgresql+asyncpg://user:password@host:port/database
        if not database_url.startswith(("postgresql://", "postgresql+asyncpg://")):
            logger.error(f"Invalid PostgreSQL URL format: {database_url}")
            return False
        
        # Extract asyncpg URL (remove +asyncpg suffix if present)
        asyncpg_url = database_url.replace("postgresql+asyncpg://", "postgresql://")
        
        conn = await asyncpg.connect(asyncpg_url)
        
        # Check if column already exists
        column_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'data_sources' 
                AND column_name = 'source_id'
            )
        """)
        
        if column_exists:
            logger.info("PostgreSQL: source_id column already exists")
        else:
            # Add the column
            logger.info("PostgreSQL: Adding source_id column to data_sources table")
            await conn.execute("ALTER TABLE data_sources ADD COLUMN source_id VARCHAR(50)")
            logger.info("PostgreSQL: Column added successfully")
        
        # Create index for the new column
        try:
            await conn.execute("CREATE INDEX IF NOT EXISTS ix_data_sources_source_id ON data_sources (source_id)")
            logger.info("PostgreSQL: Index created for source_id column")
        except Exception as e:
            logger.warning(f"PostgreSQL: Could not create index: {e}")
        
        # Populate source_id from config for records that have it
        rows_to_update = await conn.fetch("""
            SELECT id, config 
            FROM data_sources 
            WHERE source_id IS NULL 
            AND config IS NOT NULL
        """)
        
        logger.info(f"PostgreSQL: Found {len(rows_to_update)} rows to populate with source_id")
        
        updated_count = 0
        for row in rows_to_update:
            row_id = row['id']
            config_json = row['config']
            
            if config_json:
                try:
                    # PostgreSQL stores JSON as dict already if using JSON column type
                    if isinstance(config_json, dict):
                        config = config_json
                    else:
                        config = json.loads(config_json)
                        
                    if isinstance(config, dict) and 'source_id' in config:
                        source_id = config['source_id']
                        await conn.execute(
                            "UPDATE data_sources SET source_id = $1 WHERE id = $2",
                            source_id, row_id
                        )
                        updated_count += 1
                        logger.info(f"PostgreSQL: Updated row {row_id} with source_id: {source_id}")
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"PostgreSQL: Could not parse config for row {row_id}: {e}")
        
        await conn.close()
        
        logger.info(f"PostgreSQL migration completed successfully. Updated {updated_count} records.")
        return True
        
    except ImportError:
        logger.error("PostgreSQL migration requires 'asyncpg' package. Install with: pip install asyncpg")
        return False
    except Exception as e:
        logger.error(f"PostgreSQL migration failed: {e}")
        return False


async def detect_and_migrate() -> bool:
    """Auto-detect database type and run appropriate migration."""
    try:
        # Import config to detect database type
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        
        from config import get_settings
        
        settings = get_settings()
        database_url = settings.DATABASE_URL
        
        logger.info(f"Detected database URL: {database_url.split('://')[0]}://...")
        
        if database_url.startswith("sqlite"):
            # Extract database path from SQLite URL
            # Format: sqlite+aiosqlite:///./mlops_database.db
            db_path = database_url.split("///", 1)[1] if "///" in database_url else database_url.split("//", 1)[1]
            db_path = Path(db_path)
            
            logger.info(f"Running SQLite migration for: {db_path}")
            return await migrate_sqlite(db_path)
            
        elif database_url.startswith(("postgresql://", "postgresql+asyncpg://")):
            logger.info("Running PostgreSQL migration")
            return await migrate_postgresql(database_url)
            
        else:
            logger.error(f"Unsupported database type: {database_url.split('://')[0]}")
            return False
            
    except Exception as e:
        logger.error(f"Auto-detection failed: {e}")
        return False


async def main():
    """Run the comprehensive database migration."""
    logger.info("Starting comprehensive database migration (SQLite + PostgreSQL support)...")
    
    success = await detect_and_migrate()
    
    if success:
        logger.info("🎉 Migration completed successfully!")
        return 0
    else:
        logger.error("❌ Migration failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)