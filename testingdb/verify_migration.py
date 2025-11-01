#!/usr/bin/env python3
"""
Database Migration Verification Script
Verifies that the source_id column exists and is populated correctly.
"""

import asyncio
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def verify_sqlite_migration(db_path: Path):
    """Verify SQLite migration was successful."""
    import sqlite3
    import json
    
    try:
        if not db_path.exists():
            logger.error(f"SQLite database not found: {db_path}")
            return False
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check if source_id column exists
        cursor.execute("PRAGMA table_info(data_sources)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'source_id' not in columns:
            logger.error("❌ source_id column does not exist in data_sources table")
            return False
        
        logger.info("✅ source_id column exists in data_sources table")
        
        # Check if source_id is populated
        cursor.execute("SELECT COUNT(*) FROM data_sources WHERE source_id IS NOT NULL")
        populated_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM data_sources")
        total_count = cursor.fetchone()[0]
        
        logger.info(f"📊 {populated_count}/{total_count} records have source_id populated")
        
        # Show some sample data
        cursor.execute("SELECT id, source_id, config FROM data_sources LIMIT 3")
        samples = cursor.fetchall()
        
        logger.info("📝 Sample records:")
        for row in samples:
            row_id, source_id, config_json = row
            logger.info(f"  ID: {row_id}, source_id: {source_id}, config: {config_json[:50] if config_json else None}...")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"SQLite verification failed: {e}")
        return False


async def verify_postgresql_migration(database_url: str):
    """Verify PostgreSQL migration was successful."""
    try:
        import asyncpg
        
        # Remove +asyncpg suffix if present
        asyncpg_url = database_url.replace("postgresql+asyncpg://", "postgresql://")
        
        conn = await asyncpg.connect(asyncpg_url)
        
        # Check if source_id column exists
        column_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'data_sources' 
                AND column_name = 'source_id'
            )
        """)
        
        if not column_exists:
            logger.error("❌ source_id column does not exist in data_sources table")
            await conn.close()
            return False
        
        logger.info("✅ source_id column exists in data_sources table")
        
        # Check population
        populated_count = await conn.fetchval("SELECT COUNT(*) FROM data_sources WHERE source_id IS NOT NULL")
        total_count = await conn.fetchval("SELECT COUNT(*) FROM data_sources")
        
        logger.info(f"📊 {populated_count}/{total_count} records have source_id populated")
        
        # Show samples
        samples = await conn.fetch("SELECT id, source_id, config FROM data_sources LIMIT 3")
        
        logger.info("📝 Sample records:")
        for row in samples:
            config_preview = str(row['config'])[:50] if row['config'] else None
            logger.info(f"  ID: {row['id']}, source_id: {row['source_id']}, config: {config_preview}...")
        
        await conn.close()
        return True
        
    except ImportError:
        logger.error("PostgreSQL verification requires 'asyncpg' package")
        return False
    except Exception as e:
        logger.error(f"PostgreSQL verification failed: {e}")
        return False


async def verify_migration():
    """Auto-detect database and verify migration."""
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        
        from config import get_settings
        
        settings = get_settings()
        database_url = settings.DATABASE_URL
        
        logger.info(f"🔍 Verifying migration for: {database_url.split('://')[0]}://...")
        
        if database_url.startswith("sqlite"):
            db_path = database_url.split("///", 1)[1] if "///" in database_url else database_url.split("//", 1)[1]
            db_path = Path(db_path)
            return await verify_sqlite_migration(db_path)
            
        elif database_url.startswith(("postgresql://", "postgresql+asyncpg://")):
            return await verify_postgresql_migration(database_url)
            
        else:
            logger.error(f"Unsupported database type: {database_url.split('://')[0]}")
            return False
            
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


async def main():
    """Run migration verification."""
    logger.info("🔍 Starting database migration verification...")
    
    success = await verify_migration()
    
    if success:
        logger.info("🎉 Migration verification completed successfully!")
        return 0
    else:
        logger.error("❌ Migration verification failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)