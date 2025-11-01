#!/usr/bin/env python3
"""
Database migration script to add source_id column to data_sources table.
This will add the column and populate it from existing config data.
"""

import asyncio
import logging
import sqlite3
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def add_source_id_column():
    """Add source_id column to data_sources table and populate it."""
    try:
        # Connect to the SQLite database directly
        db_path = Path(__file__).parent / "mlops_database.db"
        
        if not db_path.exists():
            logger.error(f"Database file not found: {db_path}")
            return False
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check if column already exists
        cursor.execute("PRAGMA table_info(data_sources)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'source_id' in columns:
            logger.info("source_id column already exists")
        else:
            # Add the column
            logger.info("Adding source_id column to data_sources table")
            cursor.execute("ALTER TABLE data_sources ADD COLUMN source_id VARCHAR(50)")
            logger.info("Column added successfully")
        
        # Create index for the new column
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS ix_data_sources_source_id ON data_sources (source_id)")
            logger.info("Index created for source_id column")
        except sqlite3.Error as e:
            logger.warning(f"Could not create index: {e}")
        
        # Populate source_id from config for records that have it
        cursor.execute("""
            SELECT id, config 
            FROM data_sources 
            WHERE source_id IS NULL 
            AND config IS NOT NULL
        """)
        
        rows_to_update = cursor.fetchall()
        logger.info(f"Found {len(rows_to_update)} rows to populate with source_id")
        
        updated_count = 0
        for row_id, config_json in rows_to_update:
            if config_json:
                try:
                    import json
                    config = json.loads(config_json)
                    if isinstance(config, dict) and 'source_id' in config:
                        source_id = config['source_id']
                        cursor.execute(
                            "UPDATE data_sources SET source_id = ? WHERE id = ?",
                            (source_id, row_id)
                        )
                        updated_count += 1
                        logger.info(f"Updated row {row_id} with source_id: {source_id}")
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Could not parse config for row {row_id}: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Migration completed successfully. Updated {updated_count} records.")
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


async def main():
    """Run the migration."""
    logger.info("Starting database migration to add source_id column...")
    
    success = await add_source_id_column()
    
    if success:
        logger.info("Migration completed successfully!")
        return 0
    else:
        logger.error("Migration failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)