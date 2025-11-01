#!/usr/bin/env python3
"""
Migration script to populate source_id column from existing config data.
This script will extract source_id values from the config JSON and set them as the proper source_id column.
"""

import asyncio
import logging
from sqlalchemy import text
from core.database.engine import get_async_session, init_db
from core.database.models import DataSource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def migrate_source_ids():
    """Migrate source_id values from config to source_id column."""
    try:
        # Initialize database first
        await init_db()
        
        async for session in get_async_session():
            # Get all data sources that have source_id in config but no source_id column value
            result = await session.execute(
                text("""
                    SELECT id, name, config 
                    FROM data_sources 
                    WHERE source_id IS NULL 
                    AND JSON_EXTRACT(config, '$.source_id') IS NOT NULL
                """)
            )
            sources_to_update = result.fetchall()
            
            logger.info(f"Found {len(sources_to_update)} sources to migrate")
            
            updated_count = 0
            for row in sources_to_update:
                source_id_from_config = row.config.get('source_id') if isinstance(row.config, dict) else None
                
                if source_id_from_config:
                    # Update the source_id column
                    await session.execute(
                        text("UPDATE data_sources SET source_id = :source_id WHERE id = :id"),
                        {"source_id": source_id_from_config, "id": row.id}
                    )
                    
                    logger.info(f"Updated source '{row.name}' (ID: {row.id}) with source_id: {source_id_from_config}")
                    updated_count += 1
            
            if updated_count > 0:
                await session.commit()
                logger.info(f"Successfully migrated {updated_count} source_id values")
            else:
                logger.info("No sources needed migration")
            
            return updated_count
            
    except Exception as e:
        logger.error(f"Error during migration: {e}")
        return 0


async def main():
    """Run the migration."""
    logger.info("Starting source_id migration...")
    
    try:
        updated = await migrate_source_ids()
        logger.info(f"Migration completed. Updated {updated} records.")
        return 0
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)