#!/usr/bin/env python3
"""
Migration script to update existing data sources with proper created_by user IDs.
This script will map existing created_by values to actual user IDs.
"""

import asyncio
import logging
from sqlalchemy import text, select
from core.database.engine import get_async_session, init_db
from core.database.models import DataSource, User

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def migrate_created_by_fields():
    """Migrate existing data sources to use proper user IDs."""
    try:
        # Initialize database first
        await init_db()
        
        async for session in get_async_session():
            # First, get all users to create a mapping
            users_result = await session.execute(select(User))
            users = users_result.scalars().all()
            
            # Create username -> user_id mapping
            username_to_id = {user.username: user.id for user in users}
            logger.info(f"Found {len(users)} users: {list(username_to_id.keys())}")
            
            # Get all data sources
            sources_result = await session.execute(select(DataSource))
            sources = sources_result.scalars().all()
            
            logger.info(f"Found {len(sources)} data sources to potentially update")
            
            # Track updates
            updated_count = 0
            
            for source in sources:
                if source.created_by is None:
                    # For sources with no created_by, try to infer from common usernames
                    # or set to a default admin user if one exists
                    default_user_id = None
                    
                    # Try common admin usernames
                    for admin_name in ['admin', 'admin1', 'administrator', 'system']:
                        if admin_name in username_to_id:
                            default_user_id = username_to_id[admin_name]
                            break
                    
                    # If no admin found, use the first user
                    if not default_user_id and username_to_id:
                        default_user_id = list(username_to_id.values())[0]
                        logger.info(f"Using first available user ID {default_user_id} as default")
                    
                    if default_user_id:
                        source.created_by = default_user_id
                        updated_count += 1
                        logger.info(f"Updated data source '{source.name}' (ID: {source.id}) with created_by = {default_user_id}")
            
            # Commit the changes
            if updated_count > 0:
                await session.commit()
                logger.info(f"Successfully updated {updated_count} data sources")
            else:
                logger.info("No data sources needed updating")
            
            # Verify the updates
            sources_result = await session.execute(select(DataSource))
            sources = sources_result.scalars().all()
            
            logger.info("\n=== Data Sources after migration ===")
            for source in sources:
                creator_name = "Unknown"
                if source.created_by:
                    for username, user_id in username_to_id.items():
                        if user_id == source.created_by:
                            creator_name = username
                            break
                
                logger.info(f"- {source.name} (ID: {source.id}) created by: {creator_name} (user_id: {source.created_by})")
            
            return updated_count
            
    except Exception as e:
        logger.error(f"Error migrating created_by fields: {e}")
        import traceback
        traceback.print_exc()
        return 0


async def main():
    """Run the migration."""
    logger.info("🚀 Starting created_by migration...")
    
    try:
        updated_count = await migrate_created_by_fields()
        if updated_count > 0:
            logger.info(f"✅ Migration completed successfully! Updated {updated_count} data sources")
        else:
            logger.info("✅ Migration completed - no updates needed")
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)