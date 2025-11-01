#!/usr/bin/env python3
"""
Check what's in the database currently
"""

import asyncio
from core.database.engine import get_async_session, init_db
from core.database.models import DataSource
from core.utils.file_utils import extract_file_path_from_source
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def check_current_database():
    """Check what data sources are currently in the database"""
    
    try:
        # Initialize database
        await init_db()
        
        async for session in get_async_session():
            from sqlalchemy import select
            stmt = select(DataSource)
            result = await session.execute(stmt)
            sources = result.scalars().all()
            
            print(f"Found {len(sources)} data sources in database:")
            print("=" * 50)
            
            for source in sources:
                print(f"\nID: {source.id}")
                print(f"Name: {source.name}")
                print(f"Type: {source.type}")
                print(f"Config: {json.dumps(source.config, indent=2) if source.config else 'None'}")
                
                # Test path extraction
                source_dict = {
                    'file_path': getattr(source, 'file_path', None),
                    'path': getattr(source, 'path', None),
                    'source_path': getattr(source, 'source_path', None),
                    'location': getattr(source, 'location', None),
                    'file_location': getattr(source, 'file_location', None),
                    'source_name': source.name,
                    'connection_info': getattr(source, 'config', {}),
                }
                
                extracted_path = extract_file_path_from_source(source_dict)
                print(f"Extracted path: {extracted_path}")
                
                if extracted_path:
                    print(f"  Exists: {extracted_path.exists()}")
                    if extracted_path.exists():
                        print(f"  Is file: {extracted_path.is_file()}")
                        print(f"  Is directory: {extracted_path.is_dir()}")
                        print(f"  Parent: {extracted_path.parent}")
                
                print("-" * 30)
            
            break
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(check_current_database())