#!/usr/bin/env python3
"""
Check for data sources that might have directory paths
"""

import asyncio
from core.database.engine import get_async_session, init_db
from core.database.models import DataSource
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def check_all_sources_and_paths():
    """Check DataSource table for directory paths"""
    
    try:
        await init_db()
        
        async for session in get_async_session():
            from sqlalchemy import select
            
            # Check DataSource table
            print("=" * 80)
            print("CHECKING DataSource TABLE")
            print("=" * 80)
            
            stmt = select(DataSource)
            result = await session.execute(stmt)
            sources = result.scalars().all()
            
            for source in sources:
                print(f"\nDataSource ID {source.id}:")
                print(f"  Name: {source.name}")
                print(f"  Type: {source.type}")
                print(f"  Config: {json.dumps(source.config, indent=2) if source.config else 'None'}")
                
                # Check if config contains any directory-like paths
                if source.config and isinstance(source.config, dict):
                    for key, value in source.config.items():
                        if isinstance(value, str) and ('/' in value or '\\' in value):
                            from pathlib import Path
                            path = Path(value)
                            if path.exists():
                                print(f"    🔍 Found path in {key}: {value}")
                                print(f"      - Exists: {path.exists()}")
                                print(f"      - Is file: {path.is_file()}")
                                print(f"      - Is directory: {path.is_dir()}")
                                if path.is_dir():
                                    print(f"      - ⚠️  THIS IS A DIRECTORY!")
            
            break
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(check_all_sources_and_paths())