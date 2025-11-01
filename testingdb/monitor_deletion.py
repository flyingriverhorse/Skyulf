#!/usr/bin/env python3
"""
Monitor and log exactly what paths are being deleted
"""

import asyncio
from pathlib import Path
from core.database.engine import get_async_session, init_db
from core.database.models import DataSource
from core.utils.file_utils import extract_file_path_from_source, safe_delete_path
import logging
import json

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def monitor_current_database_and_paths():
    """Check all current database records and their paths"""
    
    try:
        await init_db()
        
        async for session in get_async_session():
            from sqlalchemy import select
            stmt = select(DataSource)
            result = await session.execute(stmt)
            sources = result.scalars().all()
            
            print("=" * 80)
            print(f"CURRENT DATABASE ANALYSIS - Found {len(sources)} records")
            print("=" * 80)
            
            for i, source in enumerate(sources, 1):
                print(f"\n📊 SOURCE {i}:")
                print(f"  ID: {source.id}")
                print(f"  Name: {source.name}")
                print(f"  Type: {source.type}")
                print(f"  Config: {json.dumps(source.config, indent=4) if source.config else 'None'}")
                
                # Test path extraction exactly like the service does
                source_dict = {
                    'file_path': getattr(source, 'file_path', None),
                    'path': getattr(source, 'path', None), 
                    'source_path': getattr(source, 'source_path', None),
                    'location': getattr(source, 'location', None),
                    'file_location': getattr(source, 'file_location', None),
                    'source_name': source.name,
                    'connection_info': getattr(source, 'config', {}),
                }
                
                print(f"  Source dict for extraction:")
                for key, value in source_dict.items():
                    print(f"    {key}: {value}")
                
                extracted_path = extract_file_path_from_source(source_dict)
                print(f"  🎯 EXTRACTED PATH: {extracted_path}")
                
                if extracted_path:
                    print(f"  📁 Path analysis:")
                    print(f"    - Exists: {extracted_path.exists()}")
                    print(f"    - Is file: {extracted_path.is_file()}")
                    print(f"    - Is directory: {extracted_path.is_dir()}")
                    print(f"    - Parent: {extracted_path.parent}")
                    
                    if extracted_path.exists():
                        if extracted_path.is_dir():
                            print(f"    - ⚠️  WARNING: This is a DIRECTORY!")
                            print(f"    - Contents: {list(extracted_path.iterdir())}")
                        else:
                            print(f"    - ✓ This is a file")
                            print(f"    - Size: {extracted_path.stat().st_size} bytes")
                
                print(f"  🗑️  DELETION TEST (dry run):")
                
                # Test what would happen if we called safe_delete_path
                if extracted_path and extracted_path.exists():
                    print(f"    - Would call: safe_delete_path('{extracted_path}', files_only=True)")
                    
                    # Check our logic
                    if extracted_path.is_dir():
                        print(f"    - ❌ WOULD SKIP: Directory deletion blocked by files_only=True")
                    else:
                        print(f"    - ✅ WOULD DELETE: File deletion allowed")
                else:
                    print(f"    - ℹ️  NO ACTION: Path doesn't exist or wasn't found")
                
                print("-" * 60)
            
            break
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

async def test_specific_deletion_monitoring():
    """Test deletion with detailed monitoring"""
    
    print("\n" + "=" * 80)
    print("TESTING DELETION WITH MONITORING")
    print("=" * 80)
    
    try:
        # Check uploads directory structure
        uploads_dir = Path(__file__).parent / "uploads"
        print(f"\n📂 UPLOADS DIRECTORY: {uploads_dir}")
        print(f"  Exists: {uploads_dir.exists()}")
        
        if uploads_dir.exists():
            print(f"  Contents:")
            for item in uploads_dir.iterdir():
                item_type = "📁 DIR " if item.is_dir() else "📄 FILE"
                print(f"    {item_type}: {item.name}")
                if item.is_dir():
                    print(f"      Contents: {list(item.iterdir())}")
        
    except Exception as e:
        logger.error(f"Error in deletion monitoring: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(monitor_current_database_and_paths())
    asyncio.run(test_specific_deletion_monitoring())