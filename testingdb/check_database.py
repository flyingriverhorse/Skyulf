#!/usr/bin/env python3
"""
Debug Database Content - Check what's actually stored in the database
"""

import os
import sys
import asyncio
import json
from pathlib import Path

sys.path.append('.')

async def check_database_content():
    """Check what's actually stored in the database"""
    print("🔍 Checking Database Content")
    print("=" * 50)
    
    try:
        from core.database.engine import get_async_session
        from sqlalchemy import text
        
        async for session in get_async_session():
            # Get all data sources with their connection_info
            result = await session.execute(text("""
                SELECT id, name, source_type, connection_info, created_at 
                FROM data_sources 
                ORDER BY created_at DESC 
                LIMIT 5
            """))
            
            rows = result.fetchall()
            
            if not rows:
                print("❌ No data sources found in database")
                return
            
            print(f"Found {len(rows)} data sources:")
            print()
            
            for row in rows:
                print(f"ID: {row[0]}")
                print(f"Name: {row[1]}")
                print(f"Type: {row[2]}")
                print(f"Connection Info: {row[3]}")
                print(f"Created: {row[4]}")
                
                # Parse connection_info if it's JSON
                try:
                    if row[3]:
                        conn_info = json.loads(row[3]) if isinstance(row[3], str) else row[3]
                        print(f"Parsed Connection Info: {conn_info}")
                        
                        # Check if file exists
                        if isinstance(conn_info, dict) and 'filepath' in conn_info:
                            filepath = Path(conn_info['filepath'])
                            print(f"File Path: {filepath}")
                            print(f"File Exists: {filepath.exists()}")
                            if filepath.exists():
                                print(f"File Size: {filepath.stat().st_size} bytes")
                        else:
                            print("❌ No 'filepath' key in connection_info")
                    else:
                        print("❌ No connection_info stored")
                except Exception as e:
                    print(f"❌ Error parsing connection_info: {e}")
                
                print("-" * 40)
            
            break
            
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_database_content())