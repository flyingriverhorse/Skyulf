#!/usr/bin/env python3
"""
Test script to add sample data sources and verify synchronization
"""
import asyncio
import sys
sys.path.append('.')

async def add_test_data_sources():
    """Add some test data sources to verify synchronization"""
    try:
        from core.database.engine import init_db, get_async_session
        from core.data_ingestion.service import DataIngestionService
        from core.data_ingestion.models import DataSourceCreate, SourceType
        
        print("=== Initializing Database ===")
        await init_db()
        print("✓ Database initialized")
        
        print("\n=== Adding Test Data Sources ===")
        
        # Test data sources to create
        test_sources = [
            {
                "name": "Customer Database",
                "source_type": "postgresql",
                "description": "Main customer database connection",
                "connection_info": {"host": "localhost", "database": "customers"},
                "category": "Database",
                "created_by": "admin"
            },
            {
                "name": "Sales Data Export",
                "source_type": "csv",
                "description": "CSV export of sales data",
                "connection_info": {"file_path": "/data/sales.csv"},
                "category": "Files",
                "created_by": "admin"
            },
            {
                "name": "Product Inventory DB",
                "source_type": "mysql",
                "description": "Product inventory tracking system",
                "connection_info": {"host": "inventory.example.com", "database": "products"},
                "category": "Database",
                "created_by": "user1"
            }
        ]
        
        async for session in get_async_session():
            service = DataIngestionService(session)
            
            # Check current data sources
            existing = await service.list_data_sources()
            print(f"Current data sources: {len(existing)}")
            
            # Add test sources
            created_count = 0
            for source_data in test_sources:
                try:
                    # Create DataSourceCreate object
                    source_create = DataSourceCreate(
                        name=source_data["name"],
                        source_type=SourceType(source_data["source_type"]),
                        category=source_data["category"],
                        connection_info=source_data["connection_info"],
                        metadata={"description": source_data["description"]}
                    )
                    
                    # Create the source
                    created_source = await service.create_data_source(source_create, source_data["created_by"])
                    print(f"✓ Created: {created_source.name} (ID: {created_source.id})")
                    created_count += 1
                    
                except Exception as e:
                    print(f"✗ Failed to create {source_data['name']}: {e}")
            
            # Verify final count
            final_sources = await service.list_data_sources()
            print(f"Total data sources after creation: {len(final_sources)}")
            print(f"Successfully created {created_count} test data sources")
            
            # List all sources
            print("\n=== Current Data Sources ===")
            for source in final_sources:
                print(f"- {source.name} (ID: {source.id}, Type: {source.type}, Created by: {source.created_by})")
            
            return len(final_sources)
            
    except Exception as e:
        print(f"Error adding test data sources: {e}")
        import traceback
        traceback.print_exc()
        return 0

async def test_data_source_apis():
    """Test the data source APIs to verify they work correctly"""
    import aiohttp
    import json
    
    print("\n=== Testing Data Source APIs ===")
    
    base_url = "http://127.0.0.1:8000"
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test the data sources API
            print("Testing /data/api/sources...")
            async with session.get(f"{base_url}/data/api/sources") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✓ Data sources API: {data['total']} sources found")
                    if data['sources']:
                        print(f"  First source: {data['sources'][0]['name']}")
                else:
                    print(f"✗ Data sources API failed: {resp.status}")
            
            # Test the admin data sources API
            print("Testing /admin/api/data-ingestion/sources...")
            async with session.get(f"{base_url}/admin/api/data-ingestion/sources") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✓ Admin data sources API: {data['total']} sources found")
                    if data['sources']:
                        print(f"  First source: {data['sources'][0]['name']}")
                else:
                    print(f"✗ Admin data sources API failed: {resp.status}")
    
    except Exception as e:
        print(f"Error testing APIs (server may not be running): {e}")

if __name__ == "__main__":
    # Add test data sources
    total_sources = asyncio.run(add_test_data_sources())
    
    if total_sources > 0:
        print(f"\n✅ Test data ready! You now have {total_sources} data sources to test with.")
        print("💡 Start the FastAPI server and check both:")
        print("   - Data Ingestion Management section")
        print("   - All Data Sources section")
        print("   - Try deleting sources from both sections")
        
        # Optionally test APIs if server is running
        print("\n🔄 Attempting to test APIs (requires server to be running)...")
        try:
            asyncio.run(test_data_source_apis())
        except:
            print("⚠️  API tests skipped - start the server first: python -m uvicorn main:app --reload")
    else:
        print("❌ No test data was created. Check the errors above.")