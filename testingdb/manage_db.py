"""
Database Management Script for FastAPI

This script provides utilities for database initialization, migration,
and maintenance operations.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi_app.config import get_settings
from fastapi_app.core.database.engine import init_db, close_db, create_tables, health_check
from fastapi_app.core.database.models import User, DataSource, DataIngestionJob, SystemLog


async def initialize_database():
    """Initialize the database with tables and basic data."""
    print("🔧 Initializing FastAPI database...")
    
    try:
        # Initialize database connection
        await init_db()
        print("✅ Database connection established")
        
        # Create all tables
        await create_tables()
        print("✅ Database tables created/updated")
        
        # Test database connectivity
        is_healthy = await health_check()
        if is_healthy:
            print("✅ Database health check passed")
        else:
            print("❌ Database health check failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False
    finally:
        await close_db()


async def check_database_status():
    """Check database status and connection."""
    print("🔍 Checking database status...")
    
    try:
        await init_db()
        
        # Check connectivity
        is_healthy = await health_check()
        if is_healthy:
            print("✅ Database is accessible")
        else:
            print("❌ Database is not accessible")
            return False
        
        # Count records in main tables
        from fastapi_app.core.database.engine import get_async_session
        
        async for session in get_async_session():
            from sqlalchemy import select, func
            
            # Count users
            result = await session.execute(select(func.count(User.id)))
            user_count = result.scalar()
            print(f"📊 Users: {user_count}")
            
            # Count data sources
            result = await session.execute(select(func.count(DataSource.id)))
            source_count = result.scalar()
            print(f"📊 Data Sources: {source_count}")
            
            # Count ingestion jobs
            result = await session.execute(select(func.count(DataIngestionJob.id)))
            job_count = result.scalar()
            print(f"📊 Ingestion Jobs: {job_count}")
            
            # Count system logs
            result = await session.execute(select(func.count(SystemLog.id)))
            log_count = result.scalar()
            print(f"📊 System Logs: {log_count}")
            
            break
        
        return True
        
    except Exception as e:
        print(f"❌ Database status check failed: {e}")
        return False
    finally:
        await close_db()


async def create_admin_user():
    """Create a default admin user for testing."""
    print("👤 Creating admin user...")
    
    try:
        await init_db()
        
        from fastapi_app.core.database.engine import get_async_session
        from fastapi_app.core.database.repository import get_user_repository
        from passlib.context import CryptContext
        
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        async for session in get_async_session():
            user_repo = get_user_repository(session)
            
            # Check if admin user already exists
            existing_admin = await user_repo.get_by_username("admin")
            if existing_admin:
                print("⚠️  Admin user already exists")
                return True
            
            # Create admin user
            admin_data = {
                "username": "admin",
                "email": "admin@mlops.local",
                "password_hash": pwd_context.hash("admin123"),  # Change this in production!
                "full_name": "System Administrator",
                "is_active": True,
                "is_admin": True,
                "is_verified": True,
            }
            
            admin_user = await user_repo.create(admin_data)
            print(f"✅ Admin user created with ID: {admin_user.id}")
            print("📧 Email: admin@mlops.local")
            print("🔑 Password: admin123 (change this immediately!)")
            
            break
        
        return True
        
    except Exception as e:
        print(f"❌ Admin user creation failed: {e}")
        return False
    finally:
        await close_db()


async def reset_database():
    """Reset database (drop and recreate all tables)."""
    print("🗑️  Resetting database...")
    print("⚠️  This will DELETE ALL DATA!")
    
    # Ask for confirmation
    response = input("Are you sure? Type 'yes' to continue: ").lower().strip()
    if response != "yes":
        print("❌ Database reset cancelled")
        return False
    
    try:
        await init_db()
        
        from fastapi_app.core.database.engine import get_engine, Base
        
        engine = get_engine()
        
        # Drop all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        print("🗑️  All tables dropped")
        
        # Recreate tables
        await create_tables()
        print("✅ Tables recreated")
        
        return True
        
    except Exception as e:
        print(f"❌ Database reset failed: {e}")
        return False
    finally:
        await close_db()


def main():
    """Main CLI interface for database management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FastAPI Database Management")
    parser.add_argument(
        "command",
        choices=["init", "status", "create-admin", "reset"],
        help="Database management command"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("🏗️  FastAPI Database Management")
    print("=" * 40)
    
    # Run the appropriate command
    if args.command == "init":
        success = asyncio.run(initialize_database())
    elif args.command == "status":
        success = asyncio.run(check_database_status())
    elif args.command == "create-admin":
        success = asyncio.run(create_admin_user())
    elif args.command == "reset":
        success = asyncio.run(reset_database())
    else:
        print(f"❌ Unknown command: {args.command}")
        success = False
    
    print("=" * 40)
    if success:
        print("🎉 Database management operation completed successfully!")
    else:
        print("❌ Database management operation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()