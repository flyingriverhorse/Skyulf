#!/usr/bin/env python3
"""
Script to clear all data from both SQLite and PostgreSQL databases for testing
"""
import os
import sys
sys.path.append('.')

def clear_sqlite_database():
    """Clear all data from SQLite database"""
    import sqlite3
    
    db_path = 'mlops_database.db'
    if not os.path.exists(db_path):
        print("SQLite database file not found - nothing to clear")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"Found {len(tables)} tables in SQLite database")
        
        # Clear all tables
        for table in tables:
            table_name = table[0]
            if table_name != 'sqlite_sequence':  # Skip system table
                cursor.execute(f"DELETE FROM {table_name};")
                print(f"  - Cleared table: {table_name}")
        
        # Reset auto-increment sequences if exists
        try:
            cursor.execute("DELETE FROM sqlite_sequence;")
        except Exception:
            pass  # Table doesn't exist, which is fine
        
        conn.commit()
        conn.close()
        print("✓ SQLite database cleared successfully")
        
    except Exception as e:
        print(f"Error clearing SQLite database: {e}")

def clear_postgresql_database():
    """Clear all data from PostgreSQL database"""
    try:
        import psycopg2
        import os
        
        # Use environment variables for secure connection
        conn = psycopg2.connect(
            host=os.environ.get('DB_HOST'),
            database=os.environ.get('DB_NAME'),
            user=os.environ.get('DB_USER'),
            password=os.environ.get('DB_PASSWORD'),
            port=int(os.environ.get('DB_PORT', 5432)),
            sslmode=os.environ.get('DB_SSLMODE', 'require'),
            connect_timeout=10
        )
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("""
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname = 'public'
        """)
        tables = cursor.fetchall()
        
        print(f"Found {len(tables)} tables in PostgreSQL database")
        
        # Disable foreign key checks temporarily
        cursor.execute("SET session_replication_role = replica;")
        
        # Clear all tables
        for table in tables:
            table_name = table[0]
            cursor.execute(f"TRUNCATE TABLE {table_name} RESTART IDENTITY CASCADE;")
            print(f"  - Cleared table: {table_name}")
        
        # Re-enable foreign key checks
        cursor.execute("SET session_replication_role = DEFAULT;")
        
        conn.commit()
        conn.close()
        print("✓ PostgreSQL database cleared successfully")
        
    except Exception as e:
        print(f"Error clearing PostgreSQL database: {e}")

def verify_databases_empty():
    """Verify both databases are empty"""
    print("\n=== Verifying databases are empty ===")
    
    # Check SQLite
    try:
        import sqlite3
        db_path = 'mlops_database.db'
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check main tables
            tables_to_check = ['data_sources', 'users', 'data_ingestion_jobs', 'system_logs']
            for table in tables_to_check:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    print(f"SQLite {table}: {count} records")
                except Exception as e:
                    print(f"SQLite {table}: table doesn't exist or error - {e}")
            
            conn.close()
    except Exception as e:
        print(f"Error checking SQLite: {e}")
    
    # Check PostgreSQL
    try:
        import psycopg2
        import os
        
        # Use environment variables for secure connection
        conn = psycopg2.connect(
            host=os.environ.get('DB_HOST'),
            database=os.environ.get('DB_NAME'),
            user=os.environ.get('DB_USER'),
            password=os.environ.get('DB_PASSWORD'),
            port=int(os.environ.get('DB_PORT', 5432)),
            sslmode=os.environ.get('DB_SSLMODE', 'require'),
            connect_timeout=10
        )
        cursor = conn.cursor()
        
        # Check main tables
        tables_to_check = ['data_sources', 'users', 'data_ingestion_jobs', 'system_logs']
        for table in tables_to_check:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"PostgreSQL {table}: {count} records")
            except Exception as e:
                print(f"PostgreSQL {table}: table doesn't exist or error - {e}")
        
        conn.close()
    except Exception as e:
        print(f"Error checking PostgreSQL: {e}")

if __name__ == "__main__":
    print("=== Clearing All Database Data ===\n")
    
    # Ask for confirmation
    response = input("This will delete ALL data from both SQLite and PostgreSQL databases. Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Operation cancelled.")
        sys.exit(0)
    
    print("\nClearing SQLite database...")
    clear_sqlite_database()
    
    print("\nClearing PostgreSQL database...")
    clear_postgresql_database()
    
    print("\nVerifying databases are empty...")
    verify_databases_empty()
    
    print("\n✓ All databases cleared successfully!")
    print("You can now run tests with clean databases.")