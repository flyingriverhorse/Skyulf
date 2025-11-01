#!/usr/bin/env python3
"""
Verify databases are empty after clearing
"""
import os
import sys
sys.path.append('.')

def check_sqlite():
    import sqlite3
    
    db_path = 'mlops_database.db'
    if not os.path.exists(db_path):
        print("SQLite database file not found")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    tables = ['data_sources', 'users', 'data_ingestion_jobs', 'system_logs']
    print("=== SQLite Database Status ===")
    total_records = 0
    
    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"{table}: {count} records")
            total_records += count
        except Exception as e:
            print(f"{table}: error - {e}")
    
    print(f"Total SQLite records: {total_records}")
    conn.close()

def check_postgresql():
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
        
        tables = ['data_sources', 'users', 'data_ingestion_jobs', 'system_logs']
        print("\n=== PostgreSQL Database Status ===")
        total_records = 0
        
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"{table}: {count} records")
                total_records += count
            except Exception as e:
                print(f"{table}: error - {e}")
        
        print(f"Total PostgreSQL records: {total_records}")
        conn.close()
        
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")

if __name__ == "__main__":
    check_sqlite()
    check_postgresql()
    print("\n✓ Database verification complete")