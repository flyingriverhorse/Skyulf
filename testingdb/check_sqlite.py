#!/usr/bin/env python3
"""
Check SQLite database structure and data
"""
import sqlite3

def check_sqlite_database():
    print("=== SQLite Database Structure ===\n")
    
    try:
        conn = sqlite3.connect('mlops_database.db')
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print("Tables:", [t[0] for t in tables])
        
        # Check each table
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"- {table_name}: {count} records")
            
            if table_name == 'data_sources' and count > 0:
                print("  First few data_sources records:")
                cursor.execute("SELECT id, source_id, filename, status FROM data_sources LIMIT 3")
                records = cursor.fetchall()
                for record in records:
                    print(f"    ID: {record[0]}, Source ID: {record[1]}, Filename: {record[2]}, Status: {record[3]}")
            
            elif table_name == 'data_ingestion_jobs' and count > 0:
                print("  First few data_ingestion_jobs records:")
                cursor.execute("SELECT id, status, records_processed FROM data_ingestion_jobs LIMIT 3")
                records = cursor.fetchall()
                for record in records:
                    print(f"    ID: {record[0]}, Source ID: {record[1]}, Name: {record[2]}, Status: {record[3]}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_sqlite_database()