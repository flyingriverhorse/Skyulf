#!/usr/bin/env python3
"""
Script to clean database and uploads for fresh testing
"""
import sqlite3
import os
import shutil

# Database path
db_path = 'mlops_database.db'
uploads_path = 'uploads'

print("=== Database and Uploads Cleanup ===\n")

# 1. Check if database exists
if os.path.exists(db_path):
    print(f"Database found: {db_path}")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check existing tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"Current tables: {[t[0] for t in tables]}")
        
        # Check data sources
        if any('data_source' in t[0].lower() for t in tables):
            cursor.execute("SELECT COUNT(*) FROM data_sources")
            count = cursor.fetchone()[0]
            print(f"Data sources count: {count}")
            
            if count > 0:
                cursor.execute("SELECT id, source_id, filename FROM data_sources")
                sources = cursor.fetchall()
                print("Current data sources:")
                for source in sources:
                    print(f"  ID: {source[0]}, Source ID: {source[1]}, Filename: {source[2]}")
                
                # Clear data sources table
                cursor.execute("DELETE FROM data_sources")
                conn.commit()
                print("✓ Cleared data_sources table")
            else:
                print("Data sources table is already empty")
        else:
            print("No data_sources table found")
            
    except Exception as e:
        print(f"Database error: {e}")
    finally:
        conn.close()
else:
    print(f"Database not found: {db_path}")

# 2. Clean uploads directory
if os.path.exists(uploads_path):
    print(f"\nUploads directory found: {uploads_path}")
    
    # List current files
    files = []
    for root, dirs, filenames in os.walk(uploads_path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            files.append(file_path)
    
    if files:
        print(f"Current uploaded files ({len(files)}):")
        for file_path in files:
            print(f"  {file_path}")
        
        # Remove all files and subdirectories
        for root, dirs, filenames in os.walk(uploads_path, topdown=False):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                os.remove(file_path)
                print(f"  ✓ Removed: {file_path}")
            for dirname in dirs:
                dir_path = os.path.join(root, dirname)
                os.rmdir(dir_path)
                print(f"  ✓ Removed directory: {dir_path}")
        
        print("✓ Cleared uploads directory")
    else:
        print("Uploads directory is already empty")
else:
    print(f"Uploads directory not found: {uploads_path}")

print("\n=== Cleanup Complete ===")
print("Database and uploads are now clean for fresh testing.")