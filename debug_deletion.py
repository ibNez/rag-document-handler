#!/usr/bin/env python3
"""
Debug script to investigate file deletion issue.
Check database records, staging folder, and deleted folder.
"""

import os
import logging
from ingestion.core.postgres_manager import PostgreSQLManager, PostgreSQLConfig
from rag_manager.core.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_deletion_issue():
    """Check what happened with the file deletion."""
    try:
        config = Config()
        postgres_config = PostgreSQLConfig(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            database=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD
        )
        postgres = PostgreSQLManager(postgres_config)
        
        filename = "WORLD_HISTORY_-_chap01.pdf"
        
        print("=" * 70)
        print("FILE DELETION INVESTIGATION")
        print("=" * 70)
        
        # Check database records
        with postgres.get_connection() as conn:
            with conn.cursor() as cursor:
                print(f"\n1. DATABASE RECORDS FOR '{filename}':")
                cursor.execute("""
                    SELECT id, filename, file_path, processing_status, document_type, created_at
                    FROM documents 
                    WHERE filename = %s OR filename LIKE %s OR file_path LIKE %s
                    ORDER BY created_at DESC
                """, (filename, f'%{filename}%', f'%{filename}%'))
                
                records = cursor.fetchall()
                if records:
                    for i, row in enumerate(records, 1):
                        print(f"   {i}. ID: {row['id']}")
                        print(f"      Filename: '{row['filename']}'")
                        print(f"      File Path: '{row['file_path']}'")
                        print(f"      Status: {row['processing_status']}")
                        print(f"      Type: {row['document_type']}")
                        print(f"      Created: {row['created_at']}")
                        print()
                else:
                    print("   No database records found")
        
        # Check filesystem locations
        staging_path = os.path.join(config.UPLOAD_FOLDER, filename)
        uploaded_path = os.path.join(config.UPLOADED_FOLDER, filename)
        deleted_path = os.path.join(config.DELETED_FOLDER, filename)
        
        print(f"\n2. FILESYSTEM CHECK:")
        print(f"   Staging folder: {config.UPLOAD_FOLDER}")
        print(f"   Uploaded folder: {config.UPLOADED_FOLDER}")
        print(f"   Deleted folder: {config.DELETED_FOLDER}")
        print()
        
        print(f"   File locations for '{filename}':")
        print(f"   - Staging ({staging_path}): {'EXISTS' if os.path.exists(staging_path) else 'NOT FOUND'}")
        print(f"   - Uploaded ({uploaded_path}): {'EXISTS' if os.path.exists(uploaded_path) else 'NOT FOUND'}")
        print(f"   - Deleted ({deleted_path}): {'EXISTS' if os.path.exists(deleted_path) else 'NOT FOUND'}")
        
        # Check what files are actually in each folder
        print(f"\n3. FOLDER CONTENTS:")
        
        for folder_name, folder_path in [
            ("Staging", config.UPLOAD_FOLDER),
            ("Uploaded", config.UPLOADED_FOLDER), 
            ("Deleted", config.DELETED_FOLDER)
        ]:
            print(f"\n   {folder_name} folder ({folder_path}):")
            if os.path.exists(folder_path):
                files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
                if files:
                    for file in files:
                        print(f"     - {file}")
                else:
                    print(f"     (no PDF files)")
            else:
                print(f"     FOLDER DOES NOT EXIST")
        
        # Check if there are any related files (with different extensions)
        print(f"\n4. RELATED FILES CHECK:")
        base_name = filename.rsplit('.', 1)[0]  # Remove .pdf extension
        
        for folder_name, folder_path in [
            ("Staging", config.UPLOAD_FOLDER),
            ("Uploaded", config.UPLOADED_FOLDER),
            ("Deleted", config.DELETED_FOLDER)
        ]:
            if os.path.exists(folder_path):
                related_files = [f for f in os.listdir(folder_path) if base_name in f]
                if related_files:
                    print(f"   {folder_name} - files containing '{base_name}':")
                    for file in related_files:
                        print(f"     - {file}")
        
        # Check total document count
        print(f"\n5. OVERALL DATABASE STATE:")
        with postgres.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) as total FROM documents")
                total = cursor.fetchone()['total']
                print(f"   Total documents in database: {total}")
                
                cursor.execute("""
                    SELECT processing_status, COUNT(*) as count 
                    FROM documents 
                    GROUP BY processing_status
                """)
                for row in cursor.fetchall():
                    print(f"   - {row['processing_status']}: {row['count']}")
                
    except Exception as e:
        logger.exception(f"Debug failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(debug_deletion_issue())
