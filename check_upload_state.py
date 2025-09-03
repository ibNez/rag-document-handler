#!/usr/bin/env python3
"""
Check the state after clean install and file upload to staging.
"""

import os
import logging
from ingestion.core.postgres_manager import PostgreSQLManager, PostgreSQLConfig
from rag_manager.core.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_upload_state():
    """Check the state after file upload."""
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
        print("STATE CHECK AFTER CLEAN INSTALL + FILE UPLOAD")
        print("=" * 70)
        
        # Check database state
        with postgres.get_connection() as conn:
            with conn.cursor() as cursor:
                print(f"\n1. DATABASE STATE:")
                cursor.execute("SELECT COUNT(*) as total FROM documents")
                total = cursor.fetchone()['total']
                print(f"   Total documents in database: {total}")
                
                if total > 0:
                    cursor.execute("""
                        SELECT id, document_type, title, filename, file_path, 
                               processing_status, created_at
                        FROM documents 
                        ORDER BY created_at DESC
                    """)
                    
                    print(f"\n   All documents:")
                    for i, row in enumerate(cursor.fetchall(), 1):
                        print(f"   {i}. ID: {row['id']}")
                        print(f"      Document Type: '{row['document_type']}'")
                        print(f"      Title: '{row['title']}'")
                        print(f"      Filename: '{row['filename']}'")
                        print(f"      File Path: '{row['file_path']}'")
                        print(f"      Processing Status: '{row['processing_status']}'")
                        print(f"      Created: {row['created_at']}")
                        print()
                else:
                    print("   ❌ No documents found in database!")
        
        # Check filesystem state
        print(f"2. FILESYSTEM STATE:")
        staging_path = os.path.join(config.UPLOAD_FOLDER, filename)
        uploaded_path = os.path.join(config.UPLOADED_FOLDER, filename)
        deleted_path = os.path.join(config.DELETED_FOLDER, filename)
        
        print(f"   File locations for '{filename}':")
        print(f"   - Staging: {'✅ EXISTS' if os.path.exists(staging_path) else '❌ NOT FOUND'}")
        print(f"   - Uploaded: {'✅ EXISTS' if os.path.exists(uploaded_path) else '❌ NOT FOUND'}")
        print(f"   - Deleted: {'✅ EXISTS' if os.path.exists(deleted_path) else '❌ NOT FOUND'}")
        
        # Check folder contents
        print(f"\n3. FOLDER CONTENTS:")
        for folder_name, folder_path in [
            ("Staging", config.UPLOAD_FOLDER),
            ("Uploaded", config.UPLOADED_FOLDER),
            ("Deleted", config.DELETED_FOLDER)
        ]:
            print(f"   {folder_name} ({folder_path}):")
            if os.path.exists(folder_path):
                files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
                if files:
                    for file in files:
                        print(f"     - {file}")
                else:
                    print(f"     (no PDF files)")
            else:
                print(f"     ❌ FOLDER DOES NOT EXIST")
        
        # Summary
        print(f"\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        # Expected vs actual
        expected_db_records = 1
        expected_staging_file = True
        expected_uploaded_file = False
        expected_deleted_file = False
        
        actual_db_records = total
        actual_staging_file = os.path.exists(staging_path)
        actual_uploaded_file = os.path.exists(uploaded_path)
        actual_deleted_file = os.path.exists(deleted_path)
        
        print(f"Expected state:")
        print(f"  ✓ Database records: {expected_db_records}")
        print(f"  ✓ File in staging: {expected_staging_file}")
        print(f"  ✓ File in uploaded: {expected_uploaded_file}")
        print(f"  ✓ File in deleted: {expected_deleted_file}")
        
        print(f"\nActual state:")
        status_db = "✅" if actual_db_records == expected_db_records else "❌"
        status_staging = "✅" if actual_staging_file == expected_staging_file else "❌"
        status_uploaded = "✅" if actual_uploaded_file == expected_uploaded_file else "❌"
        status_deleted = "✅" if actual_deleted_file == expected_deleted_file else "❌"
        
        print(f"  {status_db} Database records: {actual_db_records}")
        print(f"  {status_staging} File in staging: {actual_staging_file}")
        print(f"  {status_uploaded} File in uploaded: {actual_uploaded_file}")
        print(f"  {status_deleted} File in deleted: {actual_deleted_file}")
        
        if (actual_db_records == expected_db_records and 
            actual_staging_file == expected_staging_file and
            actual_uploaded_file == expected_uploaded_file and
            actual_deleted_file == expected_deleted_file):
            print(f"\n🎉 UPLOAD STATE IS CORRECT!")
        else:
            print(f"\n⚠️  UPLOAD STATE HAS ISSUES!")
        
    except Exception as e:
        logger.exception(f"Check failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(check_upload_state())
