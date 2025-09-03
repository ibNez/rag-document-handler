#!/usr/bin/env python3
"""
Clean up all records for WORLD_HISTORY_-_chap01.pdf so we can test clean upload.
"""

import logging
from ingestion.core.postgres_manager import PostgreSQLManager, PostgreSQLConfig
from rag_manager.core.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_for_fresh_test():
    """Remove all records for the test file."""
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
        
        print("=" * 60)
        print("CLEANING FOR FRESH UPLOAD TEST")
        print("=" * 60)
        
        with postgres.get_connection() as conn:
            with conn.cursor() as cursor:
                # Show what we're about to delete
                print(f"BEFORE: Records for '{filename}':")
                cursor.execute("""
                    SELECT id, filename, file_path, processing_status
                    FROM documents 
                    WHERE filename = %s OR file_path LIKE %s
                """, (filename, f'%{filename}%'))
                
                before_records = cursor.fetchall()
                for i, row in enumerate(before_records, 1):
                    print(f"   {i}. ID: {row['id']}, Filename: '{row['filename']}', Status: {row['processing_status']}")
                
                # Delete all records related to this file
                print(f"\nDELETING all records for '{filename}'...")
                cursor.execute("""
                    DELETE FROM documents 
                    WHERE filename = %s OR file_path LIKE %s
                """, (filename, f'%{filename}%'))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                print(f"   Deleted {deleted_count} records")
                
                # Show final state
                print(f"\nFINAL STATE:")
                cursor.execute("SELECT COUNT(*) as total FROM documents")
                total = cursor.fetchone()['total']
                print(f"   Total documents remaining: {total}")
                
                if total > 0:
                    cursor.execute("""
                        SELECT filename, processing_status FROM documents 
                        ORDER BY created_at DESC
                    """)
                    for row in cursor.fetchall():
                        print(f"   - '{row['filename']}' ({row['processing_status']})")
                
        print(f"\n✅ Database cleaned! Ready for fresh upload test of '{filename}'")
        
    except Exception as e:
        logger.exception(f"Cleanup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(clean_for_fresh_test())
