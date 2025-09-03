#!/usr/bin/env python3
"""
Clean up orphaned database records for the deleted file.
"""

import logging
from ingestion.core.postgres_manager import PostgreSQLManager, PostgreSQLConfig
from rag_manager.core.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_orphaned_records():
    """Clean up the orphaned database records."""
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
        print("CLEANING UP ORPHANED RECORDS")
        print("=" * 60)
        
        with postgres.get_connection() as conn:
            with conn.cursor() as cursor:
                # Show records before deletion
                print(f"\nBEFORE: Records for '{filename}':")
                cursor.execute("""
                    SELECT id, filename, file_path, processing_status
                    FROM documents 
                    WHERE filename = %s OR file_path LIKE %s
                """, (filename, f'%{filename}%'))
                
                before_records = cursor.fetchall()
                for i, row in enumerate(before_records, 1):
                    print(f"   {i}. ID: {row['id']}, Filename: '{row['filename']}', Status: {row['processing_status']}")
                
                # Delete the orphaned records
                print(f"\nDELETING orphaned records...")
                cursor.execute("""
                    DELETE FROM documents 
                    WHERE filename = %s OR file_path LIKE %s
                """, (filename, f'%{filename}%'))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                print(f"   Deleted {deleted_count} orphaned records")
                
                # Verify deletion
                print(f"\nAFTER: Checking for remaining records:")
                cursor.execute("""
                    SELECT id, filename, file_path, processing_status
                    FROM documents 
                    WHERE filename = %s OR file_path LIKE %s
                """, (filename, f'%{filename}%'))
                
                after_records = cursor.fetchall()
                if after_records:
                    print("   WARNING: Still found records:")
                    for row in after_records:
                        print(f"     - ID: {row['id']}, Filename: '{row['filename']}'")
                else:
                    print("   ✅ No remaining records found")
                
                # Show total document count
                cursor.execute("SELECT COUNT(*) as total FROM documents")
                total = cursor.fetchone()['total']
                print(f"\n   Total documents remaining in database: {total}")
        
        print(f"\n✅ Cleanup complete! You should now be able to upload '{filename}' again.")
        
    except Exception as e:
        logger.exception(f"Cleanup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(cleanup_orphaned_records())
