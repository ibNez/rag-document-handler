#!/usr/bin/env python3
"""
Clean up the broken record with filename = 'None'
"""

import logging
from ingestion.core.postgres_manager import PostgreSQLManager, PostgreSQLConfig
from rag_manager.core.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_broken_record():
    """Clean up the broken record."""
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
        
        print("=" * 60)
        print("CLEANING UP BROKEN RECORD")
        print("=" * 60)
        
        with postgres.get_connection() as conn:
            with conn.cursor() as cursor:
                # Delete record with filename = 'None'
                print(f"Deleting record with filename = 'None'...")
                cursor.execute("""
                    DELETE FROM documents 
                    WHERE filename IS NULL OR filename = 'None'
                """)
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                print(f"   Deleted {deleted_count} broken records")
                
                # Show remaining records
                print(f"\nRemaining records:")
                cursor.execute("""
                    SELECT id, filename, file_path, processing_status
                    FROM documents 
                    ORDER BY created_at DESC
                """)
                
                remaining = cursor.fetchall()
                for i, row in enumerate(remaining, 1):
                    print(f"   {i}. Filename: '{row['filename']}', Status: {row['processing_status']}")
                
    except Exception as e:
        logger.exception(f"Cleanup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(cleanup_broken_record())
