#!/usr/bin/env python3
"""
Check if there are still orphaned records causing the filename constraint violation.
"""

import logging
from ingestion.core.postgres_manager import PostgreSQLManager, PostgreSQLConfig
from rag_manager.core.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_current_state():
    """Check current database state."""
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
        print("CURRENT DATABASE STATE CHECK")
        print("=" * 60)
        
        with postgres.get_connection() as conn:
            with conn.cursor() as cursor:
                # Check for exact filename matches
                print(f"\n1. Records with filename = '{filename}':")
                cursor.execute("""
                    SELECT id, filename, file_path, processing_status, created_at
                    FROM documents 
                    WHERE filename = %s
                """, (filename,))
                
                exact_matches = cursor.fetchall()
                if exact_matches:
                    for i, row in enumerate(exact_matches, 1):
                        print(f"   {i}. ID: {row['id']}")
                        print(f"      Filename: '{row['filename']}'") 
                        print(f"      File Path: '{row['file_path']}'")
                        print(f"      Status: {row['processing_status']}")
                        print(f"      Created: {row['created_at']}")
                else:
                    print("   No exact matches found")
                
                # Check for any similar matches
                print(f"\n2. Records with filename LIKE '%{filename}%':")
                cursor.execute("""
                    SELECT id, filename, file_path, processing_status, created_at
                    FROM documents 
                    WHERE filename LIKE %s
                """, (f'%{filename}%',))
                
                similar_matches = cursor.fetchall()
                if similar_matches:
                    for i, row in enumerate(similar_matches, 1):
                        print(f"   {i}. ID: {row['id']}")
                        print(f"      Filename: '{row['filename']}'")
                        print(f"      File Path: '{row['file_path']}'")
                        print(f"      Status: {row['processing_status']}")
                        print(f"      Created: {row['created_at']}")
                else:
                    print("   No similar matches found")
                
                # Show ALL documents in the table
                print(f"\n3. ALL DOCUMENTS in database:")
                cursor.execute("""
                    SELECT id, filename, file_path, processing_status, created_at
                    FROM documents 
                    ORDER BY created_at DESC
                """)
                
                all_docs = cursor.fetchall()
                if all_docs:
                    for i, row in enumerate(all_docs, 1):
                        print(f"   {i}. ID: {row['id']}")
                        print(f"      Filename: '{row['filename']}'")
                        print(f"      File Path: '{row['file_path']}'")
                        print(f"      Status: {row['processing_status']}")
                        print(f"      Created: {row['created_at']}")
                        print()
                else:
                    print("   Database is empty")
                
    except Exception as e:
        logger.exception(f"Check failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(check_current_state())
