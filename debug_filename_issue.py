#!/usr/bin/env python3
"""
Debug script to check filename conflicts in the database.
"""

import logging
from ingestion.core.postgres_manager import PostgreSQLManager, PostgreSQLConfig
from rag_manager.core.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_filename_issue():
    """Check what's causing the filename constraint violation."""
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
        
        with postgres.get_connection() as conn:
            with conn.cursor() as cursor:
                print("=" * 60)
                print("FILENAME CONSTRAINT VIOLATION DEBUG")
                print("=" * 60)
                
                # Check existing records with this filename
                print(f"\n1. Records with filename = '{filename}':")
                cursor.execute("""
                    SELECT id, filename, file_path, processing_status, document_type, created_at
                    FROM documents 
                    WHERE filename = %s
                    ORDER BY created_at
                """, (filename,))
                
                exact_matches = cursor.fetchall()
                for i, row in enumerate(exact_matches, 1):
                    print(f"   {i}. ID: {row['id']}")
                    print(f"      Filename: '{row['filename']}'")
                    print(f"      File Path: '{row['file_path']}'")
                    print(f"      Status: {row['processing_status']}")
                    print(f"      Type: {row['document_type']}")
                    print(f"      Created: {row['created_at']}")
                    print()
                
                # Check similar records
                print(f"\n2. Records with filename LIKE '%WORLD_HISTORY%':")
                cursor.execute("""
                    SELECT id, filename, file_path, processing_status, document_type
                    FROM documents 
                    WHERE filename LIKE %s
                    ORDER BY created_at
                """, ('%WORLD_HISTORY%',))
                
                similar_matches = cursor.fetchall()
                for i, row in enumerate(similar_matches, 1):
                    print(f"   {i}. Filename: '{row['filename']}'")
                    print(f"      File Path: '{row['file_path']}'")
                    print(f"      Status: {row['processing_status']}")
                    print()
                
                # Check if filename field has UNIQUE constraint
                print(f"\n3. Database constraints on documents table:")
                cursor.execute("""
                    SELECT 
                        tc.constraint_name,
                        tc.constraint_type,
                        kcu.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu 
                        ON tc.constraint_name = kcu.constraint_name
                    WHERE tc.table_name = 'documents'
                    AND tc.constraint_type IN ('UNIQUE', 'PRIMARY KEY')
                """)
                
                constraints = cursor.fetchall()
                for constraint in constraints:
                    print(f"   {constraint['constraint_type']}: {constraint['column_name']} ({constraint['constraint_name']})")
                
                print(f"\n4. Total records in documents table: ")
                cursor.execute("SELECT COUNT(*) as count FROM documents")
                total = cursor.fetchone()['count']
                print(f"   {total} total documents")
                
    except Exception as e:
        logger.exception(f"Debug failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(debug_filename_issue())
