#!/usr/bin/env python3
"""
Check what documents are currently in the table.
"""

import logging
from ingestion.core.postgres_manager import PostgreSQLManager, PostgreSQLConfig
from rag_manager.core.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_documents_table():
    """Check what's currently in the documents table."""
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
        print("CURRENT DOCUMENTS TABLE STATE")
        print("=" * 60)
        
        with postgres.get_connection() as conn:
            with conn.cursor() as cursor:
                # Get all documents
                cursor.execute("""
                    SELECT id, document_type, title, filename, file_path, processing_status, created_at
                    FROM documents 
                    ORDER BY created_at DESC
                """)
                
                all_docs = cursor.fetchall()
                
                print(f"\nTotal documents in table: {len(all_docs)}")
                
                if all_docs:
                    print("\nAll documents:")
                    for i, row in enumerate(all_docs, 1):
                        print(f"\n{i}. ID: {row['id']}")
                        print(f"   Type: {row['document_type']}")
                        print(f"   Title: '{row['title']}'")
                        print(f"   Filename: '{row['filename']}'")
                        print(f"   File Path: '{row['file_path']}'")
                        print(f"   Status: {row['processing_status']}")
                        print(f"   Created: {row['created_at']}")
                else:
                    print("\n✅ Documents table is EMPTY")
                
    except Exception as e:
        logger.exception(f"Check failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(check_documents_table())
