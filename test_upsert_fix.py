#!/usr/bin/env python3
"""
Test the fixed upsert_document_metadata to ensure it updates existing records
instead of creating duplicates.
"""

import logging
from ingestion.document.manager import DocumentManager
from ingestion.core.postgres_manager import PostgreSQLManager, PostgreSQLConfig
from rag_manager.core.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_upsert_fix():
    """Test that upsert_document_metadata now properly updates instead of creating duplicates."""
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
        doc_manager = DocumentManager(postgres)
        
        filename = "WORLD_HISTORY_-_chap01.pdf"
        
        print("=" * 60)
        print("TESTING UPSERT_DOCUMENT_METADATA FIX")
        print("=" * 60)
        
        # Count records before
        with postgres.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) as count FROM documents WHERE filename = %s OR title = %s", (filename, filename))
                before_count = cursor.fetchone()['count']
                print(f"\nBEFORE: {before_count} records matching '{filename}'")
        
        # Test upsert with minimal metadata (simulating processing start)
        print(f"\n1. Testing upsert with processing status update...")
        doc_manager.upsert_document_metadata(filename, {
            'processing_status': 'pending',
            'file_path': f'staging/{filename}'
        })
        
        # Count records after first upsert
        with postgres.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) as count FROM documents WHERE filename = %s OR title = %s", (filename, filename))
                after_first_count = cursor.fetchone()['count']
                print(f"   After first upsert: {after_first_count} records (should be same as before)")
        
        # Test upsert with full metadata (simulating processing complete)
        print(f"\n2. Testing upsert with full metadata...")
        doc_manager.upsert_document_metadata(filename, {
            'title': 'Updated Title - World History Chapter 1',
            'content_preview': 'This chapter covers ancient civilizations...',
            'file_path': f'staging/{filename}',
            'content_type': 'application/pdf',
            'file_size': 1024000,
            'word_count': 5000,
            'page_count': 25,
            'chunk_count': 50,
            'processing_status': 'completed',
            'processing_time_seconds': 15.5
        })
        
        # Count records after second upsert
        with postgres.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) as count FROM documents WHERE filename = %s OR title LIKE %s", (filename, '%World History%'))
                after_second_count = cursor.fetchone()['count']
                print(f"   After second upsert: {after_second_count} records (should still be same)")
        
        # Show final state
        print(f"\n3. Final document state:")
        with postgres.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id, title, processing_status, word_count, updated_at
                    FROM documents 
                    WHERE filename = %s OR title LIKE %s
                    ORDER BY updated_at DESC
                """, (filename, '%World History%'))
                
                for i, row in enumerate(cursor.fetchall(), 1):
                    print(f"   {i}. Status: {row['processing_status']}, Words: {row['word_count']}, Title: '{row['title'][:50]}...', Updated: {row['updated_at']}")
        
        # Summary
        print(f"\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        print(f"Records before: {before_count}")
        print(f"Records after first upsert: {after_first_count}")
        print(f"Records after second upsert: {after_second_count}")
        
        if before_count == after_first_count == after_second_count:
            print("✅ SUCCESS: No duplicate records created!")
        else:
            print("❌ FAILURE: Duplicate records were created")
            return 1
            
    except Exception as e:
        logger.exception(f"Test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(test_upsert_fix())
