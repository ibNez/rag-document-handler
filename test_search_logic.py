#!/usr/bin/env python3
"""
Test script to verify the new document search logic works correctly.
"""

import logging
from ingestion.core.postgres_manager import PostgreSQLManager, PostgreSQLConfig
from rag_manager.core.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_search_logic():
    """Test the new search logic with actual data."""
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
        
        with postgres.get_connection() as conn:
            with conn.cursor() as cursor:
                print("=" * 60)
                print("TESTING NEW SEARCH LOGIC")
                print("=" * 60)
                
                # Test cases based on actual data
                test_cases = [
                    "WORLD_HISTORY_-_chap01.pdf",  # File document
                    "The First Civilizations and Empires",  # File title
                    "news.ycombinator.com - item (Snapshot 20250903T143303Z)",  # URL title
                ]
                
                for test_filename in test_cases:
                    print(f"\nTesting search for: '{test_filename}'")
                    
                    # Test the new search logic
                    cursor.execute("""
                        SELECT id, document_type, title, file_path, filename 
                        FROM documents 
                        WHERE title = %s 
                           OR (document_type = 'file' AND file_path LIKE %s)
                           OR (document_type = 'url' AND file_path = %s)
                           OR (filename IS NOT NULL AND filename = %s)
                    """, (test_filename, f'%/{test_filename}', test_filename, test_filename))
                    
                    results = cursor.fetchall()
                    print(f"  Found {len(results)} matches:")
                    
                    for i, row in enumerate(results[:3]):  # Show max 3 results
                        print(f"    {i+1}. Type: {row['document_type']}")
                        print(f"       Title: '{row['title']}'")
                        print(f"       File Path: '{row['file_path']}'")
                        print(f"       Filename: '{row['filename']}'")
                        print()
                
                print("=" * 60)
                print("EDGE CASE TESTING")
                print("=" * 60)
                
                # Test some edge cases
                edge_cases = [
                    "nonexistent.pdf",  # Should find nothing
                    "WORLD_HISTORY",    # Partial filename
                ]
                
                for test_case in edge_cases:
                    print(f"\nTesting edge case: '{test_case}'")
                    
                    cursor.execute("""
                        SELECT id, document_type, title, file_path, filename 
                        FROM documents 
                        WHERE title = %s 
                           OR (document_type = 'file' AND file_path LIKE %s)
                           OR (document_type = 'url' AND file_path = %s)
                           OR (filename IS NOT NULL AND filename = %s)
                    """, (test_case, f'%/{test_case}', test_case, test_case))
                    
                    results = cursor.fetchall()
                    print(f"  Found {len(results)} matches")
                    
                print("\n" + "=" * 60)
                print("FILENAME FIELD ANALYSIS")
                print("=" * 60)
                
                # Check what's in the filename field
                cursor.execute("""
                    SELECT document_type, title, file_path, filename 
                    FROM documents 
                    WHERE filename IS NOT NULL 
                    LIMIT 10
                """)
                
                print("\nDocuments with filename field populated:")
                results = cursor.fetchall()
                if results:
                    for row in results:
                        print(f"  Type: {row['document_type']}, Filename: '{row['filename']}'")
                        print(f"    Title: '{row['title']}'")
                        print(f"    File Path: '{row['file_path']}'")
                        print()
                else:
                    print("  No documents have filename field populated")
                
    except Exception as e:
        logger.exception(f"Test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(test_search_logic())
