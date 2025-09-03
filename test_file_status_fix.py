#!/usr/bin/env python3
"""
Test the _get_file_database_status fix to see if it now returns 'completed' 
for the uploaded file instead of a random 'pending' status.
"""

import logging
import psycopg2
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_file_status_query():
    """Test the improved file status query."""
    try:
        # Get connection details from environment
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=os.getenv('POSTGRES_PORT', '5432'),
            database=os.getenv('POSTGRES_DB', 'rag_metadata'),
            user=os.getenv('POSTGRES_USER', 'rag_user'),
            password=os.getenv('POSTGRES_PASSWORD', 'secure_password')
        )
        
        filename = "WORLD_HISTORY_-_chap01.pdf"
        
        print("=" * 60)
        print("TESTING FILE STATUS QUERY FIX")
        print("=" * 60)
        
        with conn.cursor() as cursor:
            print(f"\nTesting filename: '{filename}'")
            
            # Old problematic query
            print("\n1. OLD QUERY (problematic):")
            cursor.execute(
                "SELECT processing_status FROM documents WHERE title = %s OR file_path LIKE %s",
                (filename, f'%{filename}%')
            )
            old_results = cursor.fetchall()
            print(f"   All results: {[r[0] for r in old_results]}")
            print(f"   fetchone() would return: '{old_results[0][0] if old_results else None}'")
            
            # New improved query
            print("\n2. NEW QUERY (improved):")
            cursor.execute("""
                SELECT processing_status FROM documents 
                WHERE title = %s 
                   OR (document_type = 'file' AND file_path LIKE %s)
                   OR (document_type = 'url' AND file_path = %s)
                   OR (filename IS NOT NULL AND filename = %s)
                ORDER BY 
                    CASE processing_status 
                        WHEN 'completed' THEN 1 
                        WHEN 'pending' THEN 2 
                        ELSE 3 
                    END
                LIMIT 1
            """, (filename, f'%/{filename}', filename, filename))
            new_result = cursor.fetchone()
            print(f"   Returns: '{new_result[0] if new_result else None}'")
            
            # Show all matching records for context
            print("\n3. ALL MATCHING RECORDS:")
            cursor.execute("""
                SELECT id, processing_status, title, file_path, filename
                FROM documents 
                WHERE title = %s 
                   OR (document_type = 'file' AND file_path LIKE %s)
                   OR (document_type = 'url' AND file_path = %s)
                   OR (filename IS NOT NULL AND filename = %s)
                ORDER BY processing_status, created_at
            """, (filename, f'%/{filename}', filename, filename))
            
            for i, row in enumerate(cursor.fetchall(), 1):
                print(f"   {i}. Status: {row[1]}, Title: '{row[2]}', File Path: '{row[3]}', Filename: '{row[4]}'")
        
        conn.close()
        
    except Exception as e:
        logger.exception(f"Test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(test_file_status_query())
