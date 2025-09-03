#!/usr/bin/env python3
"""
Debug script to investigate document count discrepancy after clean reinstall.
Expected: 1 file document + 1 URL document = 2 total
Actual: 4 documents showing
"""

import logging
from ingestion.core.postgres_manager import PostgreSQLManager, PostgreSQLConfig
from rag_manager.core.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_document_counts():
    """Debug the document count discrepancy."""
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
                print("=" * 70)
                print("DOCUMENT COUNT INVESTIGATION AFTER CLEAN REINSTALL")
                print("=" * 70)
                
                # Get total document count
                cursor.execute("SELECT COUNT(*) as count FROM documents")
                total_count = cursor.fetchone()['count']
                print(f"\nTotal documents in database: {total_count}")
                
                # Break down by document type
                cursor.execute("""
                    SELECT document_type, COUNT(*) as count 
                    FROM documents 
                    GROUP BY document_type 
                    ORDER BY document_type
                """)
                print("\nBreakdown by document type:")
                for row in cursor.fetchall():
                    doc_type = row['document_type'] or 'NULL'
                    count = row['count']
                    print(f"  {doc_type}: {count} documents")
                
                # Show all documents with key details
                cursor.execute("""
                    SELECT id, document_type, title, file_path, filename, 
                           processing_status, created_at
                    FROM documents 
                    ORDER BY created_at DESC
                """)
                
                print(f"\nAll {total_count} documents in detail:")
                print("-" * 70)
                
                for i, row in enumerate(cursor.fetchall(), 1):
                    print(f"{i}. ID: {row['id']}")
                    print(f"   Type: {row['document_type']}")
                    print(f"   Title: '{row['title']}'")
                    print(f"   File Path: '{row['file_path']}'")
                    print(f"   Filename: '{row['filename']}'")
                    print(f"   Status: {row['processing_status']}")
                    print(f"   Created: {row['created_at']}")
                    print()
                
                # Check for potential duplicates based on file_path
                print("=" * 70)
                print("CHECKING FOR DUPLICATES")
                print("=" * 70)
                
                cursor.execute("""
                    SELECT file_path, COUNT(*) as count
                    FROM documents 
                    WHERE file_path IS NOT NULL AND file_path != ''
                    GROUP BY file_path
                    HAVING COUNT(*) > 1
                    ORDER BY count DESC
                """)
                
                duplicates = cursor.fetchall()
                if duplicates:
                    print("\nDuplicate file_path entries found:")
                    for row in duplicates:
                        print(f"  '{row['file_path']}': {row['count']} entries")
                        
                        # Show details of duplicates
                        cursor.execute("""
                            SELECT id, document_type, title, processing_status, created_at
                            FROM documents 
                            WHERE file_path = %s
                            ORDER BY created_at
                        """, (row['file_path'],))
                        
                        for dup in cursor.fetchall():
                            print(f"    - ID: {dup['id']}, Type: {dup['document_type']}, Status: {dup['processing_status']}, Created: {dup['created_at']}")
                else:
                    print("\nNo duplicate file_path entries found.")
                
                # Check for orphaned records (empty or null critical fields)
                print("\n" + "=" * 70)
                print("CHECKING FOR ORPHANED RECORDS")
                print("=" * 70)
                
                cursor.execute("""
                    SELECT id, document_type, title, file_path, filename, processing_status
                    FROM documents 
                    WHERE (title IS NULL OR title = '') 
                       AND (file_path IS NULL OR file_path = '')
                       AND (filename IS NULL OR filename = '')
                """)
                
                orphans = cursor.fetchall()
                if orphans:
                    print(f"\nFound {len(orphans)} potential orphaned records:")
                    for orphan in orphans:
                        print(f"  ID: {orphan['id']}, Type: {orphan['document_type']}, Status: {orphan['processing_status']}")
                else:
                    print("\nNo obvious orphaned records found.")
                
                # Check processing status distribution
                print("\n" + "=" * 70)
                print("PROCESSING STATUS ANALYSIS")
                print("=" * 70)
                
                cursor.execute("""
                    SELECT processing_status, COUNT(*) as count
                    FROM documents 
                    GROUP BY processing_status
                    ORDER BY count DESC
                """)
                
                print("\nProcessing status breakdown:")
                for row in cursor.fetchall():
                    status = row['processing_status'] or 'NULL'
                    count = row['count']
                    print(f"  {status}: {count} documents")
                
    except Exception as e:
        logger.exception(f"Debug failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(debug_document_counts())
