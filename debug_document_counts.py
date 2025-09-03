#!/usr/bin/env python3
"""
Debug script to investigate document count discrepancy.
Following DEVELOPMENT_RULES.md for all development requirements.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion.core.postgres_manager import PostgreSQLManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_document_counts():
    """Check document counts vs URL counts to find discrepancy."""
    
    postgres = PostgreSQLManager()
    
    try:
        with postgres.get_connection() as conn:
            with conn.cursor() as cursor:
                # Count documents
                cursor.execute("SELECT COUNT(*) as count FROM documents")
                result = cursor.fetchone()
                total_docs = int(result['count'] or 0) if result else 0
                print(f"Total documents in 'documents' table: {total_docs}")
                
                # Count URLs
                cursor.execute("SELECT COUNT(*) as count FROM urls WHERE status = 'active'")
                result = cursor.fetchone()
                total_urls = int(result['count'] or 0) if result else 0
                print(f"Total URLs in 'urls' table: {total_urls}")
                
                # Count parent URLs (main URLs)
                cursor.execute("SELECT COUNT(*) as count FROM urls WHERE status = 'active' AND parent_url_id IS NULL")
                result = cursor.fetchone()
                parent_urls = int(result['count'] or 0) if result else 0
                print(f"Parent URLs: {parent_urls}")
                
                # Count child URLs (sub-URLs)
                cursor.execute("SELECT COUNT(*) as count FROM urls WHERE status = 'active' AND parent_url_id IS NOT NULL")
                result = cursor.fetchone()
                child_urls = int(result['count'] or 0) if result else 0
                print(f"Child URLs (sub-URLs): {child_urls}")
                
                print(f"Total URLs (parent + child): {parent_urls + child_urls}")
                print(f"Discrepancy: {total_docs - total_urls} documents")
                
                # Check if documents table has document_type column
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'documents' AND column_name = 'document_type'
                """)
                has_doc_type = cursor.fetchone() is not None
                print(f"Documents table has document_type column: {has_doc_type}")
                
                if has_doc_type:
                    # Count documents by type
                    cursor.execute("""
                        SELECT document_type, COUNT(*) as count 
                        FROM documents 
                        GROUP BY document_type
                        ORDER BY count DESC
                    """)
                    print("\nDocuments by type:")
                    for row in cursor.fetchall():
                        doc_type = row['document_type'] or 'NULL'
                        count = row['count']
                        print(f"  {doc_type}: {count}")
                
                # Check for documents that don't correspond to URLs
                cursor.execute("""
                    SELECT d.id, d.title, d.file_path, d.document_type,
                           u.url as matching_url, u.parent_url_id
                    FROM documents d
                    LEFT JOIN urls u ON d.file_path = u.url
                    WHERE u.url IS NULL
                    LIMIT 10
                """)
                
                orphaned_docs = cursor.fetchall()
                if orphaned_docs:
                    print(f"\nDocuments without matching URLs: {len(orphaned_docs)}")
                    for doc in orphaned_docs:
                        doc_dict = dict(doc)
                        print(f"  ID: {doc_dict['id']}")
                        print(f"  Title: {doc_dict['title']}")
                        print(f"  File Path (URL): {doc_dict['file_path']}")
                        print(f"  Document Type: {doc_dict['document_type']}")
                        print()
                else:
                    print("\nNo orphaned documents found")
                
                # Check for URLs that don't have corresponding documents
                cursor.execute("""
                    SELECT u.id, u.url, u.title, u.parent_url_id,
                           d.id as doc_id
                    FROM urls u
                    LEFT JOIN documents d ON u.url = d.file_path
                    WHERE u.status = 'active' AND d.id IS NULL
                    LIMIT 10
                """)
                
                orphaned_urls = cursor.fetchall()
                if orphaned_urls:
                    print(f"\nURLs without matching documents: {len(orphaned_urls)}")
                    for url in orphaned_urls:
                        url_dict = dict(url)
                        print(f"  URL ID: {url_dict['id']}")
                        print(f"  URL: {url_dict['url']}")
                        print(f"  Title: {url_dict['title']}")
                        print(f"  Parent URL ID: {url_dict['parent_url_id']}")
                        print()
                else:
                    print("\nNo orphaned URLs found")
                    
    except Exception as e:
        logger.error(f"Error checking document counts: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_document_counts()
