#!/usr/bin/env python3
"""
Debug script to understand document structure and file_path usage.
This will help determine the correct search logic for the document manager.
"""

import logging
from ingestion.core.postgres_manager import PostgreSQLManager, PostgreSQLConfig
from rag_manager.core.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Analyze document structure and file_path usage patterns."""
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
                print("DOCUMENTS TABLE SCHEMA ANALYSIS")
                print("=" * 60)
                
                # Check table schema
                cursor.execute("""
                    SELECT column_name, data_type, is_nullable, column_default 
                    FROM information_schema.columns 
                    WHERE table_name = 'documents' 
                    ORDER BY ordinal_position
                """)
                print("\nTable Schema:")
                for row in cursor.fetchall():
                    nullable = "NULL" if row['is_nullable'] == 'YES' else "NOT NULL"
                    default = f" (default: {row['column_default']})" if row['column_default'] else ""
                    print(f"  {row['column_name']}: {row['data_type']} {nullable}{default}")
                
                print("\n" + "=" * 60)
                print("DOCUMENT TYPE ANALYSIS")
                print("=" * 60)
                
                # Check document_type distribution
                cursor.execute("""
                    SELECT document_type, COUNT(*) as count 
                    FROM documents 
                    GROUP BY document_type 
                    ORDER BY count DESC
                """)
                print("\nDocument Types:")
                for row in cursor.fetchall():
                    doc_type = row['document_type'] or 'NULL'
                    count = row['count']
                    print(f"  {doc_type}: {count} documents")
                
                print("\n" + "=" * 60)
                print("FILE_PATH PATTERNS BY DOCUMENT TYPE")
                print("=" * 60)
                
                # Analyze file_path patterns for files vs URLs
                cursor.execute("""
                    SELECT document_type, title, file_path, content_type 
                    FROM documents 
                    ORDER BY document_type, created_at DESC 
                    LIMIT 20
                """)
                
                file_docs = []
                url_docs = []
                
                for row in cursor.fetchall():
                    doc_dict = dict(row)
                    if doc_dict['document_type'] == 'file':
                        file_docs.append(doc_dict)
                    elif doc_dict['document_type'] == 'url':
                        url_docs.append(doc_dict)
                
                print(f"\nFILE DOCUMENTS (showing up to 10):")
                for i, doc in enumerate(file_docs[:10]):
                    print(f"  {i+1}. Title: '{doc['title']}'")
                    print(f"      File Path: '{doc['file_path']}'")
                    print(f"      Content Type: '{doc['content_type']}'")
                    print()
                
                print(f"URL DOCUMENTS (showing up to 10):")
                for i, doc in enumerate(url_docs[:10]):
                    print(f"  {i+1}. Title: '{doc['title']}'")
                    print(f"      File Path: '{doc['file_path']}'")
                    print(f"      Content Type: '{doc['content_type']}'")
                    print()
                
                print("=" * 60)
                print("CONFIGURATION ANALYSIS")
                print("=" * 60)
                
                print(f"\nFile Upload Configuration:")
                print(f"  UPLOAD_FOLDER (staging): '{config.UPLOAD_FOLDER}'")
                print(f"  UPLOADED_FOLDER: '{config.UPLOADED_FOLDER}'")
                print(f"  DELETED_FOLDER: '{config.DELETED_FOLDER}'")
                print(f"  SNAPSHOT_DIR: '{config.SNAPSHOT_DIR}'")
                
                print(f"\nFile Extensions Allowed:")
                print(f"  {list(config.ALLOWED_EXTENSIONS)}")
                
                print("\n" + "=" * 60)
                print("SEARCH PATTERN ANALYSIS")
                print("=" * 60)
                
                # Test some search patterns
                test_filenames = []
                
                # Get some actual filenames from the data
                cursor.execute("""
                    SELECT DISTINCT title, file_path, document_type 
                    FROM documents 
                    WHERE document_type = 'file' 
                    LIMIT 5
                """)
                
                print("\nTesting search patterns with actual data:")
                for row in cursor.fetchall():
                    title = row['title']
                    file_path = row['file_path']
                    doc_type = row['document_type']
                    
                    print(f"\nDocument: type={doc_type}, title='{title}', file_path='{file_path}'")
                    
                    # Test different search strategies
                    if title:
                        # Search by exact title
                        cursor.execute("SELECT COUNT(*) as count FROM documents WHERE title = %s", (title,))
                        title_exact = cursor.fetchone()['count']
                        print(f"  Search by title exact: {title_exact} matches")
                        
                        # Search by title in file_path
                        cursor.execute("SELECT COUNT(*) as count FROM documents WHERE file_path LIKE %s", (f'%{title}%',))
                        title_in_path = cursor.fetchone()['count']
                        print(f"  Search by title in file_path: {title_in_path} matches")
                    
                    if file_path:
                        # Extract just filename from file_path
                        import os
                        filename_only = os.path.basename(file_path)
                        if filename_only:
                            cursor.execute("SELECT COUNT(*) as count FROM documents WHERE title = %s", (filename_only,))
                            filename_as_title = cursor.fetchone()['count']
                            print(f"  Search by filename '{filename_only}' as title: {filename_as_title} matches")
                
    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
