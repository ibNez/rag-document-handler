"""Document manager for file-based content processing."""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DocumentManager:
    """Manager for document metadata and processing operations."""
    
    def __init__(self, postgres_manager):
        """Initialize with PostgreSQL manager."""
        self.postgres = postgres_manager
        logger.info("Document Manager initialized")
    
    def upsert_document_metadata(self, filename: str, metadata: Dict[str, Any]) -> None:
        """Update or insert document metadata."""
        try:
            # Map metadata for PostgreSQL storage with explicit columns
            insert_data = {
                'document_id': filename,
                'title': metadata.get('title', filename),
                'content_preview': metadata.get('content_preview', ''),
                'file_path': metadata.get('file_path', ''),
                'content_type': metadata.get('content_type', ''),
                'file_size': metadata.get('file_size', 0),
                'word_count': metadata.get('word_count', 0),
                'page_count': metadata.get('page_count', 0),
                'chunk_count': metadata.get('chunk_count', 0),
                'avg_chunk_chars': metadata.get('avg_chunk_chars', 0),
                'median_chunk_chars': metadata.get('median_chunk_chars', 0),
                'top_keywords': metadata.get('top_keywords', []),
                'processing_time_seconds': metadata.get('processing_time_seconds', 0),
                'processing_status': metadata.get('processing_status', 'pending')
            }
            
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO documents (
                            document_id, title, content_preview, file_path, content_type, 
                            file_size, word_count, page_count, chunk_count, avg_chunk_chars,
                            median_chunk_chars, top_keywords, processing_time_seconds, processing_status
                        ) VALUES (
                            %(document_id)s, %(title)s, %(content_preview)s, %(file_path)s, 
                            %(content_type)s, %(file_size)s, %(word_count)s, %(page_count)s,
                            %(chunk_count)s, %(avg_chunk_chars)s, %(median_chunk_chars)s,
                            %(top_keywords)s, %(processing_time_seconds)s, %(processing_status)s
                        ) ON CONFLICT (document_id) DO UPDATE SET
                            title = EXCLUDED.title,
                            content_preview = EXCLUDED.content_preview,
                            file_path = EXCLUDED.file_path,
                            content_type = EXCLUDED.content_type,
                            file_size = EXCLUDED.file_size,
                            word_count = EXCLUDED.word_count,
                            page_count = EXCLUDED.page_count,
                            chunk_count = EXCLUDED.chunk_count,
                            avg_chunk_chars = EXCLUDED.avg_chunk_chars,
                            median_chunk_chars = EXCLUDED.median_chunk_chars,
                            top_keywords = EXCLUDED.top_keywords,
                            processing_time_seconds = EXCLUDED.processing_time_seconds,
                            processing_status = EXCLUDED.processing_status,
                            updated_at = NOW()
                    """, insert_data)
                    conn.commit()
            logger.info(f"Metadata stored successfully for {filename}")
        except Exception as e:
            logger.exception(f"Failed to upsert metadata for {filename}")
            raise

    def get_document_metadata(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get document metadata."""
        try:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT * FROM documents WHERE document_id = %s", (filename,))
                    row = cursor.fetchone()
                    if not row:
                        logger.info(f"Metadata lookup miss for {filename}")
                        return None
                    
                    d = dict(row)
                    # Map PostgreSQL schema back to expected format
                    d['filename'] = d['document_id']
                    
                    # Convert datetime objects to strings
                    for key, value in d.items():
                        if isinstance(value, datetime):
                            d[key] = value.isoformat() if value else None
                    
                    logger.info(f"Metadata lookup succeeded for {filename}")
                    return d
        except Exception as e:
            logger.exception(f"Metadata lookup failed for {filename}")
            return None



    def delete_document_metadata(self, filename: str) -> None:
        """Remove a document metadata row permanently."""
        try:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("DELETE FROM documents WHERE document_id = %s", (filename,))
                    removed = cursor.rowcount
                    conn.commit()
            logger.info(f"Metadata delete for {filename}; removed_row={removed > 0}")
        except Exception as e:
            logger.exception(f"Failed to delete metadata row for {filename}")

    def get_knowledgebase_metadata(self) -> Dict[str, Any]:
        """Get aggregated knowledge base metadata from PostgreSQL."""
        kb_meta = {
            'documents_total': 0,
            'avg_words_per_doc': 0,
            'avg_chunks_per_doc': 0,
            'median_chunk_chars': 0,
            'top_keywords': []
        }
        
        try:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Get document count first - simple query
                    cursor.execute("SELECT COUNT(*) as count FROM documents")
                    result = cursor.fetchone()
                    if result:
                        kb_meta['documents_total'] = int(result['count'] or 0)
                    
                    # Only try more complex queries if we have documents
                    if kb_meta['documents_total'] > 0:
                        try:
                            # Get averages from explicit columns
                            cursor.execute("""
                                SELECT 
                                    AVG(COALESCE(word_count, 0)) as avg_words,
                                    AVG(COALESCE(chunk_count, 0)) as avg_chunks,
                                    AVG(COALESCE(median_chunk_chars, 0)) as avg_median_chars
                                FROM documents
                                WHERE word_count IS NOT NULL AND chunk_count IS NOT NULL
                            """)
                            row = cursor.fetchone()
                            if row:
                                kb_meta['avg_words_per_doc'] = int(row['avg_words'] or 0)
                                kb_meta['avg_chunks_per_doc'] = int(row['avg_chunks'] or 0)
                                kb_meta['median_chunk_chars'] = int(row['avg_median_chars'] or 0)
                        except Exception as e:
                            logger.warning(f"Failed to get document averages: {e}")
                        
                        try:
                            # Get top keywords from explicit column
                            cursor.execute("""
                                SELECT top_keywords
                                FROM documents 
                                WHERE top_keywords IS NOT NULL AND array_length(top_keywords, 1) > 0
                                LIMIT 10
                            """)
                            all_keywords = []
                            for row in cursor.fetchall():
                                keywords = row['top_keywords']
                                if keywords and isinstance(keywords, list):
                                    all_keywords.extend(keywords[:3])  # Top 3 from each doc
                            
                            # Get most common
                            from collections import Counter
                            if all_keywords:
                                kb_meta['top_keywords'] = [kw for kw, _ in Counter(all_keywords).most_common(8)]
                        except Exception as e:
                            logger.warning(f"Failed to get top keywords: {e}")
                    
                    return kb_meta
        except Exception as e:
            logger.error(f"Failed to get knowledgebase metadata: {e}")
            return kb_meta
