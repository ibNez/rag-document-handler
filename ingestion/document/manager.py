"""Document manager for file-based content processing."""

import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Protocol
from ingestion.utils.file_filters import should_ignore_file
from rag_manager.core.models import DocumentMetadata

logger = logging.getLogger(__name__)

class VectorStore(Protocol):
    """Protocol for vector storage operations."""
    def store_embeddings(self, doc_id: str, embeddings: List[float]) -> None: ...
    def similarity_search(self, query_vector: List[float], limit: int) -> List[str]: ...
    def delete_document(self, doc_id: str) -> None: ...

class DocumentManager:
    """Manager for document metadata and processing operations."""
    
    def __init__(self, postgres_manager, vector_store: Optional[VectorStore] = None):
        """Initialize with PostgreSQL manager and optional vector store."""
        self.postgres = postgres_manager
        self.vector_store = vector_store
        logger.info("Document Manager initialized")
    
    def upsert_document_metadata(self, filename: str, metadata: Dict[str, Any]) -> None:
        """Update or insert document metadata."""
        # Filter out system files
        if should_ignore_file(filename):
            logger.debug(f"Ignoring system file during upsert: {filename}")
            return
            
        try:
            # Map metadata for PostgreSQL storage with explicit columns
            insert_data = {
                'title': metadata.get('title', filename),
                'content_preview': metadata.get('content_preview', ''),
                'file_path': metadata.get('file_path', ''),
                'filename': filename,  # Always include the filename for UPSERT conflict resolution
                'content_type': metadata.get('content_type', ''),
                'file_size': metadata.get('file_size', 0),
                'word_count': metadata.get('word_count', 0),
                'page_count': metadata.get('page_count', 0),
                'chunk_count': metadata.get('chunk_count', 0),
                'avg_chunk_chars': metadata.get('avg_chunk_chars', 0),
                'median_chunk_chars': metadata.get('median_chunk_chars', 0),
                'top_keywords': metadata.get('top_keywords', []),
                'processing_time_seconds': metadata.get('processing_time_seconds', 0),
                'processing_status': metadata.get('processing_status', 'pending'),
                'filename': filename,  # Store the filename parameter as-is for now
                'document_type': 'file'
            }
            
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Use proper PostgreSQL UPSERT with ON CONFLICT
                    # Conflict detection based on file_path + document_type (natural key)
                    cursor.execute("""
                        INSERT INTO documents (
                            title, content_preview, file_path, content_type, 
                            file_size, word_count, page_count, chunk_count, avg_chunk_chars,
                            median_chunk_chars, top_keywords, processing_time_seconds, processing_status,
                            filename, document_type, created_at, updated_at
                        ) VALUES (
                            %(title)s, %(content_preview)s, %(file_path)s, 
                            %(content_type)s, %(file_size)s, %(word_count)s, %(page_count)s,
                            %(chunk_count)s, %(avg_chunk_chars)s, %(median_chunk_chars)s,
                            %(top_keywords)s, %(processing_time_seconds)s, %(processing_status)s,
                            %(filename)s, %(document_type)s, NOW(), NOW()
                        )
                        ON CONFLICT (filename) 
                        DO UPDATE SET
                            title = EXCLUDED.title,
                            content_preview = EXCLUDED.content_preview,
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
                            filename = EXCLUDED.filename,
                            updated_at = NOW()
                    """, insert_data)
                    conn.commit()
            logger.info(f"Metadata stored successfully for {filename}")
        except Exception as e:
            logger.exception(f"Failed to upsert metadata for {filename}")
            raise

    def get_document_metadata(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get document metadata."""
        # Filter out system files
        if should_ignore_file(filename):
            logger.debug(f"Ignoring system file: {filename}")
            return None
            
        try:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Search strategy depends on document type:
                    # - For file documents: search by filename in file_path (basename match)
                    # - For URL documents: search by full file_path (URL match)
                    # - Also check title field as fallback for both types
                    cursor.execute("""
                        SELECT * FROM documents 
                        WHERE title = %s 
                           OR (document_type = 'file' AND file_path LIKE %s)
                           OR (document_type = 'url' AND file_path = %s)
                           OR (filename IS NOT NULL AND filename = %s)
                    """, (filename, f'%/{filename}', filename, filename))
                    row = cursor.fetchone()
                    if not row:
                        logger.info(f"Metadata lookup miss for {filename}")
                        return None
                    
                    d = dict(row)
                    # Map PostgreSQL schema back to expected format  
                    d['filename'] = filename  # Use the searched filename
                    
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
        # Filter out system files
        if should_ignore_file(filename):
            logger.debug(f"Ignoring system file during delete: {filename}")
            return
            
        try:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Delete strategy: same logic as get_document_metadata
                    # Search by multiple criteria based on document type
                    cursor.execute("""
                        DELETE FROM documents 
                        WHERE title = %s 
                           OR (document_type = 'file' AND file_path LIKE %s)
                           OR (document_type = 'url' AND file_path = %s)
                           OR (filename IS NOT NULL AND filename = %s)
                    """, (filename, f'%/{filename}', filename, filename))
                    removed = cursor.rowcount
                    conn.commit()
            logger.info(f"Metadata delete for {filename}; removed_row={removed > 0}")
        except Exception as e:
            logger.exception(f"Failed to delete metadata row for {filename}")

    def persist_chunks(self, document_id: str, chunks: list, trace_logger: Optional[logging.Logger] = None) -> int:
        """
        Persist a list of LangChain-style Document chunks into PostgreSQL.
        Returns the number of chunks successfully stored.
        """
        if not self.postgres:
            logger.warning("persist_chunks: no postgres manager provided")
            return 0

        stored = 0
        try:
            for idx, ch in enumerate(chunks, start=1):
                try:
                    chunk_text = ch.page_content or ''
                    meta = ch.metadata or {}
                    page_start = meta.get('page')
                    page_end = meta.get('page')
                    section_path = meta.get('section_path')
                    element_types = meta.get('element_types', [])
                    token_count = meta.get('token_count') or (len(chunk_text.split()) if chunk_text else 0)
                    chunk_hash = meta.get('content_hash')
                    topics = meta.get('topics')

                    chunk_id = self.store_document_chunk(
                        document_id=document_id,
                        chunk_text=chunk_text,
                        chunk_ordinal=idx,
                        page_start=page_start,
                        page_end=page_end,
                        section_path=section_path,
                        element_types=element_types,
                        token_count=token_count,
                        chunk_hash=chunk_hash,
                        topics=topics
                    )

                    # Annotate chunk metadata for downstream consumers
                    try:
                        ch.metadata['document_chunk_id'] = str(chunk_id)
                        ch.metadata['document_id'] = str(document_id)
                    except Exception:
                        pass

                    stored += 1
                    if trace_logger:
                        trace_logger.info(f"Stored chunk ordinal={idx} chunk_id={chunk_id} content_hash={chunk_hash}")
                except Exception as e:
                    logger.exception(f"Failed to persist chunk ordinal={idx} for document {document_id}: {e}")
                    if trace_logger:
                        trace_logger.exception(f"Failed to persist chunk ordinal={idx}: {e}")

            # Update document metadata with chunk count and mark completed if any chunks stored
            try:
                with self.postgres.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE documents
                            SET chunk_count = %s,
                                processing_status = CASE WHEN %s > 0 THEN 'completed' ELSE processing_status END,
                                updated_at = NOW()
                            WHERE id = %s
                            """,
                            [stored, stored, document_id]
                        )
                        conn.commit()
                if trace_logger:
                    trace_logger.info(f"Updated document {document_id} metadata: chunk_count={stored}")
            except Exception as e:
                logger.warning(f"Failed to update document metadata for {document_id}: {e}")

        except Exception as e:
            logger.exception(f"persist_chunks failed for document {document_id}: {e}")
            raise

        return stored

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
                    # Get document count first - only count completed/processed documents
                    cursor.execute("SELECT COUNT(*) as count FROM documents WHERE processing_status = 'completed'")
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

    # MOVED FROM rag_manager/managers/postgres_manager.py - Document ingestion methods
    
    def store_document(self, file_path: str, filename: str, 
                      title: Optional[str] = None, content_preview: Optional[str] = None,
                      content_type: Optional[str] = None, file_size: Optional[int] = None,
                      word_count: Optional[int] = None, document_type: str = 'file', **kwargs) -> str:
        """
        Store document metadata and return the UUID.
        
        Args:
            title: Document title (optional)
            content_preview: Preview of document content (optional)
            file_path: REQUIRED - Full path to file for files, URL for URLs
            filename: REQUIRED - Filename for files, descriptive name for URLs
            content_type: MIME type (optional)
            file_size: File size in bytes (optional)
            word_count: Number of words in document (optional)
            document_type: Type of document ('file' or 'url')
            **kwargs: Additional metadata (ignored)
            
        Returns:
            The UUID of the stored document
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Validate required fields according to DEVELOPMENT_RULES
        if not file_path or not file_path.strip():
            logger.error(f"Missing required field: file_path")
            raise ValueError("file_path is required and cannot be empty")
        
        if not filename or not filename.strip():
            logger.error(f"Missing required field: filename")
            raise ValueError("filename is required and cannot be empty")
            
        logger.info(f"Storing document: file_path='{file_path}', filename='{filename}', type='{document_type}'")
        
        query = """
            INSERT INTO documents (title, content_preview, file_path, filename,
                                 content_type, file_size, word_count, document_type, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            RETURNING id
        """
        
        with self.postgres.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, [
                    title, content_preview, file_path, filename,
                    content_type, file_size, word_count, document_type
                ])
                result = cur.fetchone()
                conn.commit()
                document_id = str(result['id'])
                logger.info(f"Stored document metadata: {document_id}")
                return document_id
    
    def update_processing_status(self, document_id: str, status: str) -> None:
        """Update document processing status by ID."""
        query = """
            UPDATE documents 
            SET processing_status = %s, 
                indexed_at = CASE WHEN %s = 'completed' THEN NOW() ELSE indexed_at END,
                updated_at = NOW()
            WHERE id = %s
        """
        
        with self.postgres.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, [status, status, document_id])
                conn.commit()
                logger.debug(f"Updated document {document_id} status to {status}")
    
    def store_document_chunk(self, document_id: str, chunk_text: str, 
                           chunk_ordinal: int, page_start: Optional[int] = None, 
                           page_end: Optional[int] = None, section_path: Optional[str] = None,
                           element_types: Optional[List[str]] = None, token_count: Optional[int] = None,
                           chunk_hash: Optional[str] = None, topics: Optional[str] = None,
                           embedding_version: str = 'mxbai-embed-large') -> str:
        """
        Store document chunk for retrieval.
        
        Args:
            document_id: Parent document identifier
            chunk_text: Text content of the chunk
            chunk_ordinal: Sequential chunk number within document
            page_start: Starting page number (optional)
            page_end: Ending page number (optional)
            section_path: Hierarchical section path (e.g., "H1 > H2 > List")
            element_types: List of element types from Unstructured
            token_count: Number of tokens in chunk
            chunk_hash: Content hash for deduplication
            topics: Comma-separated list of topics (optional)
            embedding_version: Version/model used for embeddings
            
        Returns:
            The UUID of the created chunk
        """
        query = """
            INSERT INTO document_chunks (
                document_id, chunk_text, chunk_ordinal, page_start, page_end,
                section_path, element_types, token_count, chunk_hash, topics, embedding_version
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (document_id, chunk_ordinal) 
            DO UPDATE SET 
                chunk_text = EXCLUDED.chunk_text,
                page_start = EXCLUDED.page_start,
                page_end = EXCLUDED.page_end,
                section_path = EXCLUDED.section_path,
                element_types = EXCLUDED.element_types,
                token_count = EXCLUDED.token_count,
                chunk_hash = EXCLUDED.chunk_hash,
                topics = EXCLUDED.topics,
                embedding_version = EXCLUDED.embedding_version
            RETURNING id
        """
        
        # Instrumented storage: log parameters and capture DB exceptions for debugging
        try:
            logger.info(
                "Storing document chunk: document_id=%s chunk_ordinal=%s chunk_hash=%s text_len=%s topics=%s",
                document_id,
                chunk_ordinal,
                (chunk_hash or '')[:32],
                len(chunk_text or ''),
                topics or 'None'
            )

            with self.postgres.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, [
                        document_id, chunk_text, chunk_ordinal, page_start, page_end,
                        section_path, element_types, token_count, chunk_hash, topics, embedding_version
                    ])
                    result = cur.fetchone()
                    conn.commit()
                    chunk_id = str(result['id'])
                    logger.info("Stored document chunk id=%s for document_id=%s", chunk_id, document_id)
                    return chunk_id

        except Exception as e:
            # Log full exception and re-raise to make failures visible
            logger.exception(
                "Failed to store document chunk for document_id=%s ordinal=%s chunk_hash=%s: %s",
                document_id,
                chunk_ordinal,
                (chunk_hash or '')[:32],
                str(e)
            )
            raise
    
    def delete_document_chunks(self, document_id: str) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Number of deleted chunks
        """
        query = "DELETE FROM document_chunks WHERE document_id = %s"
        
        with self.postgres.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, [document_id])
                deleted_count = cur.rowcount
                conn.commit()
                logger.info(f"Deleted {deleted_count} chunks for document: {document_id}")
                return deleted_count

    # MOVED FROM ingestion/core/database_manager.py - Document coordination methods
    
    def store_document_with_metadata(self, content: str, metadata: DocumentMetadata) -> str:
        """
        Store document in both metadata and vector databases.
        Returns the document ID.
        """
        try:
            # Store metadata in PostgreSQL
            doc_uuid = self.store_document(
                file_path=metadata.file_path,
                filename=metadata.filename,
                title=metadata.title,
                content_preview=metadata.content_preview,
                content_type=metadata.content_type,
                file_size=metadata.file_size,
                word_count=metadata.word_count
            )
            
            # Update processing status
            self.update_processing_status(metadata.document_id, 'processing')
            
            # Store embeddings in vector store if available
            if self.vector_store and content:
                try:
                    # Note: Embedding generation would be handled by the vector store
                    # or passed in as pre-computed embeddings
                    logger.info(f"Vector storage would be handled for document: {metadata.document_id}")
                    self.update_processing_status(metadata.document_id, 'completed')
                except Exception as e:
                    logger.error(f"Failed to store embeddings for {metadata.document_id}: {e}")
                    self.update_processing_status(metadata.document_id, 'failed')
                    raise
            else:
                # Mark as completed even without vector storage for now
                self.update_processing_status(metadata.document_id, 'completed')
            
            logger.info(f"Successfully stored document: {metadata.document_id}")
            return metadata.document_id
            
        except Exception as e:
            logger.error(f"Failed to store document {metadata.document_id}: {e}")
            # Update status to failed
            try:
                self.update_processing_status(metadata.document_id, 'failed')
            except:
                pass  # Don't fail if status update fails
            raise
    
    def search_documents(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search documents using PostgreSQL full-text search."""
        try:
            results = self.postgres.search_documents(query, limit)
            logger.info(f"Found {len(results)} documents for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []
    
    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by ID."""
        try:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, title, content_preview, file_path,
                               content_type, file_size, word_count, processing_status,
                               metadata, created_at, updated_at, indexed_at
                        FROM documents 
                        WHERE id = %s
                    """, [document_id])
                    result = cur.fetchone()
                    return dict(result) if result else None
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None
    
    def update_document_metadata_jsonb(self, document_id: str, metadata: Dict[str, Any]) -> bool:
        """Update document metadata."""
        try:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE documents 
                        SET metadata = metadata || %s::jsonb,
                            updated_at = NOW()
                        WHERE id = %s
                    """, [json.dumps(metadata), document_id])
                    conn.commit()
                    logger.info(f"Updated metadata for document: {document_id}")
                    return True
        except Exception as e:
            logger.error(f"Failed to update metadata for {document_id}: {e}")
            return False
    
    def delete_document_full(self, document_id: str) -> bool:
        """Delete document from both stores."""
        try:
            # Delete from vector store if available
            if self.vector_store:
                try:
                    self.vector_store.delete_document(document_id)
                except Exception as e:
                    logger.warning(f"Failed to delete from vector store: {e}")
            
            # Delete from PostgreSQL
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM documents WHERE id = %s", [document_id])
                    conn.commit()
                    
            logger.info(f"Deleted document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    # MOVED FROM rag_manager/app.py - Direct SQL operations for ingestion
    
    def get_document_id_by_filename(self, filename: str) -> Optional[str]:
        """Get document ID by filename."""
        try:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM documents WHERE filename = %s", (filename,))
                    result = cur.fetchone()
                    if result:
                        return str(result['id'])
                    return None
        except Exception as e:
            logger.error(f"Failed to get document ID for filename {filename}: {e}")
            return None
    
    def update_document_processing_results(self, document_id: str, title: str, page_count: int, 
                                         chunk_count: int, word_count: int, avg_chunk_chars: int,
                                         median_chunk_chars: int, top_keywords: List[str], 
                                         processing_time_seconds: float) -> bool:
        """Update document metadata after processing."""
        try:
            update_query = """
                UPDATE documents 
                SET 
                    title = %s,
                    page_count = %s,
                    chunk_count = %s,
                    word_count = %s,
                    avg_chunk_chars = %s,
                    median_chunk_chars = %s,
                    top_keywords = %s,
                    processing_time_seconds = %s,
                    processing_status = %s,
                    updated_at = NOW()
                WHERE id = %s
            """
            
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(update_query, [
                        title, page_count, chunk_count, word_count, avg_chunk_chars,
                        median_chunk_chars, top_keywords, processing_time_seconds, 'completed', document_id
                    ])
                    conn.commit()
                    logger.info(f"Updated processing results for document: {document_id}")
                    return True
        except Exception as e:
            logger.error(f"Failed to update processing results for {document_id}: {e}")
            return False
    
    def cleanup_orphaned_pending_documents(self, filename: str) -> int:
        """Clean up orphaned database records with pending status."""
        try:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM documents WHERE filename = %s AND processing_status = 'pending'", [filename])
                    deleted_rows = cur.rowcount
                    conn.commit()
                    if deleted_rows > 0:
                        logger.info(f"Cleaned up {deleted_rows} orphaned database record(s) for file: {filename}")
                    return deleted_rows
        except Exception as e:
            logger.error(f"Failed to clean up orphaned records for {filename}: {e}")
            return 0
    
    def delete_document_by_id(self, document_id: str) -> bool:
        """Delete document by ID."""
        try:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM documents WHERE id = %s", [document_id])
                    deleted_rows = cur.rowcount
                    conn.commit()
                    logger.info(f"Deleted document {document_id} (rows: {deleted_rows})")
                    return deleted_rows > 0
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
