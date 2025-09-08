#!/usr/bin/env python
"""
Document Data Manager

Pure data access operations for document content management.
No business logic - only database operations for PostgreSQL and Milvus.
"""

import logging
from typing import Any, Dict, List, Optional

from .base_data import BaseDataManager

logger = logging.getLogger(__name__)


class DocumentDataManager(BaseDataManager):
    """
    Pure data access manager for document operations.
    
    Handles all PostgreSQL and Milvus operations for document content
    without any business logic or orchestration.
    """
    
    def __init__(self, postgres_manager: Any, milvus_manager: Optional[Any] = None) -> None:
        """
        Initialize document data manager.
        
        Args:
            postgres_manager: PostgreSQL connection manager
            milvus_manager: Optional Milvus vector store manager
        """
        super().__init__(postgres_manager, milvus_manager)
        logger.info("DocumentDataManager initialized for pure data operations")
    
    # =============================================================================
    # Document Metadata Operations
    # =============================================================================
    
    def upsert_document_metadata(self, filename: str, metadata: Dict[str, Any]) -> None:
        """
        Insert or update document metadata in PostgreSQL.
        
        Args:
            filename: Document filename
            metadata: Document metadata dictionary
        """
        try:
            query = """
                INSERT INTO documents (
                    filename, title, content_preview, file_path, content_type,
                    file_size, word_count, page_count, chunk_count, avg_chunk_chars,
                    median_chunk_chars, keywords, processing_time_seconds, processing_status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (filename) 
                DO UPDATE SET
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
                    keywords = EXCLUDED.keywords,
                    processing_time_seconds = EXCLUDED.processing_time_seconds,
                    processing_status = EXCLUDED.processing_status,
                    updated_at = CURRENT_TIMESTAMP
            """
            
            params = (
                filename,
                metadata.get('title', filename),
                metadata.get('content_preview', ''),
                metadata.get('file_path', ''),
                metadata.get('content_type', ''),
                metadata.get('file_size', 0),
                metadata.get('word_count', 0),
                metadata.get('page_count', 0),
                metadata.get('chunk_count', 0),
                metadata.get('avg_chunk_chars', 0),
                metadata.get('median_chunk_chars', 0),
                ', '.join(metadata.get('keywords', [])) if metadata.get('keywords') else '',
                metadata.get('processing_time_seconds', 0),
                metadata.get('processing_status', 'pending')
            )
            
            self.execute_query(query, params)
            logger.debug(f"Successfully upserted document metadata: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to upsert document metadata {filename}: {e}")
            raise
    
    def get_document_metadata(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve document metadata by filename.
        
        Args:
            filename: Document filename
            
        Returns:
            Document metadata dictionary or None if not found
        """
        query = "SELECT * FROM documents WHERE filename = %s"
        result = self.execute_query(query, (filename,), fetch_one=True)
        if result:
            result_dict = dict(result)
            # Convert keywords back to list for UI compatibility
            keywords_str = result_dict.get('keywords', '')
            result_dict['keywords'] = [k.strip() for k in keywords_str.split(',') if k.strip()] if keywords_str else []
            return result_dict
        return None
    
    def delete_document_metadata(self, filename: str) -> bool:
        """
        Delete document metadata by filename.
        
        Args:
            filename: Document filename
            
        Returns:
            True if deleted, False if not found
        """
        try:
            query = "DELETE FROM documents WHERE filename = %s"
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (filename,))
                    deleted = cur.rowcount > 0
                conn.commit()
            
            if deleted:
                logger.debug(f"Deleted document metadata: {filename}")
            else:
                logger.warning(f"Document metadata not found for deletion: {filename}")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to delete document metadata {filename}: {e}")
            raise
    
    # =============================================================================
    # Document Statistics Operations
    # =============================================================================
    
    def get_knowledgebase_metadata(self) -> Dict[str, Any]:
        """
        Get aggregated knowledge base metadata from PostgreSQL.
        
        Returns:
            Dictionary with knowledge base statistics
        """
        kb_meta = {
            'documents_total': 0,
            'avg_words_per_doc': 0,
            'avg_chunks_per_doc': 0,
            'median_chunk_chars': 0,
            'keywords': []
        }
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Get document count first - only count completed/processed file documents (not URLs)
                    cursor.execute("SELECT COUNT(*) as count FROM documents WHERE processing_status = 'completed' AND document_type = 'file'")
                    result = cursor.fetchone()
                    if result:
                        kb_meta['documents_total'] = int(result['count'] or 0)
                    
                    # Only try more complex queries if we have documents
                    if kb_meta['documents_total'] > 0:
                        try:
                            # Get averages from explicit columns - only for file documents
                            cursor.execute("""
                                SELECT 
                                    AVG(COALESCE(word_count, 0)) as avg_words,
                                    AVG(COALESCE(chunk_count, 0)) as avg_chunks,
                                    AVG(COALESCE(median_chunk_chars, 0)) as avg_median_chars
                                FROM documents
                                WHERE word_count IS NOT NULL AND chunk_count IS NOT NULL AND document_type = 'file'
                            """)
                            row = cursor.fetchone()
                            if row:
                                kb_meta['avg_words_per_doc'] = int(row['avg_words'] or 0)
                                kb_meta['avg_chunks_per_doc'] = int(row['avg_chunks'] or 0)
                                kb_meta['median_chunk_chars'] = int(row['avg_median_chars'] or 0)
                        except Exception as e:
                            logger.warning(f"Failed to get document averages: {e}")
                        
                        try:
                            # Get top keywords from comma-separated string
                            cursor.execute("""
                                SELECT keywords
                                FROM documents 
                                WHERE keywords IS NOT NULL AND keywords != ''
                                LIMIT 10
                            """)
                            all_keywords = []
                            for row in cursor.fetchall():
                                keywords_str = row['keywords']
                                if keywords_str:
                                    keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
                                    all_keywords.extend(keywords[:3])  # Top 3 from each doc
                            
                            # Get most common
                            from collections import Counter
                            if all_keywords:
                                kb_meta['keywords'] = [kw for kw, _ in Counter(all_keywords).most_common(8)]
                        except Exception as e:
                            logger.warning(f"Failed to get top keywords: {e}")
                    
                    return kb_meta
        except Exception as e:
            logger.error(f"Failed to get knowledgebase metadata: {e}")
            return kb_meta
    
    def get_document_statistics(self) -> Dict[str, Any]:
        """
        Get global document statistics.
        
        Returns:
            Dictionary with document statistics
        """
        try:
            query = """
                SELECT 
                    COUNT(*) as total_documents,
                    COUNT(*) FILTER (WHERE processing_status = 'completed') as processed_documents,
                    COUNT(*) FILTER (WHERE processing_status = 'pending') as pending_documents,
                    COUNT(*) FILTER (WHERE processing_status = 'error') as error_documents,
                    SUM(file_size) as total_size_bytes,
                    SUM(word_count) as total_words,
                    SUM(chunk_count) as total_chunks,
                    AVG(processing_time_seconds) as avg_processing_time
                FROM documents
            """
            
            result = self.execute_query(query, fetch_one=True)
            
            if result:
                return {
                    'total_documents': result['total_documents'],
                    'processed_documents': result['processed_documents'],
                    'pending_documents': result['pending_documents'],
                    'error_documents': result['error_documents'],
                    'total_size_bytes': result['total_size_bytes'] or 0,
                    'total_words': result['total_words'] or 0,
                    'total_chunks': result['total_chunks'] or 0,
                    'avg_processing_time': float(result['avg_processing_time']) if result['avg_processing_time'] else 0.0
                }
            else:
                return {
                    'total_documents': 0,
                    'processed_documents': 0,
                    'pending_documents': 0,
                    'error_documents': 0,
                    'total_size_bytes': 0,
                    'total_words': 0,
                    'total_chunks': 0,
                    'avg_processing_time': 0.0
                }
                
        except Exception as e:
            logger.error(f"Failed to get document statistics: {e}")
            raise
    
    # =============================================================================
    # Document Storage Operations 
    # =============================================================================
    
    def store_document(self, file_path: str, filename: str, 
                      title: Optional[str] = None, content_preview: Optional[str] = None,
                      content_type: Optional[str] = None, file_size: Optional[int] = None,
                      word_count: Optional[int] = None, document_type: str = 'file', **kwargs) -> str:
        """
        Store document metadata and return the UUID.
        
        Args:
            file_path: Full path to file for files, URL for URLs
            filename: Filename for files, descriptive name for URLs
            title: Document title (optional)
            content_preview: Preview of document content (optional)
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
        # Validate required fields
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
        
        with self.get_connection() as conn:
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
        """
        Update document processing status.
        
        Args:
            document_id: Document identifier
            status: New processing status
        """
        try:
            query = """
                UPDATE documents 
                SET processing_status = %s, updated_at = NOW() 
                WHERE id = %s
            """
            self.execute_query(query, (status, document_id))
            logger.debug(f"Updated document {document_id} status to {status}")
        except Exception as e:
            logger.error(f"Failed to update document status {document_id}: {e}")
            raise

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
        
        try:
            logger.info(
                "Storing document chunk: document_id=%s chunk_ordinal=%s chunk_hash=%s text_len=%s topics=%s",
                document_id,
                chunk_ordinal,
                (chunk_hash or '')[:32],
                len(chunk_text or ''),
                topics or 'None'
            )

            with self.get_connection() as conn:
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
            logger.exception(
                "Failed to store document chunk for document_id=%s ordinal=%s chunk_hash=%s: %s",
                document_id,
                chunk_ordinal,
                (chunk_hash or '')[:32],
                str(e)
            )
            raise

    def persist_chunks(self, document_id: str, chunks: list, trace_logger: Optional[logging.Logger] = None) -> int:
        """
        Persist a list of LangChain-style Document chunks into PostgreSQL.
        
        Args:
            document_id: Document identifier
            chunks: List of LangChain Document chunks
            trace_logger: Optional logger for detailed tracing
            
        Returns:
            Number of chunks successfully stored
        """
        if not chunks:
            logger.warning("persist_chunks: no chunks provided")
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
                with self.get_connection() as conn:
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

    def store_document_with_metadata(self, content: str, metadata: Any) -> str:
        """
        Store document with metadata object.
        
        Args:
            content: Document content
            metadata: DocumentMetadata object
            
        Returns:
            Document identifier
        """
        # Convert metadata object to dict for storage
        metadata_dict = {}
        if hasattr(metadata, '__dict__'):
            metadata_dict = metadata.__dict__
        elif isinstance(metadata, dict):
            metadata_dict = metadata
        
        return self.store_document(
            file_path=metadata_dict.get('file_path', ''),
            filename=metadata_dict.get('filename', ''),
            title=metadata_dict.get('title'),
            content_preview=content[:500] if content else '',
            content_type=metadata_dict.get('content_type'),
            file_size=metadata_dict.get('file_size'),
            word_count=len(content.split()) if content else 0,
            document_type=metadata_dict.get('document_type', 'file')
        )

    def search_documents(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search documents using full text search.
        
        Args:
            query: Search query
            limit: Maximum results to return
            
        Returns:
            List of document dictionaries
        """
        try:
            search_query = """
                SELECT id, filename, title, content_preview, file_path, 
                       content_type, word_count, chunk_count,
                       ts_rank_cd(to_tsvector('english', COALESCE(title, '') || ' ' || COALESCE(content_preview, '')), 
                                  plainto_tsquery('english', %s)) as rank
                FROM documents
                WHERE to_tsvector('english', COALESCE(title, '') || ' ' || COALESCE(content_preview, '')) 
                      @@ plainto_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT %s
            """
            
            results = self.execute_query(search_query, (query, query, limit), fetch_all=True)
            
            documents = []
            if results:
                for row in results:
                    doc = {
                        'id': str(row['id']),
                        'filename': row['filename'],
                        'title': row['title'],
                        'content_preview': row['content_preview'],
                        'file_path': row['file_path'],
                        'content_type': row['content_type'],
                        'word_count': row['word_count'],
                        'chunk_count': row['chunk_count'],
                        'fts_rank': float(row['rank'])
                    }
                    documents.append(doc)
            
            logger.debug(f"Document search returned {len(documents)} results for query: {query[:50]}...")
            return documents
            
        except Exception as e:
            logger.error(f"Document search failed for query '{query}': {e}")
            raise

    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document by identifier.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document dictionary or None if not found
        """
        query = "SELECT * FROM documents WHERE id = %s"
        result = self.execute_query(query, (document_id,), fetch_one=True)
        return dict(result) if result else None

    # =============================================================================
    # Document Enrichment Operations (for retrieval)
    # =============================================================================
    
    def normalize_documents_metadata(self, docs: List[Any]) -> None:
        """
        Normalize metadata for a list of Documents in-place.

        - If metadata contains a single nested 'metadata' dict, unwrap it.
        - Map common alias fields to canonical names:
            - 'id' or 'pk' -> 'chunk_id'
            - keep 'document_id' as-is
        """
        for doc in docs:
            try:
                meta = getattr(doc, 'metadata', None)
                if meta is None:
                    raise TypeError(
                        f"Document.metadata is None for document {getattr(doc,'metadata', None)!r}; expected a dict."
                        " Ensure your retriever returns Document objects with 'metadata' populated."
                    )
                if not isinstance(meta, dict):
                    raise TypeError(
                        f"Document.metadata has unexpected type {type(meta).__name__} for doc {doc!r}; expected dict."
                        " Fix the retriever to return dict metadata or run the dependency verification script."
                    )

                nested = meta.get('metadata')
                if nested is not None and not isinstance(nested, dict):
                    raise TypeError(
                        f"Nested 'metadata' field is not a dict for doc {doc!r}; found {type(nested).__name__}."
                    )
                if isinstance(nested, dict):
                    meta = dict(nested)

                # Map common aliases into table-specific id
                # Map Milvus 'pk' to 'document_chunk_id' if not already present
                if not meta.get('document_chunk_id'):
                    if meta.get('pk'):
                        meta['document_chunk_id'] = meta.get('pk')
                    elif meta.get('id'):
                        meta['document_chunk_id'] = meta.get('id')
                    else:
                        # Only log error if we truly have no chunk identifier
                        logger.debug("Document metadata missing chunk identifier: %s", meta)
                        # Don't raise error - let enrichment handle it
                
                # Some stores use 'doc_id' or 'docid' variants
                if not meta.get('document_id') and meta.get('doc_id'):
                    meta['document_id'] = meta.get('doc_id')
                if not meta.get('document_id') and meta.get('docid'):
                    meta['document_id'] = meta.get('docid')

                doc.metadata = meta
            except Exception:
                # Best-effort normalization; don't fail the whole retrieval
                logger.debug('Failed to normalize document metadata', exc_info=True)

    def batch_enrich_documents_from_postgres(self, docs: List[Any]) -> None:
        """
        Batch-fetch canonical chunk metadata from PostgreSQL and attach to
        LangChain Document.metadata in-place. This avoids duplicating large
        blobs in Milvus while ensuring UI/rerankers have readable metadata.

        Strategy:
        - Collect content_hash and document_id values from the provided docs.
        - Query `document_chunks` for matching chunk rows (batch by ANY(%s)).
        - Query `documents` for titles for any document_ids discovered.
        - Enrich each Document.metadata with: title, chunk_id (chunk row id),
          chunk_hash, chunk_ordinal, page_start/page_end and a short preview.
        """
        if not docs:
            return

        # Collect unique identifiers to look up
        content_hashes = {d.metadata.get('content_hash') for d in docs if d.metadata and d.metadata.get('content_hash')}
        content_hashes = {h for h in content_hashes if h}
        document_ids = {d.metadata.get('document_id') for d in docs if d.metadata and d.metadata.get('document_id')}
        document_ids = {did for did in document_ids if did}

        chunk_rows_by_hash = {}
        chunk_rows_by_doc = {}
        titles_by_doc = {}

        # Normalize incoming docs metadata to canonical shape before enrichment
        try:
            self.normalize_documents_metadata(docs)
        except Exception:
            logger.debug('Failed to normalize documents prior to enrichment', exc_info=True)

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Fetch chunk rows by chunk_hash in one query
                    if content_hashes:
                        cur.execute(
                            "SELECT id AS document_chunk_id, document_id, chunk_text, chunk_ordinal, page_start, page_end, chunk_hash, topics FROM document_chunks WHERE chunk_hash = ANY(%s)",
                            (list(content_hashes),)
                        )
                        for row in cur.fetchall():
                            chunk_rows_by_hash[row['chunk_hash']] = dict(row)

                    # Fetch chunk rows for document_ids (if any) so we can map by doc+ordinal
                    if document_ids:
                        cur.execute(
                            "SELECT id AS document_chunk_id, document_id, chunk_text, chunk_ordinal, page_start, page_end, chunk_hash, topics FROM document_chunks WHERE document_id = ANY(%s::uuid[])",
                            (list(document_ids),)
                        )
                        for row in cur.fetchall():
                            doc_id = row['document_id']
                            chunk_rows_by_doc.setdefault(doc_id, {})
                            # map by chunk_hash when available, fallback to ordinal
                            key = row.get('chunk_hash') or f"ordinal:{row.get('chunk_ordinal')}"
                            chunk_rows_by_doc[doc_id][key] = dict(row)
                            # Also index by page_start for datasets that use page metadata
                            page_start = row.get('page_start')
                            if page_start is not None:
                                chunk_rows_by_doc[doc_id][f"page:{page_start}"] = dict(row)

                    # Fetch document titles for discovered document_ids
                    if document_ids:
                        cur.execute("SELECT id AS document_id, title FROM documents WHERE id = ANY(%s::uuid[])", (list(document_ids),))
                        for row in cur.fetchall():
                            doc_id = row.get('document_id')
                            if doc_id:
                                titles_by_doc[str(doc_id)] = row.get('title')

        except Exception as e:
            logger.debug(f"Postgres enrichment query failed: {e}")
            return

        # Apply enrichment to docs
        for doc in docs:
            try:
                meta = doc.metadata or {}
                if meta.get('title') and meta.get('document_chunk_id'):
                    continue

                ch = meta.get('content_hash')
                did = meta.get('document_id')

                chosen_row = None
                # Prefer exact chunk_hash match
                if ch and ch in chunk_rows_by_hash:
                    chosen_row = chunk_rows_by_hash.get(ch)
                # Otherwise try mapping by document id + chunk_hash, page, or ordinal
                if not chosen_row and did and did in chunk_rows_by_doc:
                    # try chunk_hash key
                    if ch and ch in chunk_rows_by_doc[did]:
                        chosen_row = chunk_rows_by_doc[did][ch]
                    else:
                        # try matching by page metadata from Milvus
                        page_meta = meta.get('page')
                        if page_meta is not None:
                            # match by page_start or chunk_ordinal stored under page:<n>
                            page_key = f"page:{page_meta}"
                            if page_key in chunk_rows_by_doc[did]:
                                chosen_row = chunk_rows_by_doc[did][page_key]
                            elif f"ordinal:{page_meta}" in chunk_rows_by_doc[did]:
                                chosen_row = chunk_rows_by_doc[did][f"ordinal:{page_meta}"]
                        # fallback to first chunk for that doc (best-effort)
                        if not chosen_row:
                            first_key = next(iter(chunk_rows_by_doc[did].keys()), None)
                            if first_key:
                                chosen_row = chunk_rows_by_doc[did][first_key]

                if chosen_row:
                    if chosen_row.get('document_chunk_id'):
                        meta['document_chunk_id'] = chosen_row.get('document_chunk_id')
                    else:
                        logger.error('Enrichment returned row missing canonical document_chunk_id: %s', chosen_row)
                        raise KeyError('Enrichment missing document_chunk_id')
                    meta['chunk_hash'] = chosen_row.get('chunk_hash')
                    meta['chunk_ordinal'] = chosen_row.get('chunk_ordinal')
                    meta['page_start'] = chosen_row.get('page_start')
                    meta['page_end'] = chosen_row.get('page_end')
                    meta['topic'] = chosen_row.get('topic')  # Add LLM-generated topic from database
                    # Prefer document title from `documents` table if available
                    if did and did in titles_by_doc and titles_by_doc[did]:
                        meta['title'] = titles_by_doc[did]
                    # Provide a short preview from chunk_text if page_content is empty or short
                    if not doc.page_content or len(doc.page_content) < 50:
                        chunk_text = chosen_row.get('chunk_text') or ''
                        doc.page_content = chunk_text
                    # Short preview used by UI
                    meta['preview'] = (doc.page_content[:240] + '...') if len(doc.page_content) > 240 else doc.page_content
                    doc.metadata = meta
            except Exception:
                # Fail safe: don't break enrichment loop
                logger.debug("Failed to enrich a document from Postgres", exc_info=True)

    # =============================================================================
    # Document Chunk Operations (continued)
    # =============================================================================
    
    def upsert_document_chunk(self, chunk_data: Dict[str, Any]) -> None:
        """
        Insert or update document chunk in PostgreSQL.
        
        Args:
            chunk_data: Document chunk data dictionary
        """
        required_fields = ["chunk_id", "document_id", "chunk_text"]
        missing_fields = [field for field in required_fields if not chunk_data.get(field)]
        
        if missing_fields:
            raise ValueError(f"Chunk data missing required fields: {missing_fields}")
        
        try:
            query = """
                INSERT INTO document_chunks (
                    chunk_id, document_id, chunk_text, chunk_index, chunk_metadata
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) 
                DO UPDATE SET
                    document_id = EXCLUDED.document_id,
                    chunk_text = EXCLUDED.chunk_text,
                    chunk_index = EXCLUDED.chunk_index,
                    chunk_metadata = EXCLUDED.chunk_metadata,
                    updated_at = CURRENT_TIMESTAMP
            """
            
            params = (
                chunk_data['chunk_id'],
                chunk_data['document_id'],
                chunk_data['chunk_text'],
                chunk_data.get('chunk_index', 0),
                chunk_data.get('chunk_metadata', {})
            )
            
            self.execute_query(query, params)
            logger.debug(f"Successfully upserted document chunk: {chunk_data['chunk_id']}")
            
        except Exception as e:
            logger.error(f"Failed to upsert document chunk {chunk_data.get('chunk_id')}: {e}")
            raise
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a specific document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of chunk dictionaries
        """
        try:
            query = """
                SELECT 
                    chunk_id,
                    document_id,
                    chunk_text,
                    chunk_index,
                    chunk_metadata,
                    created_at
                FROM document_chunks 
                WHERE document_id = %s
                ORDER BY chunk_index
            """
            
            results = self.execute_query(query, (document_id,), fetch_all=True)
            
            chunks = []
            if results:
                for row in results:
                    chunk = {
                        'chunk_id': row['chunk_id'],
                        'document_id': row['document_id'],
                        'chunk_text': row['chunk_text'],
                        'chunk_index': row['chunk_index'],
                        'chunk_metadata': row['chunk_metadata'],
                        'created_at': row['created_at']
                    }
                    chunks.append(chunk)
            
            logger.debug(f"Retrieved {len(chunks)} chunks for document {document_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to get chunks for document {document_id}: {e}")
            raise
    
    def delete_document_chunks(self, document_id: str) -> int:
        """
        Delete all chunks for a specific document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Number of chunks deleted
        """
        try:
            query = "DELETE FROM document_chunks WHERE document_id = %s"
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (document_id,))
                    deleted_count = cur.rowcount
                conn.commit()
            
            logger.debug(f"Deleted {deleted_count} chunks for document {document_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete chunks for document {document_id}: {e}")
            raise
    
    # =============================================================================
    # Document Search Operations
    # =============================================================================
    
    def search_document_chunks_fts(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search document chunks using PostgreSQL Full Text Search.
        
        Args:
            query: Search query string
            k: Maximum number of results to return
            
        Returns:
            List of chunk dictionaries with search results
        """
        try:
            search_query = """
                SELECT 
                    dc.chunk_id,
                    dc.document_id,
                    dc.chunk_text,
                    dc.chunk_index,
                    dc.chunk_metadata,
                    d.filename,
                    d.title,
                    ts_rank_cd(to_tsvector('english', dc.chunk_text), plainto_tsquery('english', %s)) as rank
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.filename
                WHERE to_tsvector('english', dc.chunk_text) @@ plainto_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT %s
            """
            
            results = self.execute_query(search_query, (query, query, k), fetch_all=True)
            
            chunks = []
            if results:
                for row in results:
                    chunk = {
                        'chunk_id': row['chunk_id'],
                        'document_id': row['document_id'],
                        'chunk_text': row['chunk_text'],
                        'chunk_index': row['chunk_index'],
                        'filename': row['filename'],
                        'title': row['title'],
                        'fts_rank': float(row['rank']),
                        'metadata': row['chunk_metadata'] or {}
                    }
                    chunks.append(chunk)
            
            logger.debug(f"FTS search returned {len(chunks)} results for query: {query[:50]}...")
            return chunks
            
        except Exception as e:
            logger.error(f"Document FTS search failed for query '{query}': {e}")
            raise
    

