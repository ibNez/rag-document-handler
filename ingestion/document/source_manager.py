"""
Document Source Manager for ingestion operations.

Manages document ingestion orchestration and business logic.
Delegates all data operations to DocumentDataManager for clean separation of concerns.
"""

import logging
from typing import Dict, Any, Optional, List, Protocol
from ingestion.utils.file_filters import should_ignore_file
from rag_manager.core.models import DocumentMetadata
from rag_manager.data.document_data import DocumentDataManager

logger = logging.getLogger(__name__)


class VectorStore(Protocol):
    """Protocol for vector storage operations."""
    def store_embeddings(self, doc_id: str, embeddings: List[float]) -> None: ...
    def similarity_search(self, query_vector: List[float], limit: int) -> List[str]: ...
    def delete_document(self, doc_id: str) -> None: ...


class DocumentSourceManager:
    """
    Manager for document ingestion operations and business logic.
    
    Handles document source management, processing orchestration,
    and delegates all data operations to DocumentDataManager.
    """
    
    def __init__(self, postgres_manager, vector_store: Optional[VectorStore] = None):
        """
        Initialize with PostgreSQL manager and optional vector store.
        
        Args:
            postgres_manager: PostgreSQL connection manager
            vector_store: Optional vector storage interface
        """
        self.document_data_manager = DocumentDataManager(postgres_manager)
        self.vector_store = vector_store
        logger.info("DocumentSourceManager initialized for ingestion operations")
    
    def upsert_document_metadata(self, filename: str, metadata: Dict[str, Any]) -> None:
        """
        Update or insert document metadata with business logic filters.
        
        Args:
            filename: Document filename
            metadata: Document metadata dictionary
        """
        # Apply business logic: filter out system files
        if should_ignore_file(filename):
            logger.debug(f"Ignoring system file during upsert: {filename}")
            return
            
        try:
            # Map metadata for PostgreSQL storage with explicit columns
            mapped_metadata = {
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
                'processing_status': metadata.get('processing_status', 'pending'),
                'document_type': 'file'
            }
            
            # Delegate to data manager
            self.document_data_manager.upsert_document_metadata(filename, mapped_metadata)
            logger.info(f"Metadata stored successfully for {filename}")
            
        except Exception as e:
            logger.exception(f"Failed to upsert metadata for {filename}")
            raise

    def get_document_metadata(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get document metadata with business logic filters.
        
        Args:
            filename: Document filename
            
        Returns:
            Document metadata dictionary or None
        """
        # Apply business logic: filter out system files
        if should_ignore_file(filename):
            logger.debug(f"Ignoring system file: {filename}")
            return None
            
        try:
            # Use flexible search strategy for different document types
            result = self.document_data_manager.get_document_metadata(filename)
            if not result:
                # Try alternative search strategies for complex filenames
                result = self._search_document_by_patterns(filename)
            
            if result:
                # Convert datetime objects to strings for JSON serialization
                from datetime import datetime
                for key, value in result.items():
                    if isinstance(value, datetime):
                        result[key] = value.isoformat() if value else None
                
                logger.info(f"Metadata lookup succeeded for {filename}")
                return result
            else:
                logger.info(f"Metadata lookup miss for {filename}")
                return None
                
        except Exception as e:
            logger.exception(f"Metadata lookup failed for {filename}")
            return None

    def _search_document_by_patterns(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Search document using multiple patterns for complex filename matching.
        
        Args:
            filename: Document filename to search
            
        Returns:
            Document metadata or None
        """
        try:
            # Try searching by title, file_path patterns, and filename variants
            search_results = self.document_data_manager.search_documents(filename, limit=5)
            
            # Apply heuristics to find best match
            for doc in search_results:
                if (doc.get('filename') == filename or 
                    doc.get('title') == filename or
                    filename in doc.get('file_path', '')):
                    return doc
            
            return None
            
        except Exception as e:
            logger.debug(f"Pattern search failed for {filename}: {e}")
            return None

    def delete_document_metadata(self, filename: str) -> None:
        """
        Remove document metadata with business logic filters.
        
        Args:
            filename: Document filename
        """
        # Apply business logic: filter out system files
        if should_ignore_file(filename):
            logger.debug(f"Ignoring system file during delete: {filename}")
            return
            
        try:
            # Delegate to data manager with flexible deletion strategy
            deleted = self.document_data_manager.delete_document_metadata(filename)
            
            # If direct deletion failed, try alternative patterns
            if not deleted:
                # Try deleting by alternative search patterns
                alt_doc = self._search_document_by_patterns(filename)
                if alt_doc and alt_doc.get('filename'):
                    deleted = self.document_data_manager.delete_document_metadata(alt_doc['filename'])
            
            logger.info(f"Metadata delete for {filename}; removed_row={deleted}")
            
        except Exception as e:
            logger.exception(f"Failed to delete metadata row for {filename}")
            raise

    def persist_chunks(self, document_id: str, chunks: list, trace_logger: Optional[logging.Logger] = None) -> int:
        """
        Persist document chunks with ingestion orchestration logic.
        
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

        # Delegate chunk persistence to data manager
        return self.document_data_manager.persist_chunks(document_id, chunks, trace_logger)

    def get_knowledgebase_metadata(self) -> Dict[str, Any]:
        """
        Get aggregated knowledge base metadata for ingestion reporting.
        
        Returns:
            Knowledge base statistics dictionary
        """
        return self.document_data_manager.get_knowledgebase_metadata()

    def store_document(self, file_path: str, filename: str, 
                      title: Optional[str] = None, content_preview: Optional[str] = None,
                      content_type: Optional[str] = None, file_size: Optional[int] = None,
                      word_count: Optional[int] = None, document_type: str = 'file', **kwargs) -> str:
        """
        Store document with ingestion validation and business logic.
        
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
        """
        # Apply ingestion business rules and validation
        if should_ignore_file(filename):
            raise ValueError(f"File type not supported for ingestion: {filename}")
        
        # Delegate to data manager
        return self.document_data_manager.store_document(
            file_path=file_path,
            filename=filename,
            title=title,
            content_preview=content_preview,
            content_type=content_type,
            file_size=file_size,
            word_count=word_count,
            document_type=document_type
        )

    def update_processing_status(self, document_id: str, status: str) -> None:
        """
        Update document processing status.
        
        Args:
            document_id: Document identifier
            status: New processing status
        """
        # Validate status values (business logic)
        valid_statuses = ['pending', 'processing', 'completed', 'error', 'skipped']
        if status not in valid_statuses:
            raise ValueError(f"Invalid processing status: {status}. Must be one of {valid_statuses}")
        
        # Delegate to data manager
        self.document_data_manager.update_processing_status(document_id, status)

    def store_document_chunk(self, document_id: str, chunk_text: str, 
                           chunk_ordinal: int, page_start: Optional[int] = None, 
                           page_end: Optional[int] = None, section_path: Optional[str] = None,
                           element_types: Optional[List[str]] = None, token_count: Optional[int] = None,
                           chunk_hash: Optional[str] = None, topics: Optional[str] = None,
                           embedding_version: str = 'mxbai-embed-large') -> str:
        """
        Store document chunk with ingestion orchestration.
        
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
        # Apply ingestion business rules
        if not chunk_text or not chunk_text.strip():
            raise ValueError("Chunk text cannot be empty")
        
        if chunk_ordinal < 1:
            raise ValueError("Chunk ordinal must be positive")
        
        # Delegate to data manager
        return self.document_data_manager.store_document_chunk(
            document_id=document_id,
            chunk_text=chunk_text,
            chunk_ordinal=chunk_ordinal,
            page_start=page_start,
            page_end=page_end,
            section_path=section_path,
            element_types=element_types,
            token_count=token_count,
            chunk_hash=chunk_hash,
            topics=topics,
            embedding_version=embedding_version
        )

    def delete_document_chunks(self, document_id: str) -> int:
        """
        Delete all chunks for a document during ingestion cleanup.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Number of chunks deleted
        """
        return self.document_data_manager.delete_document_chunks(document_id)

    def store_document_with_metadata(self, content: str, metadata: DocumentMetadata) -> str:
        """
        Store document with metadata object during ingestion.
        
        Args:
            content: Document content
            metadata: DocumentMetadata object
            
        Returns:
            Document identifier
        """
        return self.document_data_manager.store_document_with_metadata(content, metadata)
