"""
Document Manager for ingestion operations.

Provides a clean interface for document ingestion with all database operations
delegated to DocumentDataManager through DocumentSourceManager.
"""

import logging
from typing import Dict, Any, Optional, List, Protocol
from rag_manager.core.models import DocumentMetadata
from .source_manager import DocumentSourceManager

logger = logging.getLogger(__name__)


class VectorStore(Protocol):
    """Protocol for vector storage operations."""
    def store_embeddings(self, doc_id: str, embeddings: List[float]) -> None: ...
    def similarity_search(self, query_vector: List[float], limit: int) -> List[str]: ...
    def delete_document(self, doc_id: str) -> None: ...


class DocumentManager:
    """
    Document Manager for ingestion operations.
    
    Delegates all data operations to DocumentSourceManager which uses
    the shared DocumentDataManager for database operations.
    """
    
    def __init__(self, postgres_manager, vector_store: Optional[VectorStore] = None):
        """
        Initialize with PostgreSQL manager and optional vector store.
        
        Args:
            postgres_manager: PostgreSQL connection manager
            vector_store: Optional vector storage interface
        """
        self.source_manager = DocumentSourceManager(postgres_manager, vector_store)
        self.postgres = postgres_manager  # Keep for backward compatibility
        self.vector_store = vector_store
        logger.info("Document Manager initialized for ingestion operations")
    
    def upsert_document_metadata(self, filename: str, metadata: Dict[str, Any]) -> None:
        """
        Update or insert document metadata.
        
        Args:
            filename: Document filename
            metadata: Document metadata dictionary
        """
        return self.source_manager.upsert_document_metadata(filename, metadata)

    def get_document_metadata(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get document metadata.
        
        Args:
            filename: Document filename
            
        Returns:
            Document metadata dictionary or None
        """
        return self.source_manager.get_document_metadata(filename)

    def delete_document_metadata(self, filename: str) -> None:
        """
        Remove document metadata.
        
        Args:
            filename: Document filename
        """
        return self.source_manager.delete_document_metadata(filename)

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
        return self.source_manager.persist_chunks(document_id, chunks, trace_logger)

    def get_knowledgebase_metadata(self) -> Dict[str, Any]:
        """
        Get aggregated knowledge base metadata.
        
        Returns:
            Knowledge base statistics dictionary
        """
        return self.source_manager.get_knowledgebase_metadata()

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
        """
        return self.source_manager.store_document(
            file_path=file_path,
            filename=filename,
            title=title,
            content_preview=content_preview,
            content_type=content_type,
            file_size=file_size,
            word_count=word_count,
            document_type=document_type,
            **kwargs
        )

    def update_processing_status(self, document_id: str, status: str) -> None:
        """
        Update document processing status.
        
        Args:
            document_id: Document identifier
            status: New processing status
        """
        return self.source_manager.update_processing_status(document_id, status)

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
        return self.source_manager.store_document_chunk(
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
        Delete all chunks for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Number of chunks deleted
        """
        return self.source_manager.delete_document_chunks(document_id)

    def store_document_with_metadata(self, content: str, metadata: DocumentMetadata) -> str:
        """
        Store document with metadata object.
        
        Args:
            content: Document content
            metadata: DocumentMetadata object
            
        Returns:
            Document identifier
        """
        return self.source_manager.store_document_with_metadata(content, metadata)

    def search_documents(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search documents using full text search.
        
        Args:
            query: Search query
            limit: Maximum results to return
            
        Returns:
            List of document dictionaries
        """
        # Delegate to source manager's data manager
        return self.source_manager.document_data_manager.search_documents(query, limit)

    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document by identifier.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document dictionary or None if not found
        """
        # Delegate to source manager's data manager
        return self.source_manager.document_data_manager.get_document_by_id(document_id)
