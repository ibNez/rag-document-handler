"""
Document Manager for retrieval operations.

Provides a clean interface for document retrieval with all database operations
delegated to DocumentDataManager through DocumentSearchManager.
"""

import logging
from typing import Any, List, Dict, Optional
from .search_manager import DocumentSearchManager

logger = logging.getLogger(__name__)


class DocumentManager:
    """
    Document Manager for retrieval operations.
    
    Delegates all data operations to DocumentSearchManager which uses
    the shared DocumentDataManager for database operations.
    """

    def __init__(self, postgres_manager: Any) -> None:
        """
        Initialize the DocumentManager for retrieval.

        Args:
            postgres_manager: Manager for PostgreSQL operations.
        """
        self.search_manager = DocumentSearchManager(postgres_manager)
        self.postgres_manager = postgres_manager  # Keep for backward compatibility
        logger.info("Document Manager initialized for retrieval operations")

    def normalize_documents_metadata(self, docs: List[Any]) -> None:
        """
        Normalize metadata for a list of Documents in-place.

        Args:
            docs: List of LangChain Document objects
        """
        return self.search_manager.normalize_documents_metadata(docs)

    def batch_enrich_documents_from_postgres(self, docs: List[Any]) -> None:
        """
        Batch-fetch canonical chunk metadata from PostgreSQL and attach to
        LangChain Document.metadata in-place.

        Args:
            docs: List of LangChain Document objects to enrich
        """
        return self.search_manager.batch_enrich_documents_from_postgres(docs)

    def search_documents(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search documents using full text search.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of document dictionaries
        """
        return self.search_manager.search_documents(query, limit)

    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get single document by ID.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document dictionary or None
        """
        return self.search_manager.get_document_by_id(document_id)

    def search_document_chunks_fts(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search document chunks using PostgreSQL Full Text Search.
        
        Args:
            query: Search query string
            k: Maximum number of results to return
            
        Returns:
            List of chunk dictionaries with search results
        """
        return self.search_manager.search_document_chunks_fts(query, k)

    def get_document_statistics(self) -> Dict[str, Any]:
        """
        Get document statistics for retrieval reporting.
        
        Returns:
            Statistics dictionary
        """
        return self.search_manager.get_document_statistics()