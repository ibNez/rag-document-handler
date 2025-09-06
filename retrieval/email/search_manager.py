#!/usr/bin/env python
"""
Email Search Manager
Following DEVELOPMENT_RULES.md for all development requirements

This module provides email search orchestration and business logic.
Uses EmailDataManager for data operations and EmailProcessor for RRF fusion.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from rag_manager.data.email_data import EmailDataManager

logger = logging.getLogger(__name__)


class EmailSearchManager:
    """
    Email search orchestration manager.
    
    Handles search business logic and coordinates between:
    - EmailDataManager for data access
    - EmailProcessor for RRF fusion
    - Vector stores for similarity search
    """
    
    def __init__(self, postgres_manager: Any, milvus_manager: Optional[Any] = None) -> None:
        """
        Initialize email search manager.
        
        Args:
            postgres_manager: PostgreSQL connection manager
            milvus_manager: Optional Milvus vector store manager
        """
        self.postgres_manager = postgres_manager
        self.milvus_manager = milvus_manager
        
        # Initialize data manager for database operations
        self.email_data_manager = EmailDataManager(postgres_manager, milvus_manager)
        
        # Search components (initialized on demand)
        self.postgres_fts_retriever: Optional[Any] = None
        self.hybrid_retriever: Optional[Any] = None
        
        logger.info("EmailSearchManager initialized for search orchestration")
    
    # =============================================================================
    # Search Orchestration Methods
    # =============================================================================
    
    def initialize_hybrid_retrieval(self, email_vector_store: Any) -> None:
        """
        Initialize retrieval system combining vector search and PostgreSQL FTS.
        
        Args:
            email_vector_store: Milvus email vector store from MilvusManager
        """
        self.email_data_manager.initialize_hybrid_retrieval(email_vector_store)
        
        # Copy references for compatibility
        self.postgres_fts_retriever = getattr(self.email_data_manager, 'postgres_fts_retriever', None)
        self.hybrid_retriever = getattr(self.email_data_manager, 'hybrid_retriever', None)
        
        logger.info("Email search system initialized successfully")

    def search_emails_hybrid(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform hybrid email search with business logic.
        
        Args:
            query: Search query
            top_k: Maximum number of results to return
            
        Returns:
            List of email chunks with relevance scores and metadata
        """
        if not query or not query.strip():
            logger.warning("Empty search query provided")
            return []
        
        # Delegate to data manager for actual search
        results = self.email_data_manager.search_emails_hybrid(query, top_k)
        
        # Apply any business logic filters here if needed
        # For example: filter by date range, sender whitelist, etc.
        
        logger.info(f"Search completed for query '{query[:50]}...' - {len(results)} results")
        return results

    def format_email_context(self, results: List[Dict[str, Any]], max_context_length: int = 10000) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Format search results for LLM context with business logic.
        
        Args:
            results: List of email search results
            max_context_length: Maximum character length for context
            
        Returns:
            Tuple of (context_text, sources) for LLM processing
        """
        if not results:
            return "", []
        
        # Get formatted context from data manager
        context_text, sources = self.email_data_manager.format_email_context(results)
        
        # Apply business logic: truncate if too long
        if len(context_text) > max_context_length:
            logger.warning(f"Context length {len(context_text)} exceeds limit {max_context_length}, truncating")
            context_text = context_text[:max_context_length] + "\n\n[Context truncated...]"
        
        # Add search metadata to sources
        for source in sources:
            source['search_timestamp'] = datetime.now().isoformat()
            source['search_method'] = 'hybrid_rrf'
        
        return context_text, sources

    def search_fts_only(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform FTS-only search for comparison or fallback.
        
        Args:
            query: Search query
            k: Maximum number of results to return
            
        Returns:
            List of search result documents
        """
        try:
            documents = self.email_data_manager.search_email_chunks_fts(query, k)
            
            # Convert Document objects to consistent format
            results = []
            for doc in documents:
                results.append({
                    'chunk_text': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': doc.metadata.get('fts_rank', 0.0)
                })
            
            logger.info(f"FTS search returned {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"FTS search failed: {e}")
            raise

    # =============================================================================
    # Email Data Operations (Delegated)
    # =============================================================================
    
    def get_email_statistics(self, email_address: str) -> Dict[str, int]:
        """Get email statistics for a specific account."""
        return self.email_data_manager.get_email_statistics(email_address)
    
    def get_global_email_statistics(self) -> Dict[str, int]:
        """Get global email statistics."""
        return self.email_data_manager.get_global_email_statistics()
    
    def upsert_email(self, record: Dict[str, Any]) -> None:
        """Upsert an email record."""
        self.email_data_manager.upsert_email(record)
    
    def update_total_emails_in_mailbox(self, account_id: int, total_emails: int) -> None:
        """Update total emails in mailbox for an account."""
        self.email_data_manager.update_total_emails_in_mailbox(account_id, total_emails)
    
    def search_chunks_for_email(self, email_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific email."""
        return self.email_data_manager.search_chunks_for_email(email_id)
    
    def get_chunk_statistics(self) -> Dict[str, Any]:
        """Get email chunk statistics."""
        return self.email_data_manager.get_chunk_statistics()
    
    def delete_email_vectors(self, email_id: str) -> bool:
        """Delete email vectors from Milvus."""
        return self.email_data_manager.delete_email_vectors(email_id)
    
    # =============================================================================
    # Search Analytics and Monitoring
    # =============================================================================
    
    def get_search_performance_metrics(self) -> Dict[str, Any]:
        """
        Get search performance metrics for monitoring.
        
        Returns:
            Dictionary with search performance data
        """
        try:
            # Get basic email statistics
            global_stats = self.email_data_manager.get_global_email_statistics()
            chunk_stats = self.email_data_manager.get_chunk_statistics()
            
            return {
                'total_searchable_emails': global_stats.get('total_emails', 0),
                'total_searchable_chunks': chunk_stats.get('total_chunks', 0),
                'avg_chunk_length': chunk_stats.get('avg_chunk_length', 0),
                'retrieval_system_ready': self.hybrid_retriever is not None,
                'fts_system_ready': self.postgres_fts_retriever is not None,
                'vector_system_ready': self.milvus_manager is not None
            }
            
        except Exception as e:
            logger.error(f"Failed to get search performance metrics: {e}")
            return {
                'total_searchable_emails': 0,
                'total_searchable_chunks': 0,
                'avg_chunk_length': 0,
                'retrieval_system_ready': False,
                'fts_system_ready': False,
                'vector_system_ready': False
            }
    
    def validate_search_system(self) -> Dict[str, bool]:
        """
        Validate that search system components are properly initialized.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'postgres_connection': False,
            'email_data_manager': False,
            'fts_retriever': False,
            'hybrid_retriever': False,
            'vector_store': False
        }
        
        try:
            # Test PostgreSQL connection
            with self.postgres_manager.get_connection() as conn:
                validation_results['postgres_connection'] = True
        except Exception as e:
            logger.error(f"PostgreSQL validation failed: {e}")
        
        # Test data manager
        validation_results['email_data_manager'] = self.email_data_manager is not None
        
        # Test retrievers
        validation_results['fts_retriever'] = self.postgres_fts_retriever is not None
        validation_results['hybrid_retriever'] = self.hybrid_retriever is not None
        
        # Test vector store
        validation_results['vector_store'] = self.milvus_manager is not None
        
        logger.info(f"Search system validation: {validation_results}")
        return validation_results
