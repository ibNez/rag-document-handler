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

