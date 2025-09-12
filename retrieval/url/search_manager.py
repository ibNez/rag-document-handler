"""
URL Search Manager
Search and retrieval operations for URL content.

This module handles URL search business logic, delegating
data operations to URLDataManager following the established pattern.
"""

import logging
from typing import Dict, List, Optional, Any

from rag_manager.data.url_data import URLDataManager

logger = logging.getLogger(__name__)


class URLSearchManager:
    """
    URL search manager for content retrieval operations.
    
    Handles search and filtering logic for URL content,
    delegating data operations to URLDataManager.
    """
    
    def __init__(self, postgres_manager=None, milvus_manager=None):
        """
        Initialize URL search manager.
        
        Args:
            postgres_manager: PostgreSQL connection manager
            milvus_manager: Optional Milvus vector store manager
        """
        self.url_data = URLDataManager(postgres_manager, milvus_manager)
        logger.info("URLSearchManager initialized")
    
