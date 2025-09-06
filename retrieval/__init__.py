"""
Retrieval Modules

This module contains retrieval systems for searching processed content.
Organized by content type (email, document) for scalable search functionality.
"""

# Import email retrieval components
from .email.processor import EmailProcessor
from .email.search_manager import EmailSearchManager
from .email.postgres_fts_retriever import PostgresFTSRetriever

__all__ = [
    "EmailProcessor",
    "EmailSearchManager", 
    "PostgresFTSRetriever"
]
