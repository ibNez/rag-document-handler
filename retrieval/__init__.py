"""
Retrieval Modules
Following DEVELOPMENT_RULES.md for all development requirements

This module contains retrieval systems for searching processed content.
Organized by content type (email, document) for scalable search functionality.
"""

# Import email retrieval components
from .email.hybrid_retriever import HybridRetriever
from .email.postgres_fts_retriever import PostgresFTSRetriever

__all__ = [
    "HybridRetriever", 
    "PostgresFTSRetriever"
]
