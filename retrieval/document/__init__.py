"""
Document Retrieval Module
Following DEVELOPMENT_RULES.md for all development requirements

This module provides document-specific retrieval capabilities including:
- PostgreSQL Full-Text Search (FTS) retrieval
- Hybrid retrieval combining vector similarity and FTS
- Advanced filtering and search options
"""

from .postgres_fts_retriever import DocumentPostgresFTSRetriever
from .hybrid_retriever import DocumentHybridRetriever

__all__ = [
    'DocumentPostgresFTSRetriever',
    'DocumentHybridRetriever'
]
