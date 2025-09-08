"""Document retrieval module for search and metadata operations.

Modern document retrieval architecture:
- DocumentSearchManager: Search orchestration and metadata enrichment
- DocumentProcessor: Hybrid search combining vector similarity and PostgreSQL FTS
"""

from .search_manager import DocumentSearchManager
from .processor import DocumentProcessor

__all__ = [
    'DocumentSearchManager',
    'DocumentProcessor'
]
