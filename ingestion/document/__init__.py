"""Document ingestion module for handling file-based content processing.

Modern document processing architecture:
- DocumentSourceManager: Document source management and database operations
- DocumentProcessor: Document content processing and embedding generation
- AdvancedChunker: Advanced text chunking with semantic awareness
"""

from .source_manager import DocumentSourceManager
from .processor import DocumentProcessor
from .advanced_chunking import AdvancedChunker, ChunkingConfig, ElementInfo

__all__ = [
    "DocumentSourceManager", 
    "DocumentProcessor",
    "AdvancedChunker",
    "ChunkingConfig",
    "ElementInfo",
]