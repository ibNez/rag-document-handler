"""
Core module initialization.

Contains core configuration, models, and utilities for the RAG Knowledgebase Manager.
"""

from .config import Config
from .models import (
    BaseProcessingStatus,
    DocumentProcessingStatus,
    URLProcessingStatus,
    EmailProcessingStatus,
    ProcessingStatus,
    ChunkMetadata,
    SearchResult
)

__all__ = [
    'Config',
    'BaseProcessingStatus',
    'DocumentProcessingStatus',
    'URLProcessingStatus',
    'EmailProcessingStatus',
    'ProcessingStatus',
    'ChunkMetadata',
    'SearchResult'
]
