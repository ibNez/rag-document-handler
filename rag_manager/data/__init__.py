#!/usr/bin/env python
"""
Data Access Layer for RAG Document Handler
Following DEVELOPMENT_RULES.md for all development requirements

This module provides pure data access operations for all content types.
No business logic - only database operations.
"""

from .base_data import BaseDataManager
from .email_data import EmailDataManager
from .document_data import DocumentDataManager
from .url_data import URLDataManager

__all__ = [
    "BaseDataManager",
    "EmailDataManager", 
    "DocumentDataManager",
    "URLDataManager"
]
