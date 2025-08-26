"""
Panel Statistics Package
Following DEVELOPMENT_RULES.md for all development requirements

This package contains individual panel statistics providers for the status dashboard.
Each panel has its own dedicated statistics provider for clean separation of concerns.
"""

from .email_panel import EmailPanelStats
from .url_panel import URLPanelStats
from .knowledgebase_panel import KnowledgebasePanelStats
from .system_panel import SystemPanelStats

__all__ = [
    'EmailPanelStats',
    'URLPanelStats', 
    'KnowledgebasePanelStats',
    'SystemPanelStats'
]
