"""
Stats Panel Data Provider
Following DEVELOPMENT_RULES.md for all development requirements

This module coordinates statistics gathering from individual panel providers
for the status dashboard. Each panel has its own dedicated statistics provider.
"""

import logging
from typing import Dict, Any

from .panels import EmailPanelStats, URLPanelStats, KnowledgebasePanelStats, SystemPanelStats

logger = logging.getLogger(__name__)


class StatsProvider:
    """Centralized statistics coordinator for the status panel."""
    
    def __init__(self, rag_manager):
        """Initialize with reference to the main RAG manager."""
        self.rag_manager = rag_manager
        
        # Initialize individual panel providers
        self.email_panel = EmailPanelStats(rag_manager)
        self.url_panel = URLPanelStats(rag_manager)
        self.knowledgebase_panel = KnowledgebasePanelStats(rag_manager)
        self.system_panel = SystemPanelStats(rag_manager)
        
    def get_all_stats(self) -> Dict[str, Any]:
        """
        Get all statistics for the status panel from individual panel providers.
        
        Returns:
            Dictionary containing all stats organized by panel
        """
        return {
            'knowledgebase': self.knowledgebase_panel.get_stats(),
            'email': self.email_panel.get_stats(),
            'url': self.url_panel.get_stats(),
            'system': self.system_panel.get_stats()
        }
