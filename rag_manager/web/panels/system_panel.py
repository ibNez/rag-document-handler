"""
System Panel Statistics Provider
Following DEVELOPMENT_RULES.md for all development requirements

This module handles all statistics for the System panel on the status dashboard.
Centralizes connection status, processing queues, and system health metrics.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class SystemPanelStats:
    """Statistics provider for the System panel."""
    
    def __init__(self, rag_manager):
        """Initialize with reference to the main RAG manager."""
        self.rag_manager = rag_manager
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get all system statistics for the system panel.
        
        Returns:
            Dictionary containing system panel statistics
        """
        try:
            return {
                # Connection status
                'sql_status': self._get_database_status(),
                'milvus_status': self._get_milvus_status(),
                
                # Processing queues
                'processing_files': len(self.rag_manager.processing_status),
                'processing_urls': len(self.rag_manager.url_processing_status),
                'processing_emails': len(self.rag_manager.email_processing_status),
                
                # System health
                'scheduler_active': self._is_scheduler_active(),
                'total_processing': (
                    len(self.rag_manager.processing_status) + 
                    len(self.rag_manager.url_processing_status) + 
                    len(self.rag_manager.email_processing_status)
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to get system panel stats: {e}")
            return self._empty_stats()
    
    def _get_database_status(self) -> str:
        """Get PostgreSQL database connection status."""
        try:
            if not self.rag_manager.postgres_manager:
                return 'disconnected'
                
            # Test the connection
            with self.rag_manager.postgres_manager.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return 'connected'
            
        except Exception as e:
            logger.debug(f"Database status check failed: {e}")
            return 'error'
    
    def _get_milvus_status(self) -> str:
        """Get Milvus vector database connection status."""
        try:
            if not self.rag_manager.milvus_manager:
                return 'disconnected'
                
            status = self.rag_manager.milvus_manager.check_connection()
            
            if isinstance(status, dict):
                return 'connected' if status.get('connected', False) else 'error'
            elif isinstance(status, bool):
                return 'connected' if status else 'error'
            else:
                return 'error'
                
        except Exception as e:
            logger.debug(f"Milvus status check failed: {e}")
            return 'error'
    
    def _is_scheduler_active(self) -> bool:
        """Check if the background scheduler is active."""
        try:
            if hasattr(self.rag_manager, 'scheduler_manager') and self.rag_manager.scheduler_manager:
                # Check if scheduler thread is running
                if hasattr(self.rag_manager, '_scheduler_thread') and self.rag_manager._scheduler_thread:
                    return self.rag_manager._scheduler_thread.is_alive()
            return False
        except Exception as e:
            logger.debug(f"Scheduler status check failed: {e}")
            return False
    
    def _empty_stats(self) -> Dict[str, Any]:
        """Return empty system panel stats."""
        return {
            'sql_status': 'error',
            'milvus_status': 'error',
            'processing_files': 0,
            'processing_urls': 0,
            'processing_emails': 0,
            'scheduler_active': False,
            'total_processing': 0
        }
