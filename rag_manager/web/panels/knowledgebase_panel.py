"""
Knowledgebase Panel Statistics Provider
Following DEVELOPMENT_RULES.md for all development requirements

This module handles all statistics for the Knowledgebase panel on the status dashboard.
Centralizes document collection statistics, embedding metrics, and indexing status.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class KnowledgebasePanelStats:
    """Statistics provider for the Knowledgebase panel."""
    
    def __init__(self, rag_manager):
        """Initialize with reference to the main RAG manager."""
        self.rag_manager = rag_manager
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get all knowledgebase statistics for the knowledgebase panel.
        
        Returns:
            Dictionary containing knowledgebase panel statistics
        """
        try:
            # Get document collection stats from Milvus
            collection_stats = self._get_collection_stats()
            
            # Get document metadata stats from PostgreSQL
            metadata_stats = self._get_metadata_stats()
            
            return {
                # Collection metrics from Milvus
                'total_documents': collection_stats.get('num_entities', 0),
                'metric_type': collection_stats.get('metric_type', 'cosine'),
                'collection_name': collection_stats.get('collection_name', 'documents'),
                
                # Metadata metrics from PostgreSQL
                'documents_total': metadata_stats.get('documents_total', 0),
                'avg_words_per_doc': metadata_stats.get('avg_words_per_doc', 0),
                'avg_chunks_per_doc': metadata_stats.get('avg_chunks_per_doc', 0),
                'median_chunk_chars': metadata_stats.get('median_chunk_chars', 0),
                'keywords': metadata_stats.get('keywords', [])
            }
            
        except Exception as e:
            logger.error(f"Failed to get knowledgebase panel stats: {e}")
            return self._empty_stats()
    
    def _get_collection_stats(self) -> Dict[str, Any]:
        """Get document collection statistics from Milvus."""
        try:
            if not self.rag_manager.milvus_manager:
                return {}
                
            return self.rag_manager.milvus_manager.get_collection_stats() or {}
            
        except Exception as e:
            logger.warning(f"Failed to get collection stats: {e}")
            return {}
    
    def _get_metadata_stats(self) -> Dict[str, Any]:
        """Get document metadata statistics from PostgreSQL."""
        try:
            if not self.rag_manager.document_source_manager:
                return {}
                
            return self.rag_manager.document_source_manager.document_data_manager.get_knowledgebase_metadata() or {}
            
        except Exception as e:
            logger.warning(f"Failed to get metadata stats: {e}")
            return {}
    
    def _empty_stats(self) -> Dict[str, Any]:
        """Return empty knowledgebase panel stats."""
        return {
            'total_documents': 0,
            'metric_type': 'cosine',
            'collection_name': 'documents',
            'documents_total': 0,
            'avg_words_per_doc': 0,
            'avg_chunks_per_doc': 0,
            'median_chunk_chars': 0,
            'keywords': []
        }
