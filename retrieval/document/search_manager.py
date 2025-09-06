"""
Document Search Manager for retrieval operations.

Manages document search orchestration and business logic.
Delegates all data operations to DocumentDataManager for clean separation of concerns.
"""

import logging
from typing import Any, List, Dict, Optional
from rag_manager.data.document_data import DocumentDataManager

logger = logging.getLogger(__name__)


class DocumentSearchManager:
    """
    Manager for document search and retrieval operations.
    
    Handles document search orchestration, metadata enrichment,
    and delegates all data operations to DocumentDataManager.
    """

    def __init__(self, postgres_manager: Any) -> None:
        """
        Initialize the DocumentSearchManager.

        Args:
            postgres_manager: Manager for PostgreSQL operations.
        """
        self.document_data_manager = DocumentDataManager(postgres_manager)
        logger.info("DocumentSearchManager initialized for retrieval operations")

    def normalize_documents_metadata(self, docs: List[Any]) -> None:
        """
        Normalize metadata for a list of Documents with search-specific logic.

        Args:
            docs: List of LangChain Document objects
        """
        # Delegate to data manager with additional search-specific validation
        try:
            self.document_data_manager.normalize_documents_metadata(docs)
            
            # Additional search-specific metadata validation
            for doc in docs:
                meta = doc.metadata or {}
                
                # Ensure required fields for search results
                if not meta.get('document_chunk_id') and not meta.get('chunk_id'):
                    logger.warning(f"Search result missing chunk identifier: {meta}")
                
                # Standardize preview field for UI display
                if not meta.get('preview') and hasattr(doc, 'page_content'):
                    content = doc.page_content or ''
                    meta['preview'] = (content[:240] + '...') if len(content) > 240 else content
                    doc.metadata = meta
                    
        except Exception as e:
            logger.error(f"Failed to normalize search result metadata: {e}")
            raise

    def batch_enrich_documents_from_postgres(self, docs: List[Any]) -> None:
        """
        Batch-enrich search results with comprehensive metadata from PostgreSQL.
        
        Optimized for search result display and reranking operations.

        Args:
            docs: List of LangChain Document objects to enrich
        """
        if not docs:
            logger.debug("No documents provided for enrichment")
            return

        try:
            # Delegate core enrichment to data manager
            self.document_data_manager.batch_enrich_documents_from_postgres(docs)
            
            # Apply search-specific enrichment
            self._apply_search_enrichment(docs)
            
        except Exception as e:
            logger.error(f"Document enrichment failed: {e}")
            # Continue with non-enriched documents rather than failing search
            
    def _apply_search_enrichment(self, docs: List[Any]) -> None:
        """
        Apply search-specific metadata enrichment.
        
        Args:
            docs: List of documents to enrich
        """
        for doc in docs:
            try:
                meta = doc.metadata or {}
                
                # Ensure consistent metadata structure for search results
                if not meta.get('source'):
                    meta['source'] = meta.get('file_path', 'unknown')
                
                # Add search result ranking hints
                if meta.get('fts_rank'):
                    meta['search_score'] = float(meta['fts_rank'])
                
                # Ensure title is available for display
                if not meta.get('title') and meta.get('filename'):
                    meta['title'] = meta['filename']
                
                # Add document type for result categorization
                if not meta.get('document_type'):
                    meta['document_type'] = 'file'  # default
                
                doc.metadata = meta
                
            except Exception as e:
                logger.debug(f"Failed to apply search enrichment to document: {e}")
                # Continue with partial enrichment

    def search_documents(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search documents with retrieval-specific business logic.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of document dictionaries optimized for search results
        """
        try:
            # Apply search business logic and validation
            if not query or not query.strip():
                logger.warning("Empty search query provided")
                return []
            
            # Sanitize query for security
            sanitized_query = self._sanitize_search_query(query.strip())
            
            # Apply search limits for performance
            effective_limit = min(limit, 100)  # Cap at 100 for performance
            
            # Delegate to data manager
            results = self.document_data_manager.search_documents(sanitized_query, effective_limit)
            
            # Apply search result post-processing
            processed_results = self._process_search_results(results, query)
            
            logger.info(f"Document search returned {len(processed_results)} results for query: {query[:50]}...")
            return processed_results
            
        except Exception as e:
            logger.error(f"Document search failed for query '{query}': {e}")
            raise

    def _sanitize_search_query(self, query: str) -> str:
        """
        Sanitize search query for security and performance.
        
        Args:
            query: Raw search query
            
        Returns:
            Sanitized query string
        """
        # Remove potentially problematic characters
        import re
        sanitized = re.sub(r'[<>"\';\\]', '', query)
        
        # Limit query length
        if len(sanitized) > 500:
            sanitized = sanitized[:500]
            logger.warning(f"Search query truncated to 500 characters")
        
        return sanitized

    def _process_search_results(self, results: List[Dict[str, Any]], original_query: str) -> List[Dict[str, Any]]:
        """
        Post-process search results for optimal display and ranking.
        
        Args:
            results: Raw search results from data manager
            original_query: Original search query for relevance scoring
            
        Returns:
            Processed search results
        """
        processed = []
        
        for result in results:
            try:
                # Enhance result with search metadata
                enhanced_result = dict(result)
                
                # Add search relevance indicators
                enhanced_result['search_query'] = original_query
                enhanced_result['result_type'] = 'document'
                
                # Ensure consistent field naming
                if 'fts_rank' in enhanced_result:
                    enhanced_result['relevance_score'] = enhanced_result['fts_rank']
                
                # Add display-friendly formatting
                if enhanced_result.get('content_preview'):
                    preview = enhanced_result['content_preview']
                    if len(preview) > 300:
                        enhanced_result['display_preview'] = preview[:300] + '...'
                    else:
                        enhanced_result['display_preview'] = preview
                
                processed.append(enhanced_result)
                
            except Exception as e:
                logger.debug(f"Failed to process search result: {e}")
                # Include unprocessed result rather than dropping it
                processed.append(result)
        
        return processed

    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get single document by ID with retrieval-specific enrichment.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Enhanced document dictionary or None
        """
        try:
            # Validate document ID format
            if not document_id or not document_id.strip():
                logger.warning("Invalid document ID provided")
                return None
            
            # Delegate to data manager
            result = self.document_data_manager.get_document_by_id(document_id.strip())
            
            if result:
                # Apply retrieval-specific enhancements
                enhanced_result = self._enhance_single_document(result)
                logger.debug(f"Retrieved document: {document_id}")
                return enhanced_result
            else:
                logger.info(f"Document not found: {document_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None

    def _enhance_single_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance single document for retrieval display.
        
        Args:
            document: Raw document from data manager
            
        Returns:
            Enhanced document dictionary
        """
        enhanced = dict(document)
        
        try:
            # Add retrieval metadata
            enhanced['retrieved_at'] = None  # Can be set by caller
            enhanced['source_type'] = 'database'
            
            # Ensure consistent datetime formatting
            from datetime import datetime
            for key, value in enhanced.items():
                if isinstance(value, datetime):
                    enhanced[key] = value.isoformat() if value else None
            
            # Add display helpers
            if enhanced.get('file_size'):
                size_kb = enhanced['file_size'] / 1024
                if size_kb < 1024:
                    enhanced['display_size'] = f"{size_kb:.1f} KB"
                else:
                    enhanced['display_size'] = f"{size_kb/1024:.1f} MB"
            
        except Exception as e:
            logger.debug(f"Failed to enhance document: {e}")
        
        return enhanced

    def search_document_chunks_fts(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search document chunks using Full Text Search with retrieval optimization.
        
        Args:
            query: Search query string
            k: Maximum number of chunks to return
            
        Returns:
            List of chunk dictionaries optimized for retrieval
        """
        try:
            # Apply retrieval-specific query validation
            if not query or not query.strip():
                return []
            
            sanitized_query = self._sanitize_search_query(query.strip())
            effective_k = min(k, 50)  # Cap for performance
            
            # Delegate to data manager
            chunks = self.document_data_manager.search_document_chunks_fts(sanitized_query, effective_k)
            
            # Apply retrieval-specific post-processing
            processed_chunks = self._process_chunk_results(chunks, query)
            
            logger.debug(f"Chunk FTS search returned {len(processed_chunks)} results")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Chunk FTS search failed for query '{query}': {e}")
            raise

    def _process_chunk_results(self, chunks: List[Dict[str, Any]], original_query: str) -> List[Dict[str, Any]]:
        """
        Process chunk search results for optimal retrieval display.
        
        Args:
            chunks: Raw chunk results
            original_query: Original search query
            
        Returns:
            Processed chunk results
        """
        processed = []
        
        for chunk in chunks:
            try:
                enhanced_chunk = dict(chunk)
                
                # Add retrieval metadata
                enhanced_chunk['search_query'] = original_query
                enhanced_chunk['result_type'] = 'chunk'
                
                # Enhance text display
                chunk_text = enhanced_chunk.get('chunk_text', '')
                if len(chunk_text) > 500:
                    enhanced_chunk['display_text'] = chunk_text[:500] + '...'
                else:
                    enhanced_chunk['display_text'] = chunk_text
                
                # Add context hints
                if enhanced_chunk.get('page_start'):
                    enhanced_chunk['context'] = f"Page {enhanced_chunk['page_start']}"
                elif enhanced_chunk.get('chunk_index'):
                    enhanced_chunk['context'] = f"Section {enhanced_chunk['chunk_index']}"
                
                processed.append(enhanced_chunk)
                
            except Exception as e:
                logger.debug(f"Failed to process chunk result: {e}")
                processed.append(chunk)
        
        return processed

    def get_document_statistics(self) -> Dict[str, Any]:
        """
        Get document statistics optimized for retrieval reporting.
        
        Returns:
            Statistics dictionary with retrieval-specific metrics
        """
        try:
            # Get base statistics from data manager
            stats = self.document_data_manager.get_document_statistics()
            
            # Add retrieval-specific metrics
            stats['search_ready_documents'] = stats.get('processed_documents', 0)
            stats['searchable_chunks'] = stats.get('total_chunks', 0)
            
            # Calculate search performance indicators
            if stats.get('total_documents', 0) > 0:
                stats['processing_completion_rate'] = (
                    stats.get('processed_documents', 0) / stats['total_documents']
                )
            else:
                stats['processing_completion_rate'] = 0.0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get document statistics: {e}")
            raise
