#!/usr/bin/env python
"""
Document Hybrid Retriever with Reciprocal Rank Fusion (RRF)
Following DEVELOPMENT_RULES.md for all development requirements

This module combines vector similarity search with PostgreSQL FTS
for documents using RRF fusion for improved retrieval performance.
"""

import logging
from typing import List, Any, Dict, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


class DocumentHybridRetriever:
    """
    Hybrid retriever combining vector similarity and PostgreSQL FTS for documents.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results from both
    retrieval methods for improved performance.
    """
    
    def __init__(
        self, 
        vector_retriever: BaseRetriever, 
        fts_retriever: Any,  # DocumentPostgresFTSRetriever
        rrf_constant: int = 60
    ) -> None:
        """
        Initialize document hybrid retriever.
        
        Args:
            vector_retriever: LangChain vector store retriever for documents
            fts_retriever: Document PostgreSQL FTS retriever
            rrf_constant: RRF constant for fusion (default 60)
        """
        self.vector_retriever = vector_retriever
        self.fts_retriever = fts_retriever
        self.rrf_constant = rrf_constant
        logger.info(f"Document hybrid retriever initialized with RRF constant: {rrf_constant}")
    
    def search(self, query: str, k: int = 10, 
              document_id: Optional[str] = None,
              filetype_filter: Optional[str] = None,
              page_range: Optional[tuple[int, int]] = None) -> List[Document]:
        """
        Perform hybrid search with RRF fusion for documents.
        
        Args:
            query: Search query
            k: Number of results to return
            document_id: Optional document ID filter
            filetype_filter: Optional content type filter
            page_range: Optional page range filter
            
        Returns:
            List of fused and ranked documents
        """
        try:
            logger.info(f"Starting document hybrid search for: '{query[:50]}...' (k={k})")
            
            # Get results from both retrievers
            # Retrieve more from each to improve fusion quality
            retrieve_k = min(k * 2, 20)  # Get more results for better fusion
            
            logger.debug(f"Retrieving {retrieve_k} results from each retriever")
            
            # Vector similarity search
            vector_docs = self.vector_retriever.get_relevant_documents(query)[:retrieve_k]
            logger.info(f"Vector search returned {len(vector_docs)} results")
            
            # PostgreSQL FTS search with filters
            fts_docs = self.fts_retriever.search(
                query, 
                k=retrieve_k,
                document_id=document_id,
                filetype_filter=filetype_filter,
                page_range=page_range
            )
            logger.info(f"FTS search returned {len(fts_docs)} results")
            
            # Apply RRF fusion
            fused_docs = self._apply_rrf_fusion(vector_docs, fts_docs, query)
            
            # Return top k results
            result_docs = fused_docs[:k]
            logger.info(f"Document hybrid search returning {len(result_docs)} fused results")
            
            return result_docs
            
        except Exception as e:
            logger.error(f"Document hybrid search failed: {e}")
            # Fallback to vector search only
            try:
                logger.warning("Falling back to vector search only")
                return self.vector_retriever.get_relevant_documents(query)[:k]
            except Exception as e2:
                logger.error(f"Vector search fallback also failed: {e2}")
                return []
    
    def search_with_filters(self, query: str, k: int = 10, 
                           filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Search with complex filtering options using hybrid retrieval.
        
        Args:
            query: Search query
            k: Number of results to return
            filters: Dictionary with filter options (passed to FTS retriever)
                
        Returns:
            List of fused and ranked documents
        """
        try:
            logger.info(f"Starting filtered document hybrid search for: '{query[:50]}...'")
            
            retrieve_k = min(k * 2, 20)
            
            # Vector similarity search (no filtering applied here)
            vector_docs = self.vector_retriever.get_relevant_documents(query)[:retrieve_k]
            logger.info(f"Vector search returned {len(vector_docs)} results")
            
            # PostgreSQL FTS search with complex filters
            fts_docs = self.fts_retriever.search_with_filters(
                query, 
                k=retrieve_k,
                filters=filters
            )
            logger.info(f"Filtered FTS search returned {len(fts_docs)} results")
            
            # Apply RRF fusion
            fused_docs = self._apply_rrf_fusion(vector_docs, fts_docs, query)
            
            # Filter vector results to match FTS filters if needed
            if filters and fts_docs:
                fused_docs = self._apply_post_fusion_filters(fused_docs, filters)
            
            result_docs = fused_docs[:k]
            logger.info(f"Filtered document hybrid search returning {len(result_docs)} results")
            
            return result_docs
            
        except Exception as e:
            logger.error(f"Filtered document hybrid search failed: {e}")
            return []
    
    def _apply_rrf_fusion(
        self, 
        vector_docs: List[Document], 
        fts_docs: List[Document], 
        query: str
    ) -> List[Document]:
        """
        Apply Reciprocal Rank Fusion to combine retrieval results.
        
        RRF Formula: score = 1 / (rank + rrf_constant)
        
        Args:
            vector_docs: Documents from vector search
            fts_docs: Documents from FTS search
            query: Original search query
            
        Returns:
            List of documents ranked by combined RRF score
        """
        logger.debug("Applying RRF fusion to combine document search results")
        
        # Create document score mapping
        doc_scores: Dict[str, Dict] = {}
        
        # Process vector search results
        for rank, doc in enumerate(vector_docs):
            # Use chunk_id as unique identifier
            doc_id = doc.metadata.get('chunk_id', f"vector_{rank}")
            
            # RRF score: 1 / (rank + constant)
            rrf_score = 1.0 / (rank + 1 + self.rrf_constant)
            
            doc_scores[doc_id] = {
                'document': doc,
                'vector_rank': rank + 1,
                'vector_rrf': rrf_score,
                'fts_rank': None,
                'fts_rrf': 0.0,
                'combined_score': rrf_score
            }
            
            # Add retrieval method info to metadata
            doc.metadata['retrieval_method'] = 'vector'
            doc.metadata['vector_rank'] = rank + 1
        
        # Process FTS search results
        for rank, doc in enumerate(fts_docs):
            doc_id = doc.metadata.get('chunk_id', f"fts_{rank}")
            
            # RRF score for FTS
            rrf_score = 1.0 / (rank + 1 + self.rrf_constant)
            
            if doc_id in doc_scores:
                # Document found in both searches - combine scores
                doc_scores[doc_id]['fts_rank'] = rank + 1
                doc_scores[doc_id]['fts_rrf'] = rrf_score
                doc_scores[doc_id]['combined_score'] += rrf_score
                doc_scores[doc_id]['document'].metadata['retrieval_method'] = 'hybrid'
                doc_scores[doc_id]['document'].metadata['fts_rank'] = rank + 1
                
                # Preserve FTS-specific metadata
                if 'fts_score' in doc.metadata:
                    doc_scores[doc_id]['document'].metadata['fts_score'] = doc.metadata['fts_score']
            else:
                # Document only found in FTS
                doc_scores[doc_id] = {
                    'document': doc,
                    'vector_rank': None,
                    'vector_rrf': 0.0,
                    'fts_rank': rank + 1,
                    'fts_rrf': rrf_score,
                    'combined_score': rrf_score
                }
                
                # Add retrieval method info to metadata
                doc.metadata['retrieval_method'] = 'fts'
                doc.metadata['fts_rank'] = rank + 1
        
        # Sort by combined RRF score (descending)
        sorted_items = sorted(
            doc_scores.values(), 
            key=lambda x: x['combined_score'], 
            reverse=True
        )
        
        # Add combined score to metadata and return documents
        fused_docs = []
        for item in sorted_items:
            doc = item['document']
            doc.metadata['combined_score'] = item['combined_score']
            doc.metadata['vector_rrf_score'] = item['vector_rrf']
            doc.metadata['fts_rrf_score'] = item['fts_rrf']
            fused_docs.append(doc)
        
        logger.debug(f"RRF fusion completed: {len(fused_docs)} unique documents ranked")
        
        # Log fusion statistics
        vector_only = sum(1 for item in sorted_items if item['fts_rank'] is None)
        fts_only = sum(1 for item in sorted_items if item['vector_rank'] is None)
        both = sum(1 for item in sorted_items if item['vector_rank'] is not None and item['fts_rank'] is not None)
        
        logger.info(f"Document RRF fusion stats - Vector only: {vector_only}, FTS only: {fts_only}, Both: {both}")
        
        return fused_docs
    
    def _apply_post_fusion_filters(self, docs: List[Document], 
                                  filters: Dict[str, Any]) -> List[Document]:
        """
        Apply additional filters to fused results to ensure consistency.
        
        Args:
            docs: Fused documents
            filters: Filter criteria
            
        Returns:
            Filtered documents
        """
        filtered_docs = []
        
        for doc in docs:
            metadata = doc.metadata
            include = True
            
            # Apply content type filter
            if filters.get('content_types'):
                if metadata.get('content_type') not in filters['content_types']:
                    include = False
            
            # Apply document ID filter
            if filters.get('document_ids'):
                if metadata.get('document_id') not in filters['document_ids']:
                    include = False
            
            # Apply element type filter
            if filters.get('element_types'):
                doc_element_types = metadata.get('element_types', [])
                if not any(et in doc_element_types for et in filters['element_types']):
                    include = False
            
            # Apply page range filter
            if filters.get('page_range'):
                min_page, max_page = filters['page_range']
                page_start = metadata.get('page_start')
                page_end = metadata.get('page_end')
                if page_start is not None and page_end is not None:
                    if page_start < min_page or page_end > max_page:
                        include = False
            
            if include:
                filtered_docs.append(doc)
        
        logger.debug(f"Post-fusion filtering: {len(docs)} -> {len(filtered_docs)} documents")
        return filtered_docs
    
    def get_fusion_statistics(self, query: str) -> Dict[str, Any]:
        """
        Get detailed statistics about fusion performance for a query.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary with fusion statistics
        """
        try:
            # Get results from both retrievers
            vector_docs = self.vector_retriever.get_relevant_documents(query)[:10]
            fts_docs = self.fts_retriever.search(query, k=10)
            
            # Calculate overlap
            vector_ids = {doc.metadata.get('chunk_id', f"v_{i}") for i, doc in enumerate(vector_docs)}
            fts_ids = {doc.metadata.get('chunk_id', f"f_{i}") for i, doc in enumerate(fts_docs)}
            
            overlap = vector_ids.intersection(fts_ids)
            
            # Get content type distribution
            vector_types = {}
            fts_types = {}
            
            for doc in vector_docs:
                content_type = doc.metadata.get('content_type', 'unknown')
                vector_types[content_type] = vector_types.get(content_type, 0) + 1
                
            for doc in fts_docs:
                content_type = doc.metadata.get('content_type', 'unknown')
                fts_types[content_type] = fts_types.get(content_type, 0) + 1
            
            return {
                'query': query,
                'vector_results': len(vector_docs),
                'fts_results': len(fts_docs),
                'overlap_count': len(overlap),
                'overlap_percentage': len(overlap) / max(len(vector_ids), 1) * 100,
                'unique_results': len(vector_ids.union(fts_ids)),
                'rrf_constant': self.rrf_constant,
                'vector_content_types': vector_types,
                'fts_content_types': fts_types
            }
            
        except Exception as e:
            logger.error(f"Failed to get document fusion statistics: {e}")
            return {
                'error': str(e),
                'query': query,
                'vector_results': 0,
                'fts_results': 0,
                'overlap_count': 0,
                'overlap_percentage': 0.0,
                'unique_results': 0,
                'rrf_constant': self.rrf_constant,
                'vector_content_types': {},
                'fts_content_types': {}
            }
