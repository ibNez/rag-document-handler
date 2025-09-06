#!/usr/bin/env python
"""
Retriever with Reciprocal Rank Fusion (RRF)
Following DEVELOPMENT_RULES.md for all development requirements

This module combines vector similarity search with PostgreSQL FTS
using RRF fusion for improved retrieval performance.
"""

import logging
from typing import List, Any, Dict
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


class EmailRetriever:
    """
    Hybrid retriever combining vector similarity and PostgreSQL FTS.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results from both
    retrieval methods for improved performance.
    """
    
    def __init__(
        self, 
        vector_retriever: BaseRetriever, 
        fts_retriever: Any,  # Our custom PostgresFTSRetriever
        rrf_constant: int = 60
    ) -> None:
        """
        Initialize retriever.
        
        Args:
            vector_retriever: LangChain vector store retriever
            fts_retriever: PostgreSQL FTS retriever
            rrf_constant: RRF constant for fusion (default 60)
        """
        self.vector_retriever = vector_retriever
        self.fts_retriever = fts_retriever
        self.rrf_constant = rrf_constant
        logger.info(f"Hybrid retriever initialized with RRF constant: {rrf_constant}")
    
    def search(self, query: str, k: int = 10) -> List[Document]:
        """
        Perform search with RRF fusion.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of fused and ranked documents
        """
        try:
            logger.info(f"Starting search for: '{query[:50]}...' (k={k})")
            
            # Get results from both retrievers
            # Retrieve more from each to improve fusion quality
            retrieve_k = min(k * 2, 20)  # Get more results for better fusion
            
            logger.debug(f"Retrieving {retrieve_k} results from each retriever")
            
            # Vector similarity search
            vector_docs = self.vector_retriever.get_relevant_documents(query)[:retrieve_k]
            logger.info(f"Vector search returned {len(vector_docs)} results")
            
            # PostgreSQL FTS search  
            fts_docs = self.fts_retriever.search(query, k=retrieve_k)
            logger.info(f"FTS search returned {len(fts_docs)} results")
            
            # Apply RRF fusion
            fused_docs = self._apply_rrf_fusion(vector_docs, fts_docs, query)
            
            # Return top k results
            result_docs = fused_docs[:k]
            logger.info(f"Hybrid search returning {len(result_docs)} fused results")
            
            return result_docs
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to vector search only
            try:
                logger.warning("Falling back to vector search only")
                return self.vector_retriever.get_relevant_documents(query)[:k]
            except Exception as e2:
                logger.error(f"Vector search fallback also failed: {e2}")
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
        logger.debug("Applying RRF fusion to combine search results")
        
        # Create document score mapping
        doc_scores: Dict[str, Dict] = {}
        
        # Process vector search results
        for rank, doc in enumerate(vector_docs):
            # Use email_chunk_id as unique identifier
            doc_id = doc.metadata.get('email_chunk_id') or doc.metadata.get('chunk_id') or f"vector_{rank}"
            
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
            doc_id = doc.metadata.get('email_chunk_id') or doc.metadata.get('chunk_id') or f"fts_{rank}"
            
            # RRF score for FTS
            rrf_score = 1.0 / (rank + 1 + self.rrf_constant)
            
            if doc_id in doc_scores:
                # Document found in both searches - combine scores
                doc_scores[doc_id]['fts_rank'] = rank + 1
                doc_scores[doc_id]['fts_rrf'] = rrf_score
                doc_scores[doc_id]['combined_score'] += rrf_score
                doc_scores[doc_id]['document'].metadata['retrieval_method'] = 'hybrid'
                doc_scores[doc_id]['document'].metadata['fts_rank'] = rank + 1
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
        
        logger.info(f"RRF fusion stats - Vector only: {vector_only}, FTS only: {fts_only}, Both: {both}")
        
        return fused_docs
    
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
            
            # Calculate overlap using canonical email_chunk_id
            vector_ids = {doc.metadata.get('email_chunk_id') or doc.metadata.get('chunk_id', f"v_{i}") for i, doc in enumerate(vector_docs)}
            fts_ids = {doc.metadata.get('email_chunk_id') or doc.metadata.get('chunk_id', f"f_{i}") for i, doc in enumerate(fts_docs)}
            
            overlap = vector_ids.intersection(fts_ids)
            
            return {
                'vector_results': len(vector_docs),
                'fts_results': len(fts_docs),
                'overlap_count': len(overlap),
                'overlap_percentage': len(overlap) / max(len(vector_ids), 1) * 100,
                'unique_results': len(vector_ids.union(fts_ids)),
                'rrf_constant': self.rrf_constant
            }
            
        except Exception as e:
            logger.error(f"Failed to get fusion statistics: {e}")
            return {
                'error': str(e),
                'vector_results': 0,
                'fts_results': 0,
                'overlap_count': 0,
                'overlap_percentage': 0.0,
                'unique_results': 0,
                'rrf_constant': self.rrf_constant
            }
