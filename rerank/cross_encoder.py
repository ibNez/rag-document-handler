"""
Cross-Encoder Reranking for RAG Knowledgebase Manager.

This module implements cross-encoder reranking to improve the final relevance
ranking of retrieved documents and chunks after initial retrieval and fusion.

Cross-encoders provide higher quality relevance scoring by jointly encoding
the query and candidate text, unlike bi-encoders that encode them separately.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

try:
    from sentence_transformers import CrossEncoder as SentenceTransformersCrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformersCrossEncoder = None

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result from cross-encoder reranking."""
    chunk_id: str
    text: str
    original_score: float
    rerank_score: float
    final_rank: int
    metadata: Dict[str, Any]


class CrossEncoderReranker:
    """
    Cross-encoder reranker for improving retrieval relevance.
    
    This class uses pre-trained cross-encoder models to rerank retrieved
    documents/chunks for better relevance to the user query.
    
    Follows development rules with proper type hints, error handling,
    and comprehensive logging.
    """
    
    def __init__(
        self, 
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        device: Optional[str] = None
    ) -> None:
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Name of the cross-encoder model to use
            max_length: Maximum sequence length for the model
            device: Device to run the model on ('cpu', 'cuda', etc.)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self.model: Optional[Any] = None
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the cross-encoder model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning(
                "sentence-transformers not available. Cross-encoder reranking disabled. "
                "Install with: pip install sentence-transformers"
            )
            return
        
        try:
            logger.info(f"Initializing cross-encoder model: {self.model_name}")
            start_time = time.time()
            
            if SentenceTransformersCrossEncoder is None:
                raise ImportError("sentence-transformers not available")
                
            self.model = SentenceTransformersCrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device
            )
            
            load_time = time.time() - start_time
            logger.info(f"Cross-encoder model loaded in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to initialize cross-encoder model: {e}")
            self.model = None

    def _get_canonical_chunk_id(self, candidate: Dict[str, Any]) -> str:
        """Return the canonical chunk id for a candidate.

        Enforces table-specific keys only: `document_chunk_id` or `email_chunk_id`.
        """
        doc_cid = candidate.get('document_chunk_id')
        if doc_cid:
            return doc_cid
        email_cid = candidate.get('email_chunk_id')
        if email_cid:
            return email_cid
        # Log the problematic row for debugging and fail fast
        logger.error(f"Missing canonical chunk id for candidate metadata: {candidate}")
        raise KeyError('Missing canonical chunk id (document_chunk_id or email_chunk_id)')
    
    def is_available(self) -> bool:
        """Check if reranker is available for use."""
        return self.model is not None
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank candidates using cross-encoder model.
        
        Args:
            query: The search query
            candidates: List of candidate documents/chunks to rerank
            top_k: Number of top results to return (default: all)
        
        Returns:
            List of reranked results ordered by relevance
        """
        if not self.is_available():
            raise RuntimeError("Cross-encoder model not available - ensure model is properly loaded before attempting reranking")
        
        if not candidates:
            return []
        
        try:
            logger.debug(f"Reranking {len(candidates)} candidates for query: {query[:100]}...")
            start_time = time.time()
            
            # Prepare query-candidate pairs for cross-encoder
            pairs = []
            candidate_metadata = []
            
            for candidate in candidates:
                text = candidate.get('text', '')
                # Enforce canonical chunk id keys
                chunk_id = self._get_canonical_chunk_id(candidate)
                original_score = candidate.get('score', candidate.get('similarity_score', 0.0))
                
                # Truncate text if too long
                if len(text) > self.max_length * 2:  # Rough character estimate
                    text = text[:self.max_length * 2]
                
                pairs.append([query, text])
                candidate_metadata.append({
                    'chunk_id': chunk_id,
                    'text': text,
                    'original_score': original_score,
                    'metadata': candidate
                })
            
            # Get cross-encoder scores
            if self.model is None:
                raise ValueError("Cross-encoder model is not initialized")
            scores = self.model.predict(pairs)
            
            # Create results with rerank scores
            results = []
            for i, score in enumerate(scores):
                meta = candidate_metadata[i]
                result = RerankResult(
                    chunk_id=meta['chunk_id'],
                    text=meta['text'],
                    original_score=meta['original_score'],
                    rerank_score=float(score),
                    final_rank=0,  # Will be set after sorting
                    metadata=meta['metadata']
                )
                results.append(result)
            
            # Sort by rerank score (higher is better)
            results.sort(key=lambda x: x.rerank_score, reverse=True)
            
            # Set final ranks
            for i, result in enumerate(results):
                result.final_rank = i + 1
            
            # Apply top_k limit if specified
            if top_k and top_k > 0:
                results = results[:top_k]
            
            rerank_time = time.time() - start_time
            logger.info(f"Reranked {len(candidates)} candidates in {rerank_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            raise RuntimeError(f"Cross-encoder reranking failed: {e}") from e


class RerankerFactory:
    """Factory for creating different types of rerankers."""
    
    SUPPORTED_MODELS = {
        'ms-marco-minilm': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        'ms-marco-large': 'cross-encoder/ms-marco-TinyBERT-L-2-v2',
        'bge-reranker': 'BAAI/bge-reranker-base',
        'bge-reranker-large': 'BAAI/bge-reranker-large'
    }
    
    @classmethod
    def create_reranker(
        cls, 
        model_type: str = 'ms-marco-minilm',
        **kwargs
    ) -> CrossEncoderReranker:
        """
        Create a reranker instance.
        
        Args:
            model_type: Type of model to use (key from SUPPORTED_MODELS)
            **kwargs: Additional arguments to pass to the reranker
        
        Returns:
            CrossEncoderReranker instance
        """
        if model_type not in cls.SUPPORTED_MODELS:
            logger.warning(f"Unknown model type: {model_type}, using default")
            model_type = 'ms-marco-minilm'
        
        model_name = cls.SUPPORTED_MODELS[model_type]
        return CrossEncoderReranker(model_name=model_name, **kwargs)

