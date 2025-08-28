"""
Unit tests for Cross-Encoder Reranking functionality.

Following DEVELOPMENT_RULES.md for test requirements.
"""

import sys
import os
import pytest
from unittest.mock import Mock, patch, MagicMock

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rerank.cross_encoder import (
    CrossEncoderReranker, 
    RerankerFactory, 
    RerankResult,
    rerank_documents,
    SENTENCE_TRANSFORMERS_AVAILABLE
)


class TestCrossEncoderReranker:
    """Test cases for CrossEncoderReranker class."""
    
    def test_init_without_sentence_transformers(self):
        """Test initialization when sentence-transformers is not available."""
        with patch('rerank.cross_encoder.SENTENCE_TRANSFORMERS_AVAILABLE', False):
            reranker = CrossEncoderReranker()
            assert not reranker.is_available()
            assert reranker.model is None
    
    @patch('rerank.cross_encoder.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('rerank.cross_encoder.SentenceTransformersCrossEncoder')
    def test_init_with_sentence_transformers(self, mock_cross_encoder):
        """Test successful initialization with sentence-transformers."""
        mock_model = Mock()
        mock_cross_encoder.return_value = mock_model
        
        reranker = CrossEncoderReranker()
        assert reranker.is_available()
        assert reranker.model == mock_model
        mock_cross_encoder.assert_called_once()
    
    @patch('rerank.cross_encoder.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('rerank.cross_encoder.SentenceTransformersCrossEncoder')
    def test_init_failure(self, mock_cross_encoder):
        """Test initialization failure handling."""
        mock_cross_encoder.side_effect = Exception("Model load failed")
        
        reranker = CrossEncoderReranker()
        assert not reranker.is_available()
        assert reranker.model is None
    
    def test_rerank_without_model(self):
        """Test reranking fallback when model is not available."""
        reranker = CrossEncoderReranker()
        reranker.model = None
        
        candidates = [
            {'chunk_id': '1', 'text': 'first document', 'score': 0.8},
            {'chunk_id': '2', 'text': 'second document', 'score': 0.6}
        ]
        
        results = reranker.rerank("test query", candidates)
        
        assert len(results) == 2
        assert isinstance(results[0], RerankResult)
        assert results[0].chunk_id == '1'
        assert results[0].rerank_score == 0.8  # Should use original score
        assert results[0].final_rank == 1
    
    @patch('rerank.cross_encoder.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    def test_rerank_with_model(self):
        """Test reranking with available model."""
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.3]  # Second doc gets higher score
        
        reranker = CrossEncoderReranker()
        reranker.model = mock_model
        
        candidates = [
            {'chunk_id': '1', 'text': 'first document', 'score': 0.8},
            {'chunk_id': '2', 'text': 'second document', 'score': 0.6}
        ]
        
        results = reranker.rerank("test query", candidates)
        
        assert len(results) == 2
        assert results[0].chunk_id == '1'  # First should still be first (0.9 > 0.3)
        assert results[0].rerank_score == 0.9
        assert results[1].chunk_id == '2'
        assert results[1].rerank_score == 0.3
        
        mock_model.predict.assert_called_once()
        call_args = mock_model.predict.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0] == ["test query", "first document"]
        assert call_args[1] == ["test query", "second document"]
    
    def test_rerank_empty_candidates(self):
        """Test reranking with empty candidate list."""
        reranker = CrossEncoderReranker()
        results = reranker.rerank("test query", [])
        assert results == []
    
    @patch('rerank.cross_encoder.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    def test_rerank_with_top_k(self):
        """Test reranking with top_k limit."""
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.8, 0.7]
        
        reranker = CrossEncoderReranker()
        reranker.model = mock_model
        
        candidates = [
            {'chunk_id': '1', 'text': 'first', 'score': 0.5},
            {'chunk_id': '2', 'text': 'second', 'score': 0.6},
            {'chunk_id': '3', 'text': 'third', 'score': 0.7}
        ]
        
        results = reranker.rerank("test query", candidates, top_k=2)
        
        assert len(results) == 2
        assert results[0].chunk_id == '1'  # Highest rerank score
        assert results[1].chunk_id == '2'  # Second highest
    
    @patch('rerank.cross_encoder.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    def test_rerank_exception_handling(self):
        """Test exception handling during reranking."""
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Prediction failed")
        
        reranker = CrossEncoderReranker()
        reranker.model = mock_model
        
        candidates = [
            {'chunk_id': '1', 'text': 'test', 'score': 0.8}
        ]
        
        results = reranker.rerank("test query", candidates)
        
        # Should fallback to original ranking
        assert len(results) == 1
        assert results[0].rerank_score == 0.8


class TestRerankerFactory:
    """Test cases for RerankerFactory class."""
    
    def test_list_supported_models(self):
        """Test listing supported models."""
        models = RerankerFactory.list_supported_models()
        assert isinstance(models, dict)
        assert 'ms-marco-minilm' in models
        assert 'bge-reranker' in models
    
    @patch('rerank.cross_encoder.CrossEncoderReranker')
    def test_create_reranker_default(self, mock_reranker_class):
        """Test creating reranker with default model."""
        mock_reranker = Mock()
        mock_reranker_class.return_value = mock_reranker
        
        reranker = RerankerFactory.create_reranker()
        
        mock_reranker_class.assert_called_once_with(
            model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'
        )
        assert reranker == mock_reranker
    
    @patch('rerank.cross_encoder.CrossEncoderReranker')
    def test_create_reranker_specific_model(self, mock_reranker_class):
        """Test creating reranker with specific model."""
        mock_reranker = Mock()
        mock_reranker_class.return_value = mock_reranker
        
        reranker = RerankerFactory.create_reranker('bge-reranker', max_length=256)
        
        mock_reranker_class.assert_called_once_with(
            model_name='BAAI/bge-reranker-base',
            max_length=256
        )
    
    @patch('rerank.cross_encoder.CrossEncoderReranker')
    def test_create_reranker_unknown_model(self, mock_reranker_class):
        """Test creating reranker with unknown model type."""
        mock_reranker = Mock()
        mock_reranker_class.return_value = mock_reranker
        
        reranker = RerankerFactory.create_reranker('unknown-model')
        
        # Should fallback to default
        mock_reranker_class.assert_called_once_with(
            model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'
        )


class TestRerankResult:
    """Test cases for RerankResult dataclass."""
    
    def test_rerank_result_creation(self):
        """Test creating RerankResult instance."""
        result = RerankResult(
            chunk_id="test_chunk",
            text="test text",
            original_score=0.8,
            rerank_score=0.9,
            final_rank=1,
            metadata={"key": "value"}
        )
        
        assert result.chunk_id == "test_chunk"
        assert result.text == "test text"
        assert result.original_score == 0.8
        assert result.rerank_score == 0.9
        assert result.final_rank == 1
        assert result.metadata == {"key": "value"}


class TestConvenienceFunction:
    """Test cases for convenience functions."""
    
    @patch('rerank.cross_encoder.RerankerFactory')
    def test_rerank_documents_function(self, mock_factory):
        """Test the convenience rerank_documents function."""
        mock_reranker = Mock()
        mock_result = [Mock()]
        mock_reranker.rerank.return_value = mock_result
        mock_factory.create_reranker.return_value = mock_reranker
        
        documents = [{'chunk_id': '1', 'text': 'test', 'score': 0.8}]
        results = rerank_documents("test query", documents, "bge-reranker", 5)
        
        mock_factory.create_reranker.assert_called_once_with("bge-reranker")
        mock_reranker.rerank.assert_called_once_with("test query", documents, 5)
        assert results == mock_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
