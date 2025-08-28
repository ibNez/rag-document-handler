"""
Unit tests for Advanced Chunking Strategies functionality.

Following DEVELOPMENT_RULES.md for test requirements.
"""

import sys
import os
import pytest
from unittest.mock import Mock, patch, MagicMock

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from ingestion.document.advanced_chunking import (
    AdvancedChunker,
    ChunkingConfig,
    ElementInfo,
    create_chunker
)


class TestChunkingConfig:
    """Test cases for ChunkingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ChunkingConfig()
        assert config.strategy == "title_aware"
        assert config.target_tokens == 850
        assert config.overlap_percentage == 0.125
        assert config.max_tokens == 1200
        assert config.min_tokens == 100
        assert config.preserve_tables is True
        assert config.track_section_paths is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ChunkingConfig(
            strategy="page_aware",
            target_tokens=1000,
            overlap_percentage=0.15,
            preserve_tables=False
        )
        assert config.strategy == "page_aware"
        assert config.target_tokens == 1000
        assert config.overlap_percentage == 0.15
        assert config.preserve_tables is False


class TestElementInfo:
    """Test cases for ElementInfo dataclass."""
    
    def test_element_info_creation(self):
        """Test creating ElementInfo instance."""
        element = ElementInfo(
            text="Test content",
            element_type="Title",
            page_number=1,
            section_path="Chapter 1",
            is_title=True
        )
        
        assert element.text == "Test content"
        assert element.element_type == "Title"
        assert element.page_number == 1
        assert element.section_path == "Chapter 1"
        assert element.is_title is True
        assert element.is_table is False
        assert element.metadata == {}
    
    def test_element_info_post_init(self):
        """Test post-init behavior for metadata."""
        element = ElementInfo(text="Test", element_type="Text")
        assert isinstance(element.metadata, dict)
        assert element.metadata == {}


class TestAdvancedChunker:
    """Test cases for AdvancedChunker class."""
    
    def test_init_default_config(self):
        """Test initialization with default config."""
        chunker = AdvancedChunker()
        assert chunker.config.strategy == "title_aware"
        assert chunker.config.target_tokens == 850
    
    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = ChunkingConfig(strategy="page_aware", target_tokens=1000)
        chunker = AdvancedChunker(config)
        assert chunker.config.strategy == "page_aware"
        assert chunker.config.target_tokens == 1000
    
    @patch('ingestion.document.advanced_chunking.TIKTOKEN_AVAILABLE', False)
    def test_init_without_tiktoken(self):
        """Test initialization when tiktoken is not available."""
        chunker = AdvancedChunker()
        assert chunker.tokenizer is None
    
    def test_count_tokens_fallback(self):
        """Test token counting with fallback method."""
        chunker = AdvancedChunker()
        chunker.tokenizer = None  # Force fallback
        
        tokens = chunker._count_tokens("This is a test sentence.")
        assert tokens == 6  # 24 chars / 4 = 6 tokens (rough estimate)
    
    def test_process_elements(self):
        """Test processing documents into element info."""
        chunker = AdvancedChunker()
        
        documents = [
            Document(
                page_content="Chapter 1: Introduction",
                metadata={"category": "Title", "page_number": 1}
            ),
            Document(
                page_content="This is the introduction text.",
                metadata={"category": "Text", "page_number": 1}
            ),
            Document(
                page_content="Table data here",
                metadata={"category": "Table", "page_number": 2}
            )
        ]
        
        elements = chunker._process_elements(documents)
        
        assert len(elements) == 3
        assert elements[0].is_title is True
        assert elements[1].is_title is False
        assert elements[2].is_table is True
        assert elements[0].page_number == 1
        assert elements[2].page_number == 2
    
    def test_basic_chunking(self):
        """Test basic chunking strategy."""
        config = ChunkingConfig(strategy="basic", max_tokens=50)
        chunker = AdvancedChunker(config)
        
        elements = [
            ElementInfo(text="Short text", element_type="Text"),
            ElementInfo(text="This is a longer piece of text that should be in its own chunk", element_type="Text"),
            ElementInfo(text="Another piece", element_type="Text")
        ]
        
        chunks = chunker._basic_chunking(elements)
        
        assert len(chunks) >= 1
        assert all(isinstance(chunk, list) for chunk in chunks)
        assert all(isinstance(element, ElementInfo) for chunk in chunks for element in chunk)
    
    def test_title_aware_chunking_with_titles(self):
        """Test title-aware chunking with title elements."""
        config = ChunkingConfig(strategy="title_aware", max_tokens=20, min_tokens=5)  # Low limits to force splitting
        chunker = AdvancedChunker(config)
        
        elements = [
            ElementInfo(text="Chapter 1", element_type="Title", is_title=True),
            ElementInfo(text="Introduction text here with more content", element_type="Text"),
            ElementInfo(text="Chapter 2", element_type="Title", is_title=True),
            ElementInfo(text="More content text here", element_type="Text")
        ]
        
        chunks = chunker._title_aware_chunking(elements)
        
        # Should create separate chunks for each chapter with these token limits
        assert len(chunks) >= 2
        # First chunk should start with "Chapter 1"
        assert chunks[0][0].text == "Chapter 1"
        # Should find "Chapter 2" in one of the chunks
        chapter2_found = any(any(e.text == "Chapter 2" for e in chunk) for chunk in chunks)
        assert chapter2_found
    
    def test_page_aware_chunking(self):
        """Test page-aware chunking strategy."""
        config = ChunkingConfig(strategy="page_aware", max_tokens=100)
        chunker = AdvancedChunker(config)
        
        elements = [
            ElementInfo(text="Page 1 content", element_type="Text", page_number=1),
            ElementInfo(text="More page 1", element_type="Text", page_number=1),
            ElementInfo(text="Page 2 content", element_type="Text", page_number=2),
            ElementInfo(text="More page 2", element_type="Text", page_number=2)
        ]
        
        chunks = chunker._page_aware_chunking(elements)
        
        # Should group by page
        assert len(chunks) >= 2
        # Check that elements from same page are grouped
        page1_elements = [e for chunk in chunks for e in chunk if e.page_number == 1]
        page2_elements = [e for chunk in chunks for e in chunk if e.page_number == 2]
        assert len(page1_elements) == 2
        assert len(page2_elements) == 2
    
    def test_create_document_chunks(self):
        """Test converting element chunks to Document objects."""
        chunker = AdvancedChunker()
        
        element_chunks = [
            [
                ElementInfo(text="First chunk text", element_type="Text", page_number=1),
                ElementInfo(text="More text", element_type="Text", page_number=1)
            ],
            [
                ElementInfo(text="Second chunk", element_type="Text", page_number=2)
            ]
        ]
        
        documents = chunker._create_document_chunks(element_chunks, "doc123")
        
        assert len(documents) == 2
        assert isinstance(documents[0], Document)
        assert documents[0].metadata['chunk_id'] == "doc123#c1"
        assert documents[1].metadata['chunk_id'] == "doc123#c2"
        assert documents[0].metadata['page_start'] == 1
        assert documents[0].metadata['page_end'] == 1
        assert documents[1].metadata['page_start'] == 2
    
    def test_chunk_documents_integration(self):
        """Test full document chunking integration."""
        chunker = AdvancedChunker()
        
        input_documents = [
            Document(
                page_content="Introduction Chapter",
                metadata={"category": "Title", "page_number": 1}
            ),
            Document(
                page_content="This is the introduction content with enough text to be meaningful.",
                metadata={"category": "Text", "page_number": 1}
            ),
            Document(
                page_content="Conclusion Chapter",
                metadata={"category": "Title", "page_number": 2}
            ),
            Document(
                page_content="This concludes our document with final thoughts.",
                metadata={"category": "Text", "page_number": 2}
            )
        ]
        
        result = chunker.chunk_documents(input_documents, "test_doc")
        
        assert len(result) > 0
        assert all(isinstance(doc, Document) for doc in result)
        assert all('chunk_id' in doc.metadata for doc in result)
        assert all('document_id' in doc.metadata for doc in result)
        assert all(doc.metadata['document_id'] == "test_doc" for doc in result)
    
    def test_get_chunking_stats(self):
        """Test chunking statistics calculation."""
        chunker = AdvancedChunker()
        
        chunks = [
            Document(
                page_content="Test content",
                metadata={
                    'token_count': 50,
                    'page_start': 1,
                    'page_end': 1,
                    'element_types': ['Text']
                }
            ),
            Document(
                page_content="More content",
                metadata={
                    'token_count': 75,
                    'page_start': 2,
                    'page_end': 3,
                    'element_types': ['Title', 'Text']
                }
            )
        ]
        
        stats = chunker.get_chunking_stats(chunks)
        
        assert stats['total_chunks'] == 2
        assert stats['total_tokens'] == 125
        assert stats['avg_tokens_per_chunk'] == 62.5
        assert stats['min_tokens'] == 50
        assert stats['max_tokens'] == 75
        assert 'element_type_distribution' in stats
    
    def test_metadata_preservation(self):
        """Test that document-level metadata is preserved in chunks."""
        chunker = AdvancedChunker()
        
        # Create documents with metadata
        documents = [
            Document(
                page_content="This is the first paragraph.",
                metadata={'category': 'Text', 'page_number': 1}
            ),
            Document(
                page_content="This is the second paragraph.",
                metadata={'category': 'Text', 'page_number': 1}
            )
        ]
        
        # Define preserve metadata
        preserve_metadata = {
            'filename': 'test_document.pdf',
            'source': 'test_document.pdf',
            'content_type': 'application/pdf',
            'filetype': 'pdf'
        }
        
        chunks = chunker.chunk_documents(documents, "test_doc", preserve_metadata)
        
        assert len(chunks) > 0
        
        # Check that preserved metadata is in all chunks
        for chunk in chunks:
            assert chunk.metadata['filename'] == 'test_document.pdf'
            assert chunk.metadata['source'] == 'test_document.pdf'
            assert chunk.metadata['content_type'] == 'application/pdf'
            assert chunk.metadata['filetype'] == 'pdf'
            
            # Check chunk-specific metadata is also present
            assert 'chunk_id' in chunk.metadata
            assert 'document_id' in chunk.metadata
            assert 'chunk_ordinal' in chunk.metadata
            assert chunk.metadata['document_id'] == 'test_doc'

    def test_empty_document_handling(self):
        """Test handling of empty documents."""
        chunker = AdvancedChunker()
        result = chunker.chunk_documents([], "empty_doc")
        assert result == []
    
    def test_section_path_tracking(self):
        """Test section path tracking functionality."""
        config = ChunkingConfig(track_section_paths=True)
        chunker = AdvancedChunker(config)
        
        section_stack = []
        
        # Test adding sections - section_stack is modified in place
        path1 = chunker._update_section_path("Introduction", "Title", section_stack)
        assert path1 is not None
        assert "Introduction" in path1
        
        # After first call, section_stack should contain "Introduction"
        path2 = chunker._update_section_path("Overview", "Title", section_stack)
        assert path2 is not None
        # Overview should replace Introduction for same-level heading
        assert "Overview" in path2


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_create_chunker_default(self):
        """Test create_chunker with default strategy."""
        chunker = create_chunker()
        assert chunker.config.strategy == "title_aware"
    
    def test_create_chunker_custom_strategy(self):
        """Test create_chunker with custom strategy."""
        chunker = create_chunker("page_aware", target_tokens=1000)
        assert chunker.config.strategy == "page_aware"
        assert chunker.config.target_tokens == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
