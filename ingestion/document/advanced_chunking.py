"""
Advanced Chunking Strategies for RAG Knowledgebase Manager.

This module implements advanced chunking strategies according to the POC specification:
- Title-aware chunking: Groups by headings with optimal token sizes
- Page-aware chunking: Page-bounded chunks for precise citations  
- Section path tracking: Hierarchical structure preservation
- Element types preservation: Rich metadata from Unstructured

Following DEVELOPMENT_RULES.md for all development requirements.
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from langchain_core.documents import Document

# tiktoken import with fallback
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies."""
    strategy: str = "title_aware"  # title_aware, page_aware, basic
    target_tokens: int = 850  # Target ~800-1,000 tokens
    overlap_percentage: float = 0.125  # 10-15% overlap
    max_tokens: int = 1200  # Hard limit
    min_tokens: int = 100  # Minimum viable chunk size
    preserve_tables: bool = True
    preserve_lists: bool = True
    track_section_paths: bool = True
    encoding_model: str = "cl100k_base"  # GPT tokenizer


@dataclass
class ElementInfo:
    """Information about a document element from Unstructured."""
    text: str
    element_type: str
    page_number: Optional[int] = None
    section_path: Optional[str] = None
    is_title: bool = False
    is_table: bool = False
    is_list: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.metadata:
            self.metadata = {}


class AdvancedChunker:
    """
    Advanced document chunking with multiple strategies.
    
    Implements title-aware, page-aware, and element-preserving chunking
    strategies for better document structure preservation and citation accuracy.
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize advanced chunker.
        
        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig()
        
        # Initialize tokenizer
        if TIKTOKEN_AVAILABLE and tiktoken:
            try:
                self.tokenizer = tiktoken.get_encoding(self.config.encoding_model)
            except Exception as e:
                logger.warning(f"Failed to load tokenizer {self.config.encoding_model}: {e}")
                self.tokenizer = None
        else:
            logger.warning("tiktoken not available, using fallback token counting")
            self.tokenizer = None
            
        logger.info(f"Advanced chunker initialized with strategy: {self.config.strategy}")
    
    def chunk_documents(self, documents: List[Document], document_id: str, preserve_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Chunk documents using the configured strategy.
        
        Args:
            documents: Raw documents from Unstructured loader
            document_id: Document identifier for chunk IDs
            preserve_metadata: Document-level metadata to preserve in all chunks
            
        Returns:
            List of chunked documents with enhanced metadata
        """
        if not documents:
            return []
        
        logger.info(f"Chunking {len(documents)} elements using {self.config.strategy} strategy")
        
        # Extract common metadata from first document to preserve
        if not preserve_metadata and documents:
            first_doc_metadata = documents[0].metadata or {}
            preserve_metadata = {
                'filename': first_doc_metadata.get('filename'),
                'filetype': first_doc_metadata.get('filetype'),
                'content_type': first_doc_metadata.get('content_type'),
                'file_path': first_doc_metadata.get('file_path'),
            }
            # Filter out None values
            preserve_metadata = {k: v for k, v in preserve_metadata.items() if v is not None}
        
        # Convert to element info objects
        elements = self._process_elements(documents)
        
        # Apply chunking strategy
        if self.config.strategy == "title_aware":
            chunks = self._title_aware_chunking(elements)
        elif self.config.strategy == "page_aware":
            chunks = self._page_aware_chunking(elements)
        else:
            chunks = self._basic_chunking(elements)
        
        # Convert back to Document objects with enhanced metadata
        result_documents = self._create_document_chunks(chunks, document_id, preserve_metadata or {})
        
        logger.info(f"Created {len(result_documents)} chunks from {len(elements)} elements")
        return result_documents
    
    def _process_elements(self, documents: List[Document]) -> List[ElementInfo]:
        """Process raw documents into structured element info."""
        elements = []
        section_stack = []  # Track hierarchical sections
        
        for doc in documents:
            metadata = doc.metadata or {}
            element_type = metadata.get('category', 'Text')
            page_number = metadata.get('page_number') or metadata.get('page')
            
            # Determine element characteristics
            is_title = element_type in ['Title', 'Header']
            is_table = element_type in ['Table']
            is_list = element_type in ['ListItem', 'List']
            
            # Update section path tracking
            section_path = None
            if self.config.track_section_paths:
                section_path = self._update_section_path(
                    doc.page_content, element_type, section_stack
                )
            
            element = ElementInfo(
                text=doc.page_content or '',
                element_type=element_type,
                page_number=page_number,
                section_path=section_path,
                is_title=is_title,
                is_table=is_table,
                is_list=is_list,
                metadata=metadata
            )
            elements.append(element)
        
        return elements
    
    def _update_section_path(self, text: str, element_type: str, section_stack: List[str]) -> Optional[str]:
        """Update section path tracking for hierarchical structure."""
        if element_type == 'Title':
            # Determine heading level (simplified heuristic)
            heading_level = self._estimate_heading_level(text, element_type)
            
            # Update section stack - keep building the path
            # Don't truncate unless we detect a higher-level heading
            if heading_level == 1:
                # Top level - clear stack and start fresh
                section_stack.clear()
            elif heading_level <= len(section_stack):
                # Same or higher level - truncate to appropriate level
                section_stack = section_stack[:heading_level-1]
            
            # Add current heading to stack
            section_stack.append(text.strip()[:50])  # Limit length
            
        return " > ".join(section_stack) if section_stack else None
    
    def _estimate_heading_level(self, text: str, element_type: str) -> int:
        """Estimate heading level based on text characteristics."""
        # Simple heuristic - can be enhanced with more sophisticated logic
        text_clean = text.strip()
        
        if len(text_clean) < 10:
            return 1  # Very short = high level
        elif len(text_clean) < 30:
            return 2  # Short = medium level
        else:
            return 3  # Longer = lower level
    
    def _title_aware_chunking(self, elements: List[ElementInfo]) -> List[List[ElementInfo]]:
        """
        Implement title-aware chunking strategy.
        
        Groups content by headings with optimal token sizes,
        preserving tables and lists as complete units.
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for element in elements:
            element_tokens = self._count_tokens(element.text)
            
            # Start new chunk on titles (except very first element)
            if element.is_title and current_chunk and current_tokens >= self.config.min_tokens:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            
            # Special handling for tables and lists
            if (element.is_table or element.is_list) and self.config.preserve_tables:
                # If adding this would exceed max, finalize current chunk first
                if current_tokens + element_tokens > self.config.max_tokens and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_tokens = 0
                
                # Add as complete unit
                current_chunk.append(element)
                current_tokens += element_tokens
                
                # If table/list is very large, finalize immediately
                if current_tokens > self.config.target_tokens:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_tokens = 0
                    
                continue
            
            # Regular text handling with overlap
            if current_tokens + element_tokens > self.config.max_tokens and current_chunk:
                # Create overlap from last elements
                overlap_chunk = self._create_overlap_chunk(current_chunk)
                chunks.append(current_chunk)
                current_chunk = overlap_chunk
                current_tokens = sum(self._count_tokens(e.text) for e in overlap_chunk)
            
            current_chunk.append(element)
            current_tokens += element_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _page_aware_chunking(self, elements: List[ElementInfo]) -> List[List[ElementInfo]]:
        """
        Implement page-aware chunking strategy.
        
        Creates page-bounded chunks for precise citations,
        with fallback to token limits within pages.
        """
        # Group elements by page
        page_groups = {}
        for element in elements:
            page = element.page_number or 1
            if page not in page_groups:
                page_groups[page] = []
            page_groups[page].append(element)
        
        chunks = []
        
        # Process each page
        for page_num in sorted(page_groups.keys()):
            page_elements = page_groups[page_num]
            page_tokens = sum(self._count_tokens(e.text) for e in page_elements)
            
            if page_tokens <= self.config.max_tokens:
                # Entire page fits in one chunk
                chunks.append(page_elements)
            else:
                # Split page into multiple chunks
                page_chunks = self._split_page_elements(page_elements)
                chunks.extend(page_chunks)
        
        return chunks
    
    def _basic_chunking(self, elements: List[ElementInfo]) -> List[List[ElementInfo]]:
        """Basic chunking strategy with token limits."""
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for element in elements:
            element_tokens = self._count_tokens(element.text)
            
            if current_tokens + element_tokens > self.config.max_tokens and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(element)
            current_tokens += element_tokens
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_page_elements(self, page_elements: List[ElementInfo]) -> List[List[ElementInfo]]:
        """Split page elements into token-bounded chunks."""
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for element in page_elements:
            element_tokens = self._count_tokens(element.text)
            
            if current_tokens + element_tokens > self.config.max_tokens and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(element)
            current_tokens += element_tokens
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _create_overlap_chunk(self, chunk: List[ElementInfo]) -> List[ElementInfo]:
        """Create overlap from the end of a chunk."""
        if not chunk:
            return []
        
        overlap_tokens_target = int(self.config.target_tokens * self.config.overlap_percentage)
        overlap_elements = []
        overlap_tokens = 0
        
        # Take elements from the end until we reach overlap target
        for element in reversed(chunk):
            element_tokens = self._count_tokens(element.text)
            if overlap_tokens + element_tokens > overlap_tokens_target:
                break
            overlap_elements.insert(0, element)
            overlap_tokens += element_tokens
        
        return overlap_elements
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        
        # Fallback: rough estimation (4 chars per token)
        return len(text) // 4
    
    def _create_document_chunks(self, chunks: List[List[ElementInfo]], document_id: str, preserve_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Convert element chunks back to Document objects with enhanced metadata."""
        documents = []
        preserve_metadata = preserve_metadata or {}
        
        for chunk_index, chunk_elements in enumerate(chunks):
            if not chunk_elements:
                continue
            
            # Combine text from all elements in chunk
            chunk_text = "\n\n".join(element.text for element in chunk_elements if element.text)
            
            if not chunk_text.strip():
                continue
            
            # Calculate chunk metadata
            page_numbers = [e.page_number for e in chunk_elements if e.page_number is not None]
            page_start = min(page_numbers) if page_numbers else None
            page_end = max(page_numbers) if page_numbers else None
            
            # Collect element types
            element_types = list(set(e.element_type for e in chunk_elements))
            
            # Get section path (from first title element, or first element)
            section_paths = [e.section_path for e in chunk_elements if e.section_path]
            section_path = section_paths[0] if section_paths else None
            
            # Create content hash
            content_hash = hashlib.sha1(chunk_text.encode('utf-8')).hexdigest()[:16]
            
            # Create chunk ID
            chunk_id = f"{document_id}#c{chunk_index + 1}"
            
            # Calculate token count
            token_count = self._count_tokens(chunk_text)
            
            # Enhanced metadata - start with preserved document metadata
            metadata = preserve_metadata.copy()
            
            # Add chunk-specific metadata
            metadata.update({
                'chunk_id': chunk_id,
                'document_id': document_id,
                'chunk_ordinal': chunk_index + 1,
                'page_start': page_start,
                'page_end': page_end,
                'section_path': section_path,
                'element_types': element_types,
                'token_count': token_count,
                'chunk_hash': content_hash,
                'chunking_strategy': self.config.strategy,
                'element_count': len(chunk_elements)
            })
            
            # Add POC-compliant metadata
            if page_start and page_end:
                metadata['pages'] = f"{page_start}-{page_end}" if page_start != page_end else str(page_start)
            
            document = Document(
                page_content=chunk_text,
                metadata=metadata
            )
            documents.append(document)
        
        return documents
    
    def get_chunking_stats(self, chunks: List[Document]) -> Dict[str, Any]:
        """Get statistics about chunking results."""
        if not chunks:
            return {}
        
        token_counts = [chunk.metadata.get('token_count', 0) for chunk in chunks]
        page_counts = []
        element_type_counts = {}
        
        for chunk in chunks:
            # Page span calculation
            page_start = chunk.metadata.get('page_start')
            page_end = chunk.metadata.get('page_end')
            if page_start and page_end:
                page_counts.append(page_end - page_start + 1)
            
            # Element type counting
            element_types = chunk.metadata.get('element_types', [])
            for element_type in element_types:
                element_type_counts[element_type] = element_type_counts.get(element_type, 0) + 1
        
        return {
            'total_chunks': len(chunks),
            'total_tokens': sum(token_counts),
            'avg_tokens_per_chunk': sum(token_counts) / len(token_counts) if token_counts else 0,
            'min_tokens': min(token_counts) if token_counts else 0,
            'max_tokens': max(token_counts) if token_counts else 0,
            'avg_pages_per_chunk': sum(page_counts) / len(page_counts) if page_counts else 0,
            'element_type_distribution': element_type_counts,
            'chunking_strategy': self.config.strategy
        }

