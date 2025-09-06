"""
Document processing module for RAG Knowledgebase Manager.

This module handles document processing operations including text extraction,
advanced chunking strategies, and embedding generation using UnstructuredLoader.
"""

import json
import hashlib
import logging
from typing import Dict, List, Optional, Any

from langchain_unstructured import UnstructuredLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document

from rag_manager.core.config import Config
from .advanced_chunking import AdvancedChunker, ChunkingConfig

# Configure logging
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Handles document processing operations including text extraction,
    advanced chunking strategies, and embedding generation using UnstructuredLoader.
    
    This class follows the development rules with proper type hints,
    error handling, and comprehensive logging.
    """
    
    def __init__(self, config: Config) -> None:
        """
        Initialize document processor.
        
        Args:
            config: Application configuration instance
        """
        self.config = config
        self.embedding_provider = OllamaEmbeddings(
            model=self.config.EMBEDDING_MODEL, 
            base_url=f"http://{self.config.OLLAMA_EMBEDDING_HOST}:{self.config.OLLAMA_EMBEDDING_PORT}"
        )
        
        # Initialize advanced chunker
        chunking_strategy = getattr(config, 'DOCUMENT_CHUNKING_STRATEGY', 'title_aware')
        target_tokens = getattr(config, 'DOCUMENT_TARGET_TOKENS', 850)
        overlap_percentage = getattr(config, 'DOCUMENT_OVERLAP_PERCENTAGE', 0.125)
        
        chunking_config = ChunkingConfig(
            strategy=chunking_strategy,
            target_tokens=target_tokens,
            overlap_percentage=overlap_percentage
        )
        self.advanced_chunker = AdvancedChunker(chunking_config)
        
        logger.info(f"DocumentProcessor initialized with {self.config.EMBEDDING_MODEL} and {chunking_strategy} chunking")
    
    def load_and_chunk(self, file_path: str, filename: str, document_id: str) -> List[Document]:
        """
        Load document and chunk using UnstructuredLoader with lean metadata.
        
        Args:
            file_path: Path to the file to process
            filename: Name of the file
            document_id: Unique document identifier
            
        Returns:
            List of Document chunks with metadata
            
        Raises:
            Exception: If document loading or chunking fails
        """
        logger.info(f"Loading and chunking document: {filename}")
        
        try:
            # Use UnstructuredLoader for document extraction
            loader = UnstructuredLoader(
                file_path,
                chunking_strategy="basic",  # Let advanced chunker handle the strategy
                max_characters=self.config.UNSTRUCTURED_MAX_CHARACTERS,
                overlap=self.config.UNSTRUCTURED_OVERLAP,
                include_orig_elements=self.config.UNSTRUCTURED_INCLUDE_ORIG,
            )
            raw_documents = loader.load()
            logger.info(f"Loaded {len(raw_documents)} elements via UnstructuredLoader")
            
            # Validate extraction success
            if raw_documents:
                total_chars = sum(len(doc.page_content or '') for doc in raw_documents)
                avg_chars = total_chars // len(raw_documents) if raw_documents else 0
                logger.info(f"Successfully extracted {len(raw_documents)} elements ({total_chars} chars, avg {avg_chars} chars/element) from {filename}")
                
                if total_chars < 100:  # Flag potentially poor extraction
                    logger.warning(f"Low text content extracted from {filename} ({total_chars} chars) - may need manual review or different extraction method")
                elif total_chars < 500:  # Flag moderately low extraction
                    logger.info(f"Moderate text content extracted from {filename} ({total_chars} chars) - extraction may be suboptimal")
            else:
                logger.error(f"No content extracted from {filename} - document may be corrupted, empty, or unsupported format")
                return []
            
            # Apply advanced chunking strategies with preserved metadata
            preserve_metadata = {
                'filename': filename,
                'document_id': document_id,
                'file_path': file_path,
                'content_type': 'application/pdf' if filename.lower().endswith('.pdf') else 'text/plain',
                'filetype': filename.split('.')[-1].lower() if '.' in filename else 'unknown'
            }
            chunks = self.advanced_chunker.chunk_documents(raw_documents, document_id, preserve_metadata)
            
            # Update metadata with additional document-level info
            for chunk in chunks:
                chunk.metadata.update({
                    'content_length': len(chunk.page_content),
                })
            
            # Log chunking statistics
            if chunks:
                stats = self.advanced_chunker.get_chunking_stats(chunks)
                logger.info(f"Advanced chunking created {stats['total_chunks']} chunks "
                           f"(avg {stats['avg_tokens_per_chunk']:.1f} tokens/chunk, "
                           f"strategy: {stats['chunking_strategy']})")
            
            logger.info(f"Created {len(chunks)} chunks with enhanced metadata")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to load and chunk document {filename}: {str(e)}")
            raise

    def extract_keywords(self, chunks: List[Document], max_keywords: int = 30) -> Dict[str, Any]:
        """
        LLM-based keyword extraction for document analysis.
        
        Args:
            chunks: List of document chunks to analyze
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            Dictionary with keys: global_keywords (list), llm_title (str|None)
            
        Raises:
            ValueError: If chunks list is empty
            Exception: If keyword extraction fails
        """
        if not chunks:
            raise ValueError("Cannot extract keywords from empty chunks list")
            
        logger.info(f"Starting LLM keyword extraction for {len(chunks)} chunks (max_keywords={max_keywords})")

        # Build page texts with proper error handling
        pages: Dict[int, List[str]] = {}
        for chunk in chunks:
            try:
                page_num = int(chunk.metadata.get('page', 1))
                content = chunk.page_content or ''
                if content.strip():
                    pages.setdefault(page_num, []).append(content)
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid page number in chunk metadata: {e}")
                pages.setdefault(1, []).append(chunk.page_content or '')
        
        if not pages:
            logger.warning("No valid page content found in chunks")
            return {
                'global_keywords': [],
                'llm_title': None
            }

        page_texts = {p: '\n'.join(texts) for p, texts in pages.items() if texts}

        # LLM extraction from first page for global keywords and title
        llm_keywords: List[str] = []
        llm_title: Optional[str] = None
        
        try:
            if page_texts:
                first_page_num = min(page_texts.keys())
                # Use more content for better keyword extraction
                first_page_text = page_texts[first_page_num][:3000]
                
                if first_page_text.strip():
                    llm = ChatOllama(
                        model=self.config.CLASSIFICATION_MODEL, 
                        base_url=self.config.CLASSIFICATION_BASE_URL, 
                        temperature=0
                    )
                    
                    prompt = (
                        "You extract document metadata. Analyze the text and return JSON with keys: title, keywords.\n"
                        f"title: Clear, descriptive document title (max 100 characters)\n"
                        f"keywords: {max_keywords} most important topical terms (1-3 words each), "
                        "lowercase unless proper noun, no duplicates, ordered by relevance.\n"
                        f"Text:\n{first_page_text}"
                    )
                    
                    response = llm.invoke(prompt)
                    response_text = str(response.content).strip()
                    
                    # Parse JSON response with better error handling
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    
                    if json_start != -1 and json_end > json_start:
                        try:
                            parsed_data = json.loads(response_text[json_start:json_end])
                            
                            # Extract and validate keywords
                            if 'keywords' in parsed_data:
                                raw_keywords = parsed_data['keywords']
                                if isinstance(raw_keywords, list):
                                    llm_keywords = [
                                        str(kw).strip().lower() for kw in raw_keywords 
                                        if isinstance(kw, str) and kw.strip()
                                    ][:max_keywords]  # Enforce max limit
                            
                            # Extract and validate title
                            if 'title' in parsed_data and isinstance(parsed_data['title'], str):
                                title_candidate = parsed_data['title'].strip()
                                if title_candidate:
                                    llm_title = title_candidate[:100]  # Enforce max length
                                    
                            logger.info(f"LLM extracted {len(llm_keywords)} keywords and title: {llm_title is not None}")
                            
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse LLM JSON response: {e}")
                            logger.debug(f"Raw LLM response: {response_text}")
                            
        except Exception as e:
            logger.error(f"LLM keyword extraction failed: {e}")
            raise

        logger.info(f"Keyword extraction complete: {len(llm_keywords)} keywords extracted")
        
        return {
            'global_keywords': llm_keywords,
            'llm_title': llm_title
        }

    def load_and_chunk_url(self, url: str, url_id: str) -> List[Document]:
        """
        Load a URL using UnstructuredLoader and create lean chunks with metadata.
        
        Args:
            url: URL to process
            url_id: Unique URL identifier
            
        Returns:
            List of Document chunks with metadata
            
        Raises:
            Exception: If URL loading or chunking fails
        """
        logger.info(f"Loading and chunking URL: {url}")
        try:
            loader = UnstructuredLoader(
                web_url=url,
                chunking_strategy=self.config.UNSTRUCTURED_CHUNKING_STRATEGY,
                max_characters=self.config.UNSTRUCTURED_MAX_CHARACTERS,
                overlap=self.config.UNSTRUCTURED_OVERLAP,
                include_orig_elements=self.config.UNSTRUCTURED_INCLUDE_ORIG,
            )
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} elements from URL via UnstructuredLoader(web_url)")

            # Validate URL extraction success
            if documents:
                total_chars = sum(len(doc.page_content or '') for doc in documents)
                avg_chars = total_chars // len(documents) if documents else 0
                logger.info(f"Successfully extracted {len(documents)} chunks ({total_chars} chars, avg {avg_chars} chars/chunk) from URL: {url}")

                if total_chars < 200:  # URLs might have less content than documents
                    logger.warning(
                        f"Low text content extracted from URL {url} ({total_chars} chars) - page may have dynamic content, be protected, or have extraction issues"
                    )
                elif total_chars < 1000:  # Flag moderately low extraction for URLs
                    logger.info(
                        f"Moderate text content extracted from URL {url} ({total_chars} chars) - extraction may be suboptimal"
                    )
            else:
                logger.error(
                    f"No content extracted from URL {url} - page may be inaccessible, protected, or have no readable content"
                )

            chunks: List[Document] = []
            for i, d in enumerate(documents):
                text = (d.page_content or '').strip()
                if not text:
                    continue
                content_hash = hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]
                meta = d.metadata or {}
                page = meta.get('page') or meta.get('page_number') or (i + 1)
                meta.update({
                    'page': page,
                    'document_id': url_id,
                    'chunk_id': f"{url_id}:{content_hash}",
                    'content_hash': content_hash,
                    'content_length': len(text),
                    'category_type': 'url',  # Set category_type for URL content
                })
                d.metadata = meta
                d.page_content = text
                chunks.append(d)

            logger.info(f"Created {len(chunks)} URL chunks with metadata")

            # Optional: write to per-run ingestion trace log if available
            try:
                trace_logger = logging.getLogger(f"ingest_trace_{url_id}")
                if trace_logger and trace_logger.handlers:
                    trace_logger.info(f"DocumentProcessor: created_chunks={len(chunks)} for url_id={url_id}")
                    if len(chunks) > 0:
                        # Log a small sample of the first chunk metadata for debugging
                        sample = chunks[0]
                        sample_meta = {k: sample.metadata.get(k) for k in ['page', 'content_hash', 'content_length', 'chunk_id']}
                        trace_logger.info(f"Sample chunk metadata: {sample_meta}")
            except Exception:
                # Tracing is best-effort; don't fail the processing if tracing fails
                pass

            return chunks
        except Exception as e:
            logger.error(f"Failed to load and chunk URL {url}: {e}")
            raise

    def enrich_topics(self, chunks: List[Document]) -> List[Document]:
        """
        Topic classification for document chunks.
        
        This method uses advanced logic to assign coherent multi-topics:
        1. Content-type aware paragraph boundary detection
        2. Groups chunks into semantic paragraphs
        3. Assigns 3-8 searchable topics per paragraph for maximum discoverability
        4. Handles edge cases (short chunks, cross-paragraph chunks)
        5. Uses smart topic propagation for single sentences
        
        Args:
            chunks: List of document chunks to enrich with topic metadata
            
        Returns:
            List of chunks with multi-topic metadata populated
        """
        logger.info(f"Starting smart multi-topic enrichment for {len(chunks)} chunks...")
        
        if not chunks:
            return chunks
            
        llm = ChatOllama(
            model=self.config.CLASSIFICATION_MODEL, 
            base_url=self.config.CLASSIFICATION_BASE_URL, 
            temperature=self.config.CHAT_TEMPERATURE
        )
        
        try:
            # Step 1: Content-type aware paragraph segmentation
            document_metadata = chunks[0].metadata if chunks else {}
            paragraph_segments = self._create_paragraph_segments(chunks, document_metadata)
            
            if not paragraph_segments:
                logger.warning("No paragraph segments created, falling back to simple classification")
                self._fallback_topic_assignment(chunks, llm)
            else:
                logger.info(f"Created {len(paragraph_segments)} paragraph segments for multi-topic classification")
                
                # Step 2: Multi-topic classification for each paragraph
                paragraph_topics = self._classify_paragraph_topics(paragraph_segments, llm)
                
                # Step 3: Assign multi-topics to chunks
                self._assign_topics_to_chunks(chunks, paragraph_segments, paragraph_topics)
                
                # Step 4: Handle orphaned chunks and edge cases
                self._handle_topic_edge_cases(chunks, llm)
            
            # Generate topic statistics for debugging
            stats = self.get_topic_statistics(chunks)
            logger.info(f"Multi-topic enrichment complete - Coverage: {stats['coverage_percentage']}%, "
                       f"Unique topics: {stats['unique_topics']}, "
                       f"Avg topics per chunk: {stats['avg_topics_per_chunk']}")
            
            # Log sample topics for verification
            topic_samples = [chunk.metadata.get('topics', 'None') for chunk in chunks[:3] if chunk.metadata.get('topics')]
            if topic_samples:
                logger.debug(f"Sample multi-topics: {[t[:60] + '...' if len(t) > 60 else t for t in topic_samples]}")
            
        except Exception as e:
            logger.error(f"Multi-topic enrichment failed: {e}")
            # Fallback to simple topic assignment
            self._fallback_topic_assignment(chunks, llm)
        
        return chunks

    def _create_paragraph_segments(self, chunks: List[Document], document_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create content-type aware paragraph segments for topic classification.
        
        Args:
            chunks: Document chunks
            document_metadata: Metadata from the document
            
        Returns:
            List of paragraph segments with metadata
        """
        logger.debug("Creating content-type aware paragraph segments...")
        
        # Determine content type from metadata
        content_type = document_metadata.get('content_type', 'text/plain')
        category_type = document_metadata.get('category_type', 'document')
        
        logger.debug(f"Processing content_type: {content_type}, category_type: {category_type}")
        
        # Route to appropriate segmentation method based on content type
        if 'html' in content_type.lower() or category_type == 'url':
            return self._segment_html_content(chunks)
        elif content_type == 'application/pdf':
            return self._segment_pdf_content(chunks)
        elif content_type in ['text/markdown', 'text/x-markdown']:
            return self._segment_markdown_content(chunks)
        else:
            return self._segment_text_content(chunks)

    def _segment_html_content(self, chunks: List[Document]) -> List[Dict[str, Any]]:
        """
        Segment HTML/URL content using semantic structure.
        
        Args:
            chunks: Document chunks from HTML content
            
        Returns:
            List of paragraph segments
        """
        logger.debug("Segmenting HTML content using semantic structure...")
        
        segments = []
        current_segment = []
        current_content = ""
        
        for i, chunk in enumerate(chunks):
            content = chunk.page_content or ""
            
            # HTML-specific break detection
            should_break = (
                # Content length suggests new section
                len(current_content) > 600 or
                # Look for structural breaks in content
                (current_content and self._detect_html_structure_break(content)) or
                # Header-like content (short, title-case, no punctuation)
                (len(content.strip()) < 100 and 
                 content.strip() and 
                 content.strip()[0].isupper() and 
                 current_content and
                 not content.strip().endswith('.'))
            )
            
            if should_break and current_segment:
                segments.append({
                    'chunk_indices': current_segment.copy(),
                    'combined_content': current_content.strip(),
                    'content_type': 'html'
                })
                current_segment = []
                current_content = ""
            
            current_segment.append(i)
            current_content += content + " "
        
        # Add final segment
        if current_segment:
            segments.append({
                'chunk_indices': current_segment,
                'combined_content': current_content.strip(),
                'content_type': 'html'
            })
        
        logger.debug(f"Created {len(segments)} HTML segments")
        return segments

    def _detect_html_structure_break(self, content: str) -> bool:
        """Detect structural breaks in HTML content."""
        # Look for indicators that suggest new sections
        indicators = [
            # List items
            content.strip().startswith(('• ', '- ', '* ', '1. ', '2. ', '3.')),
            # Navigation or menu items
            'nav' in content.lower() or 'menu' in content.lower(),
            # Headers or titles (short content, title case)
            (len(content.strip()) < 80 and 
             any(word[0].isupper() for word in content.split() if word)),
            # Contact or footer information
            any(term in content.lower() for term in ['contact', 'phone', 'email', 'address', 'copyright'])
        ]
        return any(indicators)

    def _segment_pdf_content(self, chunks: List[Document]) -> List[Dict[str, Any]]:
        """
        Segment PDF content handling extraction artifacts.
        
        Args:
            chunks: Document chunks from PDF
            
        Returns:
            List of paragraph segments
        """
        logger.debug("Segmenting PDF content with artifact handling...")
        
        segments = []
        current_segment = []
        current_content = ""
        
        for i, chunk in enumerate(chunks):
            content = chunk.page_content or ""
            
            # Clean PDF artifacts
            cleaned_content = self._clean_pdf_text(content)
            
            # PDF-specific break detection
            should_break = (
                # Page break indicators
                (i > 0 and 
                 chunk.metadata.get('page', 0) != chunks[i-1].metadata.get('page', 0)) or
                # Content length
                len(current_content) > 500 or
                # Section headers (short lines, title case, no period)
                (len(cleaned_content.strip()) < 80 and 
                 cleaned_content.strip() and 
                 cleaned_content.strip()[0].isupper() and 
                 current_content and
                 not cleaned_content.strip().endswith('.'))
            )
            
            if should_break and current_segment:
                segments.append({
                    'chunk_indices': current_segment.copy(),
                    'combined_content': current_content.strip(),
                    'content_type': 'pdf'
                })
                current_segment = []
                current_content = ""
            
            current_segment.append(i)
            current_content += cleaned_content + " "
        
        # Add final segment
        if current_segment:
            segments.append({
                'chunk_indices': current_segment,
                'combined_content': current_content.strip(),
                'content_type': 'pdf'
            })
        
        logger.debug(f"Created {len(segments)} PDF segments")
        return segments

    def _clean_pdf_text(self, text: str) -> str:
        """Clean common PDF extraction artifacts."""
        import re
        
        # Fix broken words across lines
        text = re.sub(r'-\n\s*', '', text)
        # Fix broken sentences
        text = re.sub(r'\n(?=[a-z])', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove excessive spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        
        return text.strip()

    def _segment_markdown_content(self, chunks: List[Document]) -> List[Dict[str, Any]]:
        """
        Segment Markdown content using headers and structure.
        
        Args:
            chunks: Document chunks from Markdown
            
        Returns:
            List of paragraph segments
        """
        logger.debug("Segmenting Markdown content using headers and structure...")
        
        segments = []
        current_segment = []
        current_content = ""
        
        for i, chunk in enumerate(chunks):
            content = chunk.page_content or ""
            
            # Markdown-specific break detection
            should_break = (
                # Header indicators
                content.strip().startswith('#') or
                # Content length
                len(current_content) > 500 or
                # List boundaries
                (current_content and content.strip().startswith(('- ', '* ', '+ ', '1. ', '2. '))) or
                # Code block boundaries
                '```' in content or
                # Table boundaries
                '|' in content and current_content
            )
            
            if should_break and current_segment:
                segments.append({
                    'chunk_indices': current_segment.copy(),
                    'combined_content': current_content.strip(),
                    'content_type': 'markdown'
                })
                current_segment = []
                current_content = ""
            
            current_segment.append(i)
            current_content += content + " "
        
        # Add final segment
        if current_segment:
            segments.append({
                'chunk_indices': current_segment,
                'combined_content': current_content.strip(),
                'content_type': 'markdown'
            })
        
        logger.debug(f"Created {len(segments)} Markdown segments")
        return segments

    def _segment_text_content(self, chunks: List[Document]) -> List[Dict[str, Any]]:
        """
        Segment plain text content using semantic heuristics.
        
        Args:
            chunks: Document chunks from text
            
        Returns:
            List of paragraph segments
        """
        logger.debug("Segmenting text content using semantic heuristics...")
        
        segments = []
        current_segment = []
        current_content = ""
        
        for i, chunk in enumerate(chunks):
            content = chunk.page_content or ""
            
            # Text-specific break detection using semantic cues
            should_break = (
                # Content length
                len(current_content) > 500 or
                # Paragraph break indicators (ends with period, next starts with capital)
                (current_content and content and 
                 current_content.rstrip().endswith('.') and
                 content.lstrip() and content.lstrip()[0].isupper()) or
                # Very short chunk that might be a header
                (len(current_segment) > 0 and len(content.strip()) < 50) or
                # Topic shift indicators (certain transition words)
                (current_content and any(content.strip().lower().startswith(phrase) for phrase in 
                 ['however,', 'furthermore,', 'in conclusion,', 'on the other hand,', 'meanwhile,']))
            )
            
            if should_break and current_segment:
                segments.append({
                    'chunk_indices': current_segment.copy(),
                    'combined_content': current_content.strip(),
                    'content_type': 'text'
                })
                current_segment = []
                current_content = ""
            
            current_segment.append(i)
            current_content += content + " "
        
        # Add final segment
        if current_segment:
            segments.append({
                'chunk_indices': current_segment,
                'combined_content': current_content.strip(),
                'content_type': 'text'
            })
        
        logger.debug(f"Created {len(segments)} text segments")
        return segments

    def _classify_paragraph_topics(self, paragraph_segments: List[Dict[str, Any]], llm: ChatOllama) -> Dict[int, str]:
        """
        Classify multi-topics for each paragraph segment.
        
        Args:
            paragraph_segments: List of paragraph segments
            llm: Language model for classification
            
        Returns:
            Dictionary mapping segment index to comma-separated topics
        """
        logger.debug(f"Classifying multi-topics for {len(paragraph_segments)} paragraph segments...")
        
        paragraph_topics = {}
        
        for i, segment in enumerate(paragraph_segments):
            content = segment['combined_content']
            content_type = segment.get('content_type', 'text')
            
            # Skip very short segments
            if len(content.strip()) < 80:
                paragraph_topics[i] = None  # Will be handled in edge cases
                continue
            
            # Multi-topic classification prompt
            topic_prompt = f"""Extract 3-8 topics that users might search for to find this content.

Think about different ways people would search for this information:
- Specific technical concepts and terms
- Broader application domains and fields
- Industry or academic areas
- Alternative terminology and synonyms
- Practical applications and use cases

Return ONLY a comma-separated list of topics (no JSON, quotes, or extra formatting):
Example: Machine Learning, Fraud Detection, Financial Security, Neural Networks, Banking Technology

Content type: {content_type}
Content: {content[:1200]}

Topics:"""
            
            try:
                response = llm.invoke(topic_prompt)
                topics_text = str(response.content).strip()
                
                # Clean the response
                topics_text = topics_text.replace('"', '').replace("'", '').strip()
                
                # Handle cases where LLM returns formatted text
                if topics_text.startswith('Topics:'):
                    topics_text = topics_text.replace('Topics:', '').strip()
                
                # Split and clean individual topics
                topic_list = [t.strip() for t in topics_text.split(',') if t.strip()]
                
                # Limit to prevent topic explosion and filter quality
                quality_topics = []
                for topic in topic_list[:8]:  # Max 8 topics
                    # Basic quality filters
                    if (len(topic) > 2 and len(topic) < 50 and 
                        not topic.lower() in ['the', 'and', 'or', 'but', 'data', 'analysis', 'content']):
                        quality_topics.append(topic)
                
                if quality_topics:
                    final_topics = ', '.join(quality_topics)
                    paragraph_topics[i] = final_topics
                    logger.debug(f"Segment {i}: {len(quality_topics)} topics: {final_topics[:60]}...")
                else:
                    paragraph_topics[i] = None
                    
            except Exception as e:
                logger.warning(f"Multi-topic classification failed for segment {i}: {e}")
                paragraph_topics[i] = None
        
        return paragraph_topics

    def _assign_topics_to_chunks(self, chunks: List[Document], paragraph_segments: List[Dict[str, Any]], paragraph_topics: Dict[int, str]) -> None:
        """
        Assign paragraph multi-topics to individual chunks.
        
        Args:
            chunks: Document chunks to update
            paragraph_segments: Paragraph segment definitions
            paragraph_topics: Multi-topics for each segment
        """
        logger.debug("Assigning paragraph multi-topics to individual chunks...")
        
        # Initialize all chunks without topics
        for chunk in chunks:
            if 'topics' not in chunk.metadata:
                chunk.metadata['topics'] = None
        
        # Assign topics from paragraph segments
        for segment_idx, segment in enumerate(paragraph_segments):
            topics = paragraph_topics.get(segment_idx)
            if topics:
                for chunk_idx in segment['chunk_indices']:
                    if chunk_idx < len(chunks):
                        chunks[chunk_idx].metadata['topics'] = topics
                        chunks[chunk_idx].metadata['paragraph_segment'] = segment_idx
                        chunks[chunk_idx].metadata['topic_source'] = 'paragraph_segment'

    def _handle_topic_edge_cases(self, chunks: List[Document], llm: ChatOllama) -> None:
        """
        Handle chunks without topics and apply smart propagation rules for multi-topics.
        
        Args:
            chunks: Document chunks
            llm: Language model for fallback classification
        """
        logger.debug("Handling topic edge cases and orphaned chunks...")
        
        for i, chunk in enumerate(chunks):
            if not chunk.metadata.get('topics'):
                content = chunk.page_content or ""
                
                # Case 1: Very short chunk (single sentence) - use nearby topics
                if len(content.strip()) < 80:
                    # Look backward for previous topics
                    for j in range(i-1, max(-1, i-3), -1):
                        if j >= 0 and chunks[j].metadata.get('topics'):
                            chunk.metadata['topics'] = chunks[j].metadata['topics']
                            chunk.metadata['topic_source'] = 'propagated_backward'
                            logger.debug(f"Chunk {i}: Propagated topics from chunk {j}")
                            break
                    
                    # If no previous topics, look forward
                    if not chunk.metadata.get('topics'):
                        for j in range(i+1, min(len(chunks), i+3)):
                            if chunks[j].metadata.get('topics'):
                                chunk.metadata['topics'] = chunks[j].metadata['topics']
                                chunk.metadata['topic_source'] = 'propagated_forward'
                                logger.debug(f"Chunk {i}: Propagated topics from chunk {j}")
                                break
                
                # Case 2: Medium chunk without topics - classify individually
                elif len(content.strip()) >= 80:
                    try:
                        simple_prompt = f"""What are the main topics for this text? List 2-5 topics that someone might search for.
                        
                        Return as comma-separated list:
                        {content[:600]}
                        
                        Topics:"""
                        
                        response = llm.invoke(simple_prompt)
                        topics_text = str(response.content).strip().replace('"', '')
                        
                        # Clean and validate
                        if topics_text and ',' in topics_text:
                            topic_list = [t.strip() for t in topics_text.split(',') if t.strip()]
                            if topic_list:
                                chunk.metadata['topics'] = ', '.join(topic_list[:6])
                                chunk.metadata['topic_source'] = 'individual_classification'
                                logger.debug(f"Chunk {i}: Individual topics: {chunk.metadata['topics'][:50]}...")
                        elif topics_text and len(topics_text) < 50:
                            chunk.metadata['topics'] = topics_text
                            chunk.metadata['topic_source'] = 'individual_classification'
                            
                    except Exception as e:
                        logger.warning(f"Individual topic classification failed for chunk {i}: {e}")

    def _fallback_topic_assignment(self, chunks: List[Document], llm: ChatOllama) -> None:
        """
        Simple fallback multi-topic assignment if advanced method fails.
        
        Args:
            chunks: Document chunks
            llm: Language model
        """
        logger.warning("Using fallback multi-topic assignment...")
        
        # Group chunks into windows and assign topics
        window_size = 3
        for i in range(0, len(chunks), window_size):
            window_chunks = chunks[i:i + window_size]
            combined_content = ' '.join([c.page_content for c in window_chunks if c.page_content])
            
            if len(combined_content.strip()) < 50:
                continue
                
            try:
                prompt = f"List 3-5 topics for this text (comma-separated): {combined_content[:800]}"
                response = llm.invoke(prompt)
                topics_text = str(response.content).strip().replace('"', '')
                
                # Basic cleanup and validation
                if topics_text and (',' in topics_text or len(topics_text.split()) <= 6):
                    for chunk in window_chunks:
                        chunk.metadata['topics'] = topics_text
                        chunk.metadata['topic_source'] = 'fallback_window'
                        
            except Exception as e:
                logger.error(f"Fallback topic assignment failed: {e}")

    def get_topic_statistics(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Generate statistics about multi-topic assignment for debugging and analysis.
        
        Args:
            chunks: List of processed chunks with topic metadata
            
        Returns:
            Dictionary with topic statistics
        """
        if not chunks:
            return {'total_chunks': 0, 'topics': {}}
        
        topic_counts = {}
        topic_sources = {}
        chunks_without_topics = 0
        all_individual_topics = []
        
        for i, chunk in enumerate(chunks):
            topics_str = chunk.metadata.get('topics')
            source = chunk.metadata.get('topic_source', 'paragraph_segment')
            
            if topics_str:
                # Split multi-topics and count each
                individual_topics = [t.strip() for t in topics_str.split(',') if t.strip()]
                all_individual_topics.extend(individual_topics)
                
                for topic in individual_topics:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                    if topic not in topic_sources:
                        topic_sources[topic] = []
                    topic_sources[topic].append(source)
            else:
                chunks_without_topics += 1
        
        # Calculate coverage and coherence metrics
        total_chunks = len(chunks)
        coverage = (total_chunks - chunks_without_topics) / total_chunks if total_chunks > 0 else 0
        unique_topics = len(topic_counts)
        avg_topics_per_chunk = len(all_individual_topics) / (total_chunks - chunks_without_topics) if (total_chunks - chunks_without_topics) > 0 else 0
        
        return {
            'total_chunks': total_chunks,
            'unique_topics': unique_topics,
            'chunks_without_topics': chunks_without_topics,
            'coverage_percentage': round(coverage * 100, 2),
            'avg_topics_per_chunk': round(avg_topics_per_chunk, 2),
            'total_topic_assignments': len(all_individual_topics),
            'topic_distribution': topic_counts,
            'topic_sources': topic_sources,
            'most_common_topics': sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }


