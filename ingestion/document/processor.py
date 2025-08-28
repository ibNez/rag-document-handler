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
                    logger.warning(f"Low text content extracted from URL {url} ({total_chars} chars) - page may have dynamic content, be protected, or have extraction issues")
                elif total_chars < 1000:  # Flag moderately low extraction for URLs
                    logger.info(f"Moderate text content extracted from URL {url} ({total_chars} chars) - extraction may be suboptimal")
            else:
                logger.error(f"No content extracted from URL {url} - page may be inaccessible, protected, or have no readable content")

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
            return chunks
        except Exception as e:
            logger.error(f"Failed to load and chunk URL {url}: {e}")
            raise

    def enrich_topics(self, chunks: List[Document]) -> None:
        """
        LLM topic classification for document chunks.
        
        Args:
            chunks: List of document chunks to enrich with topic metadata
        """
        logger.info("Enriching chunk metadata with LLM classification (topic)...")
        
        llm = ChatOllama(
            model=self.config.CLASSIFICATION_MODEL, 
            base_url=self.config.CLASSIFICATION_BASE_URL, 
            temperature=self.config.CHAT_TEMPERATURE
        )
        
        prompt_tpl = (
            "You are classifying a text chunk for RAG metadata.\n"
            "Return ONLY compact JSON with keys: topic.\n"
            "topic: concise subject title (3-6 words).\n"
            "Text:\n{chunk}\n"
        )
        
        for c in chunks:
            snippet = (c.page_content or '')[:800]
            try:
                response = llm.invoke(prompt_tpl.format(chunk=snippet))
                resp = str(response.content).strip()
                start = resp.find('{')
                end = resp.rfind('}') + 1
                if start != -1 and end > start:
                    obj = json.loads(resp[start:end])
                    if isinstance(obj, dict) and obj.get('topic'):
                        c.metadata['topic'] = obj['topic']
            except Exception as e:
                logger.error(f"Failed to extract topic from LLM enrichment: {e}")
                # Don't set default topic - let it remain unset
        
        logger.info("LLM enrichment complete")
