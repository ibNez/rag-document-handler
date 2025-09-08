#!/usr/bin/env python
"""
Text Chunking Utility
Following DEVELOPMENT_RULES.md for all development requirements

This module provides text chunking functionality for various content types
including emails, documents, and other text content. Supports token-based
chunking with configurable overlap.
"""

import hashlib
import logging
import re
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
    logger.debug("tiktoken available for accurate token counting")
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, using word-based approximation for token counting")


class TextChunker:
    """
    Text chunking utility with token-based splitting and configurable overlap.
    
    This class follows the development rules with proper type hints,
    error handling, and comprehensive logging.
    """
    
    def __init__(
        self, 
        chunk_size_tokens: int = 800,
        chunk_overlap_tokens: int = 100,
        encoding_name: str = "cl100k_base"  # Standard tokenizer encoding (used by GPT-4, but runs locally)
    ) -> None:
        """
        Initialize text chunker.
        
        Args:
            chunk_size_tokens: Target size for each chunk in tokens
            chunk_overlap_tokens: Number of tokens to overlap between chunks
            encoding_name: Tiktoken encoding name for token counting
        """
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.encoding_name = encoding_name
        
        # Initialize tokenizer if available
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding(encoding_name)
                logger.debug(f"Initialized tiktoken with encoding: {encoding_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize tiktoken encoding {encoding_name}: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None
            
        logger.info(f"TextChunker initialized: {chunk_size_tokens} tokens/chunk, {chunk_overlap_tokens} overlap")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken or word approximation.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens (or estimated tokens)
        """
        if not text:
            return 0
            
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"tiktoken encoding failed, falling back to word approximation: {e}")
        
        # Fallback: approximate 1.3 tokens per word (empirical estimate)
        words = len(text.split())
        estimated_tokens = int(words * 1.3)
        return estimated_tokens

    def split_by_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using simple regex.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting on periods, exclamation marks, question marks
        # followed by whitespace or end of string
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text.strip())
        
        # Filter out empty sentences
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text: str, preserve_sentences: bool = True) -> List[Dict[str, Any]]:
        """
        Chunk text into token-sized pieces with overlap.
        
        Args:
            text: Text to chunk
            preserve_sentences: Whether to try to preserve sentence boundaries
            
        Returns:
            List of chunk dictionaries with text, token_count, and chunk_index
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
            
        logger.debug(f"Chunking text of length {len(text)} characters")
        
        # If text is small enough to fit in one chunk, return as-is
        total_tokens = self.count_tokens(text)
        if total_tokens <= self.chunk_size_tokens:
            logger.debug(f"Text fits in single chunk ({total_tokens} tokens)")
            return [{
                'chunk_text': text,
                'token_count': total_tokens,
                'chunk_index': 0,
                'chunk_hash': self._generate_chunk_hash(text, 0)
            }]
        
        chunks = []
        chunk_index = 0
        
        if preserve_sentences:
            sentences = self.split_by_sentences(text)
            logger.debug(f"Split text into {len(sentences)} sentences")
        else:
            # Split by paragraphs or use the whole text
            sentences = [text]
        
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If single sentence is larger than chunk size, split it
            if sentence_tokens > self.chunk_size_tokens:
                # Save current chunk if it has content
                if current_chunk.strip():
                    chunks.append({
                        'chunk_text': current_chunk.strip(),
                        'token_count': current_tokens,
                        'chunk_index': chunk_index,
                        'chunk_hash': self._generate_chunk_hash(current_chunk.strip(), chunk_index)
                    })
                    chunk_index += 1
                    current_chunk = ""
                    current_tokens = 0
                
                # Split large sentence by words
                word_chunks = self._split_large_sentence(sentence, chunk_index)
                chunks.extend(word_chunks)
                chunk_index += len(word_chunks)
                continue
            
            # Check if adding this sentence would exceed chunk size
            if current_tokens + sentence_tokens > self.chunk_size_tokens and current_chunk.strip():
                # Save current chunk
                chunks.append({
                    'chunk_text': current_chunk.strip(),
                    'token_count': current_tokens,
                    'chunk_index': chunk_index,
                    'chunk_hash': self._generate_chunk_hash(current_chunk.strip(), chunk_index)
                })
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap_tokens)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                current_tokens = self.count_tokens(current_chunk)
            else:
                # Add sentence to current chunk
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
        
        # Add final chunk if it has content
        if current_chunk.strip():
            chunks.append({
                'chunk_text': current_chunk.strip(),
                'token_count': current_tokens,
                'chunk_index': chunk_index,
                'chunk_hash': self._generate_chunk_hash(current_chunk.strip(), chunk_index)
            })
        
        logger.info(f"Created {len(chunks)} chunks from text ({total_tokens} total tokens)")
        return chunks

    def _split_large_sentence(self, sentence: str, start_index: int) -> List[Dict[str, Any]]:
        """
        Split a sentence that's too large for a single chunk.
        
        Args:
            sentence: Large sentence to split
            start_index: Starting chunk index
            
        Returns:
            List of chunk dictionaries
        """
        words = sentence.split()
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_index = start_index
        
        for word in words:
            word_tokens = self.count_tokens(word + " ")
            
            if current_tokens + word_tokens > self.chunk_size_tokens and current_chunk.strip():
                # Save current chunk
                chunks.append({
                    'chunk_text': current_chunk.strip(),
                    'token_count': current_tokens,
                    'chunk_index': chunk_index,
                    'chunk_hash': self._generate_chunk_hash(current_chunk.strip(), chunk_index)
                })
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap_tokens)
                current_chunk = overlap_text + " " + word if overlap_text else word
                current_tokens = self.count_tokens(current_chunk)
            else:
                current_chunk += " " + word if current_chunk else word
                current_tokens += word_tokens
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'chunk_text': current_chunk.strip(),
                'token_count': current_tokens,
                'chunk_index': chunk_index,
                'chunk_hash': self._generate_chunk_hash(current_chunk.strip(), chunk_index)
            })
        
        return chunks

    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """
        Get the last N tokens from text for overlap.
        
        Args:
            text: Source text
            overlap_tokens: Number of tokens to extract
            
        Returns:
            Overlap text
        """
        if not text or overlap_tokens <= 0:
            return ""
        
        words = text.split()
        if not words:
            return ""
        
        # Start from the end and work backwards
        overlap_text = ""
        current_tokens = 0
        
        for word in reversed(words):
            word_tokens = self.count_tokens(word + " ")
            if current_tokens + word_tokens > overlap_tokens:
                break
            overlap_text = word + " " + overlap_text
            current_tokens += word_tokens
        
        return overlap_text.strip()

    def _generate_chunk_hash(self, chunk_text: str, chunk_index: int) -> str:
        """
        Generate a hash for the chunk content.
        
        Args:
            chunk_text: Text content of the chunk
            chunk_index: Index of the chunk
            
        Returns:
            Hash string
        """
        content = f"{chunk_text}:{chunk_index}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

    def chunk_email(self, email_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk email content with proper metadata handling.
        
        Args:
            email_content: Dictionary containing email fields
            
        Returns:
            List of email chunk dictionaries
        """
        logger.debug(f"Chunking email: {email_content.get('subject', 'No Subject')}")
        
        # Extract text content from email
        text_parts = []
        
        # Add subject as important content
        subject = email_content.get('subject', '').strip()
        if subject:
            text_parts.append(f"Subject: {subject}")
        
        # Add body content - using the standardized 'content' field
        body = email_content.get('content', '').strip()
        if body:
            text_parts.append(body)
        
        # Combine all text parts
        full_text = '\n\n'.join(text_parts)
        
        if not full_text.strip():
            logger.warning("No text content found in email for chunking")
            return []
        
        # Chunk the combined text
        text_chunks = self.chunk_text(full_text, preserve_sentences=True)
        
        # Add email metadata to each chunk
        email_chunks = []
        for chunk in text_chunks:
            email_chunk = {
                'email_id': email_content.get('id'),  # Use database UUID id, not message_id
                'chunk_text': chunk['chunk_text'],
                'chunk_index': chunk['chunk_index'],
                'token_count': chunk['token_count'],
                'chunk_hash': chunk['chunk_hash'],
                'chunk_id': f"{email_content.get('message_id', 'unknown')}:chunk:{chunk['chunk_index']}",
                # Preserve email metadata
                'subject': email_content.get('subject', ''),
                'from_addr': email_content.get('from_addr', ''),
                'date_utc': email_content.get('date_utc'),
            }
            email_chunks.append(email_chunk)
        
        logger.info(f"Created {len(email_chunks)} chunks for email {email_content.get('message_id', 'unknown')}")
        return email_chunks


def create_email_chunker(chunk_size: int = 800, overlap: int = 100) -> TextChunker:
    """
    Factory function to create a TextChunker configured for email processing.
    
    Args:
        chunk_size: Token size for each chunk
        overlap: Token overlap between chunks
        
    Returns:
        Configured TextChunker instance
    """
    return TextChunker(
        chunk_size_tokens=chunk_size,
        chunk_overlap_tokens=overlap,
        encoding_name="cl100k_base"
    )
