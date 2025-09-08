#!/usr/bin/env python
"""
PostgreSQL Full-Text Search Retriever
Following DEVELOPMENT_RULES.md for all development requirements

This module provides PostgreSQL-based full-text search retrieval for email chunks.
"""

import logging
from typing import List, Any, Dict, Optional
from langchain_core.documents import Document

from rag_manager.managers.postgres_manager import PostgreSQLManager

logger = logging.getLogger(__name__)


class PostgresFTSRetriever:
    """
    PostgreSQL Full-Text Search retriever for email chunks.
    
    Uses PostgreSQL's built-in FTS capabilities with GIN indexes
    for fast text search across email content.
    """
    
    def __init__(self, postgres_manager: PostgreSQLManager) -> None:
        """
        Initialize PostgreSQL FTS retriever.
        
        Args:
            postgres_manager: PostgreSQL manager instance
        """
        self.db_manager = postgres_manager
        logger.info("PostgreSQL FTS retriever initialized")
    
    def search(self, query: str, k: int = 10) -> List[Document]:
        """
        Search email chunks using PostgreSQL FTS.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of Document objects with FTS scores
        """
        if not query.strip():
            logger.warning("Empty query provided to FTS search")
            return []
        
        try:
            logger.info(f"Performing PostgreSQL FTS search for: '{query[:50]}...' (k={k})")
            
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    # Use PostgreSQL FTS with ranking
                    # ts_rank provides relevance scoring
                    cur.execute("""
                        SELECT 
                            ec.id AS email_chunk_id,
                            ec.email_id,
                            ec.chunk_text,
                            ec.chunk_index,
                            ec.token_count,
                            ts_rank(to_tsvector('english', ec.chunk_text), plainto_tsquery('english', %s)) as fts_score,
                            e.subject,
                            e.from_addr,
                            e.to_addrs,
                            e.date_utc,
                            e.message_id
                        FROM email_chunks ec
                        JOIN emails e ON ec.email_id = e.id
                        WHERE to_tsvector('english', ec.chunk_text) @@ plainto_tsquery('english', %s)
                        ORDER BY fts_score DESC
                        LIMIT %s
                    """, (query, query, k))
                    
                    results = cur.fetchall()
                    logger.info(f"PostgreSQL FTS returned {len(results)} results")
                    
                    documents = []
                    for row in results:
                        # Support both tuple rows and RealDictCursor dict rows.
                        if isinstance(row, dict):
                            email_chunk_id = row.get('email_chunk_id')
                            email_id = row.get('email_id')
                            chunk_text = row.get('chunk_text')
                            chunk_index = row.get('chunk_index')
                            token_count = row.get('token_count')
                            fts_score = row.get('fts_score')
                            subject = row.get('subject')
                            from_addr = row.get('from_addr')
                            to_addrs = row.get('to_addrs')
                            date_utc = row.get('date_utc')
                            message_id = row.get('message_id')
                        else:
                            email_chunk_id, email_id, chunk_text, chunk_index, token_count, fts_score, \
                            subject, from_addr, to_addrs, date_utc, message_id = row

                        # Ensure fts_score is present — fail fast if DB returned NULL unexpectedly
                        if fts_score is None:
                            # Log the problematic row for diagnostics before failing
                            logger.error(
                                "FTS returned NULL score for email_chunk_id=%s; row=%r",
                                email_chunk_id,
                                row,
                            )
                            raise RuntimeError(f"FTS returned NULL score for email_chunk_id={email_chunk_id}")
                        fts_score_value = float(fts_score)

                        # Create LangChain Document with metadata using the table-specific id key
                        doc = Document(
                            page_content=chunk_text,
                                metadata={
                                'email_chunk_id': email_chunk_id,
                                'email_id': email_id,
                                'message_id': message_id,
                                'chunk_index': chunk_index,
                                'token_count': token_count,
                                'fts_score': fts_score_value,
                                'subject': subject,
                                'from_addr': from_addr,
                                'to_addrs': to_addrs,  # JSONB field with recipient addresses
                                'date_utc': date_utc,
                                'category': 'email',
                                'category_type': 'email',
                                'retrieval_method': 'postgresql_fts'
                            }
                        )
                        documents.append(doc)
                    
                    return documents
                    
        except Exception as e:
            logger.error(f"PostgreSQL FTS search failed: {e}")
            return []
    
    def search_chunks_for_email(self, email_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a specific email.
        
        Args:
            email_id: Email message ID
            
        Returns:
            List of chunk dictionaries
        """
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id AS email_chunk_id, email_id, chunk_text, chunk_index, 
                               token_count, chunk_hash, created_at, updated_at
                        FROM email_chunks
                        WHERE email_id = %s
                        ORDER BY chunk_index
                    """, (email_id,))
                    
                    results = cur.fetchall()
                    
                    chunks = []
                    for row in results:
                        email_chunk_id, email_id, chunk_text, chunk_index, \
                        token_count, chunk_hash, created_at, updated_at = row
                        
                        chunks.append({
                            'email_chunk_id': email_chunk_id,
                            'email_id': email_id,
                            'chunk_text': chunk_text,
                            'chunk_index': chunk_index,
                            'token_count': token_count,
                            'chunk_hash': chunk_hash,
                            'created_at': created_at,
                            'updated_at': updated_at
                        })
                    
                    return chunks
                    
        except Exception as e:
            logger.error(f"Failed to retrieve chunks for email {email_id}: {e}")
            return []
    
    def get_chunk_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored email chunks.
        
        Returns:
            Dictionary with chunk statistics
        """
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            COUNT(*) as total_chunks,
                            COUNT(DISTINCT email_id) as unique_emails,
                            AVG(token_count) as avg_token_count,
                            MIN(token_count) as min_token_count,
                            MAX(token_count) as max_token_count
                        FROM email_chunks
                    """)
                    
                    row = cur.fetchone()
                    if row:
                        total_chunks, unique_emails, avg_tokens, min_tokens, max_tokens = row
                        return {
                            'total_chunks': total_chunks,
                            'unique_emails': unique_emails,
                            'average_token_count': float(avg_tokens) if avg_tokens else 0.0,
                            'min_token_count': min_tokens,
                            'max_token_count': max_tokens
                        }
                    else:
                        return {
                            'total_chunks': 0,
                            'unique_emails': 0,
                            'average_token_count': 0.0,
                            'min_token_count': 0,
                            'max_token_count': 0
                        }
                        
        except Exception as e:
            logger.error(f"Failed to get chunk statistics: {e}")
            return {
                'total_chunks': 0,
                'unique_emails': 0,
                'average_token_count': 0.0,
                'min_token_count': 0,
                'max_token_count': 0,
                'error': str(e)
            }
