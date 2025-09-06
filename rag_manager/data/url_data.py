#!/usr/bin/env python
"""
URL Data Manager
Following DEVELOPMENT_RULES.md for all development requirements

This module provides pure data access operations for URL content.
No business logic - only database operations.
"""

import logging
from typing import Any, Dict, List, Optional

from .base_data import BaseDataManager

logger = logging.getLogger(__name__)


class URLDataManager(BaseDataManager):
    """
    Pure data access manager for URL operations.
    
    Handles all PostgreSQL and Milvus operations for URL content
    without any business logic or orchestration.
    """
    
    def __init__(self, postgres_manager: Any, milvus_manager: Optional[Any] = None) -> None:
        """
        Initialize URL data manager.
        
        Args:
            postgres_manager: PostgreSQL connection manager
            milvus_manager: Optional Milvus vector store manager
        """
        super().__init__(postgres_manager, milvus_manager)
        logger.info("URLDataManager initialized for pure data operations")
    
    # =============================================================================
    # URL Metadata Operations
    # =============================================================================
    
    def upsert_url_metadata(self, url: str, metadata: Dict[str, Any]) -> None:
        """
        Insert or update URL metadata in PostgreSQL.
        
        Args:
            url: URL string
            metadata: URL metadata dictionary
        """
        try:
            query = """
                INSERT INTO urls (
                    url, title, description, content_preview, domain,
                    content_type, file_size, word_count, chunk_count,
                    processing_status, last_crawled, robots_allowed
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (url) 
                DO UPDATE SET
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    content_preview = EXCLUDED.content_preview,
                    domain = EXCLUDED.domain,
                    content_type = EXCLUDED.content_type,
                    file_size = EXCLUDED.file_size,
                    word_count = EXCLUDED.word_count,
                    chunk_count = EXCLUDED.chunk_count,
                    processing_status = EXCLUDED.processing_status,
                    last_crawled = EXCLUDED.last_crawled,
                    robots_allowed = EXCLUDED.robots_allowed,
                    updated_at = CURRENT_TIMESTAMP
            """
            
            params = (
                url,
                metadata.get('title', ''),
                metadata.get('description', ''),
                metadata.get('content_preview', ''),
                metadata.get('domain', ''),
                metadata.get('content_type', ''),
                metadata.get('file_size', 0),
                metadata.get('word_count', 0),
                metadata.get('chunk_count', 0),
                metadata.get('processing_status', 'pending'),
                metadata.get('last_crawled'),
                metadata.get('robots_allowed', True)
            )
            
            self.execute_query(query, params)
            logger.debug(f"Successfully upserted URL metadata: {url}")
            
        except Exception as e:
            logger.error(f"Failed to upsert URL metadata {url}: {e}")
            raise
    
    def get_url_metadata(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve URL metadata by URL.
        
        Args:
            url: URL string
            
        Returns:
            URL metadata dictionary or None if not found
        """
        query = "SELECT * FROM urls WHERE url = %s"
        result = self.execute_query(query, (url,), fetch_one=True)
        return dict(result) if result else None
    
    def delete_url_metadata(self, url: str) -> bool:
        """
        Delete URL metadata by URL.
        
        Args:
            url: URL string
            
        Returns:
            True if deleted, False if not found
        """
        try:
            query = "DELETE FROM urls WHERE url = %s"
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (url,))
                    deleted = cur.rowcount > 0
                conn.commit()
            
            if deleted:
                logger.debug(f"Deleted URL metadata: {url}")
            else:
                logger.warning(f"URL metadata not found for deletion: {url}")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to delete URL metadata {url}: {e}")
            raise
    
    # =============================================================================
    # URL Statistics Operations
    # =============================================================================
    
    def get_url_statistics(self) -> Dict[str, Any]:
        """
        Get global URL statistics.
        
        Returns:
            Dictionary with URL statistics
        """
        try:
            query = """
                SELECT 
                    COUNT(*) as total_urls,
                    COUNT(*) FILTER (WHERE processing_status = 'completed') as processed_urls,
                    COUNT(*) FILTER (WHERE processing_status = 'pending') as pending_urls,
                    COUNT(*) FILTER (WHERE processing_status = 'error') as error_urls,
                    COUNT(DISTINCT domain) as unique_domains,
                    SUM(word_count) as total_words,
                    SUM(chunk_count) as total_chunks
                FROM urls
            """
            
            result = self.execute_query(query, fetch_one=True)
            
            if result:
                return {
                    'total_urls': result['total_urls'],
                    'processed_urls': result['processed_urls'],
                    'pending_urls': result['pending_urls'],
                    'error_urls': result['error_urls'],
                    'unique_domains': result['unique_domains'],
                    'total_words': result['total_words'] or 0,
                    'total_chunks': result['total_chunks'] or 0
                }
            else:
                return {
                    'total_urls': 0,
                    'processed_urls': 0,
                    'pending_urls': 0,
                    'error_urls': 0,
                    'unique_domains': 0,
                    'total_words': 0,
                    'total_chunks': 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get URL statistics: {e}")
            raise
    
    def get_domain_statistics(self, domain: str) -> Dict[str, Any]:
        """
        Get statistics for a specific domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Dictionary with domain statistics
        """
        try:
            query = """
                SELECT 
                    COUNT(*) as total_urls,
                    COUNT(*) FILTER (WHERE processing_status = 'completed') as processed_urls,
                    COUNT(*) FILTER (WHERE processing_status = 'pending') as pending_urls,
                    COUNT(*) FILTER (WHERE processing_status = 'error') as error_urls,
                    SUM(word_count) as total_words,
                    SUM(chunk_count) as total_chunks
                FROM urls
                WHERE domain = %s
            """
            
            result = self.execute_query(query, (domain,), fetch_one=True)
            
            if result:
                return {
                    'domain': domain,
                    'total_urls': result['total_urls'],
                    'processed_urls': result['processed_urls'],
                    'pending_urls': result['pending_urls'],
                    'error_urls': result['error_urls'],
                    'total_words': result['total_words'] or 0,
                    'total_chunks': result['total_chunks'] or 0
                }
            else:
                return {
                    'domain': domain,
                    'total_urls': 0,
                    'processed_urls': 0,
                    'pending_urls': 0,
                    'error_urls': 0,
                    'total_words': 0,
                    'total_chunks': 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get domain statistics for {domain}: {e}")
            raise
    
    # =============================================================================
    # URL Chunk Operations
    # =============================================================================
    
    def upsert_url_chunk(self, chunk_data: Dict[str, Any]) -> None:
        """
        Insert or update URL chunk in PostgreSQL.
        
        Args:
            chunk_data: URL chunk data dictionary
        """
        required_fields = ["chunk_id", "url", "chunk_text"]
        missing_fields = [field for field in required_fields if not chunk_data.get(field)]
        
        if missing_fields:
            raise ValueError(f"Chunk data missing required fields: {missing_fields}")
        
        try:
            query = """
                INSERT INTO url_chunks (
                    chunk_id, url, chunk_text, chunk_index, chunk_metadata
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) 
                DO UPDATE SET
                    url = EXCLUDED.url,
                    chunk_text = EXCLUDED.chunk_text,
                    chunk_index = EXCLUDED.chunk_index,
                    chunk_metadata = EXCLUDED.chunk_metadata,
                    updated_at = CURRENT_TIMESTAMP
            """
            
            params = (
                chunk_data['chunk_id'],
                chunk_data['url'],
                chunk_data['chunk_text'],
                chunk_data.get('chunk_index', 0),
                chunk_data.get('chunk_metadata', {})
            )
            
            self.execute_query(query, params)
            logger.debug(f"Successfully upserted URL chunk: {chunk_data['chunk_id']}")
            
        except Exception as e:
            logger.error(f"Failed to upsert URL chunk {chunk_data.get('chunk_id')}: {e}")
            raise
    
    def get_url_chunks(self, url: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a specific URL.
        
        Args:
            url: URL string
            
        Returns:
            List of chunk dictionaries
        """
        try:
            query = """
                SELECT 
                    chunk_id,
                    url,
                    chunk_text,
                    chunk_index,
                    chunk_metadata,
                    created_at
                FROM url_chunks 
                WHERE url = %s
                ORDER BY chunk_index
            """
            
            results = self.execute_query(query, (url,), fetch_all=True)
            
            chunks = []
            if results:
                for row in results:
                    chunk = {
                        'chunk_id': row['chunk_id'],
                        'url': row['url'],
                        'chunk_text': row['chunk_text'],
                        'chunk_index': row['chunk_index'],
                        'chunk_metadata': row['chunk_metadata'],
                        'created_at': row['created_at']
                    }
                    chunks.append(chunk)
            
            logger.debug(f"Retrieved {len(chunks)} chunks for URL {url}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to get chunks for URL {url}: {e}")
            raise
    
    def delete_url_chunks(self, url: str) -> int:
        """
        Delete all chunks for a specific URL.
        
        Args:
            url: URL string
            
        Returns:
            Number of chunks deleted
        """
        try:
            query = "DELETE FROM url_chunks WHERE url = %s"
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (url,))
                    deleted_count = cur.rowcount
                conn.commit()
            
            logger.debug(f"Deleted {deleted_count} chunks for URL {url}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete chunks for URL {url}: {e}")
            raise
    
    # =============================================================================
    # URL Search Operations
    # =============================================================================
    
    def search_url_chunks_fts(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search URL chunks using PostgreSQL Full Text Search.
        
        Args:
            query: Search query string
            k: Maximum number of results to return
            
        Returns:
            List of chunk dictionaries with search results
        """
        try:
            search_query = """
                SELECT 
                    uc.chunk_id,
                    uc.url,
                    uc.chunk_text,
                    uc.chunk_index,
                    uc.chunk_metadata,
                    u.title,
                    u.domain,
                    ts_rank_cd(to_tsvector('english', uc.chunk_text), plainto_tsquery('english', %s)) as rank
                FROM url_chunks uc
                JOIN urls u ON uc.url = u.url
                WHERE to_tsvector('english', uc.chunk_text) @@ plainto_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT %s
            """
            
            results = self.execute_query(search_query, (query, query, k), fetch_all=True)
            
            chunks = []
            if results:
                for row in results:
                    chunk = {
                        'chunk_id': row['chunk_id'],
                        'url': row['url'],
                        'chunk_text': row['chunk_text'],
                        'chunk_index': row['chunk_index'],
                        'title': row['title'],
                        'domain': row['domain'],
                        'fts_rank': float(row['rank']),
                        'metadata': row['chunk_metadata'] or {}
                    }
                    chunks.append(chunk)
            
            logger.debug(f"FTS search returned {len(chunks)} results for query: {query[:50]}...")
            return chunks
            
        except Exception as e:
            logger.error(f"URL FTS search failed for query '{query}': {e}")
            raise
    
    # =============================================================================
    # Vector Operations (Milvus)
    # =============================================================================
    
    def delete_url_vectors(self, url: str) -> bool:
        """
        Delete URL vectors from Milvus.
        
        Args:
            url: URL string
            
        Returns:
            True if successful, False otherwise
        """
        if not self.milvus_manager:
            logger.warning("Milvus manager not available for vector deletion")
            return False
        
        try:
            # Implementation depends on Milvus manager interface
            # This is a placeholder for future Milvus operations
            logger.debug(f"Would delete vectors for URL {url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors for URL {url}: {e}")
            return False
