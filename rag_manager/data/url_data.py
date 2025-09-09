#!/usr/bin/env python
"""
URL Data Manager
Following DEVELOPMENT_RULES.md for all development requirements

This module provides pure data access operations for URL content.
No business logic - only database operations.
"""

import logging
import os
import os
import uuid
import urllib.parse
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_data import BaseDataManager

import requests
from bs4 import BeautifulSoup
EXTERNAL_DEPS_AVAILABLE = True

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
    # URL Title Extraction Helper
    # =============================================================================
    
    def extract_title_from_url(self, url: str) -> str:
        """
        Extract the title from a web page by scraping the <title> tag.
        
        Args:
            url: URL to extract title from
            
        Returns:
            Extracted title or fallback title
        """
        if not EXTERNAL_DEPS_AVAILABLE:
            logger.warning("External dependencies not available for title extraction")
            try:
                parsed = urllib.parse.urlparse(url)
                return f"Page from {parsed.netloc}"
            except:
                return "Unknown Page"
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            title_tag = soup.find('title')
            
            if title_tag and title_tag.string:
                title = title_tag.string.strip()
                # Clean up title
                title = re.sub(r'\s+', ' ', title)
                title = title[:255]  # Limit length
                return title
            else:
                # Fallback to URL parsing
                parsed = urllib.parse.urlparse(url)
                return f"Page from {parsed.netloc}"
                
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch title from {url}: {e}")
            # Return a fallback title based on URL
            try:
                parsed = urllib.parse.urlparse(url)
                return f"Page from {parsed.netloc}"
            except:
                return "Unknown Page"
        except Exception as e:
            logger.error(f"Error extracting title from {url}: {e}")
            return "Error Loading Page"
    
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
    # URL Table CRUD Operations (moved from PostgreSQLURLManager)
    # =============================================================================
    
    def add_url(self, url: str, title: Optional[str] = None, description: Optional[str] = None,
                refresh_interval_minutes: Optional[int] = None, crawl_domain: bool = False,
                ignore_robots: bool = False, snapshot_retention_days: int = 0,
                snapshot_max_snapshots: int = 0, parent_url_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a new URL to the database with validation and title extraction.
        
        Args:
            url: The URL to add
            title: Optional title for the URL (will be extracted if not provided)
            description: Optional description
            refresh_interval_minutes: How often to refresh (defaults to 1440 minutes/24 hours)
            crawl_domain: Whether to crawl the entire domain
            ignore_robots: Whether to ignore robots.txt for this URL
            snapshot_retention_days: How long to keep snapshots
            snapshot_max_snapshots: Maximum number of snapshots to keep
            parent_url_id: Optional parent URL ID for child URLs
            
        Returns:
            Dict with success status and URL details
        """
        # Validate URL format
        try:
            result = urllib.parse.urlparse(url)
            is_valid_url = all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except Exception:
            is_valid_url = False
            
        if not is_valid_url:
            return {
                'success': False,
                'error': 'Invalid URL format',
                'url_id': None,
                'url_data': None
            }
        
        # Extract title if not provided
        if not title:
            try:
                title = self.extract_title_from_url(url)
            except Exception as e:
                logger.warning(f"Could not extract title for {url}: {e}")
                title = "Unknown Page"
        
        # Validate parent URL if provided
        if parent_url_id:
            # Validate UUID format
            try:
                uuid_obj = uuid.UUID(parent_url_id)
                is_valid_uuid = str(uuid_obj) == parent_url_id
            except (ValueError, TypeError):
                is_valid_uuid = False
                
            if not is_valid_uuid:
                return {
                    'success': False,
                    'error': 'Invalid parent URL ID format',
                    'url_id': None,
                    'url_data': None
                }
            
            # Check if parent exists
            parent_data = self.get_url_by_id(parent_url_id)
            if not parent_data:
                return {
                    'success': False,
                    'error': 'Parent URL not found',
                    'url_id': None,
                    'url_data': None
                }
            
        # Set defaults
        if refresh_interval_minutes is None:
            refresh_interval_minutes = 1440  # 24 hours default
        if snapshot_retention_days == 0:
            snapshot_retention_days = 1
        if snapshot_max_snapshots == 0:
            snapshot_max_snapshots = 1

        try:
            query = """
                INSERT INTO urls (url, title, description, status, refresh_interval_minutes, 
                                crawl_domain, ignore_robots, snapshot_retention_days, 
                                snapshot_max_snapshots, parent_url_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
            """
            params = (url, title, description, 'active', refresh_interval_minutes, 
                     crawl_domain, ignore_robots, snapshot_retention_days, 
                     snapshot_max_snapshots, parent_url_id)
            
            result = self.execute_query(query, params, fetch_one=True)
            if not result:
                raise Exception("No result returned from INSERT query")
                
            url_id = result['id']
            
            logger.info(f"Added URL: {url} with title: {title} (ID: {url_id}) {f'(parent: {parent_url_id})' if parent_url_id else '(root URL)'}")
            return {
                "success": True, 
                "message": "URL added successfully", 
                "url_id": str(url_id), 
                "title": title,
                "url_data": {
                    "url_id": str(url_id),
                    "url": url,
                    "title": title,
                    "description": description
                }
            }
            
        except Exception as e:
            if "duplicate key" in str(e).lower():
                return {"success": False, "message": "URL already exists", "url_id": None, "url_data": None}
            logger.error(f"Error adding URL: {str(e)}")
            return {"success": False, "message": f"Database error: {str(e)}", "url_id": None, "url_data": None}
            
        except Exception as e:
            if "duplicate key" in str(e).lower():
                return {"success": False, "message": "URL already exists"}
            logger.error(f"Error adding URL: {str(e)}")
            return {"success": False, "message": f"Database error: {str(e)}"}

    def get_all_urls(self) -> List[Dict[str, Any]]:
        """Retrieve all parent URLs from the database (excludes auto-discovered child URLs)."""
        try:
            query = """
                SELECT id AS url_id, url, title, description, status, refresh_interval_minutes, 
                       crawl_domain, ignore_robots, snapshot_retention_days, snapshot_max_snapshots,
                       last_crawled, is_refreshing, last_refresh_started, last_content_hash, last_update_status,
                       created_at, updated_at, parent_url_id,
                        CASE 
                            WHEN refresh_interval_minutes IS NOT NULL 
                                 AND refresh_interval_minutes > 0 
                                 AND last_crawled IS NOT NULL
                            THEN last_crawled + INTERVAL '1 minute' * refresh_interval_minutes
                            ELSE NULL
                        END AS next_refresh
                FROM urls 
                WHERE status = 'active' AND parent_url_id IS NULL
                ORDER BY created_at DESC
            """
            
            result = self.execute_query(query, fetch_all=True)
            if not result:
                return []
                
            urls = []
            
            for row in result:
                url_dict = dict(row)
                
                # Convert boolean to int for SQLite compatibility
                url_dict['crawl_domain'] = 1 if url_dict.get('crawl_domain') else 0
                url_dict['ignore_robots'] = 1 if url_dict.get('ignore_robots') else 0
                
                # Map PostgreSQL columns used by the UI
                url_dict['last_scraped'] = url_dict['last_crawled']
                
                # Convert datetime objects to strings for JSON serialization
                for key, value in url_dict.items():
                    if isinstance(value, datetime):
                        url_dict[key] = value.isoformat() if value else None
                
                urls.append(url_dict)
            
            return urls
            
        except Exception as e:
            logger.error(f"Error retrieving URLs: {str(e)}")
            return []

    def get_due_urls(self) -> List[Dict[str, Any]]:
        """Get URLs that are due for refresh."""
        try:
            query = """
                SELECT * FROM urls
                WHERE status = 'active'
                    AND refresh_interval_minutes IS NOT NULL
                    AND refresh_interval_minutes > 0
                    AND (is_refreshing IS NOT TRUE OR is_refreshing IS NULL)
                    AND NOT EXISTS (
                        SELECT 1 FROM urls child_urls 
                        WHERE child_urls.parent_url_id = urls.id 
                        AND child_urls.is_refreshing = TRUE
                    )
                    AND (
                        last_crawled IS NULL OR
                        last_crawled + INTERVAL '1 minute' * refresh_interval_minutes <= NOW()
                    )
            """
            
            result = self.execute_query(query, fetch_all=True)
            if not result:
                logger.debug("Scheduler: no due URLs this cycle")
                return []
                
            rows = []
            
            for row in result:
                url_dict = dict(row)
                
                # Convert boolean to int for SQLite compatibility
                url_dict['crawl_domain'] = 1 if url_dict.get('crawl_domain') else 0
                url_dict['ignore_robots'] = 1 if url_dict.get('ignore_robots') else 0
                
                # Map PostgreSQL columns used by the UI
                url_dict['last_scraped'] = url_dict['last_crawled']
                
                # Convert datetime objects to strings
                for key, value in url_dict.items():
                    if isinstance(value, datetime):
                        url_dict[key] = value.isoformat() if value else None
                
                rows.append(url_dict)
            
            return rows
            
        except Exception as e:
            logger.error(f"Error fetching due URLs: {e}")
            return []

    def get_url_by_id(self, url_id: str) -> Optional[Dict[str, Any]]:
        """Get URL by ID."""
        import uuid
        
        logger.debug(f"Getting URL by ID: {url_id}")
        
        # Input validation
        if not url_id:
            logger.error("get_url_by_id failed: No URL ID provided")
            return None
            
        # Convert to string for UUID handling
        url_id_str = str(url_id)
        
        # Basic UUID format validation
        try:
            uuid.UUID(url_id_str)
        except (ValueError, TypeError):
            logger.error(f"get_url_by_id failed: Invalid UUID format for ID: {url_id_str}")
            return None
            
        try:
            query = "SELECT * FROM urls WHERE id = %s"
            result = self.execute_query(query, (url_id_str,), fetch_one=True)
            
            if result:
                url_dict = dict(result)
                url_dict['url_id'] = str(url_dict.get('id'))
                if 'id' in url_dict:
                    del url_dict['id']
                
                # Normalize booleans for UI compatibility
                url_dict['crawl_domain'] = 1 if url_dict.get('crawl_domain') else 0
                url_dict['ignore_robots'] = 1 if url_dict.get('ignore_robots') else 0
                
                # Map PostgreSQL columns used by the UI
                url_dict['last_scraped'] = url_dict['last_crawled']

                # Convert datetime objects to strings
                from datetime import datetime
                for key, value in url_dict.items():
                    if isinstance(value, datetime):
                        url_dict[key] = value.isoformat() if value else None
                
                logger.debug(f"Found URL: {url_dict['url']} (ID: {url_id_str})")
                return url_dict
            else:
                logger.warning(f"No URL found with ID: {url_id_str}")
                return None
                
        except Exception as e:
            logger.error(f"Database error getting URL by ID {url_id_str}: {str(e)}", exc_info=True)
            return None

    def get_child_urls(self, parent_url_id: str) -> List[Dict[str, Any]]:
        """Retrieve all child URLs discovered from domain crawling for a specific parent URL."""
        try:
            query = """
                SELECT id, url, title, description, status, last_crawled, last_update_status,
                       created_at, updated_at
                FROM urls 
                WHERE status = 'active' AND parent_url_id = %s
                ORDER BY created_at DESC
            """
            
            result = self.execute_query(query, (parent_url_id,), fetch_all=True)
            if not result:
                return []
                
            child_urls = []
            
            for row in result:
                url_dict = dict(row)
                
                # Convert datetime objects to strings for JSON serialization
                for key, value in url_dict.items():
                    if isinstance(value, datetime):
                        url_dict[key] = value.isoformat() if value else None
                
                child_urls.append(url_dict)
            
            return child_urls
            
        except Exception as e:
            logger.error(f"Error retrieving child URLs for parent {parent_url_id}: {str(e)}")
            return []

    def get_child_url_stats(self, parent_url_id: str) -> Dict[str, Any]:
        """
        Get statistics for child URLs.
        
        Args:
            parent_url_id: Parent URL ID
            
        Returns:
            Dictionary with child URL statistics
        """
        try:
            child_urls = self.get_child_urls(parent_url_id)
            
            total_children = len(child_urls)
            status_counts = {}
            
            for child in child_urls:
                status = child.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                'success': True,
                'parent_url_id': parent_url_id,
                'total_children': total_children,
                'status_breakdown': status_counts,
                'child_urls': child_urls
            }
            
        except Exception as e:
            logger.error(f"Failed to get child URL stats for {parent_url_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'parent_url_id': parent_url_id,
                'total_children': 0,
                'status_breakdown': {},
                'child_urls': []
            }

    def update_url_metadata(self, url_id: str, title: Optional[str], description: Optional[str], 
                           refresh_interval_minutes: Optional[int], crawl_domain: Optional[int], 
                           ignore_robots: Optional[int],
                           snapshot_retention_days: Optional[int] = None, 
                           snapshot_max_snapshots: Optional[int] = None) -> Dict[str, Any]:
        """Update URL metadata."""
        try:
            # Check if URL exists
            existing_url = self.execute_query("SELECT id FROM urls WHERE id = %s", (str(url_id),), fetch_one=True)
            if not existing_url:
                return {"success": False, "message": "URL not found"}
            
            # Prepare update query with only non-None values
            update_fields = []
            params = []
            
            if title is not None:
                update_fields.append("title = %s")
                params.append(title)
            if description is not None:
                update_fields.append("description = %s")
                params.append(description)
            if refresh_interval_minutes is not None:
                update_fields.append("refresh_interval_minutes = %s")
                params.append(refresh_interval_minutes)
            if crawl_domain is not None:
                update_fields.append("crawl_domain = %s")
                params.append(bool(crawl_domain))
            if ignore_robots is not None:
                update_fields.append("ignore_robots = %s")
                params.append(bool(ignore_robots))
            if snapshot_retention_days is not None:
                update_fields.append("snapshot_retention_days = %s")
                params.append(snapshot_retention_days)
            if snapshot_max_snapshots is not None:
                update_fields.append("snapshot_max_snapshots = %s")
                params.append(snapshot_max_snapshots)
            
            if not update_fields:
                return {"success": False, "message": "No fields to update"}
            
            update_fields.append("updated_at = NOW()")
            params.append(str(url_id))
            
            query = f"UPDATE urls SET {', '.join(update_fields)} WHERE id = %s"
            self.execute_query(query, tuple(params))
            
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Error updating URL metadata: {e}")
            return {"success": False, "message": str(e)}

    def delete_url(self, url_id: str) -> Dict[str, Any]:
        """Delete a URL from the database and clean up all associated data."""
        import uuid
        
        logger.info(f"Starting URL deletion process for ID: {url_id}")
        
        # Input validation
        if not url_id:
            logger.error("URL deletion failed: No URL ID provided")
            return {"success": False, "message": "Invalid URL ID: ID cannot be empty"}
        
        # Basic UUID format validation
        try:
            uuid.UUID(url_id)
        except (ValueError, TypeError):
            logger.error(f"URL deletion failed: Invalid UUID format for ID: {url_id}")
            return {"success": False, "message": "Invalid URL ID format"}
        
        try:
            logger.debug(f"Attempting to delete URL with ID: {url_id}")
            
            # First check if the URL exists
            existing_url = self.execute_query("SELECT url, title FROM urls WHERE id = %s", (url_id,), fetch_one=True)
            
            if not existing_url:
                logger.warning(f"URL deletion failed: No URL found with ID: {url_id}")
                return {"success": False, "message": "URL not found"}
            
            url_string = existing_url['url']
            logger.info(f"Found URL to delete - ID: {url_id}, URL: {url_string}, Title: {existing_url['title']}")
            
            # Step 1: Clean up snapshots using snapshot service
            logger.info(f"Cleaning up snapshots for URL ID: {url_id}")
            snapshot_cleanup_result = {"deleted": 0}
            try:
                # Import snapshot service to handle proper cleanup
                from ingestion.url.utils.snapshot_service import URLSnapshotService
                from rag_manager.core.config import Config
                
                config = Config()
                snapshot_service = URLSnapshotService(self.postgres_manager, config)
                
                # Use the comprehensive snapshot cleanup
                snapshot_cleanup_result = snapshot_service.cleanup_all_snapshots(url_id)
                if snapshot_cleanup_result.get("success"):
                    logger.info(f"Successfully cleaned up {snapshot_cleanup_result.get('deleted', 0)} snapshots")
                else:
                    logger.warning(f"Snapshot cleanup warning: {snapshot_cleanup_result.get('error', 'Unknown error')}")
            except Exception as snapshot_error:
                logger.warning(f"Failed to use snapshot service for cleanup: {snapshot_error}")
                # Fall back to basic cleanup
                snapshot_cleanup_result = {"deleted": 0, "error": str(snapshot_error)}

            # Step 2: Find and delete any remaining documents (fallback)
            logger.info(f"Cleaning up any remaining documents for URL: {url_string}")
            documents_query = """
                SELECT d.id, d.file_path 
                FROM documents d 
                LEFT JOIN url_snapshots us ON d.id = us.document_id 
                WHERE us.url_id = %s OR (d.document_type = 'url' AND d.file_path LIKE %s)
            """
            url_documents = self.execute_query(documents_query, (url_id, f"%{url_string}%"), fetch_all=True)
            
            if url_documents:
                document_ids = [doc['id'] for doc in url_documents]
                remaining_files = [doc['file_path'] for doc in url_documents]
                
                logger.info(f"Found {len(url_documents)} remaining documents to clean up")
                
                # Step 3: Delete document chunks first (foreign key constraint)
                chunk_delete_query = "DELETE FROM document_chunks WHERE document_id = ANY(%s)"
                self.execute_query(chunk_delete_query, (document_ids,))
                logger.info(f"Deleted document chunks for {len(document_ids)} documents")
                
                # Step 4: Delete remaining documents
                doc_delete_query = "DELETE FROM documents WHERE id = ANY(%s)"
                self.execute_query(doc_delete_query, (document_ids,))
                logger.info(f"Deleted {len(document_ids)} remaining documents")
                
                # Step 5: Clean up any remaining files on disk
                files_deleted = 0
                for file_path in remaining_files:
                    try:
                        if file_path and os.path.exists(file_path):
                            os.remove(file_path)
                            files_deleted += 1
                            logger.debug(f"Deleted remaining file: {file_path}")
                            # Also delete JSON sidecar if it exists
                            json_path = file_path.replace(".pdf", ".json")
                            if os.path.exists(json_path):
                                os.remove(json_path)
                                logger.debug(f"Deleted remaining JSON file: {json_path}")
                    except Exception as file_error:
                        logger.warning(f"Failed to delete remaining file {file_path}: {file_error}")
            else:
                document_ids = []
                files_deleted = 0
                logger.info("No remaining documents found to clean up")
            
            # Step 6: Delete child URLs first (if any)
            child_delete_query = "DELETE FROM urls WHERE parent_url_id = %s"
            self.execute_query(child_delete_query, (url_id,))
            logger.info(f"Deleted child URLs for parent {url_id}")
            
            # Step 7: Delete the main URL
            url_delete_query = "DELETE FROM urls WHERE id = %s"
            self.execute_query(url_delete_query, (url_id,))
            
            # Calculate total files deleted (snapshots + remaining files)
            total_files_deleted = snapshot_cleanup_result.get("deleted", 0) + files_deleted
            
            logger.info(f"URL deletion completed successfully - ID: {url_id}, URL: {url_string}, Total files cleaned: {total_files_deleted}")
            return {
                "success": True, 
                "message": "URL and associated data deleted successfully",
                "details": {
                    "url": url_string,
                    "documents_deleted": len(document_ids),
                    "snapshot_files_deleted": snapshot_cleanup_result.get("deleted", 0),
                    "remaining_files_deleted": files_deleted,
                    "total_files_deleted": total_files_deleted
                }
            }
            
        except Exception as e:
            logger.error(f"URL deletion failed for ID {url_id}: {str(e)}", exc_info=True)
            return {"success": False, "message": f"Deletion failed: {str(e)}"}

    def mark_scraped(self, url_id: str, refresh_interval_minutes: Optional[int]) -> None:
        """Mark URL as scraped."""
        try:
            query = "UPDATE urls SET last_crawled = NOW() WHERE id = %s"
            self.execute_query(query, (str(url_id),))
        except Exception as e:
            logger.error(f"Error marking URL scraped: {e}")

    def set_refreshing(self, url_id: str, refreshing: bool) -> None:
        """Set or clear the refreshing flag."""
        try:
            if refreshing:
                query = "UPDATE urls SET is_refreshing = %s, last_refresh_started = NOW() WHERE id = %s"
                self.execute_query(query, (refreshing, str(url_id)))
            else:
                query = "UPDATE urls SET is_refreshing = %s WHERE id = %s"
                self.execute_query(query, (refreshing, str(url_id)))
        except Exception as e:
            logger.error(f"Error updating refreshing flag: {e}")

    def update_url_hash_status(self, url_id: str, content_hash: Optional[str], status: str) -> None:
        """Update URL content hash and status."""
        try:
            query = "UPDATE urls SET last_content_hash = %s, last_update_status = %s, last_crawled = NOW() WHERE id = %s"
            self.execute_query(query, (content_hash, status, str(url_id)))
        except Exception as e:
            logger.error(f"Error updating URL hash/status: {e}")

    def get_url_count(self) -> int:
        """Get the total number of active URLs."""
        try:
            result = self.execute_query("SELECT COUNT(*) as count FROM urls WHERE status = 'active'", fetch_one=True)
            if result:
                return result['count']
            return 0
        except Exception as e:
            logger.error(f"Error getting URL count: {str(e)}")
            return 0

    def get_robots_status(self, url_id: str) -> Dict[str, Any]:
        """
        Get robots.txt configuration and status for a specific URL.
        
        Args:
            url_id: The UUID of the URL to check
            
        Returns:
            Dict with robots.txt status information
        """
        try:
            query = """
                SELECT id, url, ignore_robots, last_update_status, created_at
                FROM urls 
                WHERE id = %s AND status = 'active'
            """
            result = self.execute_query(query, (url_id,), fetch_one=True)
            
            if not result:
                return {"success": False, "message": "URL not found"}
            
            url_info = dict(result)
            
            return {
                "success": True,
                "id": str(url_info['id']),
                "url": url_info['url'],
                "ignore_robots": bool(url_info['ignore_robots']),
                "robots_enforcement": "disabled" if url_info['ignore_robots'] else "enabled",
                "last_update_status": url_info['last_update_status'],
                "created_at": url_info['created_at'].isoformat() if url_info['created_at'] else None
            }
                
        except Exception as e:
            logger.error(f"Error getting robots status for URL {url_id}: {str(e)}")
            return {"success": False, "message": f"Database error: {str(e)}"}

    def update_robots_setting(self, url_id: str, ignore_robots: bool) -> Dict[str, Any]:
        """
        Update the robots.txt ignore setting for a specific URL.
        
        Args:
            url_id: The UUID of the URL to update
            ignore_robots: Whether to ignore robots.txt for this URL
            
        Returns:
            Dict with update result
        """
        try:
            query = """
                UPDATE urls 
                SET ignore_robots = %s, updated_at = NOW()
                WHERE id = %s AND status = 'active'
                RETURNING id, url, ignore_robots
            """
            result = self.execute_query(query, (ignore_robots, url_id), fetch_one=True)
            
            if not result:
                return {"success": False, "message": "URL not found"}
            
            url_info = dict(result)
            
            logger.info(f"Updated robots setting for {url_info['url']}: ignore_robots={ignore_robots}")
            
            return {
                "success": True,
                "id": str(url_info['id']),
                "url": url_info['url'],
                "ignore_robots": bool(url_info['ignore_robots']),
                "message": f"Robots.txt enforcement {'disabled' if ignore_robots else 'enabled'}"
            }
                
        except Exception as e:
            logger.error(f"Error updating robots setting for URL {url_id}: {str(e)}")
            return {"success": False, "message": f"Database error: {str(e)}"}
