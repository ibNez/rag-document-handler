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
    @staticmethod
    def get_snapshot_folder_for_url(url: str, base_dir: str = "snapshots") -> str:
        """
        Given a URL, return the full folder path for storing its snapshots/documents.
        Uses the domain as the first folder, and the safe page name as the subfolder.
        Example: https://news.ycombinator.com/from?site=sourcetable.com
        → snapshots/news.ycombinator.com/from_QuestionMark_site-Equals-sourcetable.com/
        """
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc or "unknown-domain"
        page_folder = URLDataManager.url_to_safe_folder(url)
        return os.path.join(base_dir, domain, page_folder)
    # =============================================================================
    # URL Path to Filesystem-safe Folder Name
    # =============================================================================

    @staticmethod
    def url_to_safe_folder(url: str) -> str:
        """
        Convert a URL path and query into a filesystem-safe folder name.
        Replaces unsafe characters with _CharName_ (e.g., ? -> _QuestionMark_).
        Example: /from?site=sourcetable.com -> from_QuestionMark_site-Equals-sourcetable.com
        """
        char_map = {
            '/': '_ForwardSlash_',
            '?': '_QuestionMark_',
            '=': '_Equals_',
            '&': '_Ampersand_',
            '#': '_Hash_',
            ':': '_Colon_',
            ';': '_Semicolon_',
            '\\': '_Backslash_',
            ' ': '_Space_',
            '<': '_LessThan_',
            '>': '_GreaterThan_',
            '|': '_Pipe_',
            '"': '_Quote_',
            "'": '_Apostrophe_',
            '*': '_Asterisk_',
            '%': '_Percent_',
            '$': '_Dollar_',
            ',': '_Comma_',
            '@': '_At_',
            '+': '_Plus_',
            '`': '_Backtick_',
            '^': '_Caret_',
            '~': '_Tilde_',
            '(': '_LeftParen_',
            ')': '_RightParen_',
            '{': '_LeftBrace_',
            '}': '_RightBrace_',
            '[': '_LeftBracket_',
            ']': '_RightBracket_',
        }
        parsed = urllib.parse.urlparse(url)
        # Combine path and query for uniqueness
        page = parsed.path
        if parsed.query:
            page += '?' + parsed.query
        # Remove leading slash for folder name
        if page.startswith('/'):
            page = page[1:]
        # Replace each unsafe character
        safe = []
        for c in page:
            if c in char_map:
                safe.append(char_map[c])
            else:
                safe.append(c)
        return ''.join(safe) or 'root'
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
            
            if title_tag and title_tag:
                title = title_tag.get_text().strip()
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
                snapshot_max_snapshots: int = 0) -> Dict[str, Any]:
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
                                snapshot_max_snapshots)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
            """
            params = (url, title, description, 'active', refresh_interval_minutes, 
                     crawl_domain, ignore_robots, snapshot_retention_days, 
                     snapshot_max_snapshots)
            
            result = self.execute_query(query, params, fetch_one=True)
            if not result:
                raise Exception("No result returned from INSERT query")
                
            url_id = result['id']
            
            logger.info(f"Added URL: {url} with title: {title} (ID: {url_id}) (root URL)")
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
                       created_at, updated_at,
                        CASE 
                            WHEN refresh_interval_minutes IS NOT NULL 
                                 AND refresh_interval_minutes > 0 
                                 AND last_crawled IS NOT NULL
                            THEN last_crawled + INTERVAL '1 minute' * refresh_interval_minutes
                            ELSE NULL
                        END AS next_refresh
                FROM urls 
                WHERE status = 'active'
                ORDER BY created_at DESC
            """
            
            result = self.execute_query(query, fetch_all=True)
            if not result:
                return []
                
            return [dict(row) for row in result]
            
        except Exception as e:
            logger.error(f"Error retrieving URLs: {str(e)}")
            return []

    def get_due_urls(self) -> List[Dict[str, Any]]:
        """Get URLs that are due for refresh."""
        try:
            query = """
                SELECT * FROM urls
                WHERE refresh_interval_minutes > 0
                    AND (is_refreshing IS NOT TRUE)
                    AND (
                        last_crawled IS NULL OR
                        last_crawled + INTERVAL '1 minute' * refresh_interval_minutes <= NOW()
                    )
            """
            
            result = self.execute_query(query, fetch_all=True)
            if not result:
                logger.debug("Scheduler: no due URLs this cycle")
                return []
                
            return [dict(row) for row in result]
            
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
            query = "SELECT id AS url_id, * FROM urls WHERE id = %s"
            result = self.execute_query(query, (url_id_str,), fetch_one=True)
            
            if result:
                url_dict = dict(result)
                logger.debug(f"Found URL: {url_dict['url']} (ID: {url_id_str})")
                return url_dict
            else:
                logger.warning(f"No URL found with ID: {url_id_str}")
                return None
                
        except Exception as e:
            logger.error(f"Database error getting URL by ID {url_id_str}: {str(e)}", exc_info=True)
            return None

    def get_child_url_documents(self, parent_url_id: str) -> List[Dict[str, Any]]:
        """Retrieve all child documents discovered from domain crawling for a specific parent URL."""
        try:
            query = """
                SELECT id AS document_id, title, file_path, filename, content_preview, document_type,
                       processing_status, created_at, updated_at
                FROM documents 
                WHERE parent_url_id = %s
                ORDER BY created_at DESC
            """
            
            result = self.execute_query(query, (parent_url_id,), fetch_all=True)
            if not result:
                return []
                
            return [dict(row) for row in result]
            
        except Exception as e:
            logger.error(f"Error retrieving child URL documents for parent {parent_url_id}: {str(e)}")
            return []

    def get_child_document_stats(self, parent_url_id: str) -> Dict[str, Any]:
        """
        Get statistics for child documents.
        
        Args:
            parent_url_id: Parent URL ID
            
        Returns:
            Dictionary with child document statistics
        """
        try:
            child_documents = self.get_child_url_documents(parent_url_id)
            
            total_children = len(child_documents)
            status_counts = {}
            
            for child in child_documents:
                status = child.get('processing_status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                'success': True,
                'parent_url_id': parent_url_id,
                'total_children': total_children,
                'status_breakdown': status_counts,
                'child_documents': child_documents
            }
            
        except Exception as e:
            logger.error(f"Failed to get child document stats for {parent_url_id}: {e}")
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
            existing_url = self.execute_query("SELECT id AS url_id FROM urls WHERE id = %s", (str(url_id),), fetch_one=True)
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
        """Delete a URL and all associated data."""
        import uuid
        
        logger.info(f"Starting URL deletion process for ID: {url_id}")
        
        # Input validation
        if not url_id:
            logger.error("URL deletion failed: No URL ID provided")
            return {"success": False, "message": "Invalid URL ID: ID cannot be empty"}
        
        try:
            uuid.UUID(url_id)
        except (ValueError, TypeError):
            logger.error(f"URL deletion failed: Invalid UUID format for ID: {url_id}")
            return {"success": False, "message": "Invalid URL ID format"}
        
        try:
            # Step 1: Get document IDs and snapshot file info for this URL
            url_context = self._get_complete_url_context(url_id)
            if url_context is None:
                return {"success": False, "message": "Error retrieving URL context"}
            
            document_ids = url_context["document_ids"]
            snapshots = url_context["snapshots"]
            
            # Step 2: Delete files from disk
            files_deleted = self._delete_snapshot_files_from_disk(snapshots)
            
            # Step 3: Delete embeddings from Milvus
            embeddings_deleted = self._delete_embeddings_by_document_ids(document_ids)
            
            # Step 4: Delete document chunks from PostgreSQL
            chunks_deleted = self._delete_document_chunks_by_document_ids(document_ids)
            
            # Step 5: Delete URL snapshots
            snapshots_deleted = self._delete_url_snapshots_by_document_ids(document_ids)
            
            # Step 6: Delete documents
            documents_deleted = self._delete_documents_by_document_ids(document_ids)
            
            # Step 7: Delete URL record
            self._delete_url_record(url_id)
            
            logger.info(f"URL deletion completed successfully - ID: {url_id}")
            return {
                "success": True,
                "message": "URL and associated data deleted successfully",
                "details": {
                    "url_id": url_id,
                    "files_deleted": files_deleted,
                    "embeddings_deleted": embeddings_deleted,
                    "chunks_deleted": chunks_deleted,
                    "snapshots_deleted": snapshots_deleted,
                    "documents_deleted": documents_deleted
                }
            }
            
        except Exception as e:
            logger.error(f"URL deletion failed for ID {url_id}: {str(e)}")
            raise

    def _get_complete_url_context(self, url_id: str) -> Optional[Dict[str, Any]]:
        """Get document IDs and snapshot file info for URL deletion operations."""
        try:
            # Get document IDs for this URL
            documents_query = """
                SELECT id AS document_id
                FROM documents 
                WHERE parent_url_id = %s
            """
            documents = self.execute_query(documents_query, (url_id,), fetch_all=True)
            document_ids = [doc['document_id'] for doc in documents] if documents else []
            
            # Get snapshot file info for this URL
            snapshots_query = """
                SELECT file_path, pdf_file, json_file
                FROM url_snapshots 
                WHERE url_id = %s
            """
            snapshots = self.execute_query(snapshots_query, (url_id,), fetch_all=True)
            snapshot_files = [dict(snap) for snap in snapshots] if snapshots else []
            
            return {
                "document_ids": document_ids,
                "snapshots": snapshot_files
            }
            
        except Exception as e:
            logger.error(f"Error getting URL context for {url_id}: {e}")
            raise

    def _delete_snapshot_files_from_disk(self, snapshots: List[Dict[str, Any]]) -> int:
        """Delete snapshot files from filesystem."""
        if not snapshots:
            return 0
        
        files_deleted = 0
        
        try:
            for snapshot in snapshots:
                file_path = snapshot.get("file_path")
                pdf_file = snapshot.get("pdf_file")
                json_file = snapshot.get("json_file")
                
                if file_path and (pdf_file or json_file):
                    # Delete PDF file
                    if pdf_file:
                        pdf_path = os.path.join(file_path, pdf_file)
                        if os.path.exists(pdf_path):
                            os.remove(pdf_path)
                            files_deleted += 1
                            logger.debug(f"Deleted PDF file: {pdf_path}")
                    
                    # Delete JSON file
                    if json_file:
                        json_path = os.path.join(file_path, json_file)
                        if os.path.exists(json_path):
                            os.remove(json_path)
                            files_deleted += 1
                            logger.debug(f"Deleted JSON file: {json_path}")
            
            logger.info(f"Deleted {files_deleted} files from disk")
            return files_deleted
            
        except Exception as e:
            logger.error(f"Error deleting files from disk: {e}")
            raise

    def _delete_embeddings_by_document_ids(self, document_ids: List[str]) -> int:
        """Delete embeddings from Milvus by document IDs."""
        if not self.milvus_manager or not document_ids:
            return 0
        
        try:
            total_deleted = 0
            for document_id in document_ids:
                result = self.milvus_manager.delete_document(document_id=document_id)
                if result.get('success', False):
                    deleted_count = result.get('deleted_count', 0)
                    total_deleted += deleted_count
                    logger.debug(f"Deleted {deleted_count} embeddings for document {document_id}")
                else:
                    logger.warning(f"Failed to delete embeddings for document {document_id}")
            
            logger.info(f"Deleted {total_deleted} embeddings from Milvus")
            return total_deleted
            
        except Exception as e:
            logger.error(f"Error deleting embeddings: {e}")
            raise

    def _delete_document_chunks_by_document_ids(self, document_ids: List[str]) -> int:
        """Delete document chunks from PostgreSQL by document IDs."""
        if not document_ids:
            return 0
        
        try:
            query = "DELETE FROM document_chunks WHERE document_id = ANY(%s::uuid[]) RETURNING id"
            results = self.execute_query(query, (document_ids,), fetch_all=True)
            
            chunks_deleted = len(results) if results else 0
            logger.info(f"Deleted {chunks_deleted} document chunks")
            return chunks_deleted
            
        except Exception as e:
            logger.error(f"Error deleting document chunks: {e}")
            raise

    def _delete_url_snapshots_by_document_ids(self, document_ids: List[str]) -> int:
        """Delete URL snapshots by document IDs."""
        if not document_ids:
            return 0
        
        try:
            query = "DELETE FROM url_snapshots WHERE document_id = ANY(%s::uuid[]) RETURNING id"
            results = self.execute_query(query, (document_ids,), fetch_all=True)
            
            snapshots_deleted = len(results) if results else 0
            logger.info(f"Deleted {snapshots_deleted} URL snapshots")
            return snapshots_deleted
            
        except Exception as e:
            logger.error(f"Error deleting URL snapshots: {e}")
            raise

    def _delete_documents_by_document_ids(self, document_ids: List[str]) -> int:
        """Delete documents by document IDs."""
        if not document_ids:
            return 0
        
        try:
            query = "DELETE FROM documents WHERE id = ANY(%s::uuid[]) RETURNING id"
            results = self.execute_query(query, (document_ids,), fetch_all=True)
            
            documents_deleted = len(results) if results else 0
            logger.info(f"Deleted {documents_deleted} documents")
            return documents_deleted
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise

    def _delete_url_record(self, url_id: str) -> None:
        """Delete the URL record from urls table."""
        try:
            query = "DELETE FROM urls WHERE id = %s"
            self.execute_query(query, (url_id,))
            logger.info(f"Deleted URL record: {url_id}")
            
        except Exception as e:
            logger.error(f"Error deleting URL record: {e}")
            raise

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
                SELECT id AS url_id, url, ignore_robots, last_update_status, created_at
                FROM urls 
                WHERE id = %s AND status = 'active'
            """
            result = self.execute_query(query, (url_id,), fetch_one=True)
            
            if not result:
                return {"success": False, "message": "URL not found"}
            
            url_info = dict(result)
            
            return {
                "success": True,
                "id": str(url_info['url_id']),
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
                RETURNING id AS url_id, url, ignore_robots
            """
            result = self.execute_query(query, (ignore_robots, url_id), fetch_one=True)
            
            if not result:
                return {"success": False, "message": "URL not found"}
            
            url_info = dict(result)
            
            logger.info(f"Updated robots setting for {url_info['url']}: ignore_robots={ignore_robots}")
            
            return {
                "success": True,
                "id": str(url_info['url_id']),
                "url": url_info['url'],
                "ignore_robots": bool(url_info['ignore_robots']),
                "message": f"Robots.txt enforcement {'disabled' if ignore_robots else 'enabled'}"
            }
                
        except Exception as e:
            logger.error(f"Error updating robots setting for URL {url_id}: {str(e)}")
            return {"success": False, "message": f"Database error: {str(e)}"}
