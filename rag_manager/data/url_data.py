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
            

    def get_all_urls(self) -> List[Dict[str, Any]]:
        """Retrieve all parent URLs from the database."""
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
