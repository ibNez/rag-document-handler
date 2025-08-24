"""
PostgreSQL URL Manager - Simple adaptation of SQLite URLManager interface
Uses existing PostgreSQL schema from postgres_manager.py
"""

import logging
import os
import urllib.parse
import requests
import json
import hashlib
import uuid
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class PostgreSQLURLManager:
    """PostgreSQL-based URL manager that adapts SQLite URLManager interface to existing PostgreSQL schema."""
    
    def __init__(self, postgres_manager):
        """Initialize with PostgreSQL manager."""
        self.postgres = postgres_manager
        # For compatibility with SQLite URLManager
        self.db_path = "postgresql://rag_metadata"  # Fake path for compatibility
        logger.info("PostgreSQL URLManager initialized")
    
    def _is_valid_uuid_format(self, uuid_string: str) -> bool:
        """Validate if a string is a valid UUID format."""
        try:
            uuid_obj = uuid.UUID(uuid_string)
            return str(uuid_obj) == uuid_string
        except (ValueError, TypeError):
            return False
    
    def validate_url(self, url: str) -> bool:
        """Validate if the provided string is a valid URL."""
        try:
            result = urllib.parse.urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except Exception:
            return False
    
    def extract_title_from_url(self, url: str) -> str:
        """Extract the title from a web page by scraping the <title> tag."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            title_tag = soup.find('title')
            if title_tag and title_tag.get_text():
                title = title_tag.get_text().strip()
                title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
                title = title[:200]  # Limit length
                return title
            
            # Fallback to domain name if no title found
            parsed_url = urllib.parse.urlparse(url)
            return parsed_url.netloc
            
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch title from {url}: {str(e)}")
            try:
                parsed_url = urllib.parse.urlparse(url)
                return parsed_url.netloc
            except Exception:
                return "Unknown Title"
        except Exception as e:
            logger.error(f"Error extracting title from {url}: {str(e)}")
            return "Unknown Title"
    
    def add_url(self, url: str, title: Optional[str] = None, description: Optional[str] = None) -> Dict[str, Any]:
        """Add a new URL to the database with automatic title extraction."""
        if not self.validate_url(url):
            return {"success": False, "message": "Invalid URL format"}
        
        if not title:
            logger.info(f"Extracting title from URL: {url}")
            title = self.extract_title_from_url(url)
            logger.info(f"Extracted title: {title}")
        
        try:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Determine default per-URL snapshot flag from environment
                    default_snapshot_enabled = os.getenv('SNAPSHOT_DEFAULT_ENABLED', 'false').lower() == 'true'
                    cursor.execute(
                        """INSERT INTO urls (url, title, description, status, refresh_interval_minutes, crawl_domain, ignore_robots, snapshot_enabled)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                        (url, title, description, 'active', 1440, False, False, default_snapshot_enabled)
                    )
                    url_id = cursor.fetchone()['id']
                    conn.commit()
                    logger.info(f"Added URL: {url} with title: {title} (ID: {url_id})")
                    return {"success": True, "message": "URL added successfully", "id": str(url_id), "title": title}
        except Exception as e:
            if "duplicate key" in str(e).lower():
                return {"success": False, "message": "URL already exists"}
            logger.error(f"Error adding URL: {str(e)}")
            return {"success": False, "message": f"Database error: {str(e)}"}
    
    def get_all_urls(self) -> List[Dict[str, Any]]:
        """Retrieve all URLs from the database."""
        try:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
               SELECT id, url, title, description, status, refresh_interval_minutes, 
                   crawl_domain, ignore_robots, snapshot_enabled, snapshot_retention_days, snapshot_max_snapshots,
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
                    )
                    urls = []
                    for row in cursor.fetchall():
                        url_dict = dict(row)
                        # Convert UUID to string
                        url_dict['id'] = str(url_dict['id'])
                        
                        # Convert boolean to int for SQLite compatibility
                        url_dict['crawl_domain'] = 1 if url_dict.get('crawl_domain') else 0
                        url_dict['ignore_robots'] = 1 if url_dict.get('ignore_robots') else 0
                        
                        # Map PostgreSQL columns to SQLite equivalents
                        url_dict['added_date'] = url_dict['created_at']
                        url_dict['last_scraped'] = url_dict['last_crawled']
                        # Normalize snapshot flags
                        url_dict['snapshot_enabled'] = 1 if url_dict.get('snapshot_enabled') else 0
                        
                        # last_update_status is now a direct column
                        # No need to extract from metadata
                        
                        # Convert datetime objects to strings for JSON serialization
                        for key, value in url_dict.items():
                            if isinstance(value, datetime):
                                url_dict[key] = value.isoformat() if value else None
                        
                        urls.append(url_dict)
                    return urls
        except Exception as e:
            logger.error(f"Error retrieving URLs: {str(e)}")
            return []
    
    def delete_url(self, url_id: str) -> Dict[str, Any]:
        """Delete a URL from the database."""
        logger.info(f"Starting URL deletion process for ID: {url_id}")
        
        # Input validation
        if not url_id:
            logger.error("URL deletion failed: No URL ID provided")
            return {"success": False, "message": "Invalid URL ID: ID cannot be empty"}
        
        # Basic UUID format validation
        if not self._is_valid_uuid_format(url_id):
            logger.error(f"URL deletion failed: Invalid UUID format for ID: {url_id}")
            return {"success": False, "message": "Invalid URL ID format"}
        
        try:
            logger.debug(f"Attempting to delete URL with ID: {url_id}")
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cursor:
                    # First check if the URL exists
                    cursor.execute("SELECT url, title FROM urls WHERE id = %s", (url_id,))
                    existing_url = cursor.fetchone()
                    
                    if not existing_url:
                        logger.warning(f"URL deletion failed: No URL found with ID: {url_id}")
                        return {"success": False, "message": "URL not found"}
                    
                    logger.info(f"Found URL to delete - ID: {url_id}, URL: {existing_url['url']}, Title: {existing_url['title']}")
                    
                    # Perform the deletion
                    cursor.execute("DELETE FROM urls WHERE id = %s", (url_id,))
                    deleted_count = cursor.rowcount
                    
                    if deleted_count > 0:
                        conn.commit()
                        logger.info(f"Successfully deleted URL with ID: {url_id} (URL: {existing_url['url']})")
                        return {"success": True, "message": "URL deleted successfully"}
                    else:
                        logger.error(f"URL deletion failed: No rows affected for ID: {url_id}")
                        return {"success": False, "message": "URL not found"}
                        
        except Exception as e:
            logger.error(f"Database error during URL deletion for ID {url_id}: {str(e)}", exc_info=True)
            return {"success": False, "message": f"Database error: {str(e)}"}

    def get_pages_for_parent(self, parent_url_id: str) -> List[str]:
        """Get all pages for a parent URL (not implemented in current PostgreSQL schema)."""
        logger.debug(f"Getting pages for parent URL ID: {parent_url_id}")
        # The current PostgreSQL schema doesn't have url_pages table
        # Return empty list for now
        return []

    def delete_pages_for_parent(self, parent_url_id: str) -> None:
        """Delete all pages for a parent URL (not implemented in current PostgreSQL schema)."""
        logger.debug(f"Deleting pages for parent URL ID: {parent_url_id}")
        # The current PostgreSQL schema doesn't have url_pages table
        # No-op for now
        pass
    
    def get_url_count(self) -> int:
        """Get the total number of active URLs."""
        try:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) as count FROM urls WHERE status = 'active'")
                    return cursor.fetchone()['count']
        except Exception as e:
            logger.error(f"Error getting URL count: {str(e)}")
            return 0

    def get_url_by_id(self, url_id: Union[str, int]) -> Optional[Dict[str, Any]]:
        """Get URL by ID."""
        logger.debug(f"Getting URL by ID: {url_id}")
        
        # Input validation
        if not url_id:
            logger.error("get_url_by_id failed: No URL ID provided")
            return None
            
        # Convert to string for UUID handling
        url_id_str = str(url_id)
        
        # Basic UUID format validation
        if not self._is_valid_uuid_format(url_id_str):
            logger.error(f"get_url_by_id failed: Invalid UUID format for ID: {url_id_str}")
            return None
            
        try:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT * FROM urls WHERE id = %s", (url_id_str,))
                    row = cursor.fetchone()
                    if row:
                        url_dict = dict(row)
                        url_dict['id'] = str(url_dict['id'])
                        
                        # Normalize booleans for UI compatibility
                        url_dict['crawl_domain'] = 1 if url_dict.get('crawl_domain') else 0
                        url_dict['ignore_robots'] = 1 if url_dict.get('ignore_robots') else 0
                        url_dict['snapshot_enabled'] = 1 if url_dict.get('snapshot_enabled') else 0
                        
                        # Map PostgreSQL columns to SQLite equivalents
                        url_dict['added_date'] = url_dict['created_at']
                        url_dict['last_scraped'] = url_dict['last_crawled']

                        # last_update_status is now a direct column
                        # No need to extract from metadata
                        
                        # Convert datetime objects to strings
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

    def update_url_metadata(self, url_id: str, title: Optional[str], description: Optional[str], 
                           refresh_interval_minutes: Optional[int], crawl_domain: Optional[int], 
                           ignore_robots: Optional[int], snapshot_enabled: Optional[int] = None,
                           snapshot_retention_days: Optional[int] = None, snapshot_max_snapshots: Optional[int] = None) -> Dict[str, Any]:
        """Update URL metadata."""
        try:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Check if URL exists
                    cursor.execute("SELECT id FROM urls WHERE id = %s", (str(url_id),))
                    row = cursor.fetchone()
                    if not row:
                        return {"success": False, "message": "URL not found"}
                    
                    # Update the record with explicit columns
                    cursor.execute(
                        """UPDATE urls SET 
                           title = %s, 
                           description = %s, 
                           refresh_interval_minutes = %s, 
                           crawl_domain = %s, 
                           ignore_robots = %s,
                           snapshot_enabled = COALESCE(%s, snapshot_enabled),
                           snapshot_retention_days = %s,
                           snapshot_max_snapshots = %s,
                           updated_at = NOW() 
                           WHERE id = %s""",
                        (
                            title, description, refresh_interval_minutes,
                            bool(crawl_domain) if crawl_domain is not None else None,
                            bool(ignore_robots) if ignore_robots is not None else None,
                            bool(snapshot_enabled) if snapshot_enabled is not None else None,
                            snapshot_retention_days, snapshot_max_snapshots,
                            str(url_id)
                        )
                    )
                    conn.commit()
                    return {"success": cursor.rowcount > 0}
        except Exception as e:
            logger.error(f"Error updating URL metadata: {e}")
            return {"success": False, "message": str(e)}

    def mark_scraped(self, url_id: str, refresh_interval_minutes: Optional[int]) -> None:
        """Mark URL as scraped."""
        try:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "UPDATE urls SET last_crawled = NOW() WHERE id = %s",
                        (str(url_id),)
                    )
                    conn.commit()
        except Exception as e:
            logger.error(f"Error marking URL scraped: {e}")

    def set_refreshing(self, url_id: str, refreshing: bool) -> None:
        """Set or clear the refreshing flag."""
        try:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cursor:
                    if refreshing:
                        cursor.execute(
                            "UPDATE urls SET is_refreshing = %s, last_refresh_started = NOW() WHERE id = %s",
                            (refreshing, str(url_id))
                        )
                    else:
                        cursor.execute(
                            "UPDATE urls SET is_refreshing = %s WHERE id = %s",
                            (refreshing, str(url_id))
                        )
                    conn.commit()
        except Exception as e:
            logger.error(f"Error updating refreshing flag: {e}")

    def update_url_hash_status(self, url_id: str, content_hash: Optional[str], status: str) -> None:
        """Update URL content hash and status."""
        try:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "UPDATE urls SET last_content_hash = %s, last_update_status = %s, last_crawled = NOW() WHERE id = %s",
                        (content_hash, status, str(url_id))
                    )
                    conn.commit()
        except Exception as e:
            logger.error(f"Error updating URL hash/status: {e}")

    def get_page_hash(self, page_url: str) -> Optional[str]:
        """Get content hash for a page (not implemented in current PostgreSQL schema)."""
        # The current PostgreSQL schema doesn't have url_pages table
        return None

    def set_page_hash(self, parent_url_id: str, page_url: str, content_hash: str) -> None:
        """Set content hash for a page (not implemented in current PostgreSQL schema)."""
        # The current PostgreSQL schema doesn't have url_pages table
        pass

    def get_due_urls(self) -> List[Dict[str, Any]]:
        """Get URLs that are due for refresh."""
        try:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT * FROM urls
                        WHERE status = 'active'
                            AND refresh_interval_minutes IS NOT NULL
                            AND refresh_interval_minutes > 0
                            AND (is_refreshing IS NOT TRUE OR is_refreshing IS NULL)
                            AND (
                                last_crawled IS NULL OR
                                last_crawled + INTERVAL '1 minute' * refresh_interval_minutes <= NOW()
                            )
                        """
                    )
                    rows = []
                    for row in cursor.fetchall():
                        url_dict = dict(row)
                        url_dict['id'] = str(url_dict['id'])
                        
                        # Convert boolean to int for SQLite compatibility
                        url_dict['crawl_domain'] = 1 if url_dict.get('crawl_domain') else 0
                        url_dict['ignore_robots'] = 1 if url_dict.get('ignore_robots') else 0
                        
                        # Map PostgreSQL columns to SQLite equivalents
                        url_dict['added_date'] = url_dict['created_at']
                        url_dict['last_scraped'] = url_dict['last_crawled']
                        
                        # Convert datetime objects to strings
                        for key, value in url_dict.items():
                            if isinstance(value, datetime):
                                url_dict[key] = value.isoformat() if value else None
                        
                        rows.append(url_dict)
                    
                    if not rows:
                        logger.debug("Scheduler: no due URLs this cycle")
                    return rows
        except Exception as e:
            logger.error(f"Error fetching due URLs: {e}")
            return []

    def check_connection(self) -> Dict[str, Any]:
        """Check PostgreSQL connection health."""
        try:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT version()")
                    ver_row = cursor.fetchone()
                    ver = ver_row['version'] if ver_row else None
            return {"connected": True, "version": ver}
        except Exception as e:
            return {"connected": False, "error": str(e)}
    
    def get_url_metadata_stats(self) -> Dict[str, Any]:
        """Get aggregated URL metadata statistics from PostgreSQL."""
        url_meta = {
            'total': 0,
            'active': 0,
            'crawl_on': 0,
            'robots_ignored': 0,
            'scraped': 0,
            'never_scraped': 0,
            'due_now': 0,
        }
        
        try:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Start with simple count
                    cursor.execute("SELECT COUNT(*) as count FROM urls")
                    result = cursor.fetchone()
                    if result:
                        url_meta['total'] = int(result['count'] or 0)
                    
                    # Only try more complex queries if we have URLs
                    if url_meta['total'] > 0:
                        try:
                            # Get status counts
                            cursor.execute("""
                                SELECT
                                    SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active,
                                    SUM(CASE WHEN last_crawled IS NOT NULL THEN 1 ELSE 0 END) AS scraped,
                                    SUM(CASE WHEN last_crawled IS NULL THEN 1 ELSE 0 END) AS never_scraped
                                FROM urls
                            """)
                            row = cursor.fetchone()
                            if row:
                                url_meta['active'] = int(row['active'] or 0)
                                url_meta['scraped'] = int(row['scraped'] or 0)
                                url_meta['never_scraped'] = int(row['never_scraped'] or 0)
                        except Exception as e:
                            logger.warning(f"Failed to get URL status counts: {e}")
                        
                        try:
                            # Get metadata-based counts using explicit columns
                            cursor.execute("""
                                SELECT
                                    SUM(CASE WHEN crawl_domain = true THEN 1 ELSE 0 END) AS crawl_on,
                                    SUM(CASE WHEN ignore_robots = true THEN 1 ELSE 0 END) AS robots_ignored
                                FROM urls
                            """)
                            row = cursor.fetchone()
                            if row:
                                url_meta['crawl_on'] = int(row['crawl_on'] or 0)
                                url_meta['robots_ignored'] = int(row['robots_ignored'] or 0)
                        except Exception as e:
                            logger.warning(f"Failed to get URL metadata counts: {e}")
                        
                        try:
                            # Get due_now count using explicit column
                            cursor.execute("""
                                SELECT COUNT(*) as count FROM urls 
                                WHERE refresh_interval_minutes IS NOT NULL 
                                AND refresh_interval_minutes > 0
                            """)
                            result = cursor.fetchone()
                            if result:
                                url_meta['due_now'] = int(result['count'] or 0)
                        except Exception as e:
                            logger.warning(f"Failed to get due URLs count: {e}")
                    
                    return url_meta
        except Exception as e:
            logger.error(f"Failed to get URL metadata stats: {e}")
            return url_meta
