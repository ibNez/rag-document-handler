"""
URL Source Manager
Business logic for URL crawling and content ingestion.

This module handles URL crawling business logic, delegating
data operations to URLDataManager following the established pattern.
"""

import logging
import re
import urllib.parse
import requests
import uuid
from typing import Dict, List, Optional, Any, Union
from bs4 import BeautifulSoup

from rag_manager.data.url_data import URLDataManager

logger = logging.getLogger(__name__)


class URLSourceManager:
    """
    URL source manager for crawling and content ingestion.
    
    Handles business logic for URL operations including validation,
    title extraction, and crawling orchestration.
    """
    
    def __init__(self, postgres_manager=None, milvus_manager=None):
        """
        Initialize URL source manager.
        
        Args:
            postgres_manager: PostgreSQL connection manager
            milvus_manager: Optional Milvus vector store manager
        """
        self.url_data = URLDataManager(postgres_manager, milvus_manager)
        logger.info("URLSourceManager initialized")
    
    def _is_valid_uuid_format(self, uuid_string: str) -> bool:
        """
        Validate if a string is a valid UUID format.
        
        Args:
            uuid_string: String to validate
            
        Returns:
            True if valid UUID format, False otherwise
        """
        try:
            uuid_obj = uuid.UUID(uuid_string)
            return str(uuid_obj) == uuid_string
        except (ValueError, TypeError):
            return False
    
    def validate_url(self, url: str) -> bool:
        """
        Validate if the provided string is a valid URL.
        
        Args:
            url: URL string to validate
            
        Returns:
            True if valid URL, False otherwise
        """
        try:
            result = urllib.parse.urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except Exception:
            return False
    
    def extract_title_from_url(self, url: str) -> str:
        """
        Extract the title from a web page by scraping the <title> tag.
        
        Args:
            url: URL to extract title from
            
        Returns:
            Extracted title or fallback title
        """
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
                title = re.sub(r'\\s+', ' ', title)
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
    
    def add_url(self, url: str, title: Optional[str] = None, description: Optional[str] = None,
                refresh_interval_minutes: Optional[int] = None, crawl_domain: bool = False,
                ignore_robots: bool = False, snapshot_retention_days: int = 0,
                snapshot_max_snapshots: int = 0, parent_url_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a new URL with validation and title extraction.
        
        Args:
            url: The URL to add
            title: Optional title for the URL
            description: Optional description
            refresh_interval_minutes: How often to refresh
            crawl_domain: Whether to crawl the entire domain
            ignore_robots: Whether to ignore robots.txt
            snapshot_retention_days: How long to keep snapshots
            snapshot_max_snapshots: Maximum number of snapshots to keep
            parent_url_id: Parent URL ID if this is a child URL
            
        Returns:
            Dictionary with operation result and URL data
        """
        # Validate URL format
        if not self.validate_url(url):
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
            if not self._is_valid_uuid_format(parent_url_id):
                return {
                    'success': False,
                    'error': 'Invalid parent URL ID format',
                    'url_id': None,
                    'url_data': None
                }
            
            # Check if parent exists
            parent_data = self.url_data.get_url_by_id(parent_url_id)
            if not parent_data:
                return {
                    'success': False,
                    'error': 'Parent URL not found',
                    'url_id': None,
                    'url_data': None
                }
        
        # Delegate to data layer
        return self.url_data.add_url(
            url=url,
            title=title,
            description=description,
            refresh_interval_minutes=refresh_interval_minutes,
            crawl_domain=crawl_domain,
            ignore_robots=ignore_robots,
            snapshot_retention_days=snapshot_retention_days,
            snapshot_max_snapshots=snapshot_max_snapshots,
            parent_url_id=parent_url_id
        )
    
    def get_child_urls(self, parent_url_id: str) -> List[Dict[str, Any]]:
        """
        Get child URLs for a parent URL.
        
        Args:
            parent_url_id: Parent URL ID
            
        Returns:
            List of child URL dictionaries
        """
        return self.url_data.get_child_urls(parent_url_id)
    
    def get_child_url_stats(self, parent_url_id: str) -> Dict[str, Any]:
        """
        Get statistics for child URLs.
        
        Args:
            parent_url_id: Parent URL ID
            
        Returns:
            Dictionary with child URL statistics
        """
        try:
            child_urls = self.url_data.get_child_urls(parent_url_id)
            
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
                           refresh_interval_minutes: Optional[int], crawl_domain: Optional[bool],
                           ignore_robots: Optional[bool], snapshot_retention_days: Optional[int],
                           snapshot_max_snapshots: Optional[int]) -> Dict[str, Any]:
        """
        Update URL metadata with validation.
        
        Args:
            url_id: URL ID to update
            title: New title
            description: New description
            refresh_interval_minutes: New refresh interval
            crawl_domain: Whether to crawl domain
            ignore_robots: Whether to ignore robots.txt
            snapshot_retention_days: Snapshot retention period
            snapshot_max_snapshots: Maximum snapshots to keep
            
        Returns:
            Dictionary with operation result
        """
        # Validate URL ID format
        if not self._is_valid_uuid_format(url_id):
            return {
                'success': False,
                'error': 'Invalid URL ID format'
            }
        
        # Validate refresh interval if provided
        if refresh_interval_minutes is not None and refresh_interval_minutes < 1:
            return {
                'success': False,
                'error': 'Refresh interval must be at least 1 minute'
            }
        
        # Delegate to data layer
        return self.url_data.update_url_metadata(
            url_id, title, description, refresh_interval_minutes,
            crawl_domain, ignore_robots, snapshot_retention_days,
            snapshot_max_snapshots
        )
    
    def delete_url(self, url_id: str) -> Dict[str, Any]:
        """
        Delete URL with validation and cleanup.
        
        Args:
            url_id: URL ID to delete
            
        Returns:
            Dictionary with operation result
        """
        # Validate URL ID format
        if not self._is_valid_uuid_format(url_id):
            return {
                'success': False,
                'error': 'Invalid URL ID format'
            }
        
        # Check if URL has children
        child_urls = self.url_data.get_child_urls(url_id)
        if child_urls:
            return {
                'success': False,
                'error': f'Cannot delete URL with {len(child_urls)} child URLs. Delete children first.'
            }
        
        # Delegate to data layer
        return self.url_data.delete_url(url_id)
    
    def mark_scraped(self, url_id: str, refresh_interval_minutes: Optional[int] = None) -> None:
        """
        Mark URL as scraped with business logic validation.
        
        Args:
            url_id: URL ID to mark as scraped
            refresh_interval_minutes: Refresh interval in minutes
        """
        self.url_data.mark_scraped(url_id, refresh_interval_minutes)
    
    def set_refreshing(self, url_id: str, refreshing: bool) -> None:
        """
        Set URL refreshing status.
        
        Args:
            url_id: URL ID to update
            refreshing: Whether URL is currently being refreshed
        """
        self.url_data.set_refreshing(url_id, refreshing)
    
    def update_url_hash_status(self, url_id: str, content_hash: Optional[str], status: str) -> None:
        """
        Update URL content hash and status.
        
        Args:
            url_id: URL ID to update
            content_hash: New content hash
            status: New status
        """
        self.url_data.update_url_hash_status(url_id, content_hash, status)
    
    def get_due_urls(self) -> List[Dict[str, Any]]:
        """
        Get URLs that are due for refresh with business logic filtering.
        
        Returns:
            List of URL dictionaries that need refreshing
        """
        return self.url_data.get_due_urls()
    
    def update_robots_setting(self, url_id: str, ignore_robots: bool) -> Dict[str, Any]:
        """
        Update robots.txt setting for URL.
        
        Args:
            url_id: URL ID to update
            ignore_robots: Whether to ignore robots.txt
            
        Returns:
            Dictionary with operation result
        """
        return self.url_data.update_robots_setting(url_id, ignore_robots)
    
    def get_all_urls_including_children(self) -> List[Dict[str, Any]]:
        """
        Get all URLs including children with hierarchy information.
        
        Returns:
            List of all URL dictionaries with parent-child relationships
        """
        all_urls = self.url_data.get_all_urls()
        
        # Add hierarchy information
        for url in all_urls:
            if url.get('parent_url_id'):
                # Mark as child
                url['is_child'] = True
                url['hierarchy_level'] = 1  # Could be extended for deeper nesting
            else:
                # Mark as parent
                url['is_child'] = False
                url['hierarchy_level'] = 0
        
        return all_urls
    
    def get_pages_for_parent(self, parent_url_id: str) -> List[str]:
        """
        Get pages for a parent URL.
        
        Args:
            parent_url_id: Parent URL ID
            
        Returns:
            List of page URLs
        """
        # This would need additional implementation for page tracking
        # For now, return child URLs as pages
        child_urls = self.url_data.get_child_urls(parent_url_id)
        return [child['url'] for child in child_urls if child.get('url')]
    
    def delete_pages_for_parent(self, parent_url_id: str) -> None:
        """
        Delete pages for a parent URL.
        
        Args:
            parent_url_id: Parent URL ID
        """
        # Delete all child URLs for the parent
        child_urls = self.url_data.get_child_urls(parent_url_id)
        for child in child_urls:
            self.url_data.delete_url(child['id'])
    
    def get_page_hash(self, page_url: str) -> Optional[str]:
        """
        Get stored hash for a page URL.
        
        Args:
            page_url: Page URL to look up
            
        Returns:
            Hash string or None if not found
        """
        return self.url_data.get_page_hash(page_url)
    
    def check_connection(self) -> Dict[str, Any]:
        """
        Check database connection health.
        
        Returns:
            Dictionary with connection status
        """
        try:
            # Test connection by getting URL count
            count = self.url_data.get_url_count()
            
            return {
                'success': True,
                'status': 'healthy',
                'url_count': count,
                'message': 'Database connection is working'
            }
            
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return {
                'success': False,
                'status': 'error',
                'error': str(e),
                'message': 'Database connection failed'
            }
