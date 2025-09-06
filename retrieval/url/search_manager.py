"""
URL Search Manager
Search and retrieval operations for URL content.

This module handles URL search business logic, delegating
data operations to URLDataManager following the established pattern.
"""

import logging
from typing import Dict, List, Optional, Any

from rag_manager.data.url_data import URLDataManager

logger = logging.getLogger(__name__)


class URLSearchManager:
    """
    URL search manager for content retrieval operations.
    
    Handles search and filtering logic for URL content,
    delegating data operations to URLDataManager.
    """
    
    def __init__(self, postgres_manager=None, milvus_manager=None):
        """
        Initialize URL search manager.
        
        Args:
            postgres_manager: PostgreSQL connection manager
            milvus_manager: Optional Milvus vector store manager
        """
        self.url_data = URLDataManager(postgres_manager, milvus_manager)
        logger.info("URLSearchManager initialized")
    
    def search_urls(self, query: str, filters: Optional[Dict] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search URLs by title, description, or content.
        
        Args:
            query: Search query string
            filters: Optional filters to apply
            limit: Maximum number of results to return
            
        Returns:
            List of matching URL dictionaries
        """
        try:
            # Get all URLs first (we can add FTS search later)
            all_urls = self.url_data.get_all_urls()
            
            # Simple text search in title and description
            query_lower = query.lower()
            matching_urls = []
            
            for url in all_urls:
                # Search in title
                title_match = url.get('title', '').lower().find(query_lower) != -1
                
                # Search in description
                desc_match = url.get('description', '').lower().find(query_lower) != -1
                
                # Search in URL itself
                url_match = url.get('url', '').lower().find(query_lower) != -1
                
                if title_match or desc_match or url_match:
                    # Add relevance score (simple implementation)
                    score = 0
                    if title_match:
                        score += 3
                    if desc_match:
                        score += 2
                    if url_match:
                        score += 1
                    
                    url['relevance_score'] = score
                    matching_urls.append(url)
            
            # Apply filters if provided
            if filters:
                filtered_urls = []
                for url in matching_urls:
                    if self._apply_filters(url, filters):
                        filtered_urls.append(url)
                matching_urls = filtered_urls
            
            # Sort by relevance score and limit results
            matching_urls.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            return matching_urls[:limit]
            
        except Exception as e:
            logger.error(f"Failed to search URLs: {e}")
            return []
    
    def _apply_filters(self, url: Dict[str, Any], filters: Dict) -> bool:
        """
        Apply filters to a URL record.
        
        Args:
            url: URL dictionary to check
            filters: Filters to apply
            
        Returns:
            True if URL matches all filters, False otherwise
        """
        try:
            # Status filter
            if 'status' in filters:
                if url.get('status') != filters['status']:
                    return False
            
            # Crawl domain filter
            if 'crawl_domain' in filters:
                if url.get('crawl_domain') != filters['crawl_domain']:
                    return False
            
            # Parent/child filter
            if 'is_parent' in filters:
                has_children = url.get('child_url_count', 0) > 0
                if filters['is_parent'] and not has_children:
                    return False
                if not filters['is_parent'] and has_children:
                    return False
            
            # Date range filters could be added here
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return False
    
    def get_url_with_children(self, url_id: str) -> Optional[Dict[str, Any]]:
        """
        Get URL with its child URLs included.
        
        Args:
            url_id: URL ID to retrieve
            
        Returns:
            URL dictionary with children included or None if not found
        """
        try:
            url = self.url_data.get_url_by_id(url_id)
            if not url:
                return None
            
            # Get child URLs
            children = self.url_data.get_child_urls(url_id)
            url['children'] = children
            url['child_count'] = len(children)
            
            return url
            
        except Exception as e:
            logger.error(f"Failed to get URL with children for {url_id}: {e}")
            return None
    
    def get_url_hierarchy(self, url_id: str) -> Dict[str, Any]:
        """
        Get complete URL hierarchy starting from a URL.
        
        Args:
            url_id: URL ID to start from
            
        Returns:
            Dictionary with complete hierarchy information
        """
        try:
            url = self.url_data.get_url_by_id(url_id)
            if not url:
                return {
                    'success': False,
                    'error': 'URL not found'
                }
            
            # Build hierarchy
            hierarchy = {
                'root': url,
                'children': [],
                'total_descendants': 0
            }
            
            # Get direct children
            children = self.url_data.get_child_urls(url_id)
            hierarchy['children'] = children
            hierarchy['total_descendants'] = len(children)
            
            # For now, we support only one level of hierarchy
            # This could be extended for deeper nesting
            
            return {
                'success': True,
                'hierarchy': hierarchy
            }
            
        except Exception as e:
            logger.error(f"Failed to get URL hierarchy for {url_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_urls_by_status(self, status: str) -> List[Dict[str, Any]]:
        """
        Get URLs filtered by status.
        
        Args:
            status: Status to filter by
            
        Returns:
            List of URLs with matching status
        """
        try:
            all_urls = self.url_data.get_all_urls()
            return [url for url in all_urls if url.get('status') == status]
            
        except Exception as e:
            logger.error(f"Failed to get URLs by status {status}: {e}")
            return []
    
    def get_urls_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """
        Get URLs filtered by domain.
        
        Args:
            domain: Domain to filter by
            
        Returns:
            List of URLs from the specified domain
        """
        try:
            all_urls = self.url_data.get_all_urls()
            matching_urls = []
            
            for url in all_urls:
                url_string = url.get('url', '')
                if domain.lower() in url_string.lower():
                    matching_urls.append(url)
            
            return matching_urls
            
        except Exception as e:
            logger.error(f"Failed to get URLs by domain {domain}: {e}")
            return []
    
    def get_recent_urls(self, days: int = 7, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recently added URLs.
        
        Args:
            days: Number of days to look back
            limit: Maximum number of results
            
        Returns:
            List of recent URLs
        """
        try:
            all_urls = self.url_data.get_all_urls()
            
            # Sort by creation date (most recent first)
            sorted_urls = sorted(
                all_urls,
                key=lambda x: x.get('created_at', ''),
                reverse=True
            )
            
            return sorted_urls[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get recent URLs: {e}")
            return []
    
    def get_urls_due_for_refresh(self) -> List[Dict[str, Any]]:
        """
        Get URLs that are due for refresh with additional metadata.
        
        Returns:
            List of URLs due for refresh with timing information
        """
        try:
            due_urls = self.url_data.get_due_urls()
            
            # Add timing information
            for url in due_urls:
                last_crawled = url.get('last_crawled')
                refresh_interval = url.get('refresh_interval_minutes', 1440)
                
                if last_crawled:
                    # Calculate how overdue this URL is
                    # This would need proper datetime handling in a real implementation
                    url['refresh_status'] = 'overdue'
                else:
                    url['refresh_status'] = 'never_crawled'
                
                url['priority'] = 'high' if url['refresh_status'] == 'never_crawled' else 'normal'
            
            return due_urls
            
        except Exception as e:
            logger.error(f"Failed to get URLs due for refresh: {e}")
            return []
    
    def get_url_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive URL statistics.
        
        Returns:
            Dictionary with URL statistics
        """
        try:
            all_urls = self.url_data.get_all_urls()
            total_count = len(all_urls)
            
            # Status breakdown
            status_counts = {}
            for url in all_urls:
                status = url.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Parent/child breakdown
            parent_count = sum(1 for url in all_urls if url.get('parent_url_id') is None)
            child_count = sum(1 for url in all_urls if url.get('parent_url_id') is not None)
            
            # Domain breakdown (top 10)
            domain_counts = {}
            for url in all_urls:
                url_string = url.get('url', '')
                try:
                    import urllib.parse
                    parsed = urllib.parse.urlparse(url_string)
                    domain = parsed.netloc
                    if domain:
                        domain_counts[domain] = domain_counts.get(domain, 0) + 1
                except:
                    pass
            
            top_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                'total_urls': total_count,
                'parent_urls': parent_count,
                'child_urls': child_count,
                'status_breakdown': status_counts,
                'top_domains': top_domains,
                'crawl_settings': {
                    'crawl_domain_enabled': sum(1 for url in all_urls if url.get('crawl_domain')),
                    'robots_ignored': sum(1 for url in all_urls if url.get('ignore_robots'))
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get URL statistics: {e}")
            return {
                'total_urls': 0,
                'parent_urls': 0,
                'child_urls': 0,
                'status_breakdown': {},
                'top_domains': [],
                'crawl_settings': {
                    'crawl_domain_enabled': 0,
                    'robots_ignored': 0
                }
            }
