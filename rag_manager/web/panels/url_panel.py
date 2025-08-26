"""
URL Panel Statistics Provider
Following DEVELOPMENT_RULES.md for all development requirements

This module handles all statistics for the URL panel on the status dashboard.
Centralizes URL management statistics, scraping counts, and scheduling metrics.
"""

import logging
from typing import Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class URLPanelStats:
    """Statistics provider for the URL panel."""
    
    def __init__(self, rag_manager):
        """Initialize with reference to the main RAG manager."""
        self.rag_manager = rag_manager
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get all URL statistics for the URL panel.
        
        Returns:
            Dictionary containing URL panel statistics
        """
        try:
            if not self.rag_manager.url_manager:
                return self._empty_stats()
                
            # Get all URLs for analysis
            urls = self.rag_manager.url_manager.get_all_urls()
            
            # Calculate basic counts
            total_urls = len(urls)
            scraped = sum(1 for url in urls if url.get('last_scraped'))
            never_scraped = total_urls - scraped
            due_now = sum(1 for url in urls if self._is_url_due_for_scraping(url))
            
            # Additional URL metrics
            active_urls = sum(1 for url in urls if url.get('refresh_interval_minutes', 0) > 0)
            robots_ignored = sum(1 for url in urls if url.get('ignore_robots_txt', False))
            crawl_on = sum(1 for url in urls if url.get('refresh_interval_minutes', 0) > 0)
            
            return {
                'total': total_urls,
                'active': active_urls,
                'crawl_on': crawl_on,
                'robots_ignored': robots_ignored,
                'scraped': scraped,
                'never_scraped': never_scraped,
                'due_now': due_now
            }
            
        except Exception as e:
            logger.error(f"Failed to get URL panel stats: {e}")
            return self._empty_stats()
    
    def _is_url_due_for_scraping(self, url: Dict[str, Any]) -> bool:
        """Check if a URL is due for scraping based on refresh interval."""
        try:
            refresh_interval = url.get('refresh_interval_minutes', 0)
            
            # If no refresh interval, not due
            if not refresh_interval or refresh_interval <= 0:
                return False
                
            last_scraped = url.get('last_scraped')
            
            # If never scraped but has refresh interval, it's due
            if not last_scraped:
                return True
                
            # Parse last scraped time and check if due
            if isinstance(last_scraped, str):
                last_scraped_dt = datetime.fromisoformat(last_scraped.replace('Z', '+00:00'))
            else:
                last_scraped_dt = last_scraped
                
            interval = timedelta(minutes=refresh_interval)
            next_scrape = last_scraped_dt + interval
            
            return datetime.now() >= next_scrape
            
        except Exception as e:
            logger.debug(f"Error checking if URL is due: {e}")
            return False
    
    def _empty_stats(self) -> Dict[str, Any]:
        """Return empty URL panel stats."""
        return {
            'total': 0,
            'active': 0,
            'crawl_on': 0,
            'robots_ignored': 0,
            'scraped': 0,
            'never_scraped': 0,
            'due_now': 0
        }
