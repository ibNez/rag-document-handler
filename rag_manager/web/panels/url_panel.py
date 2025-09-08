"""
URL Panel Statistics Provider
Following DEVELOPMENT_RULES.md for all development requirements

This module handles all statistics for the URL panel on the status dashboard.
Centralizes URL management statistics, scraping counts, and scheduling metrics.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timedelta
from urllib.parse import urlparse

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
            Dictionary containing URL panel statistics including pages processed 
            and snapshot disk usage
        """
        try:
            if not self.rag_manager.url_manager:
                return self._empty_stats()
                
            # Get all URLs for analysis
            urls = self.rag_manager.url_manager.url_data.get_all_urls()
            
            # Calculate basic counts
            total_urls = len(urls)
            due_now = sum(1 for url in urls if self._is_url_due_for_scraping(url))
            
            # Calculate threshold status for due_now (Warning: >10, Critical: >50)
            due_now_status = self._get_due_now_threshold_status(due_now)
            
            # Additional URL metrics
            robots_ignored = sum(1 for url in urls if url.get('ignore_robots', 0) == 1)
            robots_ignored_status = self._get_robots_ignored_threshold_status(robots_ignored)
            crawl_on = sum(1 for url in urls if url.get('crawl_domain', 0) == 1)
            
            # Get sub_urls count directly from database since get_all_urls() only returns parent URLs
            sub_urls = self._get_sub_urls_count()
            
            # Calculate pages processed for all URLs (count chunks/documents in vector database)
            total_pages_processed = self._calculate_total_pages_processed(urls)
            
            # Calculate total snapshot disk usage
            snapshot_size_human = self._calculate_snapshot_disk_usage()
            
            return {
                'total': total_urls,
                'sub_urls': sub_urls,
                'crawl_on': crawl_on,
                'robots_ignored': robots_ignored,
                'robots_ignored_status': robots_ignored_status,
                'due_now': due_now,
                'due_now_status': due_now_status,
                'pages_processed': total_pages_processed,
                'snapshot_disk_usage': snapshot_size_human
            }
            
        except Exception as e:
            logger.error(f"Failed to get URL panel stats: {e}")
            return self._empty_stats()
    
    def _calculate_total_pages_processed(self, urls: list) -> int:
        """
        Calculate total pages processed across all URLs by counting chunks/documents 
        in the vector database for each URL.
        
        Args:
            urls: List of URL dictionaries
            
        Returns:
            Total number of pages/chunks processed
        """
        try:
            if not self.rag_manager.milvus_manager:
                return 0
                
            total_pages = 0
            
            for url in urls:
                url_string = url.get('url', '')
                url_id = url.get('url_id', '')
                
                if url_string:
                    # Count chunks for this URL in Milvus
                    try:
                        chunk_count = self.rag_manager.milvus_manager.get_chunk_count_for_url(url_string, url_id)
                        total_pages += chunk_count
                    except Exception as e:
                        logger.debug(f"Failed to get chunk count for URL {url_string}: {e}")
                        continue
                        
            return total_pages
            
        except Exception as e:
            logger.error(f"Failed to calculate total pages processed: {e}")
            return 0
    
    def _calculate_snapshot_disk_usage(self) -> str:
        """
        Calculate total disk space usage of the snapshot folder in human-readable format.
        
        Returns:
            Human-readable string of total snapshot disk usage (e.g., "15.3 MB")
        """
        try:
            # Get snapshot directory from config
            snapshot_dir = getattr(self.rag_manager.config, 'SNAPSHOT_DIR', 
                                 os.path.join('uploaded', 'snapshots'))
            snapshot_path = Path(snapshot_dir)
            
            if not snapshot_path.exists():
                return "0 B"
            
            total_size = 0
            
            # Walk through all files in snapshot directory
            for root, dirs, files in os.walk(snapshot_path):
                for file in files:
                    file_path = Path(root) / file
                    try:
                        total_size += file_path.stat().st_size
                    except (OSError, IOError):
                        # Skip files that can't be accessed
                        continue
            
            # Convert to human-readable format
            return self._format_bytes(total_size)
            
        except Exception as e:
            logger.error(f"Failed to calculate snapshot disk usage: {e}")
            return "0 B"
    
    def _format_bytes(self, bytes_count: int) -> str:
        """
        Convert bytes to human-readable format.
        
        Args:
            bytes_count: Number of bytes
            
        Returns:
            Human-readable string (e.g., "1.5 MB", "3.2 GB")
        """
        if bytes_count == 0:
            return "0 B"
        
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        unit_index = 0
        size = float(bytes_count)
        
        while size >= 1024.0 and unit_index < len(units) - 1:
            size /= 1024.0
            unit_index += 1
        
        # Format with appropriate decimal places
        if unit_index == 0:  # Bytes
            return f"{int(size)} {units[unit_index]}"
        else:
            return f"{size:.1f} {units[unit_index]}"
    
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
    
    def _get_sub_urls_count(self) -> int:
        """
        Get the count of sub-URLs (child URLs) directly from the database.
        
        Returns:
            Number of URLs that have a parent_url_id (i.e., child URLs discovered via domain crawling)
        """
        try:
            if not self.rag_manager.url_manager:
                return 0
                
            # Use URLSourceManager's url_data (URLDataManager) to access database
            url_data_manager = self.rag_manager.url_manager.url_data
            if not url_data_manager:
                return 0
                
            query = """
                SELECT COUNT(*) as count 
                FROM urls 
                WHERE status = 'active' AND parent_url_id IS NOT NULL
            """
            result = url_data_manager.execute_query(query, fetch_one=True)
            return int(result['count'] or 0) if result else 0
                    
        except Exception as e:
            logger.error(f"Failed to get sub-URLs count: {e}")
            return 0
    
    def _get_due_now_threshold_status(self, due_now: int) -> str:
        """
        Get threshold status for URLs due now.
        
        Args:
            due_now: Number of URLs due for processing
            
        Returns:
            Status string: 'success', 'warning', 'critical'
        """
        # Based on documentation: Warning: >10, Critical: >50
        if due_now >= 50:
            return 'critical'
        elif due_now >= 10:
            return 'warning'
        else:
            return 'success'
    
    def _get_robots_ignored_threshold_status(self, robots_ignored: int) -> str:
        """
        Get threshold status for robots ignored count.
        
        Args:
            robots_ignored: Number of URLs ignoring robots.txt
            
        Returns:
            Status string: 'success', 'warning', 'critical'  
        """
        # High count indicates risk surface - warn if >5, critical if >20
        if robots_ignored >= 20:
            return 'critical'
        elif robots_ignored >= 5:
            return 'warning'
        else:
            return 'success'
    
    def _empty_stats(self) -> Dict[str, Any]:
        """Return empty URL panel stats."""
        return {
            'total': 0,
            'sub_urls': 0,
            'crawl_on': 0,
            'robots_ignored': 0,
            'robots_ignored_status': 'success',
            'scraped': 0,
            'due_now': 0,
            'due_now_status': 'success',
            'pages_processed': 0,
            'snapshot_disk_usage': '0 B'
        }
